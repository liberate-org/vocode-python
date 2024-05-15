from __future__ import annotations

import asyncio
import queue
import random
import threading
from typing import Any, Awaitable, Callable, Generic, Optional, Tuple, TypeVar, cast
import logging
import time
import typing
import requests
from io import BytesIO
import random

from vocode.streaming.action.worker import ActionsWorker

from vocode.streaming.agent.bot_sentiment_analyser import (
    BotSentimentAnalyser,
)
from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.models.actions import ActionInput
from vocode.streaming.models.events import Sender
from vocode.streaming.models.transcript import (
    Message,
    Transcript,
    TranscriptCompleteEvent,
)
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.transcriber import EndpointingConfig, TranscriberConfig
from vocode.streaming.models.synthetic_hold import SyntheticHoldConfig
from vocode.streaming.output_device.base_output_device import BaseOutputDevice
from vocode.streaming.utils import convert_wav
from vocode.streaming.utils import mp3_helper
from vocode.streaming.utils.conversation_logger_adapter import wrap_logger
from vocode.streaming.utils.events_manager import EventsManager
from vocode.streaming.utils.goodbye_model import GoodbyeModel

from vocode.streaming.models.agent import ChatGPTAgentConfig, FillerAudioConfig
from vocode.streaming.models.synthesizer import (
    SentimentConfig,
)
from vocode.streaming.constants import (
    TEXT_TO_SPEECH_CHUNK_SIZE_SECONDS,
    PER_CHUNK_ALLOWANCE_SECONDS,
    ALLOWED_IDLE_TIME,
)
from vocode.streaming.agent.base_agent import (
    AgentInput,
    AgentResponse,
    AgentResponseFillerAudio,
    AgentResponseMessage,
    AgentResponseStop,
    AgentResponseType,
    BaseAgent,
    TranscriptionAgentInput,
)
from vocode.streaming.synthesizer.base_synthesizer import (
    BaseSynthesizer,
    SynthesisResult,
    FillerAudio,
)
from vocode.streaming.utils import create_conversation_id, get_chunk_size_per_second
from vocode.streaming.transcriber.base_transcriber import (
    Transcription,
    BaseTranscriber,
)
from vocode.streaming.utils.state_manager import ConversationStateManager
from vocode.streaming.utils.worker import (
    AsyncQueueWorker,
    InterruptibleAgentResponseWorker,
    InterruptibleEvent,
    InterruptibleEventFactory,
    InterruptibleAgentResponseEvent,
    InterruptibleWorker,
)
from vocode.streaming.input_device.stream_handler import (
    AudioStreamHandler
)

OutputDeviceType = TypeVar("OutputDeviceType", bound=BaseOutputDevice)

class StreamingConversation(Generic[OutputDeviceType]):
    class QueueingInterruptibleEventFactory(InterruptibleEventFactory):
        def __init__(self, conversation: "StreamingConversation"):
            self.conversation = conversation

        def create_interruptible_event(
            self, payload: Any, is_interruptible: bool = True
        ) -> InterruptibleEvent[Any]:
            interruptible_event: InterruptibleEvent = (
                super().create_interruptible_event(payload, is_interruptible)
            )
            self.conversation.interruptible_events.put_nowait(interruptible_event)
            return interruptible_event

        def create_interruptible_agent_response_event(
            self,
            payload: Any,
            is_interruptible: bool = True,
            agent_response_tracker: Optional[asyncio.Event] = None,
        ) -> InterruptibleAgentResponseEvent:
            interruptible_event = super().create_interruptible_agent_response_event(
                payload,
                is_interruptible=is_interruptible,
                agent_response_tracker=agent_response_tracker,
            )
            self.conversation.interruptible_events.put_nowait(interruptible_event)
            return interruptible_event

    class TranscriptionsWorker(AsyncQueueWorker):
        """Processes all transcriptions: sends an interrupt if needed
        and sends final transcriptions to the output queue"""

        def __init__(
            self,
            input_queue: asyncio.Queue[Transcription],
            output_queue: asyncio.Queue[InterruptibleEvent[AgentInput]],
            conversation: "StreamingConversation",
            interruptible_event_factory: InterruptibleEventFactory,
        ):
            super().__init__(input_queue, output_queue)
            self.input_queue = input_queue
            self.output_queue = output_queue
            self.conversation = conversation
            self.interruptible_event_factory = interruptible_event_factory

        def kill_tasks_when_human_is_talking(self):
            has_task = self.conversation.synthesis_results_worker.current_task is not None
            if has_task and not self.conversation.synthesis_results_worker.current_task.done():
                self.conversation.logger.info("###### Synthesis task is running, attempting to cancel it ######")
                self.conversation.synthesis_results_worker.current_task.cancel()
                self.conversation.logger.info("###### Synthesis task is running, has been canceled ######")
            has_agent_task = self.conversation.agent_responses_worker.current_task
            if has_agent_task and not self.conversation.agent_responses_worker.current_task.done():
                self.conversation.logger.info("&&&&&&& Agent Response task is running, attempting to cancel it &&&&&&&")
                self.conversation.agent_responses_worker.current_task.cancel()
                self.conversation.logger.info("&&&&&&& Agent Response task is running, has been canceled &&&&&&&")

        async def process(self, transcription: Transcription):
            self.conversation.mark_last_action_timestamp()
            # If the message was empty (silence), we ignore it
            if transcription.message.strip() == "":
                self.conversation.logger.info("Ignoring empty transcription")
                return
            # otherwise mark the message as last time the human spoke
            else:
                self.conversation.mark_last_final_transcript_from_human()
            # special case where the human interrupts the bot
            # this is only enabled in experimental mode (though this should be relooked)
            # send interrupt and mark message as last time the human spoke
            if transcription.message.strip() == "<INTERRUPT>" and transcription.confidence == 1.0:
                if self.conversation.transcriber.get_transcriber_config().experimental:
                    # self.conversation.is_human_speaking = True
                    self.conversation.mark_last_final_transcript_from_human()
                    self.conversation.broadcast_interrupt()
                    return
                else:
                    return

            # If the human is not speaking but there is an interrupt,
            # we should send the interrupt to the agent and log that the human is speaking
            if (
                not self.conversation.is_human_speaking
                and self.conversation.is_interrupt(transcription)
            ):
                self.conversation.current_transcription_is_interrupt = (
                    self.conversation.broadcast_interrupt()
                )
                if self.conversation.current_transcription_is_interrupt:
                    self.conversation.logger.debug("sending interrupt")
                self.conversation.logger.debug("Human started speaking")

            transcription.is_interrupt = (
                self.conversation.current_transcription_is_interrupt
            )
            # self.conversation.is_human_speaking = not transcription.is_final

            # if this is the final transcript, log out data
            # create a new event and put it on the queue
            # set last human utterance time and reset `is_human_speaking` boolean
            if transcription.is_final:
                # we use getattr here to avoid the dependency cycle between VonageCall and StreamingConversation
                self.conversation.logger.debug(
                    "Got transcription: {}, confidence: {}".format(
                        transcription.message, transcription.confidence
                    )
                )                
                event = self.interruptible_event_factory.create_interruptible_event(
                    TranscriptionAgentInput(
                        transcription=transcription,
                        conversation_id=self.conversation.id,
                        vonage_uuid=getattr(self.conversation, "vonage_uuid", None),
                        twilio_sid=getattr(self.conversation, "twilio_sid", None),
                    )
                )
                self.output_queue.put_nowait(event)
                self.conversation.mark_last_final_transcript_from_human()
                # self.conversation.is_human_speaking = False
            else:
                self.kill_tasks_when_human_is_talking()
                self.conversation.broadcast_interrupt()

    class FillerAudioWorker(InterruptibleAgentResponseWorker):
        """
        - Waits for a configured number of seconds and then sends filler audio to the output
        - Exposes wait_for_filler_audio_to_finish() which the AgentResponsesWorker waits on before
          sending responses to the output queue
        """

        def __init__(
            self,
            input_queue: asyncio.Queue[InterruptibleAgentResponseEvent[FillerAudio]],
            conversation: "StreamingConversation",
        ):
            super().__init__(input_queue=input_queue)
            self.input_queue = input_queue
            self.conversation = conversation
            self.current_filler_seconds_per_chunk: Optional[int] = None
            self.filler_audio_started_event: Optional[threading.Event] = None

        async def wait_for_filler_audio_to_finish(self):
            if (
                self.filler_audio_started_event is None
                or not self.filler_audio_started_event.set()
            ):
                self.conversation.logger.debug(
                    "Not waiting for filler audio to finish since we didn't send any chunks"
                )
                return
            if self.interruptible_event and isinstance(
                self.interruptible_event, InterruptibleAgentResponseEvent
            ):
                await self.interruptible_event.agent_response_tracker.wait()

        def interrupt_current_filler_audio(self):
            return self.interruptible_event and self.interruptible_event.interrupt()

        async def process(self, item: InterruptibleAgentResponseEvent[FillerAudio]):
            try:
                filler_audio = item.payload
                assert self.conversation.filler_audio_config is not None
                filler_synthesis_result = filler_audio.create_synthesis_result()
                self.current_filler_seconds_per_chunk = filler_audio.seconds_per_chunk
                silence_threshold = (
                    self.conversation.filler_audio_config.silence_threshold_seconds
                )
                await asyncio.sleep(silence_threshold)
                self.conversation.logger.debug("Sending filler audio to output")
                self.filler_audio_started_event = threading.Event()
                await self.conversation.send_speech_to_output(
                    filler_audio.message.text,
                    filler_synthesis_result,
                    item.interruption_event,
                    filler_audio.seconds_per_chunk,
                    started_event=self.filler_audio_started_event,
                )
                item.agent_response_tracker.set()
            except asyncio.CancelledError:
                pass

    class AgentResponsesWorker(InterruptibleAgentResponseWorker):
        """Runs Synthesizer.create_speech and sends the SynthesisResult to the output queue"""

        def __init__(
            self,
            input_queue: asyncio.Queue[InterruptibleAgentResponseEvent[AgentResponse]],
            output_queue: asyncio.Queue[
                InterruptibleAgentResponseEvent[Tuple[BaseMessage, SynthesisResult]]
            ],
            conversation: "StreamingConversation",
            interruptible_event_factory: InterruptibleEventFactory,
        ):
            super().__init__(
                input_queue=input_queue,
                output_queue=output_queue,
            )
            self.input_queue = input_queue
            self.output_queue = output_queue
            self.conversation = conversation
            self.interruptible_event_factory = interruptible_event_factory
            self.chunk_size = (
                get_chunk_size_per_second(
                    self.conversation.synthesizer.get_synthesizer_config().audio_encoding,
                    self.conversation.synthesizer.get_synthesizer_config().sampling_rate,
                )
                * TEXT_TO_SPEECH_CHUNK_SIZE_SECONDS
            )

        def send_filler_audio(self, agent_response_tracker: Optional[asyncio.Event]):
            assert self.conversation.filler_audio_worker is not None
            self.conversation.logger.debug("Sending filler audio")
            if self.conversation.synthesizer.filler_audios:
                filler_audio = random.choice(
                    self.conversation.synthesizer.filler_audios
                )
                self.conversation.logger.debug(f"Chose {filler_audio.message.text}")
                event = self.interruptible_event_factory.create_interruptible_agent_response_event(
                    filler_audio,
                    is_interruptible=filler_audio.is_interruptible,
                    agent_response_tracker=agent_response_tracker,
                )
                self.conversation.filler_audio_worker.consume_nonblocking(event)
            else:
                self.conversation.logger.debug(
                    "No filler audio available for synthesizer"
                )

        def send_on_hold_audio(self, audio: FillerAudio, agent_response_tracker: Optional[asyncio.Event]):
            assert self.conversation.on_hold_audio_worker is not None
            self.conversation.logger.debug(f"Sending {audio.message.text}")
            event = self.interruptible_event_factory.create_interruptible_agent_response_event(
                audio,
                is_interruptible=audio.is_interruptible,
                agent_response_tracker=agent_response_tracker,
            )
            self.conversation.on_hold_audio_worker.consume_nonblocking(event)

        async def process(self, item: InterruptibleAgentResponseEvent[AgentResponse]):
            if not self.conversation.synthesis_enabled:
                self.conversation.logger.debug(
                    "Synthesis disabled, not synthesizing speech"
                )
                return
            try:
                agent_response = item.payload
                if isinstance(agent_response, AgentResponseFillerAudio):
                    self.send_filler_audio(item.agent_response_tracker)
                    return
                if isinstance(agent_response, AgentResponseStop):
                    self.conversation.logger.debug("Agent requested to stop")
                    item.agent_response_tracker.set()
                    await self.conversation.terminate()
                    return

                agent_response_message = typing.cast(
                    AgentResponseMessage, agent_response
                )

                if self.conversation.filler_audio_worker is not None:
                    if (
                        self.conversation.filler_audio_worker.interrupt_current_filler_audio()
                    ):
                        await self.conversation.filler_audio_worker.wait_for_filler_audio_to_finish()

                self.conversation.logger.debug("Synthesizing speech for message")
                synthesis_result = await self.conversation.synthesizer.create_speech(
                    agent_response_message.message,
                    self.chunk_size,
                    bot_sentiment=self.conversation.bot_sentiment,
                )
                self.produce_interruptible_agent_response_event_nonblocking(
                    (agent_response_message.message, synthesis_result),
                    is_interruptible=item.is_interruptible,
                    agent_response_tracker=item.agent_response_tracker,
                )
                self.conversation.mark_last_agent_response()
            except asyncio.CancelledError:
                pass

    class SynthesisResultsWorker(InterruptibleAgentResponseWorker):
        """Plays SynthesisResults from the output queue on the output device"""

        def __init__(
            self,
            input_queue: asyncio.Queue[
                InterruptibleAgentResponseEvent[Tuple[BaseMessage, SynthesisResult]]
            ],
            conversation: "StreamingConversation",
        ):
            super().__init__(input_queue=input_queue)
            self.input_queue = input_queue
            self.conversation = conversation

        async def process(
            self,
            item: InterruptibleAgentResponseEvent[Tuple[BaseMessage, SynthesisResult]],
        ):
            try:
                message, synthesis_result = item.payload
                # create an empty transcript message and attach it to the transcript
                transcript_message = Message(
                    text="",
                    sender=Sender.BOT,
                )
                self.conversation.transcript.add_message(
                    message=transcript_message,
                    conversation_id=self.conversation.id,
                    publish_to_events_manager=False,
                )
                message_sent, cut_off = await self.conversation.send_speech_to_output(
                    message.text,
                    synthesis_result,
                    item.interruption_event,
                    TEXT_TO_SPEECH_CHUNK_SIZE_SECONDS,
                    transcript_message=transcript_message,
                )
                # publish the transcript message now that it includes what was said during send_speech_to_output
                self.conversation.transcript.maybe_publish_transcript_event_from_message(
                    message=transcript_message,
                    conversation_id=self.conversation.id,
                )
                item.agent_response_tracker.set()
                self.conversation.logger.debug("Message sent: {}".format(message_sent))
                if cut_off:
                    self.conversation.agent.update_last_bot_message_on_cut_off(
                        message_sent
                    )
                if self.conversation.agent.agent_config.end_conversation_on_goodbye:
                    goodbye_detected_task = (
                        self.conversation.agent.create_goodbye_detection_task(
                            message_sent
                        )
                    )
                    try:
                        if await asyncio.wait_for(goodbye_detected_task, 0.1):
                            self.conversation.logger.debug(
                                "Agent said goodbye, ending call"
                            )
                            await self.conversation.terminate()
                    except asyncio.TimeoutError:
                        pass
            except asyncio.CancelledError:
                pass

    def __init__(
        self,
        output_device: OutputDeviceType,
        transcriber: BaseTranscriber[TranscriberConfig],
        agent: BaseAgent,
        synthesizer: BaseSynthesizer,
        conversation_id: Optional[str] = None,
        per_chunk_allowance_seconds: float = PER_CHUNK_ALLOWANCE_SECONDS,
        events_manager: Optional[EventsManager] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.id = conversation_id or create_conversation_id()
        self.logger = wrap_logger(
            logger or logging.getLogger(__name__),
            conversation_id=self.id,
        )
        self.output_device = output_device
        self.transcriber = transcriber
        # self.audio_stream_handler = AudioStreamHandler(conversation_id=self.id, transcriber=transcriber, logger=self.logger)
        self.audio_stream_handler = None
        self.agent = agent
        self.synthesizer = synthesizer
        self.synthesis_enabled = True

        self.interruptible_events: queue.Queue[InterruptibleEvent] = queue.Queue()
        self.interruptible_event_factory = self.QueueingInterruptibleEventFactory(
            conversation=self
        )
        self.agent.set_interruptible_event_factory(self.interruptible_event_factory)
        self.synthesis_results_queue: asyncio.Queue[
            InterruptibleAgentResponseEvent[Tuple[BaseMessage, SynthesisResult]]
        ] = asyncio.Queue()
        self.filler_audio_queue: asyncio.Queue[
            InterruptibleAgentResponseEvent[FillerAudio]
        ] = asyncio.Queue()
        self.state_manager = self.create_state_manager()
        self.transcriptions_worker = self.TranscriptionsWorker(
            input_queue=self.transcriber.output_queue,
            output_queue=self.agent.get_input_queue(),
            conversation=self,
            interruptible_event_factory=self.interruptible_event_factory,
        )
        self.agent.attach_conversation_state_manager(self.state_manager)
        self.agent_responses_worker = self.AgentResponsesWorker(
            input_queue=self.agent.get_output_queue(),
            output_queue=self.synthesis_results_queue,
            conversation=self,
            interruptible_event_factory=self.interruptible_event_factory,
        )
        self.actions_worker = None
        if self.agent.get_agent_config().actions:
            self.actions_worker = ActionsWorker(
                input_queue=self.agent.actions_queue,
                output_queue=self.agent.get_input_queue(),
                interruptible_event_factory=self.interruptible_event_factory,
                action_factory=self.agent.action_factory,
            )
            self.actions_worker.attach_conversation_state_manager(self.state_manager)
        self.synthesis_results_worker = self.SynthesisResultsWorker(
            input_queue=self.synthesis_results_queue, conversation=self
        )
        self.filler_audio_worker = None
        self.filler_audio_config: Optional[FillerAudioConfig] = None
        if self.agent.get_agent_config().send_filler_audio:
            self.filler_audio_worker = self.FillerAudioWorker(
                input_queue=self.filler_audio_queue, conversation=self
            )

        self.events_manager = events_manager or EventsManager()
        self.events_task: Optional[asyncio.Task] = None
        self.per_chunk_allowance_seconds = per_chunk_allowance_seconds
        self.transcript = Transcript()
        self.transcript.attach_events_manager(self.events_manager)
        self.bot_sentiment = None
        if self.agent.get_agent_config().track_bot_sentiment:
            self.sentiment_config = (
                self.synthesizer.get_synthesizer_config().sentiment_config
            )
            if not self.sentiment_config:
                self.sentiment_config = SentimentConfig()
            self.bot_sentiment_analyser = BotSentimentAnalyser(
                emotions=self.sentiment_config.emotions
            )

        self.is_human_speaking = False
        self.active = False
        self.mark_last_action_timestamp()
        self.is_caller_on_hold = False

        self.check_for_idle_task: Optional[asyncio.Task] = None
        self.track_bot_sentiment_task: Optional[asyncio.Task] = None

        self.current_transcription_is_interrupt: bool = False
        self.background_tasks = set()

        # tracing
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def create_state_manager(self) -> ConversationStateManager:
        return ConversationStateManager(conversation=self)

    async def start(self, mark_ready: Optional[Callable[[], Awaitable[None]]] = None):
        self.transcriber.start()
        self.transcriptions_worker.start()
        self.agent_responses_worker.start()
        self.synthesis_results_worker.start()
        self.output_device.start()
        if self.filler_audio_worker is not None:
            self.filler_audio_worker.start()
        if self.actions_worker is not None:
            self.actions_worker.start()
        is_ready = await self.transcriber.ready()
        if not is_ready:
            raise Exception("Transcriber startup failed")
        if self.agent.get_agent_config().send_filler_audio:
            if not isinstance(
                self.agent.get_agent_config().send_filler_audio, FillerAudioConfig
            ):
                self.filler_audio_config = FillerAudioConfig()
            else:
                self.filler_audio_config = typing.cast(
                    FillerAudioConfig, self.agent.get_agent_config().send_filler_audio
                )
            await self.synthesizer.set_filler_audios(self.filler_audio_config)

        self.agent.start()
        initial_message = self.agent.get_agent_config().initial_message
        if initial_message:
            asyncio.create_task(self.send_initial_message(initial_message))
        self.agent.attach_transcript(self.transcript)

        self.audio_stream_handler = self.audio_stream_handler = AudioStreamHandler(conversation_id=self.id, transcriber=self.transcriber, logger=self.logger)
        await self.audio_stream_handler.post_init()

        if mark_ready:
            await mark_ready()
        if self.synthesizer.get_synthesizer_config().sentiment_config:
            await self.update_bot_sentiment()
        self.active = True
        if self.synthesizer.get_synthesizer_config().sentiment_config:
            self.track_bot_sentiment_task = asyncio.create_task(
                self.track_bot_sentiment()
            )
        self.check_for_idle_task = asyncio.create_task(self.check_for_idle())
        if len(self.events_manager.subscriptions) > 0:
            self.events_task = asyncio.create_task(self.events_manager.start())
        if (
            self.agent.get_agent_config().reengage_timeout and
            (self.agent.get_agent_config().reengage_options and
             len(self.agent.get_agent_config().reengage_options) > 0)
        ):
            self.human_prompt_checker = asyncio.create_task(self.check_if_human_should_be_prompted())
        else:
            self.logger.debug(f"re-engagement disabled")

    async def send_initial_message(self, initial_message: BaseMessage):
        # TODO: configure if initial message is interruptible
        self.transcriber.mute()
        initial_message_tracker = asyncio.Event()
        agent_response_event = (
            self.interruptible_event_factory.create_interruptible_agent_response_event(
                AgentResponseMessage(message=initial_message),
                is_interruptible=False,
                agent_response_tracker=initial_message_tracker,
            )
        )
        self.agent_responses_worker.consume_nonblocking(agent_response_event)
        await initial_message_tracker.wait()
        self.transcriber.unmute()

    async def check_for_idle(self):
        """Terminates the conversation after 15 seconds if no activity is detected"""
        while self.is_active():
            if time.time() - self.last_action_timestamp > (
                self.agent.get_agent_config().allowed_idle_time_seconds
                or ALLOWED_IDLE_TIME
            ):
                self.logger.debug("Conversation idle for too long, terminating")
                await self.terminate()
                return
            await asyncio.sleep(15)

    async def track_bot_sentiment(self):
        """Updates self.bot_sentiment every second based on the current transcript"""
        prev_transcript = None
        while self.is_active():
            await asyncio.sleep(1)
            if self.transcript.to_string() != prev_transcript:
                await self.update_bot_sentiment()
                prev_transcript = self.transcript.to_string()

    async def update_bot_sentiment(self):
        new_bot_sentiment = await self.bot_sentiment_analyser.analyse(
            self.transcript.to_string()
        )
        if new_bot_sentiment.emotion:
            self.logger.debug("Bot sentiment: %s", new_bot_sentiment)
            self.bot_sentiment = new_bot_sentiment

    def receive_message(self, message: str):
        transcription = Transcription(
            message=message,
            confidence=1.0,
            is_final=True,
        )
        self.transcriptions_worker.consume_nonblocking(transcription)

    async def receive_audio(self, chunk: bytes):
        # self.transcriber.send_audio(chunk)
        await self.audio_stream_handler.receive_audio(chunk=chunk)

    def warmup_synthesizer(self):
        self.synthesizer.ready_synthesizer()

    def mark_last_action_timestamp(self):
        self.last_action_timestamp = time.time()

    def mark_last_final_transcript_from_human(self):
        self.last_final_transcript_from_human = time.time()
    
    def mark_last_agent_response(self):
        self.last_agent_response = time.time()


    def broadcast_interrupt(self):
        """Stops all inflight events and cancels all workers that are sending output

        Returns true if any events were interrupted - which is used as a flag for the agent (is_interrupt)
        """
        num_interrupts = 0
        while True:
            try:
                interruptible_event = self.interruptible_events.get_nowait()
                if not interruptible_event.is_interrupted():
                    if interruptible_event.interrupt():
                        self.logger.debug("Interrupting event")
                        num_interrupts += 1
            except queue.Empty:
                break
        self.agent.cancel_current_task()
        self.agent_responses_worker.cancel_current_task()

        # Clearing these queues cuts time from finishing interruption talking to bot talking cut by 1 second from ~4.5 to ~3.5 seconds.
        self.clear_queue(self.agent.output_queue, 'agent.output_queue')
        self.clear_queue(self.agent_responses_worker.output_queue, 'agent_responses_worker.output_queue')
        self.clear_queue(self.agent_responses_worker.input_queue, 'agent_responses_worker.input_queue')
        self.clear_queue(self.synthesis_results_worker.output_queue, 'synthesis_results_worker.output_queue')
        self.clear_queue(self.synthesis_results_worker.input_queue, 'synthesis_results_worker.input_queue')
        if hasattr(self.output_device, 'queue'):
            self.clear_queue(self.output_device.queue, 'output_device.queue')
        
        return num_interrupts > 0

    def is_interrupt(self, transcription: Transcription):
        return transcription.confidence >= (
            self.transcriber.get_transcriber_config().min_interrupt_confidence or 0
        )

    @staticmethod
    def clear_queue(q: asyncio.Queue, queue_name: str):
        while not q.empty():
            logging.info(f'Clearing queue {queue_name} with size {q.qsize()}')
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                continue

    async def send_speech_to_output(
        self,
        message: str,
        synthesis_result: SynthesisResult,
        stop_event: threading.Event,
        seconds_per_chunk: int,
        transcript_message: Optional[Message] = None,
        started_event: Optional[threading.Event] = None,
    ):
        """
        - Sends the speech chunk by chunk to the output device
          - update the transcript message as chunks come in (transcript_message is always provided for non filler audio utterances)
        - If the stop_event is set, the output is stopped
        - Sets started_event when the first chunk is sent

        Importantly, we rate limit the chunks sent to the output. For interrupts to work properly,
        the next chunk of audio can only be sent after the last chunk is played, so we send
        a chunk of x seconds only after x seconds have passed since the last chunk was sent.

        Returns the message that was sent up to, and a flag if the message was cut off
        """
        if self.transcriber.get_transcriber_config().mute_during_speech:
            self.logger.debug("Muting transcriber")
            self.transcriber.mute()
        message_sent = message
        cut_off = False
        chunk_size = seconds_per_chunk * get_chunk_size_per_second(
            self.synthesizer.get_synthesizer_config().audio_encoding,
            self.synthesizer.get_synthesizer_config().sampling_rate,
        )
        chunk_idx = 0
        seconds_spoken = 0
        async for chunk_result in synthesis_result.chunk_generator:
            start_time = time.time()
            speech_length_seconds = seconds_per_chunk * (
                len(chunk_result.chunk) / chunk_size
            )
            seconds_spoken = chunk_idx * seconds_per_chunk
            if self.transcriber.get_transcriber_config().experimental:
                if stop_event.is_set() or self.is_human_speaking:
                    self.logger.debug(
                        "Interrupted, stopping text to speech after {} chunks".format(
                            chunk_idx
                        )
                    )
                    message_sent = f"{synthesis_result.get_message_up_to(seconds_spoken)}-"
                    cut_off = True
                    break    
            elif stop_event.is_set():
                self.logger.debug(
                    "Interrupted, stopping text to speech after {} chunks".format(
                        chunk_idx
                    )
                )
                message_sent = f"{synthesis_result.get_message_up_to(seconds_spoken)}-"
                cut_off = True
                break
            if chunk_idx == 0:
                if started_event:
                    started_event.set()
            self.output_device.consume_nonblocking(chunk_result.chunk)
            end_time = time.time()
            await asyncio.sleep(
                max(
                    speech_length_seconds
                    - (end_time - start_time)
                    - self.per_chunk_allowance_seconds,
                    0,
                )
            )
            self.logger.debug(
                "Sent chunk {} with size {}".format(chunk_idx, len(chunk_result.chunk))
            )
            self.mark_last_action_timestamp()
            self.mark_last_agent_response()
            chunk_idx += 1
            seconds_spoken += seconds_per_chunk
            if transcript_message:
                transcript_message.text = synthesis_result.get_message_up_to(
                    seconds_spoken
                )
        if self.transcriber.get_transcriber_config().mute_during_speech:
            self.logger.debug("Unmuting transcriber")
            self.transcriber.unmute()
        if transcript_message:
            transcript_message.text = message_sent
        return message_sent, cut_off

    def mark_terminated(self):
        self.active = False

    async def terminate(self):
        self.mark_terminated()
        self.broadcast_interrupt()
        if self.audio_stream_handler.vad_wrapper:
            self.audio_stream_handler.vad_wrapper.reset_states()
            self.logger.debug(f"Reset VAD model states")
        self.events_manager.publish_event(
            TranscriptCompleteEvent(conversation_id=self.id, transcript=self.transcript)
        )
        if self.check_for_idle_task:
            self.logger.debug("Terminating check_for_idle Task")
            self.check_for_idle_task.cancel()
        if self.track_bot_sentiment_task:
            self.logger.debug("Terminating track_bot_sentiment Task")
            self.track_bot_sentiment_task.cancel()
        if self.events_manager and self.events_task:
            self.logger.debug("Terminating events Task")
            await self.events_manager.flush()
        self.logger.debug("Tearing down synthesizer")
        await self.synthesizer.tear_down()
        self.logger.debug("Terminating agent")
        if (
            isinstance(self.agent, ChatGPTAgent)
            and self.agent.agent_config.vector_db_config
        ):
            # Shutting down the vector db should be done in the agent's terminate method,
            # but it is done here because `vector_db.tear_down()` is async and
            # `agent.terminate()` is not async.
            self.logger.debug("Terminating vector db")
            await self.agent.vector_db.tear_down()
        self.agent.terminate()
        self.logger.debug("Terminating output device")
        self.output_device.terminate()
        self.logger.debug("Terminating speech transcriber")
        self.transcriber.terminate()
        self.logger.debug("Terminating transcriptions worker")
        self.transcriptions_worker.terminate()
        self.logger.debug("Terminating final transcriptions worker")
        self.agent_responses_worker.terminate()
        self.logger.debug("Terminating synthesis results worker")
        self.synthesis_results_worker.terminate()
        if self.filler_audio_worker is not None:
            self.logger.debug("Terminating filler audio worker")
            self.filler_audio_worker.terminate()
        if self.actions_worker is not None:
            self.logger.debug("Terminating actions worker")
            self.actions_worker.terminate()
        self.logger.debug("Successfully terminated")
        self.audio_stream_handler.terminate()

    def is_active(self):
        return self.active

    async def check_if_human_should_be_prompted(self):
        self.logger.debug("starting should prompt user task")
        self.last_agent_response = None
        self.last_final_transcript_from_human = None
        reengage_timeout = self.agent.get_agent_config().reengage_timeout
        reengage_options = self.agent.get_agent_config().reengage_options
        while self.active:
            if self.last_agent_response and self.last_final_transcript_from_human:
                last_human_touchpoint = time.time() - self.last_final_transcript_from_human
                last_agent_touchpoint = time.time() - self.last_agent_response
                if (last_human_touchpoint >= reengage_timeout) and (last_agent_touchpoint >= reengage_timeout):
                    reengage_statement = random.choice(reengage_options)
                    self.logger.debug(f"Prompting user with {reengage_statement}: no interaction has happened in {reengage_timeout} seconds")
                    await self.say_something_to_caller(message=reengage_statement, is_interruptible=True)
                    self.mark_last_agent_response()
                await asyncio.sleep(2.5)
            else:
                await asyncio.sleep(1)
        self.logger.debug("stopped check if human should be prompted")

    async def __spoof_reengagement_coroutine(self):
        """prevents the agent from attempting re-engagement while synthetic hold in place"""

        self.logger.debug("start spoofing re-engagement")
        agent_reengage_interval = self.agent.get_agent_config().reengage_timeout
        if agent_reengage_interval is not None and agent_reengage_interval < 1:
            spoof_last_agent_interval = agent_reengage_interval * .75
        else:
            spoof_last_agent_interval = 1
        while self.is_caller_on_hold:
            self.mark_last_agent_response()
            await asyncio.sleep(spoof_last_agent_interval)
        self.mark_last_agent_response()
        self.logger.debug("end spoofing re-engagement")

    def __load_remote_audio(self, audio_url: str, chunk_size = 16000):
        """loads remote wav file and returns the bytes in specified chunk size.
        Bytes are converted to 8000hz and mulaw encoded per Twilio spec"""

        # get audio content into memory
        converted_bytes = None
        try:
            if audio_url.lower().endswith(".mp3"):
                audio_bytes = requests.get(audio_url).content
                wav_bytes = mp3_helper.decode_mp3(audio_bytes)
                converted_bytes = convert_wav(file=wav_bytes, output_sample_rate=8000, output_encoding="mulaw")
            elif audio_url.lower().endswith(".wav"):
                audio_bytes = requests.get(audio_url).content
                converted_bytes = convert_wav(file=BytesIO(wav_bytes), output_sample_rate=8000, output_encoding="mulaw")
            else:
                self.logger.error(f"unsupported file extension")
        except Exception as err:
            self.logger.error(f"Unable to parse [{audio_url}]. {err}")

        if converted_bytes is None:
            return

        # generate chunks
        pnt = 0
        while pnt + chunk_size < converted_bytes.__len__():
            yield converted_bytes[pnt:pnt+chunk_size]
            pnt = pnt + chunk_size
        yield converted_bytes[pnt:]

    async def __hold_audio_coroutine(self, *, hold_config: SyntheticHoldConfig):
        """loads audio hosted at provided url and streams chunks to output device
        Currently you MUST use Twilio as output_device to get hold audio"""

        self.logger.debug("start synthetic hold")
        audio_generator = None
        if hold_config.audio_url is not None:
            # test generator for initial chunk
            audio_generator = self.__load_remote_audio(hold_config.audio_url)
            try:
                audio = audio_generator.__next__()
                self.output_device.consume_nonblocking(audio)
            except StopIteration:
                self.logger.error("No initial audio chunk received, no hold music will be played.")
                audio_generator = None

        while self.is_caller_on_hold:
            if audio_generator is not None:
                try:
                    audio = audio_generator.__next__()
                except StopIteration:
                    # reload
                    audio_generator = self.__load_remote_audio(hold_config.audio_url)
                    audio = audio_generator.__next__()
                finally:
                    self.output_device.consume_nonblocking(audio)
            await asyncio.sleep(1)
        self.output_device.clear_stream()
        self.logger.debug("end synthetic hold audio")

    def start_on_hold(self, hold_config: SyntheticHoldConfig):
        # sanity check
        if not hold_config.enabled:
            self.logger.debug("hold_config set to enabled=False")
            return
        self.is_caller_on_hold = True
        # keep agent quiet
        reengagement_handle = asyncio.create_task(self.__spoof_reengagement_coroutine())
        self.background_tasks.add(reengagement_handle)
        reengagement_handle.add_done_callback(self.background_tasks.discard)
        # keep caller entertained
        audio_handle = asyncio.create_task(self.__hold_audio_coroutine(hold_config=hold_config))
        self.background_tasks.add(audio_handle)
        audio_handle.add_done_callback(self.background_tasks.discard)

    def stop_on_hold(self):
        self.is_caller_on_hold = False

    async def say_something_to_caller(self, message: str, is_interruptible: bool = True):
        """Synthesize and send a message."""
        self.logger.debug(f"Saying to caller: \"{message}\"")
        self.chunk_size = (
            get_chunk_size_per_second(
                self.synthesizer.get_synthesizer_config().audio_encoding,
                self.synthesizer.get_synthesizer_config().sampling_rate,
            )
            * TEXT_TO_SPEECH_CHUNK_SIZE_SECONDS
        )
        message = BaseMessage(text=message)
        synthesis_result = await self.synthesizer.create_speech(
            message,
            self.chunk_size,
            bot_sentiment=self.bot_sentiment,
        )
        self.agent_responses_worker.produce_interruptible_agent_response_event_nonblocking(
            (message, synthesis_result),
            is_interruptible=is_interruptible,
            agent_response_tracker=asyncio.Event(),
        )