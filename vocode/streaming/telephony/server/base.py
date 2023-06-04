import abc
from functools import partial
import logging
from typing import List, Optional
from fastapi import APIRouter, Form, Response
from pydantic import BaseModel, Field
from vocode.streaming.agent.base_agent import BaseAgent
from vocode.streaming.agent.factory import AgentFactory
from vocode.streaming.models.agent import AgentConfig
from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.models.synthesizer import (
    AzureSynthesizerConfig,
    SynthesizerConfig,
)
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
    TranscriberConfig,
)
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer
from vocode.streaming.synthesizer.factory import SynthesizerFactory
from vocode.streaming.telephony.client.twilio_client import TwilioClient
from vocode.streaming.telephony.client.vonage_client import VonageClient
from vocode.streaming.telephony.config_manager.base_config_manager import (
    BaseConfigManager,
)
from vocode.streaming.telephony.constants import (
    DEFAULT_AUDIO_ENCODING,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SAMPLING_RATE,
)

from vocode.streaming.telephony.server.router.calls import CallsRouter
from vocode.streaming.models.telephony import (
    CallConfig,
    TwilioCallConfig,
    TwilioConfig,
    VonageCallConfig,
    VonageConfig,
)

from vocode.streaming.telephony.conversation.call import Call
from vocode.streaming.telephony.templater import Templater
from vocode.streaming.transcriber.base_transcriber import BaseTranscriber
from vocode.streaming.transcriber.factory import TranscriberFactory
from vocode.streaming.utils import create_conversation_id
from vocode.streaming.utils.events_manager import EventsManager


class InboundCallConfig(BaseModel, abc.ABC):
    url: str
    agent_config: AgentConfig
    transcriber_config: Optional[TranscriberConfig] = None
    synthesizer_config: Optional[SynthesizerConfig] = None


class TwilioInboundCallConfig(InboundCallConfig):
    twilio_config: TwilioConfig


class VonageInboundCallConfig(InboundCallConfig):
    vonage_config: VonageConfig


class VonageAnswerRequest(BaseModel):
    to: str
    from_: str = Field(..., alias="from")
    uuid: str


class TelephonyServer:
    def __init__(
        self,
        base_url: str,
        config_manager: BaseConfigManager,
        inbound_call_configs: List[InboundCallConfig] = [],
        transcriber_factory: TranscriberFactory = TranscriberFactory(),
        agent_factory: AgentFactory = AgentFactory(),
        synthesizer_factory: SynthesizerFactory = SynthesizerFactory(),
        events_manager: Optional[EventsManager] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.base_url = base_url
        self.logger = logger or logging.getLogger(__name__)
        self.router = APIRouter()
        self.config_manager = config_manager
        self.templater = Templater()
        self.events_manager = events_manager
        self.router.include_router(
            CallsRouter(
                base_url=base_url,
                config_manager=self.config_manager,
                transcriber_factory=transcriber_factory,
                agent_factory=agent_factory,
                synthesizer_factory=synthesizer_factory,
                events_manager=self.events_manager,
                logger=self.logger,
            ).get_router()
        )
        for config in inbound_call_configs:
            self.router.add_api_route(
                config.url,
                self.create_inbound_route(inbound_call_config=config),
                methods=["POST"],
            )

    def create_inbound_route(
        self,
        inbound_call_config: InboundCallConfig,
    ):
        def twilio_route(
            twilio_config: TwilioConfig,
            twilio_sid: str = Form(alias="CallSid"),
            twilio_from: str = Form(alias="From"),
            twilio_to: str = Form(alias="To"),
        ) -> Response:
            call_config = TwilioCallConfig(
                transcriber_config=inbound_call_config.transcriber_config
                or DeepgramTranscriberConfig(
                    sampling_rate=DEFAULT_SAMPLING_RATE,
                    audio_encoding=DEFAULT_AUDIO_ENCODING,
                    chunk_size=DEFAULT_CHUNK_SIZE,
                    model="phonecall",
                    tier="nova",
                    endpointing_config=PunctuationEndpointingConfig(),
                ),
                agent_config=inbound_call_config.agent_config,
                synthesizer_config=inbound_call_config.synthesizer_config
                or AzureSynthesizerConfig(
                    sampling_rate=DEFAULT_SAMPLING_RATE,
                    audio_encoding=DEFAULT_AUDIO_ENCODING,
                ),
                twilio_config=twilio_config,
                twilio_sid=twilio_sid,
                twilio_from=twilio_from,
                twilio_to=twilio_to,
            )

            conversation_id = create_conversation_id()
            self.config_manager.save_config(conversation_id, call_config)
            return self.templater.get_connection_twiml(
                base_url=self.base_url, call_id=conversation_id
            )

        def vonage_route(
            vonage_config: VonageConfig, vonage_answer_request: VonageAnswerRequest
        ):
            call_config = VonageCallConfig(
                transcriber_config=inbound_call_config.transcriber_config
                or DeepgramTranscriberConfig(
                    sampling_rate=16000,
                    audio_encoding=AudioEncoding.LINEAR16,
                    chunk_size=DEFAULT_CHUNK_SIZE,
                    model="phonecall",
                    tier="nova",
                    endpointing_config=PunctuationEndpointingConfig(),
                ),
                agent_config=inbound_call_config.agent_config,
                synthesizer_config=inbound_call_config.synthesizer_config
                or AzureSynthesizerConfig(
                    sampling_rate=16000,
                    audio_encoding=AudioEncoding.LINEAR16,
                ),
                vonage_config=vonage_config,
                vonage_uuid=vonage_answer_request.uuid,
                vonage_from=vonage_answer_request.from_,
                vonage_to=vonage_answer_request.to,
            )
            conversation_id = create_conversation_id()
            self.config_manager.save_config(conversation_id, call_config)
            return VonageClient.create_call_ncco(
                base_url=self.base_url, conversation_id=conversation_id
            )

        if isinstance(inbound_call_config, TwilioInboundCallConfig):
            self.logger.info(
                f"Set up inbound call TwiML at https://{self.base_url}{inbound_call_config.url}"
            )
            return partial(twilio_route, inbound_call_config.twilio_config)
        elif isinstance(inbound_call_config, VonageInboundCallConfig):
            self.logger.info(
                f"Set up inbound call NCCO at https://{self.base_url}{inbound_call_config.url}"
            )
            return partial(vonage_route, inbound_call_config.vonage_config)
        else:
            raise ValueError(
                f"Unknown inbound call config type {type(inbound_call_config)}"
            )

    async def end_outbound_call(self, conversation_id: str):
        # TODO validation via twilio_client
        call_config = self.config_manager.get_config(conversation_id)
        if not call_config:
            raise ValueError(f"Could not find call config for {conversation_id}")
        if not isinstance(call_config, TwilioCallConfig):
            raise ValueError(
                f"Call config for {conversation_id} is not a TwilioCallConfig"
            )
        telephony_client = TwilioClient(
            base_url=self.base_url, twilio_config=call_config.twilio_config
        )
        telephony_client.end_call(call_config.twilio_sid)
        return {"id": conversation_id}

    def get_router(self) -> APIRouter:
        return self.router
