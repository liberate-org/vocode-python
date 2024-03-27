import io
import logging
import asyncio
from typing import Any, AsyncGenerator, Optional, Tuple, Union, List
import aiohttp
from pydub import AudioSegment
import os
import time

from redis.asyncio import Redis
import hashlib
import base64
import json

from opentelemetry.trace import Span


from vocode import getenv
from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.synthesizer.base_synthesizer import (
    BaseSynthesizer,
    SynthesisResult, 
    FillerAudio, 
    tracer,
)
from vocode.streaming.models.synthesizer import (
  ElevenLabsSynthesizerConfig, 
  SynthesizerType
)
from vocode.streaming.agent.bot_sentiment_analyser import BotSentiment
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.utils import convert_wav
from vocode.streaming.utils.mp3_helper import decode_mp3

import audioop
from urllib.parse import urljoin, urlencode
import wave

ADAM_VOICE_ID = "pNInz6obpgDQGcFmaJgB"
ELEVEN_LABS_BASE_URL = "https://api.elevenlabs.io/v1/"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"
MULAW_OUTPUT_FORMAT = "ulaw_8000"

FILLER_AUDIO_PATH = os.path.join(os.path.dirname(__file__), "filler_audio")

LIB_FILLER_PHRASES = [
    BaseMessage(text="Hold on a second."),
    BaseMessage(text="One second."),
    BaseMessage(text="One minute please."),
    BaseMessage(text="Give me a second."),
    BaseMessage(text="Just a minute."),
    BaseMessage(text="Sorry, my computer is slow today."),
]

class ElevenLabsSynthesizer(BaseSynthesizer[ElevenLabsSynthesizerConfig]):
    def __init__(
            self,
            synthesizer_config: ElevenLabsSynthesizerConfig,
            logger: Optional[logging.Logger] = None,
            aiohttp_session: Optional[aiohttp.ClientSession] = None,
    ):
        super().__init__(synthesizer_config, aiohttp_session)

        import elevenlabs

        self.elevenlabs = elevenlabs

        self.api_key = synthesizer_config.api_key or getenv("ELEVEN_LABS_API_KEY")
        self.voice_id = synthesizer_config.voice_id or ADAM_VOICE_ID
        self.stability = synthesizer_config.stability
        self.similarity_boost = synthesizer_config.similarity_boost
        self.model_id = synthesizer_config.model_id
        self.optimize_streaming_latency = synthesizer_config.optimize_streaming_latency
        self.words_per_minute = 125
        self.experimental_streaming = synthesizer_config.experimental_streaming
        self.output_format = self._get_eleven_labs_format()
        self.logger = logger

        self.redis: Redis = Redis(
            host=os.environ.get("REDISHOST", "localhost"),
            port=int(os.environ.get("REDISPORT", 6379)),
            username=os.environ.get("REDISUSER", None),
            password=os.environ.get("REDISPASSWORD", None),
            db=4,
            decode_responses=True,
        )
        
    async def get_phrase_filler_audios(self) -> List[FillerAudio]:
        fillers = []
        for phrase in LIB_FILLER_PHRASES:
            cache_key = "-".join(
                (
                    str(phrase.text),
                    str(self.synthesizer_config.type),
                    str(self.synthesizer_config.audio_encoding),
                    str(self.synthesizer_config.sampling_rate),
                    str(self.synthesizer_config.model_id),
                    str(self.voice_id),
                )
            )
            filler_audio_path = os.path.join(FILLER_AUDIO_PATH, f"{cache_key}.bytes")
            if os.path.exists(filler_audio_path):
                audio_data = open(filler_audio_path, "rb").read()
            else:
              wav = await create_wav(
                message=phrase,
                voice_id=self.voice_id,
                stability=self.stability,
                similarity_boost=self.similarity_boost,
                api_key=self.api_key,
                optimize_streaming_latency=self.synthesizer_config.optimize_streaming_latency,
              )
              audio_data = convert_wav(
                wav,
                output_sample_rate=self.synthesizer_config.sampling_rate,
                output_encoding=self.synthesizer_config.audio_encoding,
              )
              with open(filler_audio_path, "wb") as f:
                  f.write(audio_data)
            fillers.append(
                FillerAudio(
                    message=phrase,
                    audio_data=audio_data,
                    synthesizer_config=self.synthesizer_config,
                    is_interruptible=True,
                    seconds_per_chunk=2
                )
            )
        return fillers

    async def create_speech(
        self,
        message: BaseMessage,
        chunk_size: int,
        bot_sentiment: Optional[BotSentiment] = None,
    ) -> SynthesisResult:
        voice = self.elevenlabs.Voice(voice_id=self.voice_id)
        if self.stability is not None and self.similarity_boost is not None:
            voice.settings = self.elevenlabs.VoiceSettings(
                stability=self.stability, similarity_boost=self.similarity_boost
            )
        # Check for cached audio before calling 11 Labs
        cached_audio = await self.get_cached_audio(message=message.text)
        if cached_audio is not None:
            self.logger.debug(f"%% Using cached audio. %%")
            async def generator():
                yield SynthesisResult.ChunkResult(cached_audio, True)
            
            return SynthesisResult(
                generator(),
                lambda seconds: self.get_message_cutoff_from_voice_speed(
                    message, seconds, self.words_per_minute
                ),
            )

        # Add additional path if experimental streaming is enabled

        base_url = urljoin(ELEVEN_LABS_BASE_URL, f"text-to-speech/{self.voice_id}")

        if self.experimental_streaming:
            base_url = urljoin(base_url + "/", "stream")

        # Construct query parameters
        query_params = {"output_format": self.output_format}

        if self.optimize_streaming_latency:
            query_params["optimize_streaming_latency"] = self.optimize_streaming_latency

        url = f"{base_url}?{urlencode(query_params)}"

        headers = {"xi-api-key": self.api_key}
        body = {
            "text": message.text,
            "voice_settings": voice.settings.dict() if voice.settings else None,
        }
        if self.model_id:
            body["model_id"] = self.model_id

        create_speech_span = tracer.start_span(
            f"synthesizer.{SynthesizerType.ELEVEN_LABS.value.split('_', 1)[-1]}.create_total",
        )

        session = self.aiohttp_session
        start_processing_time = time.time()
        response = await session.request(
            "POST",
            url,
            json=body,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=15),
        )
        if not response.ok:
            response_text = await response.text(encoding="utf-8", errors="ignore")
            raise Exception(
                f"ElevenLabs API returned {response.status} status code with error : {response_text}"
            )

        if self.output_format == DEFAULT_OUTPUT_FORMAT and self.experimental_streaming:
            return SynthesisResult(
                self.experimental_mp3_streaming_output_generator(
                    response=response, 
                    chunk_size=chunk_size,
                    create_speech_span=create_speech_span
                ),  # should be wav
                lambda seconds: self.get_message_cutoff_from_voice_speed(
                    message, seconds, self.words_per_minute
                ),
            )
        elif (
            self.output_format == DEFAULT_OUTPUT_FORMAT
            and not self.experimental_streaming
        ):
            audio_data = await response.read()
            create_speech_span.end()
            convert_span = tracer.start_span(
                f"synthesizer.{SynthesizerType.ELEVEN_LABS.value.split('_', 1)[-1]}.convert",
            )
            output_bytes_io = decode_mp3(audio_data)

            result = self.create_synthesis_result_from_wav(
                synthesizer_config=self.synthesizer_config,
                file=output_bytes_io,
                message=message,
                chunk_size=chunk_size,
            )
            convert_span.end()

            return result
        elif (
            self.output_format == MULAW_OUTPUT_FORMAT
            and not self.experimental_streaming
        ):
            audio_data = await response.read()
            # Convert μ-law to linear PCM
            pcm_data = audioop.ulaw2lin(audio_data, 2)
            # Create a WAV file in memory
            wav_data = convert_to_wav(pcm_data)

            result = self.create_synthesis_result_from_wav(
                synthesizer_config=self.synthesizer_config,
                file=io.BytesIO(wav_data),
                message=message,
                chunk_size=chunk_size,
            )
            return result
        elif self.output_format == MULAW_OUTPUT_FORMAT and self.experimental_streaming:
            create_speech_span.end()
            convert_span = tracer.start_span(
                f"synthesizer.{SynthesizerType.ELEVEN_LABS.value.split('_', 1)[-1]}.convert",
            )
            return SynthesisResult(
                self.experimental_streaming_output_generator(
                    response=response,
                    create_speech_span=convert_span,
                    message=message.text,
                    start_processing_time=start_processing_time
                ),  # should be wav
                lambda seconds: self.get_message_cutoff_from_voice_speed(
                    message, seconds, self.words_per_minute
                ),
            )
        else:
            raise RuntimeError(
                f"Unsupported ElevenLabs configuration: {self.synthesizer_config.sampling_rate}, {self.synthesizer_config.audio_encoding}, {self.output_format}"
            )

    def _get_eleven_labs_format(self):
        sampling_rate = self.synthesizer_config.sampling_rate
        codec = self.synthesizer_config.audio_encoding

        if sampling_rate != 8000 and codec != AudioEncoding.MULAW:
            return DEFAULT_OUTPUT_FORMAT  # default
        else:
            return MULAW_OUTPUT_FORMAT

    async def experimental_streaming_output_generator(
        self,
        response: aiohttp.ClientResponse,
        create_speech_span: Optional[Span],
        message: str,
        start_processing_time: float
    ) -> AsyncGenerator[SynthesisResult.ChunkResult, None]:
        stream_reader = response.content
        try:
            # Get the wav chunk and the flag from the output queue of the MiniaudioWorker
            buffer = bytearray()
            accumulated_audio_chunks = []
            async for chunk, done in stream_reader.iter_chunks():
                buffer += chunk
                accumulated_audio_chunks.append(chunk)
                if stream_reader.is_eof():
                    yield SynthesisResult.ChunkResult(buffer, stream_reader.is_eof())
                    buffer.clear()
                # If this is the last chunk, break the loop
                if stream_reader.is_eof() and create_speech_span is not None:
                    create_speech_span.end()
                #     break
            self.logger.info("Generating audio for [%s] took %s",
                message, time.time() - start_processing_time)
            complete_audio_sample = b''.join(accumulated_audio_chunks)
            cache_key = self.create_hash_cache_key(message=message)
            await self.cache_audio(message=message, audio=complete_audio_sample, key=cache_key)
        except asyncio.CancelledError:
            pass


    def create_hash_cache_key(self, message: str):
        key = f"{self.synthesizer_config.voice_id}:{self.synthesizer_config.audio_encoding.value}:{message}"
        hash_object = hashlib.sha1(key.encode())
        hashed_text = hash_object.hexdigest()
        return hashed_text
    
    async def cache_audio(self, message: str, audio: bytes, key: str):
        try:
            self.logger.debug(f"Caching audio for future use.")
            a = base64.b64encode(audio)
            await self.redis.set(key, a, ex=7200) # TODO: Probably need to set maxmemory limit in AWS and the ensure eviction policy is volatile-lru. For now 2 hours
        except Exception as e:
            self.logger.error(f"Error caching audio: {e}")
            pass

    async def get_cached_audio(self, message: str):
        key = self.create_hash_cache_key(message=message)
        try:
            cached_audio = await self.redis.get(key)
            if cached_audio:
                cached_audio = base64.b64decode(cached_audio)
                return cached_audio
            else:
                return None
        except:
            return None


async def create_wav(
        message: BaseMessage,
        voice_id: Optional[str] = None,
        stability: Optional[float] = None,
        similarity_boost: Optional[float] = None,
        api_key: Optional[str] = None,
        optimize_streaming_latency: Optional[int] = None,

) -> SynthesisResult:
    import elevenlabs
    voice = elevenlabs.Voice(voice_id=voice_id)
    if stability is not None and similarity_boost is not None:
        voice.settings = elevenlabs.VoiceSettings(
            stability=stability, similarity_boost=similarity_boost
        )
    url = ELEVEN_LABS_BASE_URL + f"text-to-speech/{voice_id}"
    headers = {"xi-api-key": api_key}
    body = {
        "text": message.text,
        "voice_settings": voice.settings.dict() if voice.settings else None,
    }
    if optimize_streaming_latency:
        body[
            "optimize_streaming_latency"
        ] = optimize_streaming_latency

    async with aiohttp.ClientSession() as session:
        async with session.request(
                "POST",
                url,
                json=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
        ) as response:
            if not response.ok:
                raise Exception(
                    f"ElevenLabs API returned {response.status} status code"
                )
            audio_data = await response.read()
            audio_segment: AudioSegment = AudioSegment.from_mp3(
                io.BytesIO(audio_data)  # type: ignore
            )
            output_bytes_io = io.BytesIO()
            # , codec="pcm_s16le", parameters=["-ac", "1", "-ar", "8000", "-sample_fmt", "s16"]
            audio_segment.export(output_bytes_io, format="wav")

            return output_bytes_io
        
def convert_to_wav(pcm_data: bytes) -> bytes:
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(8000)
            wav_file.writeframes(pcm_data)
        return wav_io.getvalue()
