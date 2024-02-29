import logging
import torch
from importlib import resources as impresources


class SileroVAD:
    INT16_NORM_CONST = 32768.0

    def __init__(self, sample_rate: int, window_size: int, threshold: float = 0.5):
        # Silero VAD is optimized for performance on single CPU thread
        torch.set_num_threads(1)

        self.logger = logging.getLogger(__name__)
        self.model = self._load_model(use_onnx=False)
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.window_size = window_size

    def _load_model(self, use_onnx: bool = False) -> torch.nn.Module:
        try:
            model, _ = torch.hub.load(
                repo_or_dir='silero-vad',
                model='silero_vad',
                source='local',
                onnx=use_onnx,
                trust_repo=True
            )
        except FileNotFoundError:
            self.logger.warning("Could not find local VAD model, downloading from GitHub!")
            model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                source='github',
                onnx=use_onnx,
                trust_repo=True
            )
        return model

    def process_chunk(self, chunk: bytes) -> bool:
        if len(chunk) != self.window_size:
            raise ValueError(f"Chunk size must be {self.window_size} bytes")
        chunk_array = torch.frombuffer(chunk, dtype=torch.int16).to(torch.float32) / self.INT16_NORM_CONST
        speech_prob = self.model(chunk_array, self.sample_rate).item()
        if speech_prob > self.threshold:
            return True
        return False

    def reset_states(self) -> None:
        self.model.reset_states()
