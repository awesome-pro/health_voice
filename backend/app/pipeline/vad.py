"""Streaming Voice Activity Detection using silero-vad.

silero requires fixed-size windows (512 samples @ 16 kHz). Audio arrives from the
browser in arbitrary chunk sizes, so we buffer and feed exact windows, surfacing
speech start/end events that the caller uses for utterance endpointing.
"""
import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad

from .. import config


class StreamingVAD:
    WINDOW = 512  # required silero window size at 16 kHz

    def __init__(self) -> None:
        self._model = load_silero_vad()
        self._iterator = VADIterator(
            self._model,
            threshold=config.VAD_THRESHOLD,
            sampling_rate=config.SAMPLE_RATE,
            min_silence_duration_ms=config.MIN_SILENCE_MS,
            speech_pad_ms=config.SPEECH_PAD_MS,
        )
        self._buf = np.zeros(0, dtype=np.float32)

    def process(self, audio: np.ndarray) -> list[dict]:
        """Feed float32 [-1, 1] audio; return a list of {'start': s} / {'end': s} events."""
        self._buf = np.concatenate([self._buf, audio])
        events: list[dict] = []
        while len(self._buf) >= self.WINDOW:
            window = self._buf[: self.WINDOW]
            self._buf = self._buf[self.WINDOW :]
            res = self._iterator(torch.from_numpy(window.copy()), return_seconds=True)
            if res:
                events.append(res)
        return events

    def reset(self) -> None:
        self._iterator.reset_states()
        self._buf = np.zeros(0, dtype=np.float32)
