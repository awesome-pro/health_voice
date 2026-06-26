"""Whisper ASR via MLX (Metal-accelerated on Apple Silicon).

This is the "on-prem" speech-to-text stage from the case study: the model runs
locally and PHI audio never leaves the machine. Model weights are pulled once
from Hugging Face (public mlx-community repos) and cached on disk.
"""
import re

import numpy as np
import mlx_whisper

from .. import config

# Stock phrases Whisper emits on silence / non-speech (it learned them from
# YouTube-style audio: sign-offs and the ubiquitous medical disclaimer). These
# are deterministic to recognise, so we drop the segment before it reaches NER.
_HALLUCINATION_RE = re.compile(
    "|".join([
        r"thank(s| you)?(?: (?:so|very) much)?(?: for (?:watching|listening))",
        r"please (?:like|subscribe|comment)",
        r"don'?t forget to subscribe",
        r"substitute for (?:professional )?medical advice",
        r"qualified healthcare professional",
        r"for (?:educational|informational) purposes",
        r"consult (?:your|a) (?:doctor|physician|healthcare)",
    ]),
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"[a-z']+")


def _is_degenerate(text: str) -> bool:
    """True for an utterance that is just one token repeated (e.g. 'love love')."""
    words = _WORD_RE.findall(text.lower())
    return len(words) >= 2 and len(set(words)) == 1


class WhisperASR:
    def __init__(self, model_repo: str) -> None:
        self.model_repo = model_repo
        self.ready = False

    def warmup(self) -> None:
        """Force the weights to load + JIT compile on a second of silence."""
        self.transcribe(np.zeros(config.SAMPLE_RATE, dtype=np.float32))
        self.ready = True

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        """Return (text, confidence). Confidence is exp(mean avg_logprob) in [0, 1].

        Decoded segments that look like silence-hallucinations (non-speech,
        repetitive, or a known stock phrase) are dropped so they never reach the
        transcript or NER. Text + confidence are computed from the survivors.
        """
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model_repo,
            language="en",
            condition_on_previous_text=False,
        )
        kept = [s for s in (result.get("segments") or []) if self._keep(s)]
        if not kept:
            return "", 0.0
        text = " ".join((s.get("text") or "").strip() for s in kept).strip()
        if not text:
            return "", 0.0
        mean_logprob = sum(s.get("avg_logprob", -1.0) for s in kept) / len(kept)
        return text, min(float(np.exp(mean_logprob)), 1.0)

    @staticmethod
    def _keep(segment: dict) -> bool:
        """False for a decoded segment that looks like a Whisper hallucination."""
        if segment.get("no_speech_prob", 0.0) > config.NO_SPEECH_THRESHOLD:
            return False
        if segment.get("compression_ratio", 0.0) > config.COMPRESSION_RATIO_THRESHOLD:
            return False
        text = (segment.get("text") or "").strip()
        if not text:
            return False
        if _is_degenerate(text) or _HALLUCINATION_RE.search(text):
            return False
        return True
