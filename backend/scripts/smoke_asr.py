"""Offline smoke test for the ASR stage: synth speech -> WhisperASR -> text.

Run from backend/:  uv run python scripts/smoke_asr.py
Generates a synthetic clinical sentence with macOS `say`, decodes it to 16 kHz
mono float32 via ffmpeg, and verifies Whisper transcribes it.
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import SAMPLE_RATE, WHISPER_MODEL
from app.pipeline.asr import WhisperASR

SENTENCE = "Patient reports a headache for three days and took Tylenol 500 milligrams twice daily."


def synth_to_float32(text: str) -> np.ndarray:
    subprocess.run(["say", "-o", "/tmp/hv_smoke.aiff", text], check=True)
    raw = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", "/tmp/hv_smoke.aiff",
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def main() -> int:
    print("synthesizing speech...")
    audio = synth_to_float32(SENTENCE)
    print(f"audio: {len(audio) / SAMPLE_RATE:.1f}s @ {SAMPLE_RATE} Hz")

    asr = WhisperASR(WHISPER_MODEL)
    print(f"loading model {asr.model_repo} (first run downloads weights)...")
    t0 = time.perf_counter()
    asr.warmup()
    print(f"warmup/load: {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    text, conf = asr.transcribe(audio)
    dt = (time.perf_counter() - t0) * 1000
    print(f"\nexpected : {SENTENCE}")
    print(f"got      : {text}")
    print(f"confidence: {conf:.3f}   asr: {dt:.0f}ms   rtf: {dt / (len(audio) / SAMPLE_RATE * 1000):.2f}")
    return 0 if text else 1


if __name__ == "__main__":
    sys.exit(main())
