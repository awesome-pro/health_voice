"""End-to-end streaming test: feed multi-sentence clinical audio to the running
server over the real WebSocket at real-time pace and print partial/final events
with wall-clock timing.

Usage (server must be running on :8000):
    uv run python scripts/stream_client.py
"""
import asyncio
import json
import subprocess
import time

import numpy as np
import websockets

SR = 16000
WS = "ws://localhost:8000/ws/transcribe"

SENTENCES = [
    "Patient reports a headache for the past three days.",
    "She took Tylenol five hundred milligrams twice daily.",
    "Blood pressure is one thirty over eighty five, temperature ninety nine point one.",
]


def synth(text: str) -> np.ndarray:
    subprocess.run(["say", "-o", "/tmp/hv_s.aiff", text], check=True)
    raw = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", "/tmp/hv_s.aiff", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def build_audio() -> np.ndarray:
    gap = np.zeros(int(0.7 * SR), dtype=np.float32)  # pause between sentences
    parts = [np.zeros(int(0.3 * SR), dtype=np.float32)]
    for s in SENTENCES:
        parts.append(synth(s))
        parts.append(gap)
    return np.concatenate(parts)


async def main() -> None:
    print("synthesizing clinical audio with pauses...")
    audio = build_audio()
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    print(f"audio: {len(audio) / SR:.1f}s\n")

    async with websockets.connect(WS, max_size=None) as ws:
        t0 = time.perf_counter()

        async def receiver() -> None:
            async for raw in ws:
                m = json.loads(raw)
                t = time.perf_counter() - t0
                if m["type"] == "partial":
                    print(f"[{t:5.1f}s] partial : {m['text']}")
                elif m["type"] == "final":
                    flag = "  ⚠ REVIEW" if m["lowConfidence"] else ""
                    print(f"[{t:5.1f}s] FINAL   : {m['text']}  ({m['confidence']:.2f}, {m['asrMs']:.0f}ms){flag}")
                elif m["type"] in ("ready", "vad", "error"):
                    print(f"[{t:5.1f}s] {m['type']}: {m.get('state') or m.get('message') or m.get('model','')}")

        recv = asyncio.create_task(receiver())

        # Stream 30ms frames at real-time pace.
        frame = int(0.03 * SR) * 2  # bytes (int16)
        for i in range(0, len(pcm), frame):
            await ws.send(pcm[i:i + frame])
            await asyncio.sleep(0.03)

        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.sleep(4)  # let the final model finish + flush
        recv.cancel()


if __name__ == "__main__":
    asyncio.run(main())
