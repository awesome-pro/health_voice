"""End-to-end Phase 2 test: enroll a nurse voice, then stream a two-speaker
conversation and check each finalized line gets the right Nurse/Patient label.

Server must be running (default :8001):  uv run python scripts/test_phase2.py
"""
import asyncio
import json
import subprocess
import urllib.request

import numpy as np
import websockets

SR = 16000
API = "http://localhost:8001"
WS = "ws://localhost:8001/ws/transcribe"


def synth(text: str, voice: str) -> np.ndarray:
    subprocess.run(["say", "-v", voice, "-o", "/tmp/p2.aiff", text], check=True)
    raw = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", "/tmp/p2.aiff", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def pcm(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


CONVO = [
    ("Alex", "nurse", "Good morning, I am going to document your visit today."),
    ("Samantha", "patient", "I have had a sore throat and a fever since yesterday."),
    ("Alex", "nurse", "Your temperature is one hundred point four, I will note that."),
    ("Daniel", "patient", "I also feel very tired and my whole body aches a lot."),
]


async def main() -> None:
    # 1) Enroll the nurse (Alex).
    enroll = synth("Hello, I am the attending nurse and I will be documenting today's encounter in our system.", "Alex")
    req = urllib.request.Request(API + "/enroll", data=pcm(enroll),
                                 headers={"Content-Type": "application/octet-stream"}, method="POST")
    print("enroll:", json.load(urllib.request.urlopen(req)))
    print("status:", json.load(urllib.request.urlopen(API + "/enroll/status")))

    # 2) Build the conversation audio with pauses.
    gap = np.zeros(int(0.8 * SR), dtype=np.float32)
    parts, expected = [np.zeros(int(0.3 * SR), dtype=np.float32)], []
    for voice, role, text in CONVO:
        parts += [synth(text, voice), gap]
        expected.append(role)
    data = pcm(np.concatenate(parts))

    finals = []
    async with websockets.connect(WS, max_size=None) as ws:
        async def rx():
            async for raw in ws:
                m = json.loads(raw)
                if m["type"] == "final":
                    finals.append(m)
                    print(f"  FINAL [{m['speaker']:>7} {m['speakerSim']:+.2f}] {m['text']}")
        recv = asyncio.create_task(rx())
        frame = int(0.03 * SR) * 2
        for i in range(0, len(data), frame):
            await ws.send(data[i:i + frame]); await asyncio.sleep(0.03)
        await ws.send(json.dumps({"type": "end_encounter"}))
        await asyncio.sleep(5)
        recv.cancel()

    # 3) Score.
    print("\nresult:")
    got = [f["speaker"] for f in finals]
    ok = sum(1 for g, e in zip(got, expected) if g == e)
    for f, e in zip(finals, expected):
        mark = "✓" if f["speaker"] == e else "✗"
        print(f"  {mark} expected {e:>7}  got {f['speaker']:>7}  | {f['text'][:50]}")
    print(f"\n{ok}/{len(expected)} segments labeled correctly "
          f"({len(finals)} finals for {len(expected)} sentences)")


if __name__ == "__main__":
    asyncio.run(main())
