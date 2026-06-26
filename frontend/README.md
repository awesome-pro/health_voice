# HealthVoice — Frontend

The clinical console for the **HealthVoice** voice-AI scribe. It captures
microphone audio, streams it to the backend over a WebSocket, and renders the
full encounter live: transcript, nurse voice enrollment, **Nurse / Patient**
speaker labels, extracted clinical entities, an editable **SOAP note** with
safety checks, and a clinician-signed **FHIR** push.

The page downsamples mic audio to **16 kHz mono Int16 PCM** in the browser and
streams it over a WebSocket; the SOAP note, safeguards, and FHIR filing run over
REST (`POST /soap`, `POST /fhir/push` — see the backend README). UI is built
with shadcn-style primitives (`app/components/ui.tsx`) on Tailwind.

### Speaker ID

- **Nurse voice enrollment** — record ~10 seconds of the nurse's voice once.
  The console POSTs the captured Int16 PCM to the backend, which builds a voice
  profile. Status is shown as **Enrolled ✓** / **Not enrolled**, and the
  enrolled voice can be cleared / re-recorded.
- **Live speaker labels** — each finalized segment carries a `speaker`
  (`nurse` / `patient` / `unknown`) rendered as a chip and a tinted left
  border (teal for Nurse, violet for Patient). Labels only appear once the
  nurse voice is enrolled.
- **Refine pass** — when an encounter ends, the client sends
  `{"type":"end_encounter"}` and briefly keeps the socket open to receive an
  optional `diarization` message (the backend's pyannote pass), which corrects
  speaker labels in place. This step is skipped gracefully if the backend has
  no Hugging Face token.

> Prototype — synthetic data only. ASR models run locally (on-device).

## Tech

- Next.js 14 (App Router) · React 18 · TypeScript · Tailwind CSS
- Web Audio API (`AudioContext` @ 16 kHz + `AudioWorklet`) for capture/resampling
- WebSocket for bidirectional streaming

## Prerequisites

- Node 18+ (developed against Node 25 / npm 11)
- The HealthVoice backend running and exposing the transcription WebSocket
  (default `ws://localhost:8000/ws/transcribe`)
- A browser with `AudioWorklet` support (Chrome, Edge, Firefox, Safari 14.1+)
- Microphone access. Note: `getUserMedia` requires a **secure context** —
  `http://localhost` is allowed; on other hosts you need HTTPS.

## Setup & run

```bash
npm install
npm run dev
```

Then open <http://localhost:3000>.

Make sure the backend is listening on `:8000` (or whatever you configure below).

### Configuration

Two env vars configure the backend location:

- `NEXT_PUBLIC_WS_URL` — transcription WebSocket, defaults to
  `ws://localhost:8000/ws/transcribe`.
- `NEXT_PUBLIC_API_URL` — HTTP base URL for the enrollment endpoints
  (`/enroll/status`, `/enroll`), defaults to `http://localhost:8000`.

```bash
cp .env.local.example .env.local
# edit NEXT_PUBLIC_WS_URL / NEXT_PUBLIC_API_URL if your backend lives elsewhere
```

## How it works

1. Click **Start encounter** → the browser prompts for mic access.
2. `getUserMedia({ audio: { channelCount: 1, echoCancellation, noiseSuppression } })`
   captures mono audio.
3. An `AudioContext({ sampleRate: 16000 })` resamples the input to 16 kHz.
4. `public/pcm-worklet.js` (an `AudioWorklet` processor) batches the Float32
   frames into ~1024-sample chunks and posts them to the main thread.
5. The main thread converts each batch to **Int16 PCM little-endian** and sends
   it as a **binary** WebSocket frame.
6. The backend streams back JSON messages, which drive the UI.

### Wire protocol

**Client → server**

- Binary frames: raw Int16 PCM (16 kHz, mono, little-endian).
- On stop: a text frame `{"type":"end_encounter"}`. Mic capture stops
  immediately; the socket lingers up to ~8 s to receive the `diarization`
  refine message, then closes.

**Server → client** (text JSON)

| `type`        | Payload                                                                              | UI effect                                                   |
| ------------- | ----------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `ready`       | `{ model, enrolled }`                                                                | Connected pill + seeds enrollment status                   |
| `vad`         | `{ state: "speech" \| "silence" }`                                                  | Listening / Speaking indicator                             |
| `partial`     | `{ segmentId, text }`                                                                | Live italic interim line                                   |
| `final`       | `{ segmentId, text, confidence, lowConfidence, asrMs, speaker, speakerSim }`         | Permanent line + speaker chip; low-confidence flagged amber |
| `diarization` | `{ speakerCount, segments: [{ segmentId, speaker, cluster }], summary }`             | Corrects speaker labels in place; shows summary under header |
| `error`       | `{ message }`                                                                        | Error banner                                               |

When a `final` arrives for a segment, its interim line is cleared. The
`diarization` message arrives only after `end_encounter` and only when the
backend has a Hugging Face token; its absence is handled gracefully.

**Enrollment (HTTP, `NEXT_PUBLIC_API_URL`)**

| Method + path        | Body                                  | Response                                              |
| -------------------- | ------------------------------------- | ---------------------------------------------------- |
| `GET /enroll/status` | —                                     | `{ enrolled, model, threshold }`                     |
| `POST /enroll`       | raw Int16 PCM (`application/octet-stream`) | `{ ok, durationSec, model }` or `{ ok:false, error }` |
| `DELETE /enroll`     | —                                     | `{ ok: true, enrolled: false }`                      |

## Scripts

| Command         | Description                  |
| --------------- | ---------------------------- |
| `npm run dev`   | Start the dev server (:3000) |
| `npm run build` | Production build             |
| `npm run start` | Serve the production build   |

## Project layout

```
frontend/
├─ app/
│  ├─ components/
│  │  └─ TranscriptionConsole.tsx   # client component: audio + WS + UI
│  ├─ globals.css
│  ├─ layout.tsx
│  └─ page.tsx
├─ public/
│  └─ pcm-worklet.js                # AudioWorklet processor (plain JS)
├─ .env.local.example
├─ next.config.mjs
├─ postcss.config.mjs
├─ tailwind.config.ts
├─ tsconfig.json
└─ package.json
```
