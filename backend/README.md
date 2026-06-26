# HealthVoice — Inference Service

The Python backend for the HealthVoice clinical scribe. A single FastAPI process runs
the whole on-device pipeline and exposes it over a WebSocket (live audio) plus a few
REST endpoints (SOAP generation, FHIR push, enrollment, audit):

```
Mic (browser) ─▶ VAD (silero) ─▶ Whisper ASR (MLX) ─▶ speaker ID (ECAPA)
              ─▶ medical NER ─▶ SOAP note (OpenAI) ─▶ clinician review ─▶ FHIR (HAPI R4)
```

Audio and PHI never leave the machine — ASR, speaker ID, and NER all run locally. The
**only** network egress is the SOAP note-structuring call to OpenAI.

## Requirements

- [uv](https://docs.astral.sh/uv/) — pins CPython 3.12 (system Python 3.14 has no ML
  wheels yet) and installs everything. `uv` downloads the interpreter automatically.
- Docker — for the local HAPI FHIR R4 server (`docker-compose.yml`).
- An OpenAI API key (SOAP note). *Optional:* a Hugging Face token for the pyannote
  diarization refine pass.

## Run

```bash
cp .env.example .env          # add OPENAI_API_KEY (and optionally HF_TOKEN)
uv sync                       # creates .venv (py3.12) + installs deps
uv run uvicorn app.main:app --port 8000

docker compose up -d          # HAPI FHIR R4 at http://localhost:8080/fhir
```

First boot downloads the Whisper / NER / ECAPA weights and warms them (~1 min).
`GET /health` returns `{"ready": true}` once warm and also reports `enrolled`,
`soapAvailable`, `diarizationAvailable`, and `fhirServer`.

> Run **without** `--reload` for normal use: the models hold large in-memory state, so
> a reload-triggered restart re-downloads/re-warms. Kill and relaunch to apply changes.

## Configuration (env vars)

| Var | Default | Notes |
|-----|---------|-------|
| `OPENAI_API_KEY` | — | Required for SOAP generation (only egress). |
| `OPENAI_MODEL` | `gpt-5.5` | Note-structuring model. |
| `HF_TOKEN` | — | Optional; enables the gated pyannote diarization refine pass. |
| `FINAL_MODEL` | `mlx-community/whisper-large-v3-turbo` | Accurate per-sentence ASR. |
| `DRAFT_MODEL` | `mlx-community/whisper-base.en-mlx` | Fast live-partial ASR. |
| `NER_MODEL` | `d4data/biomedical-ner-all` | Biomedical entity extraction. |
| `CONFIDENCE_THRESHOLD` | `0.55` | Below this, a segment is flagged for review (and skips NER). |
| `NO_SPEECH_THRESHOLD` / `COMPRESSION_RATIO_THRESHOLD` | `0.6` / `2.4` | Drop Whisper silence-hallucinations (repeats, stock phrases). |
| `FHIR_BASE_URL` | `http://localhost:8080/fhir` | Target FHIR R4 server. |

(See `app/config.py` for the full list — VAD, voiceprint threshold, audit path, etc.)

## API surface

**WebSocket `/ws/transcribe`** — client streams binary Int16 PCM (16 kHz mono LE);
server streams JSON: `ready` · `vad` · `partial` · `final` (with `confidence`,
`speaker`, `entities`, `safety`, `correction`) · `finalsComplete` · `diarization` /
`diarizationSkipped`. Send `{"type":"end_encounter"}` to finalize.

**REST**

| Method + path | Purpose |
|---------------|---------|
| `GET /health` | Readiness + capability flags |
| `POST /soap` | Generate a SOAP note from a transcript (stateless, retryable) → note + safety alerts + completeness |
| `POST /fhir/push` | File the reviewed note as a FHIR transaction Bundle (requires clinician); writes audit + Provenance |
| `GET /fhir/status` | FHIR server reachability |
| `GET /enroll/status` · `POST /enroll` · `DELETE /enroll` | Nurse voiceprint enrollment |
| `GET /audit` | Recent approve-and-file events |

## Layout

```
app/
  main.py            FastAPI app: WebSocket orchestration + REST endpoints
  config.py          env-driven configuration
  pipeline/
    vad.py           StreamingVAD (silero endpointing)
    asr.py           WhisperASR (MLX; hybrid draft/final; hallucination filtering)
    diarization.py   ECAPA voiceprint + optional pyannote refine
    ner.py           biomedical NER → clinical buckets
    soap.py          SOAP note generation (OpenAI, strict JSON schema)
    analysis.py      deterministic safeguards (safety / completeness / corrections)
    audit.py         on-device JSONL audit log
    fhir.py          FHIR R4 transaction Bundle (+ Provenance / attester)
scripts/
  smoke_asr.py       offline ASR sanity check
  stream_client.py   WebSocket test client
```

> Prototype — synthetic data only. Not a medical device.
