"""Runtime configuration for the inference service.

All values are overridable via environment variables so the same code runs
on the Mac today and on a dedicated GPU server later without edits.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load secrets from backend/.env (gitignored) so keys never live in code or shell
# history. Real environment variables still win over .env values.
load_dotenv(BASE_DIR / ".env")

# Hybrid ASR: a fast DRAFT model streams live interim text, and an accurate
# FINAL model re-transcribes each completed sentence once for the committed note.
#
# Whisper's encoder cost is dominated by a fixed per-call overhead (large-v3-turbo
# ~2s/call on Apple Silicon), so we cannot call it many times per second. base.en
# is ~270ms/call and good enough for a live draft; turbo gives the accurate final.
#
# For maximum fidelity to the case study ("Whisper Large v3"), set
#   FINAL_MODEL=mlx-community/whisper-large-v3-mlx   (slower, slightly better)
DRAFT_MODEL = os.environ.get("DRAFT_MODEL", "mlx-community/whisper-base.en-mlx")
FINAL_MODEL = os.environ.get("FINAL_MODEL", "mlx-community/whisper-large-v3-turbo")

# Audio contract with the browser: mono, 16 kHz, Int16 PCM little-endian.
SAMPLE_RATE = 16000

# Voice Activity Detection (silero) endpointing.
VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.5"))
MIN_SILENCE_MS = int(os.environ.get("MIN_SILENCE_MS", "500"))
SPEECH_PAD_MS = int(os.environ.get("SPEECH_PAD_MS", "150"))

# Live interim cadence + the rolling window the draft model sees (seconds).
PARTIAL_INTERVAL = float(os.environ.get("PARTIAL_INTERVAL", "0.7"))
PARTIAL_MAX_SEC = float(os.environ.get("PARTIAL_MAX_SEC", "12"))

# Force-finalize a sentence if the speaker never pauses (seconds) so the buffer
# stays bounded and committed text keeps flowing during long continuous speech.
MAX_UTTERANCE_SEC = float(os.environ.get("MAX_UTTERANCE_SEC", "14"))

# Below this ASR confidence we flag a segment for nurse review. Whisper's
# exp(avg_logprob) proxy sits ~0.65 even for clean speech, so the case study's
# nominal 0.7 over-flags; 0.55 makes REVIEW fire only on genuinely shaky audio.
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.55"))

# Whisper, trained on web audio, hallucinates stock phrases or repeated tokens
# when fed silence / non-speech (e.g. "love love", "thanks for watching", a
# medical disclaimer). We drop a decoded segment when it looks like that, before
# it can pollute the transcript or the NER stage:
#   - no_speech_prob above NO_SPEECH_THRESHOLD  -> almost certainly not speech
#   - compression_ratio above this threshold     -> degenerate / repetitive text
NO_SPEECH_THRESHOLD = float(os.environ.get("NO_SPEECH_THRESHOLD", "0.6"))
COMPRESSION_RATIO_THRESHOLD = float(os.environ.get("COMPRESSION_RATIO_THRESHOLD", "2.4"))

# --- Phase 2: speaker diarization / voiceprint ---------------------------------
# Public (non-gated) ECAPA model for per-segment speaker embeddings -> live labels.
SPEAKER_EMBEDDING_MODEL = os.environ.get(
    "SPEAKER_EMBEDDING_MODEL", "speechbrain/spkrec-ecapa-voxceleb"
)
# Cosine similarity (to the enrolled nurse voiceprint) at/above which a segment is
# labeled "nurse"; below it, "patient". ECAPA same-speaker cosine ~0.4-0.7.
SPEAKER_THRESHOLD = float(os.environ.get("SPEAKER_THRESHOLD", "0.35"))
# Where the enrolled nurse voiceprint is persisted (survives restarts).
ENROLL_PATH = Path(os.environ.get("ENROLL_PATH", str(BASE_DIR / "enrollment" / "nurse_voiceprint.npy")))

# Gated pyannote pipeline for the OPTIONAL full diarization refine pass on Stop
# (Phase 2b). pyannote.audio 4.x ships "speaker-diarization-community-1"; both it
# and its dependencies are gated, so this needs HF_TOKEN + accepting the model
# terms at huggingface.co/pyannote/speaker-diarization-community-1. If not
# accepted, the refine pass is skipped silently — live ECAPA labels are unaffected.
DIARIZATION_MODEL = os.environ.get("DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
# The pyannote refine pass runs on CPU over the whole encounter and can take tens
# of seconds. It is OPTIONAL (live ECAPA labels already stand), so bound it: if it
# exceeds this many seconds the refine is skipped and never blocks the SOAP result.
REFINE_TIMEOUT = float(os.environ.get("REFINE_TIMEOUT", "45"))

# --- Phase 3: clinical entity extraction (medical NER) -------------------------
# Public BERT-family biomedical NER (~84 fine-grained clinical labels). Ungated,
# runs locally on CPU. Maps to clinical buckets (symptom/condition/medication/...)
# for the transcript highlights and feeds the Phase 4 SOAP note.
NER_MODEL = os.environ.get("NER_MODEL", "d4data/biomedical-ner-all")
# Drop entity spans below this model confidence to keep highlights clean.
NER_MIN_SCORE = float(os.environ.get("NER_MIN_SCORE", "0.5"))

# --- Phase 4: SOAP note generation (OpenAI GPT-4o) -----------------------------
# The only stage that calls a hosted API. Set OPENAI_API_KEY in backend/.env.
# Everything before this (ASR, voiceprint, NER) runs fully on-device.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")
# Low temperature keeps the note faithful. Reasoning models (gpt-5/o-series)
# reject a custom temperature; the generator falls back to the model default.
SOAP_TEMPERATURE = float(os.environ.get("SOAP_TEMPERATURE", "0.2"))

# --- Phase 5: FHIR push (local HAPI FHIR R4) -----------------------------------
# The approved + edited SOAP note is written to a local HAPI FHIR server running
# in Docker (see docker-compose.yml), so patient data never leaves the machine.
# Point FHIR_BASE_URL at any FHIR R4 endpoint to target a different server.
FHIR_BASE_URL = os.environ.get("FHIR_BASE_URL", "http://localhost:8080/fhir")
FHIR_TIMEOUT = float(os.environ.get("FHIR_TIMEOUT", "30"))

# --- Phase 6: on-device audit trail -------------------------------------------
# Every approve-and-file action is appended here (who/when/what changed). Stays
# on the machine; gitignored. The FHIR Provenance resource mirrors it server-side.
AUDIT_PATH = Path(os.environ.get("AUDIT_PATH", str(BASE_DIR / "audit" / "audit-log.jsonl")))
