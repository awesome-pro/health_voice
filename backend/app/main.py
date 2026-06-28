"""HealthVoice inference service — Phases 1-2.

Phase 1: real-time transcription (Audio Capture -> VAD -> hybrid Whisper ASR).
Phase 2: speaker diarization — live nurse/patient labels from an enrolled
         voiceprint, plus an optional full pyannote refine pass on encounter end.

Audio ingestion is decoupled from transcription via an asyncio.Queue so the mic
stream is never blocked by a model call.

Protocol
--------
Client -> server:
  * binary frames : Int16 PCM, mono, 16 kHz, little-endian
  * text JSON     : {"type":"stop"}           -> finalize current utterance
                    {"type":"end_encounter"}  -> finalize + run pyannote refine pass

Server -> client (text JSON):
  * {"type":"ready","model":...,"draftModel":...,"enrolled":bool}
  * {"type":"vad","state":"speech"|"silence"}
  * {"type":"partial","segmentId":N,"text":...}
  * {"type":"final","segmentId":N,"text":...,"confidence":c,"lowConfidence":bool,
     "asrMs":ms,"speaker":"nurse"|"patient"|"unknown","speakerSim":s,
     "entities":[{"text":...,"label":...,"category":...,"start":i,"end":j,"score":s}]}
  * {"type":"diarization","speakerCount":k,"segments":[{"segmentId":N,"speaker":...}],
     "summary":"..."}                          (refine pass; only if HF token present)
  * {"type":"soapStatus","state":"generating"} (SOAP note generation started)
  * {"type":"soap","note":{...}}               (structured SOAP note, GPT-4o)
  * {"type":"soapUnavailable","message":...}   (no OPENAI_API_KEY configured)
  * {"type":"error","message":...}
"""
import asyncio
import json
import time

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import config
from .pipeline import analysis, audit
from .pipeline.asr import WhisperASR
from .pipeline.diarization import FullDiarizer, SpeakerEmbedder, VoiceprintStore
from .pipeline.fhir import FhirPublisher
from .pipeline.ner import ClinicalNER
from .pipeline.soap import SoapGenerator
from .pipeline.vad import StreamingVAD

app = FastAPI(title="HealthVoice Inference Service", version="0.2.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

SR = config.SAMPLE_RATE

draft_asr = WhisperASR(config.DRAFT_MODEL)
final_asr = WhisperASR(config.FINAL_MODEL)
embedder = SpeakerEmbedder()
voiceprint = VoiceprintStore(config.ENROLL_PATH)
diarizer = FullDiarizer()
ner = ClinicalNER(config.NER_MODEL)
soap = SoapGenerator()
publisher = FhirPublisher()

# MLX/torch calls are blocking and not thread-safe; serialize them process-wide.
model_lock = asyncio.Lock()


@app.on_event("startup")
async def _startup() -> None:
    await asyncio.to_thread(draft_asr.warmup)
    await asyncio.to_thread(final_asr.warmup)
    await asyncio.to_thread(embedder.warmup)
    await asyncio.to_thread(ner.warmup)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "draftModel": config.DRAFT_MODEL,
        "finalModel": config.FINAL_MODEL,
        "nerModel": config.NER_MODEL,
        "ready": draft_asr.ready and final_asr.ready and embedder.ready and ner.ready,
        "enrolled": voiceprint.enrolled,
        "diarizationAvailable": diarizer.available,
        "soapAvailable": soap.available,
        "fhirServer": config.FHIR_BASE_URL,
    }


# Quiet route: something on localhost polls this; not part of HealthVoice.
@app.get("/api/v1/sessions/active")
async def _sessions_active() -> dict:
    return {"sessions": []}


# ----------------------------- enrollment ------------------------------------
@app.get("/enroll/status")
async def enroll_status() -> dict:
    return {"enrolled": voiceprint.enrolled, "model": config.SPEAKER_EMBEDDING_MODEL,
            "threshold": config.SPEAKER_THRESHOLD}


@app.post("/enroll")
async def enroll(request: Request) -> dict:
    """Body: raw Int16 PCM, mono, 16 kHz. Computes + stores the nurse voiceprint."""
    raw = await request.body()
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    seconds = len(audio) / SR
    if seconds < 3.0:
        return {"ok": False, "error": "Need at least 3 seconds of speech.", "durationSec": round(seconds, 1)}
    async with model_lock:
        emb = await asyncio.to_thread(embedder.embed, audio)
    voiceprint.set(emb)
    return {"ok": True, "durationSec": round(seconds, 1), "model": config.SPEAKER_EMBEDDING_MODEL}


@app.delete("/enroll")
async def unenroll() -> dict:
    voiceprint.clear()
    return {"ok": True, "enrolled": False}


# ------------------------------ FHIR push (Phase 5) --------------------------
class PatientIn(BaseModel):
    name: str = ""
    mrn: str = ""


class MedicationIn(BaseModel):
    name: str = ""
    dosage: str = ""
    frequency: str = ""


class SoapNoteIn(BaseModel):
    chiefComplaint: str = ""
    subjective: list[str] = []
    objective: list[str] = []
    assessment: list[str] = []
    plan: list[str] = []
    medications: list[MedicationIn] = []
    reviewFlags: list[str] = []


class AuditEditIn(BaseModel):
    field: str = ""
    detail: str = ""


class FhirPushIn(BaseModel):
    patient: PatientIn
    note: SoapNoteIn
    clinician: str = ""
    edits: list[AuditEditIn] = []


@app.get("/fhir/status")
async def fhir_status() -> dict:
    return {"server": config.FHIR_BASE_URL, "reachable": await publisher.status()}


@app.post("/fhir/push")
async def fhir_push(body: FhirPushIn) -> dict:
    """Write the clinician-approved SOAP note to FHIR as a transaction Bundle."""
    if not body.patient.name.strip():
        return {"ok": False, "error": "Patient name is required before pushing to FHIR."}
    if not body.clinician.strip():
        return {"ok": False, "error": "Clinician name is required to sign and file the note."}
    edits = [e.model_dump() for e in body.edits]
    try:
        result = await publisher.push(
            body.patient.model_dump(),
            body.note.model_dump(),
            clinician=body.clinician.strip(),
            edits=edits,
        )
        # Append to the on-device audit trail (who/when/what changed).
        audit.append_event({
            "action": "approve_and_push",
            "clinician": body.clinician.strip(),
            "patient": {"name": body.patient.name.strip(), "mrn": body.patient.mrn.strip()},
            "model": config.OPENAI_MODEL,
            "editCount": len(edits),
            "edits": edits,
            "resources": result.get("resources", []),
        })
        return {"ok": True, "clinician": body.clinician.strip(),
                "editCount": len(edits), **result}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "server": config.FHIR_BASE_URL}


@app.get("/audit")
async def audit_log(limit: int = 50) -> dict:
    """Recent approve-and-file events from the on-device audit trail."""
    return {"events": audit.recent(limit)}


# ------------------------------ SOAP generation (Phase 4) --------------------
# SOAP generation is exposed as a plain REST call (not only over the WebSocket)
# so the frontend can (re)generate the note from a stored transcript — a retry
# that needs no re-recording, and that is never coupled to the audio socket's
# lifetime or to the optional, slow pyannote refine pass.
class EntityIn(BaseModel):
    text: str = ""
    label: str = ""
    category: str = "other"
    start: int = 0
    end: int = 0
    score: float = 0.0


class TranscriptSegmentIn(BaseModel):
    segmentId: int = 0
    speaker: str = "unknown"
    text: str = ""
    entities: list[EntityIn] = []


class SoapGenerateIn(BaseModel):
    segments: list[TranscriptSegmentIn] = []


@app.post("/soap")
async def soap_generate(body: SoapGenerateIn) -> dict:
    """Generate a SOAP note from a speaker-labeled transcript (stateless / retryable)."""
    if not soap.available:
        return {
            "ok": False,
            "unavailable": True,
            "message": "Set OPENAI_API_KEY in backend/.env to generate the SOAP note.",
        }
    segs = [s.model_dump() for s in body.segments if s.text.strip()]
    if not segs:
        return {"ok": False, "error": "No transcript available to summarize."}
    ordered = sorted(segs, key=lambda x: x["segmentId"])
    lines = []
    for s in ordered:
        who = s["speaker"].capitalize() if s["speaker"] in ("nurse", "patient") else "Speaker"
        lines.append(f"{who}: {s['text']}")
    transcript = "\n".join(lines)
    entity_summary = _build_entity_summary(segs)
    try:
        # Network call, not a local model -> no model_lock.
        note = await asyncio.to_thread(soap.generate, transcript, entity_summary)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    # --- clinical safeguards over the transcript + generated note ---
    safety_alerts: list[dict] = []
    corrections: list[dict] = []
    for s in ordered:
        for alert in analysis.scan_safety(s["text"]):
            safety_alerts.append({**alert, "segmentId": s["segmentId"]})
        marker = analysis.detect_correction(s["text"])
        if marker:
            corrections.append({"segmentId": s["segmentId"], "marker": marker["marker"],
                                "text": s["text"]})

    all_entities = [e for s in segs for e in s.get("entities", [])]
    completeness = analysis.check_completeness(all_entities, note)
    # Surface any uncaptured clinical terms as a review flag on the note itself.
    if completeness["missing"]:
        note.setdefault("reviewFlags", []).append(
            "Mentioned but not in note (verify): " + ", ".join(completeness["missing"])
        )

    return {
        "ok": True,
        "note": note,
        "model": soap.model,
        "safetyAlerts": safety_alerts,
        "corrections": corrections,
        "completeness": completeness,
    }


# ------------------------------ transcription --------------------------------
async def _run(model: WhisperASR, audio: np.ndarray) -> tuple[str, float, float]:
    t0 = time.perf_counter()
    async with model_lock:
        text, conf = await asyncio.to_thread(model.transcribe, audio)
    return text, conf, (time.perf_counter() - t0) * 1000.0


@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket) -> None:
    await ws.accept()
    vad = StreamingVAD()
    jobs: asyncio.Queue = asyncio.Queue()

    # Per-encounter accumulators for the optional pyannote refine pass.
    encounter: list[np.ndarray] = []
    total = 0          # running sample count over the whole encounter
    segments: list[dict] = []  # finalized segments: {segmentId, start, end, speaker, emb}

    await ws.send_json({
        "type": "ready",
        "model": config.FINAL_MODEL,
        "draftModel": config.DRAFT_MODEL,
        "enrolled": voiceprint.enrolled,
    })

    async def worker() -> None:
        while True:
            job = await jobs.get()
            try:
                if job is None:
                    return
                kind = job[0]
                if kind == "partial":
                    _, seg_id, audio = job
                    text, _c, _ms = await _run(draft_asr, audio)
                    pending["partial"] = False
                    if text and any(c.isalnum() for c in text):
                        await ws.send_json({"type": "partial", "segmentId": seg_id, "text": text})
                else:  # final
                    _, seg_id, audio, start, end = job
                    text, conf, ms = await _run(final_asr, audio)
                    if not text:
                        continue
                    # A low-confidence (REVIEW) segment is untrusted: run the
                    # voiceprint for labeling, but skip NER so shaky audio never
                    # feeds entities into the panel, SOAP input, or completeness.
                    low_conf = conf < config.CONFIDENCE_THRESHOLD
                    async with model_lock:
                        emb = await asyncio.to_thread(embedder.embed, audio)
                        entities = [] if low_conf else await asyncio.to_thread(ner.extract, text)
                    speaker, sim = voiceprint.label(emb)
                    # Live clinical safeguards (cheap, deterministic).
                    safety = analysis.scan_safety(text)
                    correction = analysis.detect_correction(text)
                    segments.append({"segmentId": seg_id, "start": start, "end": end,
                                     "speaker": speaker, "emb": emb, "entities": entities,
                                     "text": text})
                    await ws.send_json({
                        "type": "final", "segmentId": seg_id, "text": text,
                        "confidence": round(conf, 3),
                        "lowConfidence": low_conf,
                        "asrMs": round(ms, 1),
                        "speaker": speaker, "speakerSim": round(sim, 3),
                        "entities": entities,
                        "safety": safety,
                        "correction": bool(correction),
                    })
            except Exception as exc:
                await ws.send_json({"type": "error", "message": str(exc)})
            finally:
                jobs.task_done()

    worker_task = asyncio.create_task(worker())

    utterance = np.zeros(0, dtype=np.float32)
    speech = False
    seg = 0
    utt_start = 0
    last_partial = 0.0
    pending = {"partial": False}

    def enqueue_final(audio: np.ndarray, seg_id: int, start: int, end: int) -> None:
        if len(audio) >= int(0.2 * SR):
            jobs.put_nowait(("final", seg_id, audio.copy(), start, end))

    async def refine() -> None:
        """Phase 2b: run pyannote over the whole encounter and correct labels."""
        if not diarizer.available or not segments:
            return
        try:
            full = np.concatenate(encounter) if encounter else np.zeros(0, dtype=np.float32)
            async with model_lock:
                turns = await asyncio.to_thread(diarizer.diarize, full)
            corrections = _reconcile(turns, segments, voiceprint)
            speaker_count = len({t["speaker"] for t in turns})
            await ws.send_json({
                "type": "diarization",
                "speakerCount": speaker_count,
                "segments": corrections,
                "summary": f"pyannote detected {speaker_count} speaker(s) across {len(turns)} turns.",
            })
        except Exception as exc:
            # The refine pass is an OPTIONAL accuracy polish — the live ECAPA
            # voiceprint labels already stand. Never surface it as a fatal error
            # (e.g. gated pyannote model not yet accepted); just skip it quietly.
            print(f"[diarization] refine skipped: {exc}")
            await ws.send_json({"type": "diarizationSkipped"})

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            if msg.get("text") is not None:
                ctl = json.loads(msg["text"])
                if ctl.get("type") in ("stop", "end_encounter"):
                    if speech:
                        speech = False
                        enqueue_final(utterance, seg, utt_start, total)
                        utterance = np.zeros(0, dtype=np.float32)
                        await ws.send_json({"type": "vad", "state": "silence"})
                        vad.reset()
                    if ctl["type"] == "end_encounter":
                        await jobs.join()      # ensure all finals processed
                        # Signal that the transcript is complete. The frontend
                        # generates the SOAP note via POST /soap from here — the
                        # note no longer waits behind (or is blocked by) refine.
                        await ws.send_json({"type": "finalsComplete"})
                        # Phase 2b refine is OPTIONAL + slow: bound it so it can
                        # never hang the socket. On timeout, skip it quietly.
                        try:
                            await asyncio.wait_for(refine(), timeout=config.REFINE_TIMEOUT)
                        except asyncio.TimeoutError:
                            print("[diarization] refine timed out; skipped")
                            try:
                                await ws.send_json({"type": "diarizationSkipped"})
                            except Exception:
                                pass
                continue

            data = msg.get("bytes")
            if not data:
                continue
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            encounter.append(audio)
            total += len(audio)

            for ev in vad.process(audio):
                if "start" in ev and not speech:
                    speech = True
                    seg += 1
                    utt_start = total - len(audio)
                    utterance = np.zeros(0, dtype=np.float32)
                    last_partial = time.time()
                    pending["partial"] = False
                    await ws.send_json({"type": "vad", "state": "speech"})
                elif "end" in ev and speech:
                    speech = False
                    enqueue_final(utterance, seg, utt_start, total)
                    utterance = np.zeros(0, dtype=np.float32)
                    await ws.send_json({"type": "vad", "state": "silence"})

            if not speech:
                continue

            utterance = np.concatenate([utterance, audio])

            if len(utterance) >= int(config.MAX_UTTERANCE_SEC * SR):
                enqueue_final(utterance, seg, utt_start, total)
                seg += 1
                utt_start = total
                utterance = np.zeros(0, dtype=np.float32)
                last_partial = time.time()
                pending["partial"] = False
                continue

            now = time.time()
            if (not pending["partial"] and (now - last_partial) >= config.PARTIAL_INTERVAL
                    and len(utterance) >= int(0.4 * SR)):
                last_partial = now
                pending["partial"] = True
                window = utterance[-int(config.PARTIAL_MAX_SEC * SR):]
                jobs.put_nowait(("partial", seg, window.copy()))
    except WebSocketDisconnect:
        pass
    finally:
        await jobs.put(None)
        try:
            await asyncio.wait_for(worker_task, timeout=5)
        except (asyncio.TimeoutError, Exception):
            worker_task.cancel()


def _build_transcript(segments: list[dict]) -> str:
    """Speaker-labeled transcript in spoken order, for the SOAP prompt."""
    lines = []
    for s in sorted(segments, key=lambda x: x["start"]):
        who = s["speaker"].capitalize() if s["speaker"] in ("nurse", "patient") else "Speaker"
        lines.append(f"{who}: {s['text']}")
    return "\n".join(lines)


def _build_entity_summary(segments: list[dict]) -> str:
    """Entities grouped by clinical category (deduped), as grounding hints."""
    cats: dict[str, list[str]] = {}
    for s in segments:
        for e in s.get("entities", []):
            seen = cats.setdefault(e["category"], [])
            term = e["text"].strip()
            if term and term.lower() not in {t.lower() for t in seen}:
                seen.append(term)
    return "\n".join(f"- {cat}: {', '.join(terms)}" for cat, terms in cats.items())


def _reconcile(turns: list[dict], segments: list[dict], vp: VoiceprintStore) -> list[dict]:
    """Map pyannote speaker clusters -> nurse/patient via the voiceprint, then assign
    each transcript segment the dominant cluster over its time span."""
    # 1) cluster -> role, using the mean embedding of segments dominated by each cluster.
    by_cluster: dict[str, list[np.ndarray]] = {}
    seg_cluster: dict[int, str] = {}
    for s in segments:
        cluster = _dominant_cluster(turns, s["start"] / SR, s["end"] / SR)
        if cluster is None:
            continue
        seg_cluster[s["segmentId"]] = cluster
        by_cluster.setdefault(cluster, []).append(s["emb"])

    cluster_role: dict[str, str] = {}
    if vp.enrolled:
        for cluster, embs in by_cluster.items():
            mean = np.mean(embs, axis=0)
            mean = mean / (np.linalg.norm(mean) or 1.0)
            cluster_role[cluster] = "nurse" if vp.similarity(mean) >= config.SPEAKER_THRESHOLD else "patient"

    out = []
    for s in segments:
        cluster = seg_cluster.get(s["segmentId"])
        role = cluster_role.get(cluster, s["speaker"]) if cluster else s["speaker"]
        out.append({"segmentId": s["segmentId"], "speaker": role, "cluster": cluster})
    return out


def _dominant_cluster(turns: list[dict], start: float, end: float) -> str | None:
    """The speaker label with the most overlap over [start, end] (seconds)."""
    overlap: dict[str, float] = {}
    for t in turns:
        o = max(0.0, min(end, t["end"]) - max(start, t["start"]))
        if o > 0:
            overlap[t["speaker"]] = overlap.get(t["speaker"], 0.0) + o
    return max(overlap, key=overlap.get) if overlap else None
