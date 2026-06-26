"use client";

import {
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  CircleAlert,
  CornerDownLeft,
  ExternalLink,
  FileText,
  Mic,
  Plus,
  Radio,
  RefreshCw,
  ShieldAlert,
  Square,
  Stethoscope,
  Tags,
  Trash2,
  Upload,
  User,
  Waves,
  Wifi,
  WifiOff,
  X,
} from "lucide-react";

import { cn } from "@/lib/utils";
import {
  Badge,
  Button,
  Card,
  CardBody,
  FieldLabel,
  Input,
  SectionHeader,
  StatusBadge,
} from "./ui";

/* -------------------------------------------------------------------------- */
/*  Types                                                                     */
/* -------------------------------------------------------------------------- */

type ConnectionStatus = "disconnected" | "connecting" | "connected";
type VadState = "idle" | "silence" | "speech";
type Speaker = "nurse" | "patient" | "unknown";

// A clinical concept extracted from one utterance by the biomedical NER model.
type EntityCategory =
  | "symptom"
  | "condition"
  | "medication"
  | "vital"
  | "procedure"
  | "anatomy"
  | "context"
  | "other";

interface Entity {
  text: string;
  label: string;
  category: EntityCategory;
  start: number;
  end: number;
  score: number;
}

// A safety-critical alert (self-harm / abuse / mandatory-reporting trigger).
interface SafetyAlert {
  category: string;
  label: string;
  severity: string;
  term: string;
  snippet: string;
  segmentId?: number;
}

// Completeness check: which extracted clinical entities made it into the note.
interface Completeness {
  coveredCount: number;
  totalCount: number;
  missing: string[];
}

interface FinalSegment {
  segmentId: number;
  text: string;
  confidence: number;
  lowConfidence: boolean;
  asrMs: number;
  speaker: Speaker;
  speakerSim: number;
  entities: Entity[];
  safety?: SafetyAlert[];
  correction?: boolean;
}

interface PartialSegment {
  segmentId: number;
  text: string;
}

interface DiarizationSegment {
  segmentId: number;
  speaker: "nurse" | "patient";
  cluster: string;
}

interface SoapMedication {
  name: string;
  dosage: string;
  frequency: string;
}

interface SoapNote {
  chiefComplaint: string;
  subjective: string[];
  objective: string[];
  assessment: string[];
  plan: string[];
  medications: SoapMedication[];
  reviewFlags: string[];
  requiresReview: boolean;
}

type SoapState = "idle" | "generating" | "ready" | "unavailable" | "error";

// Phase 5: the review -> FHIR push.
type FhirState = "idle" | "pushing" | "done" | "error";

interface FhirResource {
  type: string;
  id: string;
  url: string;
}

interface PatientInfo {
  name: string;
  mrn: string;
}

// Server -> client message shapes.
type ServerMessage =
  | { type: "ready"; model: string; enrolled?: boolean }
  | { type: "vad"; state: "speech" | "silence" }
  | { type: "partial"; segmentId: number; text: string }
  | {
      type: "final";
      segmentId: number;
      text: string;
      confidence: number;
      lowConfidence: boolean;
      asrMs: number;
      speaker?: Speaker;
      speakerSim?: number;
      entities?: Entity[];
      safety?: SafetyAlert[];
      correction?: boolean;
    }
  | {
      type: "diarization";
      speakerCount: number;
      segments: DiarizationSegment[];
      summary: string;
    }
  | { type: "finalsComplete" }
  | { type: "diarizationSkipped" }
  | { type: "error"; message: string };

// What we persist to localStorage so a transcript (and any note) survives a
// failed SOAP attempt or a page reload — the clinician can retry without
// re-recording the encounter.
interface PersistedSession {
  finals: FinalSegment[];
  soapNote: SoapNote | null;
  editedNote: SoapNote | null;
  soapModel: string | null;
  patient: PatientInfo;
  clinician: string;
  diarizationSummary: string | null;
  safetyAlerts: SafetyAlert[];
  completeness: Completeness | null;
  savedAt: number;
}

const WS_URL =
  process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws/transcribe";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const TARGET_SAMPLE_RATE = 16000;

// How long to keep the socket open after ending an encounter so the server's
// post-processing (pyannote refine + GPT-4o SOAP note) can arrive. The SOAP
// note is the last message; this is a fallback if it never comes.
const FINALIZE_GRACE_MS = 50000;

// localStorage key for the persisted encounter (transcript + note + patient).
const SESSION_KEY = "healthvoice.session.v1";

// Approximate duration of the enrollment recording.
const ENROLL_SECONDS = 10;

/* -------------------------------------------------------------------------- */
/*  Helpers                                                                   */
/* -------------------------------------------------------------------------- */

// Diff the AI-generated note against the clinician's edited copy → the list of
// changed fields, recorded in the audit trail (who changed WHAT).
function diffSoapNote(
  orig: SoapNote | null,
  edited: SoapNote | null
): { field: string; detail: string }[] {
  if (!orig || !edited) return [];
  const edits: { field: string; detail: string }[] = [];
  if (orig.chiefComplaint !== edited.chiefComplaint)
    edits.push({ field: "chiefComplaint", detail: "edited" });
  const listFields: (keyof SoapNote)[] = [
    "subjective",
    "objective",
    "assessment",
    "plan",
    "reviewFlags",
  ];
  for (const f of listFields) {
    const a = (orig[f] as string[]) ?? [];
    const b = (edited[f] as string[]) ?? [];
    if (JSON.stringify(a) !== JSON.stringify(b))
      edits.push({ field: f as string, detail: `${a.length} → ${b.length} line(s)` });
  }
  if (JSON.stringify(orig.medications) !== JSON.stringify(edited.medications))
    edits.push({
      field: "medications",
      detail: `${orig.medications.length} → ${edited.medications.length}`,
    });
  return edits;
}

// Convert a Float32 mono buffer (-1..1) to an Int16 PCM little-endian buffer.
function floatTo16BitPCM(float32: Float32Array): ArrayBuffer {
  const int16 = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    let s = float32[i];
    if (s > 1) s = 1;
    else if (s < -1) s = -1;
    int16[i] = s < 0 ? s * 32768 : s * 32767;
  }
  return int16.buffer;
}

/* -------------------------------------------------------------------------- */
/*  Component                                                                 */
/* -------------------------------------------------------------------------- */

export default function TranscriptionConsole() {
  const [recording, setRecording] = useState(false);
  const [finalizing, setFinalizing] = useState(false);
  const [connection, setConnection] = useState<ConnectionStatus>(
    "disconnected"
  );
  const [model, setModel] = useState<string | null>(null);
  const [vad, setVad] = useState<VadState>("idle");
  const [finals, setFinals] = useState<FinalSegment[]>([]);
  const [partial, setPartial] = useState<PartialSegment | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Enrollment status (the nurse's voice profile).
  const [enrolled, setEnrolled] = useState<boolean | null>(null);

  // The summary text from the diarization refine pass, shown under the header.
  const [diarizationSummary, setDiarizationSummary] = useState<string | null>(
    null
  );

  // Phase 4: the generated SOAP note + its lifecycle state.
  const [soapNote, setSoapNote] = useState<SoapNote | null>(null);
  const [soapState, setSoapState] = useState<SoapState>("idle");
  const [soapMessage, setSoapMessage] = useState<string | null>(null);
  const [soapModel, setSoapModel] = useState<string | null>(null);

  // Phase 5: the editable working copy, patient identity, and the FHIR push.
  const [editedNote, setEditedNote] = useState<SoapNote | null>(null);
  const [patient, setPatient] = useState<PatientInfo>({ name: "", mrn: "" });
  const [fhirState, setFhirState] = useState<FhirState>("idle");
  const [fhirResources, setFhirResources] = useState<FhirResource[]>([]);
  const [fhirMessage, setFhirMessage] = useState<string | null>(null);

  // True when the transcript on screen was restored from a previous session.
  const [restored, setRestored] = useState(false);

  // Phase 6 clinical safeguards.
  const [safetyAlerts, setSafetyAlerts] = useState<SafetyAlert[]>([]);
  const [safetyAck, setSafetyAck] = useState(false); // clinician acknowledged alerts
  const [completeness, setCompleteness] = useState<Completeness | null>(null);
  const [clinician, setClinician] = useState(""); // who signs/files the note
  const [auditSummary, setAuditSummary] = useState<
    { clinician: string; editCount: number } | null
  >(null);

  // Mutable refs for the audio + socket graph (kept out of React state so we
  // can tear them down deterministically).
  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const workletRef = useRef<AudioWorkletNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const transcriptEndRef = useRef<HTMLDivElement | null>(null);

  // Mirror of `finals` we can read synchronously (e.g. when the "finals
  // complete" signal arrives) without waiting for a React state flush.
  const finalsRef = useRef<FinalSegment[]>([]);

  // Timer that closes the socket if no diarization message arrives in time.
  const graceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  /* ---------------------------------------------------------------------- */
  /*  Auto-scroll transcript                                                */
  /* ---------------------------------------------------------------------- */
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [finals, partial]);

  /* ---------------------------------------------------------------------- */
  /*  Waveform drawing                                                      */
  /* ---------------------------------------------------------------------- */
  const drawWaveform = useCallback(() => {
    const canvas = canvasRef.current;
    const analyser = analyserRef.current;
    if (!canvas || !analyser) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Handle device pixel ratio for crisp rendering.
    const dpr = window.devicePixelRatio || 1;
    const cssWidth = canvas.clientWidth;
    const cssHeight = canvas.clientHeight;
    if (canvas.width !== cssWidth * dpr || canvas.height !== cssHeight * dpr) {
      canvas.width = cssWidth * dpr;
      canvas.height = cssHeight * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const data = new Uint8Array(analyser.frequencyBinCount);

    const render = () => {
      analyser.getByteTimeDomainData(data);

      ctx.clearRect(0, 0, cssWidth, cssHeight);

      // Background baseline.
      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, cssHeight / 2);
      ctx.lineTo(cssWidth, cssHeight / 2);
      ctx.stroke();

      // Waveform line.
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#0d9488";
      ctx.beginPath();
      const slice = cssWidth / data.length;
      let x = 0;
      for (let i = 0; i < data.length; i++) {
        const v = data[i] / 128.0; // 0..2, centered at 1
        const y = (v * cssHeight) / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += slice;
      }
      ctx.stroke();

      rafRef.current = requestAnimationFrame(render);
    };

    rafRef.current = requestAnimationFrame(render);
  }, []);

  /* ---------------------------------------------------------------------- */
  /*  Teardown                                                              */
  /* ---------------------------------------------------------------------- */

  // Tear down only the audio capture graph (mic, context, worklet). The
  // WebSocket is intentionally left untouched so the caller can decide whether
  // to keep it open (e.g. during the diarization grace period).
  const stopAudio = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    if (workletRef.current) {
      workletRef.current.port.onmessage = null;
      try {
        workletRef.current.disconnect();
      } catch {
        /* ignore */
      }
      workletRef.current = null;
    }

    if (analyserRef.current) {
      try {
        analyserRef.current.disconnect();
      } catch {
        /* ignore */
      }
      analyserRef.current = null;
    }

    if (sourceRef.current) {
      try {
        sourceRef.current.disconnect();
      } catch {
        /* ignore */
      }
      sourceRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    if (audioContextRef.current) {
      const ctx = audioContextRef.current;
      audioContextRef.current = null;
      if (ctx.state !== "closed") {
        ctx.close().catch(() => {
          /* ignore */
        });
      }
    }
  }, []);

  // Close and forget the WebSocket.
  const closeSocket = useCallback(() => {
    if (graceTimerRef.current !== null) {
      clearTimeout(graceTimerRef.current);
      graceTimerRef.current = null;
    }
    if (wsRef.current) {
      const ws = wsRef.current;
      wsRef.current = null;
      ws.onopen = null;
      ws.onmessage = null;
      ws.onerror = null;
      ws.onclose = null;
      if (
        ws.readyState === WebSocket.OPEN ||
        ws.readyState === WebSocket.CONNECTING
      ) {
        ws.close();
      }
    }
  }, []);

  // Full teardown of everything.
  const cleanup = useCallback(() => {
    stopAudio();
    closeSocket();
  }, [stopAudio, closeSocket]);

  // Clean up on unmount.
  useEffect(() => {
    return () => {
      cleanup();
    };
  }, [cleanup]);

  /* ---------------------------------------------------------------------- */
  /*  Enrollment status (on mount)                                          */
  /* ---------------------------------------------------------------------- */
  const refreshEnrollStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/enroll/status`);
      if (!res.ok) return;
      const data = (await res.json()) as { enrolled: boolean };
      setEnrolled(Boolean(data.enrolled));
    } catch {
      // Backend unreachable — leave status unknown; not fatal.
    }
  }, []);

  useEffect(() => {
    void refreshEnrollStatus();
  }, [refreshEnrollStatus]);

  /* ---------------------------------------------------------------------- */
  /*  SOAP generation (REST) — decoupled from the audio socket, retryable   */
  /* ---------------------------------------------------------------------- */
  const generateSoap = useCallback(async () => {
    const segs = finalsRef.current;
    if (segs.length === 0) {
      setSoapState("idle");
      return;
    }
    setSoapState("generating");
    setSoapMessage(null);
    setSafetyAck(false);
    try {
      const res = await fetch(`${API_URL}/soap`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          segments: segs.map((s) => ({
            segmentId: s.segmentId,
            speaker: s.speaker,
            text: s.text,
            entities: s.entities,
          })),
        }),
      });
      const data = (await res.json()) as {
        ok: boolean;
        note?: SoapNote;
        model?: string;
        unavailable?: boolean;
        message?: string;
        error?: string;
        safetyAlerts?: SafetyAlert[];
        completeness?: Completeness;
      };
      if (data.ok && data.note) {
        setSoapNote(data.note);
        setEditedNote(data.note);
        setSoapModel(data.model ?? null);
        setSafetyAlerts(data.safetyAlerts ?? []);
        setCompleteness(data.completeness ?? null);
        setSoapState("ready");
      } else if (data.unavailable) {
        setSoapState("unavailable");
        setSoapMessage(data.message ?? null);
      } else {
        setSoapState("error");
        setSoapMessage(data.error ?? "SOAP generation failed.");
      }
    } catch {
      setSoapState("error");
      setSoapMessage("Could not reach the SOAP service. Is the backend running?");
    }
  }, []);

  // Keep a ref to the latest generateSoap so the WS handler can trigger it
  // without re-subscribing the socket every render.
  const generateSoapRef = useRef(generateSoap);
  useEffect(() => {
    generateSoapRef.current = generateSoap;
  }, [generateSoap]);

  // Mirror soapState so timers / socket callbacks can read it without a stale
  // closure (used by the grace-timer SOAP fallback).
  const soapStateRef = useRef(soapState);
  useEffect(() => {
    soapStateRef.current = soapState;
  }, [soapState]);

  /* ---------------------------------------------------------------------- */
  /*  Local persistence — restore on mount, save on change                  */
  /* ---------------------------------------------------------------------- */
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = window.localStorage.getItem(SESSION_KEY);
      if (!raw) return;
      const s = JSON.parse(raw) as PersistedSession;
      if (!s.finals?.length) return;
      finalsRef.current = s.finals;
      setFinals(s.finals);
      if (s.soapNote) setSoapNote(s.soapNote);
      if (s.editedNote) {
        setEditedNote(s.editedNote);
        setSoapState("ready");
      }
      if (s.soapModel) setSoapModel(s.soapModel);
      if (s.patient) setPatient(s.patient);
      if (s.clinician) setClinician(s.clinician);
      if (s.diarizationSummary) setDiarizationSummary(s.diarizationSummary);
      if (s.safetyAlerts) setSafetyAlerts(s.safetyAlerts);
      if (s.completeness) setCompleteness(s.completeness);
      setRestored(true);
    } catch {
      /* ignore corrupt snapshot */
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    // Never overwrite a saved session with an empty (freshly reset) one.
    if (finals.length === 0) return;
    const snapshot: PersistedSession = {
      finals,
      soapNote,
      editedNote,
      soapModel,
      patient,
      clinician,
      diarizationSummary,
      safetyAlerts,
      completeness,
      savedAt: Date.now(),
    };
    try {
      window.localStorage.setItem(SESSION_KEY, JSON.stringify(snapshot));
    } catch {
      /* storage full / unavailable — non-fatal */
    }
  }, [
    finals,
    soapNote,
    editedNote,
    soapModel,
    patient,
    clinician,
    diarizationSummary,
    safetyAlerts,
    completeness,
  ]);

  // Discard the stored transcript + note and reset the console to empty.
  const clearSession = useCallback(() => {
    try {
      window.localStorage.removeItem(SESSION_KEY);
    } catch {
      /* ignore */
    }
    finalsRef.current = [];
    setFinals([]);
    setPartial(null);
    setSoapNote(null);
    setEditedNote(null);
    setSoapState("idle");
    setSoapMessage(null);
    setSoapModel(null);
    setDiarizationSummary(null);
    setFhirState("idle");
    setFhirResources([]);
    setFhirMessage(null);
    setRestored(false);
    setSafetyAlerts([]);
    setSafetyAck(false);
    setCompleteness(null);
    setAuditSummary(null);
  }, []);

  /* ---------------------------------------------------------------------- */
  /*  Incoming WebSocket messages                                           */
  /* ---------------------------------------------------------------------- */
  const handleServerMessage = useCallback(
    (raw: string) => {
      let msg: ServerMessage;
      try {
        msg = JSON.parse(raw) as ServerMessage;
      } catch {
        return;
      }

      switch (msg.type) {
        case "ready":
          setModel(msg.model);
          setConnection("connected");
          if (typeof msg.enrolled === "boolean") setEnrolled(msg.enrolled);
          break;
        case "vad":
          setVad(msg.state);
          break;
        case "partial":
          setPartial({ segmentId: msg.segmentId, text: msg.text });
          break;
        case "final": {
          // Append the finalized line and clear the live interim line.
          const seg: FinalSegment = {
            segmentId: msg.segmentId,
            text: msg.text,
            confidence: msg.confidence,
            lowConfidence: msg.lowConfidence,
            asrMs: msg.asrMs,
            speaker: msg.speaker ?? "unknown",
            speakerSim: msg.speakerSim ?? 0,
            entities: msg.entities ?? [],
            safety: msg.safety ?? [],
            correction: msg.correction ?? false,
          };
          finalsRef.current = [...finalsRef.current, seg];
          setFinals((prev) => [...prev, seg]);
          setPartial((prev) =>
            prev && prev.segmentId === msg.segmentId ? null : prev
          );
          break;
        }
        case "finalsComplete":
          // Transcript is fully committed. Generate the SOAP note now via REST
          // — independent of the (optional, slow) refine pass that follows.
          setFinalizing(false);
          void generateSoapRef.current();
          break;
        case "diarization": {
          // The refine pass: correct speaker labels in place. This is the last
          // post-processing step, so close the socket once it lands.
          const bySegment = new Map<number, "nurse" | "patient">();
          for (const s of msg.segments) bySegment.set(s.segmentId, s.speaker);
          const relabel = (seg: FinalSegment) => {
            const corrected = bySegment.get(seg.segmentId);
            return corrected ? { ...seg, speaker: corrected } : seg;
          };
          finalsRef.current = finalsRef.current.map(relabel);
          setFinals((prev) => prev.map(relabel));
          if (msg.summary) setDiarizationSummary(msg.summary);
          setFinalizing(false);
          closeSocket();
          setConnection("disconnected");
          break;
        }
        case "diarizationSkipped":
          // Optional refine pass was skipped/timed out (e.g. gated model). Live
          // ECAPA labels already stand — not an error. Last step → close socket.
          setFinalizing(false);
          closeSocket();
          setConnection("disconnected");
          break;
        case "error":
          setError(msg.message);
          break;
        default:
          break;
      }
    },
    [closeSocket]
  );

  /* ---------------------------------------------------------------------- */
  /*  Start encounter                                                       */
  /* ---------------------------------------------------------------------- */
  const start = useCallback(async () => {
    setError(null);
    finalsRef.current = [];
    setFinals([]);
    setRestored(false);
    setPartial(null);
    setModel(null);
    setVad("idle");
    setDiarizationSummary(null);
    setSoapNote(null);
    setSoapState("idle");
    setSoapMessage(null);
    setSoapModel(null);
    setEditedNote(null);
    setFhirState("idle");
    setFhirResources([]);
    setFhirMessage(null);
    setSafetyAlerts([]);
    setSafetyAck(false);
    setCompleteness(null);
    setAuditSummary(null);
    setFinalizing(false);
    setConnection("connecting");

    try {
      if (typeof window === "undefined" || !navigator.mediaDevices) {
        throw new Error("Microphone access is not available in this browser.");
      }

      // 1) Mic stream (mono, with clinical-friendly processing).
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      streamRef.current = stream;

      // 2) AudioContext forced to 16 kHz so the browser resamples for us.
      const AudioCtx: typeof AudioContext =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext })
          .webkitAudioContext;
      const audioContext = new AudioCtx({ sampleRate: TARGET_SAMPLE_RATE });
      audioContextRef.current = audioContext;

      // Contexts can start suspended until a user gesture resumes them.
      if (audioContext.state === "suspended") {
        await audioContext.resume();
      }

      // 3) Load the worklet module.
      await audioContext.audioWorklet.addModule("/pcm-worklet.js");

      // 4) Open the WebSocket. We wait for it to open before wiring audio so
      //    we don't drop the first batches.
      const ws = new WebSocket(WS_URL);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      await new Promise<void>((resolve, reject) => {
        ws.onopen = () => resolve();
        ws.onerror = () =>
          reject(new Error("Could not connect to the transcription server."));
      });

      ws.onmessage = (event) => {
        if (typeof event.data === "string") {
          handleServerMessage(event.data);
        }
      };
      ws.onerror = () => {
        setError("WebSocket connection error.");
      };
      ws.onclose = () => {
        setConnection("disconnected");
        setFinalizing(false);
      };

      setConnection("connected");

      // 5) Build the audio graph.
      const source = audioContext.createMediaStreamSource(stream);
      sourceRef.current = source;

      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.8;
      analyserRef.current = analyser;

      const worklet = new AudioWorkletNode(audioContext, "pcm-worklet");
      workletRef.current = worklet;

      // Convert each Float32 batch -> Int16 PCM little-endian and ship it.
      worklet.port.onmessage = (event: MessageEvent<Float32Array>) => {
        const socket = wsRef.current;
        if (!socket || socket.readyState !== WebSocket.OPEN) return;
        socket.send(floatTo16BitPCM(event.data));
      };

      // source -> analyser (for the meter) and source -> worklet (for streaming).
      source.connect(analyser);
      source.connect(worklet);
      // The worklet produces no audible output; connecting to destination keeps
      // some browsers' graphs scheduled. We route through a muted gain to be safe.
      const sink = audioContext.createGain();
      sink.gain.value = 0;
      worklet.connect(sink);
      sink.connect(audioContext.destination);

      setRecording(true);
      drawWaveform();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to start the encounter.";
      setError(message);
      setConnection("disconnected");
      setRecording(false);
      cleanup();
    }
  }, [cleanup, drawWaveform, handleServerMessage]);

  /* ---------------------------------------------------------------------- */
  /*  Stop encounter                                                        */
  /* ---------------------------------------------------------------------- */
  const stop = useCallback(() => {
    // Mic capture stops immediately; only the socket lingers for the refine pass.
    stopAudio();
    setRecording(false);
    setVad("idle");

    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify({ type: "end_encounter" }));
      } catch {
        /* ignore */
      }
      // Keep the socket open to receive the refine + SOAP-note messages.
      setFinalizing(true);
      setSoapState("generating");
      if (graceTimerRef.current !== null) clearTimeout(graceTimerRef.current);
      graceTimerRef.current = setTimeout(() => {
        graceTimerRef.current = null;
        // Post-processing socket never closed itself (server unreachable /
        // refine ran long). Close it; the SOAP note is handled over REST.
        setFinalizing(false);
        closeSocket();
        setConnection("disconnected");
        // Safety net: if the SOAP note never generated (e.g. the
        // finalsComplete signal was missed), try once from the stored
        // transcript so the clinician isn't left with nothing.
        if (
          soapStateRef.current === "generating" &&
          finalsRef.current.length > 0
        ) {
          void generateSoapRef.current();
        }
      }, FINALIZE_GRACE_MS);
    } else {
      // Socket already gone — nothing to wait for.
      closeSocket();
      setConnection("disconnected");
    }
  }, [stopAudio, closeSocket]);

  const toggle = useCallback(() => {
    if (recording) stop();
    else void start();
  }, [recording, start, stop]);

  /* ---------------------------------------------------------------------- */
  /*  Approve & push to FHIR (Phase 5)                                      */
  /* ---------------------------------------------------------------------- */
  const pushToFhir = useCallback(async () => {
    if (!editedNote) return;
    if (!patient.name.trim()) {
      setFhirMessage("Enter the patient's name before pushing to FHIR.");
      setFhirState("error");
      return;
    }
    if (!clinician.trim()) {
      setFhirMessage("Enter the clinician name to sign and file the note.");
      setFhirState("error");
      return;
    }
    if (safetyAlerts.length > 0 && !safetyAck) {
      setFhirMessage("Review and acknowledge the safety alerts before filing.");
      setFhirState("error");
      return;
    }
    const edits = diffSoapNote(soapNote, editedNote);
    setFhirState("pushing");
    setFhirMessage(null);
    try {
      const res = await fetch(`${API_URL}/fhir/push`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ patient, note: editedNote, clinician, edits }),
      });
      const data = (await res.json()) as {
        ok: boolean;
        error?: string;
        resources?: FhirResource[];
        clinician?: string;
        editCount?: number;
      };
      if (data.ok) {
        setFhirResources(data.resources ?? []);
        setAuditSummary({
          clinician: data.clinician ?? clinician,
          editCount: data.editCount ?? edits.length,
        });
        setFhirState("done");
      } else {
        setFhirMessage(data.error ?? "FHIR push failed.");
        setFhirState("error");
      }
    } catch {
      setFhirMessage("Could not reach the FHIR push service.");
      setFhirState("error");
    }
  }, [editedNote, patient, clinician, soapNote, safetyAlerts, safetyAck]);

  /* ---------------------------------------------------------------------- */
  /*  Render                                                                */
  /* ---------------------------------------------------------------------- */
  const busy = recording || finalizing;

  // Aggregate all extracted entities across the encounter, grouped by clinical
  // category and de-duplicated (case-insensitive) with an occurrence count.
  const entityGroups = useMemo(() => {
    const groups = new Map<EntityCategory, Map<string, { text: string; count: number }>>();
    for (const seg of finals) {
      for (const e of seg.entities ?? []) {
        const key = e.text.toLowerCase().trim();
        if (!key) continue;
        if (!groups.has(e.category)) groups.set(e.category, new Map());
        const bucket = groups.get(e.category)!;
        const existing = bucket.get(key);
        if (existing) existing.count += 1;
        else bucket.set(key, { text: e.text.trim(), count: 1 });
      }
    }
    return CATEGORY_ORDER.filter((c) => groups.has(c)).map((c) => ({
      category: c,
      items: Array.from(groups.get(c)!.values()),
    }));
  }, [finals]);

  const entityCount = entityGroups.reduce((n, g) => n + g.items.length, 0);

  return (
    <div className="flex flex-col gap-5">
      {/* Error banner */}
      {error && (
        <div
          role="alert"
          className="flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 shadow-card animate-fadeInUp"
        >
          <CircleAlert className="mt-0.5 h-4 w-4 shrink-0 text-red-500" />
          <span className="flex-1">{error}</span>
          <button
            onClick={() => setError(null)}
            className="shrink-0 rounded-md p-0.5 text-red-400 transition hover:bg-red-100 hover:text-red-600"
            aria-label="Dismiss error"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      {/* Setup row: nurse enrollment + patient identity, side by side on large screens */}
      <div className="grid gap-5 lg:grid-cols-2">
        {/* Enrollment panel */}
        <EnrollmentPanel
          enrolled={enrolled}
          onEnrolledChange={setEnrolled}
          disabled={busy}
          onError={setError}
          refreshStatus={refreshEnrollStatus}
        />

        {/* Patient identity (filed into FHIR on approval) */}
        <PatientCard
          patient={patient}
          onChange={setPatient}
          disabled={fhirState === "pushing"}
        />
      </div>

      {/* Control card */}
      <Card>
        <CardBody>
          <div className="flex flex-col gap-5 sm:flex-row sm:items-center sm:justify-between">
            {/* Start / Stop button */}
            <Button
              onClick={toggle}
              disabled={connection === "connecting" || finalizing}
              variant={recording ? "destructive" : "primary"}
              size="lg"
              loading={connection === "connecting" || finalizing}
            >
              {!(connection === "connecting" || finalizing) &&
                (recording ? (
                  <Square className="h-4 w-4 fill-current" />
                ) : (
                  <Mic className="h-[18px] w-[18px]" />
                ))}
              {recording
                ? "Stop encounter"
                : finalizing
                ? "Finalizing…"
                : connection === "connecting"
                ? "Connecting…"
                : "Start encounter"}
            </Button>

            {/* Status pills */}
            <div className="flex flex-wrap items-center gap-2">
              <ConnectionPill status={connection} model={model} />
              <VadPill state={vad} />
              {finalizing && <FinalizingPill />}
            </div>
          </div>

          {/* Waveform meter */}
          <div className="mt-5">
            <canvas
              ref={canvasRef}
              className="h-16 w-full rounded-xl border border-clinical-border bg-slate-50/80"
              aria-hidden="true"
            />
            {!recording && (
              <p className="mt-2 text-center text-xs text-slate-400">
                Audio level appears here while recording.
              </p>
            )}
          </div>
        </CardBody>
      </Card>

      {/* Transcript + clinical entities — side by side on large screens */}
      <div className="grid items-start gap-5 lg:grid-cols-3">
      {/* Transcript panel */}
      <Card className="lg:col-span-2">
        <SectionHeader
          divided
          icon={<FileText className="h-[18px] w-[18px]" />}
          title="Transcript"
          description={diarizationSummary ?? "Live, speaker-labeled transcript with clinical highlights."}
          actions={
            <>
              {restored && (
                <Badge tone="warning" title="Loaded from your previous session">
                  Restored
                </Badge>
              )}
              <span className="text-xs font-medium text-slate-400">
                {finals.length} segment{finals.length === 1 ? "" : "s"}
              </span>
              {!recording && !finalizing && finals.length > 0 && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={clearSession}
                  title="Discard the saved transcript and note"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                  Clear
                </Button>
              )}
            </>
          }
        />

        {/* Gentle hint when the nurse voice isn't enrolled. */}
        {enrolled === false && finals.length > 0 && (
          <p className="border-b border-clinical-border bg-amber-50/60 px-5 py-2 text-xs text-amber-700 sm:px-6">
            Enroll the nurse&rsquo;s voice to label speakers.
          </p>
        )}

        <div className="transcript-scroll max-h-[48vh] min-h-[16rem] overflow-y-auto px-5 py-4 sm:px-6">
          {finals.length === 0 && !partial && (
            <div className="flex flex-col items-center gap-2 py-12 text-center">
              <span className="flex h-10 w-10 items-center justify-center rounded-full bg-slate-100 text-slate-400">
                {recording ? (
                  <Waves className="h-5 w-5 animate-pulse" />
                ) : (
                  <Mic className="h-5 w-5" />
                )}
              </span>
              <p className="max-w-xs text-sm text-slate-400">
                {recording
                  ? "Listening… start speaking and the transcript will appear here."
                  : "Start an encounter to begin transcribing in real time."}
              </p>
            </div>
          )}

          <ul className="space-y-1.5">
            {finals.map((seg) => (
              <li
                key={seg.segmentId}
                className={cn(
                  "rounded-xl border border-transparent px-3.5 py-2.5 text-sm leading-relaxed text-slate-800 transition",
                  speakerBorder(seg.speaker),
                  seg.safety && seg.safety.length > 0
                    ? "border-red-200 bg-red-50 ring-1 ring-red-100"
                    : seg.lowConfidence
                    ? "border-amber-200 bg-amber-50"
                    : "hover:bg-slate-50"
                )}
              >
                <div className="flex items-start justify-between gap-3">
                  <span className="flex flex-col gap-1.5">
                    <span className="flex flex-wrap items-center gap-1.5">
                      <SpeakerChip speaker={seg.speaker} sim={seg.speakerSim} />
                      {seg.correction && <CorrectionChip />}
                      {seg.safety && seg.safety.length > 0 && (
                        <SafetyChip count={seg.safety.length} />
                      )}
                    </span>
                    <span>{renderWithEntities(seg.text, seg.entities)}</span>
                  </span>
                  <span className="mt-0.5 flex shrink-0 items-center gap-1.5">
                    {seg.lowConfidence && (
                      <span className="rounded-full bg-amber-200 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-amber-800">
                        Review
                      </span>
                    )}
                    <span
                      className="rounded-md bg-slate-100 px-1.5 py-0.5 text-[10px] font-medium tabular-nums text-slate-400"
                      title="Speech-recognition latency for this segment"
                    >
                      {seg.asrMs} ms
                    </span>
                    <span
                      className="rounded-md bg-slate-100 px-1.5 py-0.5 text-[10px] font-medium tabular-nums text-slate-400"
                      title="Recognition confidence for this segment"
                    >
                      {Math.round(seg.confidence * 100)}%
                    </span>
                  </span>
                </div>
              </li>
            ))}

            {/* Live interim line */}
            {partial && (
              <li className="px-3.5 py-2.5 text-sm italic leading-relaxed text-slate-400">
                {partial.text}
                <span className="ml-1 inline-block h-3 w-1.5 animate-pulseDot bg-slate-300 align-middle" />
              </li>
            )}
          </ul>

          <div ref={transcriptEndRef} />
        </div>
      </Card>

      {/* Clinical entities panel */}
      <Card className="lg:col-span-1">
        <SectionHeader
          divided
          icon={<Tags className="h-[18px] w-[18px]" />}
          title="Clinical entities"
          description="Extracted live by biomedical NER — the structured basis for the SOAP note."
          actions={
            <span className="text-xs font-medium text-slate-400">
              {entityCount} entit{entityCount === 1 ? "y" : "ies"}
            </span>
          }
        />

        <div className="px-5 py-5 sm:px-6">
          {entityCount === 0 ? (
            <p className="py-6 text-center text-sm text-slate-400">
              Clinical terms (symptoms, medications, vitals…) will appear here as
              they&rsquo;re mentioned.
            </p>
          ) : (
            <div className="flex flex-col gap-4">
              {entityGroups.map((g) => {
                const meta = CATEGORY_META[g.category];
                return (
                  <div key={g.category} className="flex flex-col gap-2">
                    <span className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                      {meta.label}
                    </span>
                    <div className="flex flex-wrap gap-1.5">
                      {g.items.map((item) => (
                        <span
                          key={item.text}
                          className={`inline-flex items-center gap-1 rounded-lg border px-2.5 py-1 text-xs font-medium ${meta.chip}`}
                        >
                          {item.text}
                          {item.count > 1 && (
                            <span className="rounded-full bg-white/70 px-1 text-[10px] font-semibold tabular-nums">
                              ×{item.count}
                            </span>
                          )}
                        </span>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </Card>
      </div>

      {/* Safety-critical alerts — must be acknowledged before filing */}
      {safetyAlerts.length > 0 && (
        <SafetyBanner
          alerts={safetyAlerts}
          acknowledged={safetyAck}
          onAcknowledge={setSafetyAck}
        />
      )}

      {/* SOAP note panel — editable, then approve & push to FHIR */}
      <SoapPanel
        state={soapState}
        note={editedNote}
        onChange={setEditedNote}
        message={soapMessage}
        model={soapModel}
        patient={patient}
        clinician={clinician}
        onClinicianChange={setClinician}
        completeness={completeness}
        safetyBlocking={safetyAlerts.length > 0 && !safetyAck}
        auditSummary={auditSummary}
        fhirState={fhirState}
        fhirResources={fhirResources}
        fhirMessage={fhirMessage}
        onPush={() => void pushToFhir()}
        onRegenerate={() => void generateSoap()}
        canRegenerate={finals.length > 0 && soapState !== "generating"}
      />
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/*  SOAP note panel                                                           */
/* -------------------------------------------------------------------------- */

function SoapPanel({
  state,
  note,
  onChange,
  message,
  model,
  patient,
  clinician,
  onClinicianChange,
  completeness,
  safetyBlocking,
  auditSummary,
  fhirState,
  fhirResources,
  fhirMessage,
  onPush,
  onRegenerate,
  canRegenerate,
}: {
  state: SoapState;
  note: SoapNote | null;
  onChange: (note: SoapNote) => void;
  message: string | null;
  model: string | null;
  patient: PatientInfo;
  clinician: string;
  onClinicianChange: (v: string) => void;
  completeness: Completeness | null;
  safetyBlocking: boolean;
  auditSummary: { clinician: string; editCount: number } | null;
  fhirState: FhirState;
  fhirResources: FhirResource[];
  fhirMessage: string | null;
  onPush: () => void;
  onRegenerate: () => void;
  canRegenerate: boolean;
}) {
  const update = (patch: Partial<SoapNote>) => {
    if (note) onChange({ ...note, ...patch });
  };
  const pushed = fhirState === "done";
  const canPush =
    state === "ready" &&
    !!note &&
    fhirState !== "pushing" &&
    patient.name.trim().length > 0 &&
    clinician.trim().length > 0 &&
    !safetyBlocking;

  return (
    <Card>
      <SectionHeader
        divided
        icon={<Stethoscope className="h-[18px] w-[18px]" />}
        title="SOAP note"
        description={`AI-generated${
          model ? ` by ${model}` : ""
        } · editable — review and correct before filing to FHIR.`}
        actions={
          <>
            {canRegenerate &&
              (state === "ready" ||
                state === "error" ||
                state === "unavailable") && (
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={onRegenerate}
                  title="Re-run SOAP generation from the transcript — no re-recording"
                >
                  <RefreshCw className="h-3.5 w-3.5" />
                  Regenerate
                </Button>
              )}
            {state === "ready" &&
              (pushed ? (
                <StatusBadge tone="success">Filed to FHIR</StatusBadge>
              ) : (
                <StatusBadge tone="warning">Draft · review</StatusBadge>
              ))}
          </>
        }
      />

      <div className="px-5 py-5 sm:px-6">
        {state === "idle" &&
          (canRegenerate ? (
            <div className="flex flex-col items-center gap-3 py-10">
              <span className="flex h-10 w-10 items-center justify-center rounded-full bg-clinical-accentSoft text-clinical-accent">
                <FileText className="h-5 w-5" />
              </span>
              <p className="text-center text-sm text-slate-500">
                Transcript ready — generate the structured SOAP note from it.
              </p>
              <Button onClick={onRegenerate}>Generate SOAP note</Button>
            </div>
          ) : (
            <p className="py-8 text-center text-sm text-slate-400">
              The structured SOAP note appears here after you end an encounter.
            </p>
          ))}

        {state === "generating" && (
          <div className="flex flex-col items-center justify-center gap-3 py-12 text-sm text-slate-500">
            <RefreshCw className="h-6 w-6 animate-spin text-clinical-accent" />
            <span>Generating SOAP note…</span>
          </div>
        )}

        {state === "unavailable" && (
          <div className="flex items-start gap-2.5 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>
              {message ??
                "SOAP generation is unavailable. Set OPENAI_API_KEY in backend/.env."}
            </span>
          </div>
        )}

        {state === "error" && (
          <div className="flex flex-col gap-3">
            <div className="flex items-start gap-2.5 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              <CircleAlert className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{message ?? "SOAP generation failed."}</span>
            </div>
            {canRegenerate && (
              <Button className="self-start" onClick={onRegenerate}>
                <RefreshCw className="h-4 w-4" />
                Try again
              </Button>
            )}
          </div>
        )}

        {state === "ready" && note && (
          <div className="flex flex-col gap-4">
            {/* Completeness — are all extracted entities reflected in the note? */}
            {completeness && completeness.totalCount > 0 && (
              <CompletenessLine completeness={completeness} />
            )}

            {/* Chief complaint (editable) */}
            <div className="rounded-xl border border-clinical-border bg-slate-50/70 px-4 py-3">
              <FieldLabel>Chief complaint</FieldLabel>
              <Input
                className="mt-1.5 font-medium"
                value={note.chiefComplaint}
                placeholder="Chief complaint…"
                onChange={(e) => update({ chiefComplaint: e.target.value })}
              />
            </div>

            {/* S / O / A / P (editable) */}
            <div className="grid gap-3 sm:grid-cols-2">
              <EditableSection
                letter="S" title="Subjective" accent="rose"
                items={note.subjective}
                onChange={(subjective) => update({ subjective })}
              />
              <EditableSection
                letter="O" title="Objective" accent="sky"
                items={note.objective}
                onChange={(objective) => update({ objective })}
              />
              <EditableSection
                letter="A" title="Assessment" accent="orange"
                items={note.assessment}
                onChange={(assessment) => update({ assessment })}
              />
              <EditableSection
                letter="P" title="Plan" accent="emerald"
                items={note.plan}
                onChange={(plan) => update({ plan })}
              />
            </div>

            {/* Structured medications (editable) */}
            <EditableMedications
              meds={note.medications}
              onChange={(medications) => update({ medications })}
            />

            {/* Review flags (editable) */}
            <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3">
              <span className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-amber-700">
                <AlertTriangle className="h-3.5 w-3.5" />
                Needs verification
              </span>
              <div className="mt-2">
                <EditableList
                  items={note.reviewFlags}
                  onChange={(reviewFlags) => update({ reviewFlags })}
                  placeholder="Item to verify…"
                  addLabel="Add flag"
                />
              </div>
            </div>

            {/* Approve & push to FHIR */}
            <div className="flex flex-col gap-3 border-t border-clinical-border pt-4">
              {fhirState === "done" && (
                <FhirResultPanel resources={fhirResources} />
              )}
              {fhirState === "done" && auditSummary && (
                <p className="text-xs text-slate-500">
                  Filed by{" "}
                  <span className="font-medium text-slate-700">
                    {auditSummary.clinician}
                  </span>
                  {" · "}
                  {auditSummary.editCount} field
                  {auditSummary.editCount === 1 ? "" : "s"} edited from the AI draft
                  {" · "}audit logged + FHIR Provenance recorded.
                </p>
              )}
              {fhirState === "error" && (
                <div className="flex items-start gap-2.5 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                  <CircleAlert className="mt-0.5 h-4 w-4 shrink-0" />
                  <span>{fhirMessage ?? "FHIR push failed."}</span>
                </div>
              )}

              {/* Clinician signature — recorded as the legal attester + in the audit trail */}
              <label className="flex flex-col gap-1.5 sm:max-w-xs">
                <FieldLabel>Signed by (clinician)</FieldLabel>
                <Input
                  value={clinician}
                  disabled={fhirState === "pushing"}
                  placeholder="e.g. Nurse Jane Smith"
                  onChange={(e) => onClinicianChange(e.target.value)}
                />
              </label>

              <div className="flex flex-wrap items-center justify-between gap-3">
                <p className="max-w-md text-xs text-slate-400">
                  {safetyBlocking
                    ? "Acknowledge the safety alert above before filing."
                    : pushed
                    ? "Note filed. Edit and push again to file a new version."
                    : !patient.name.trim()
                    ? "Enter a patient name in the Patient card above to enable filing."
                    : !clinician.trim()
                    ? "Enter the signing clinician to enable filing."
                    : `Will file under “${patient.name.trim()}”${
                        patient.mrn.trim() ? ` · MRN ${patient.mrn.trim()}` : ""
                      }.`}
                </p>
                <Button
                  onClick={onPush}
                  disabled={!canPush}
                  loading={fhirState === "pushing"}
                >
                  {fhirState !== "pushing" && <Upload className="h-4 w-4" />}
                  {fhirState === "pushing"
                    ? "Pushing…"
                    : pushed
                    ? "Push again to FHIR"
                    : "Approve & push to FHIR"}
                </Button>
              </div>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}

const SOAP_ACCENT: Record<string, string> = {
  rose: "bg-rose-100 text-rose-700",
  sky: "bg-sky-100 text-sky-700",
  orange: "bg-orange-100 text-orange-700",
  emerald: "bg-emerald-100 text-emerald-700",
};

function EditableSection({
  letter,
  title,
  accent,
  items,
  onChange,
}: {
  letter: string;
  title: string;
  accent: string;
  items: string[];
  onChange: (items: string[]) => void;
}) {
  return (
    <div className="flex flex-col gap-2.5 rounded-xl border border-clinical-border bg-white px-4 py-3.5">
      <div className="flex items-center gap-2">
        <span
          className={`flex h-6 w-6 items-center justify-center rounded-md text-xs font-bold ${SOAP_ACCENT[accent]}`}
        >
          {letter}
        </span>
        <span className="text-[13px] font-semibold text-slate-700">{title}</span>
      </div>
      <EditableList
        items={items}
        onChange={onChange}
        placeholder={`Add ${title.toLowerCase()} line…`}
        addLabel="Add line"
      />
    </div>
  );
}

// A controlled list of free-text bullet lines with add / edit / delete.
function EditableList({
  items,
  onChange,
  placeholder,
  addLabel,
}: {
  items: string[];
  onChange: (items: string[]) => void;
  placeholder: string;
  addLabel: string;
}) {
  const setAt = (i: number, v: string) =>
    onChange(items.map((it, j) => (j === i ? v : it)));
  const removeAt = (i: number) => onChange(items.filter((_, j) => j !== i));
  const add = () => onChange([...items, ""]);

  return (
    <div className="flex flex-col gap-1.5">
      {items.length === 0 && (
        <p className="pl-1 text-xs italic text-slate-400">
          Nothing documented — add a line if needed.
        </p>
      )}
      {items.map((it, i) => (
        <div key={i} className="flex items-center gap-1.5">
          <Input
            value={it}
            placeholder={placeholder}
            onChange={(e) => setAt(i, e.target.value)}
          />
          <button
            type="button"
            onClick={() => removeAt(i)}
            className="shrink-0 rounded-md p-1.5 text-slate-300 transition hover:bg-red-50 hover:text-red-500"
            aria-label="Remove line"
            title="Remove line"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      ))}
      <button
        type="button"
        onClick={add}
        className="mt-0.5 inline-flex items-center gap-1 self-start rounded-md px-1 py-0.5 text-xs font-medium text-clinical-accent transition hover:text-clinical-accentDark"
      >
        <Plus className="h-3.5 w-3.5" />
        {addLabel}
      </button>
    </div>
  );
}

// Editable medications table (name / dosage / frequency) with add / delete rows.
function EditableMedications({
  meds,
  onChange,
}: {
  meds: SoapMedication[];
  onChange: (meds: SoapMedication[]) => void;
}) {
  const setAt = (i: number, patch: Partial<SoapMedication>) =>
    onChange(meds.map((m, j) => (j === i ? { ...m, ...patch } : m)));
  const removeAt = (i: number) => onChange(meds.filter((_, j) => j !== i));
  const add = () => onChange([...meds, { name: "", dosage: "", frequency: "" }]);

  return (
    <div className="overflow-hidden rounded-xl border border-clinical-border">
      <div className="border-b border-clinical-border bg-slate-50/70 px-4 py-2.5 text-[11px] font-semibold uppercase tracking-wide text-slate-500">
        Medications
      </div>
      {meds.length === 0 ? (
        <p className="px-4 py-3 text-xs italic text-slate-400">
          No medications recorded.
        </p>
      ) : (
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-[11px] uppercase tracking-wide text-slate-400">
              <th className="px-3 py-2 font-medium">Name</th>
              <th className="px-3 py-2 font-medium">Dosage</th>
              <th className="px-3 py-2 font-medium">Frequency</th>
              <th className="px-2 py-2" />
            </tr>
          </thead>
          <tbody>
            {meds.map((m, i) => (
              <tr key={i} className="border-t border-clinical-border align-top">
                <td className="px-3 py-1.5">
                  <Input
                    value={m.name}
                    placeholder="Name"
                    onChange={(e) => setAt(i, { name: e.target.value })}
                  />
                </td>
                <td className="px-3 py-1.5">
                  <Input
                    value={m.dosage}
                    placeholder="Dosage"
                    onChange={(e) => setAt(i, { dosage: e.target.value })}
                  />
                </td>
                <td className="px-3 py-1.5">
                  <Input
                    value={m.frequency}
                    placeholder="Frequency"
                    onChange={(e) => setAt(i, { frequency: e.target.value })}
                  />
                </td>
                <td className="px-2 py-1.5 text-center">
                  <button
                    type="button"
                    onClick={() => removeAt(i)}
                    className="rounded-md p-1.5 text-slate-300 transition hover:bg-red-50 hover:text-red-500"
                    aria-label="Remove medication"
                    title="Remove medication"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <div className="border-t border-clinical-border px-4 py-2.5">
        <button
          type="button"
          onClick={add}
          className="inline-flex items-center gap-1 rounded-md px-1 py-0.5 text-xs font-medium text-clinical-accent transition hover:text-clinical-accentDark"
        >
          <Plus className="h-3.5 w-3.5" />
          Add medication
        </button>
      </div>
    </div>
  );
}

// The created FHIR resources, each a clickable link to the stored record.
function FhirResultPanel({ resources }: { resources: FhirResource[] }) {
  return (
    <div className="rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3.5">
      <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-emerald-700">
        <CheckCircle2 className="h-4 w-4" />
        Filed to FHIR · {resources.length} resource
        {resources.length === 1 ? "" : "s"} created
      </div>
      <ul className="mt-2.5 space-y-1.5 text-sm">
        {resources.map((r) => (
          <li key={`${r.type}-${r.id}`} className="flex items-center gap-2">
            <span className="w-24 shrink-0 rounded-md border border-emerald-200 bg-white px-1.5 py-0.5 text-center text-[11px] font-semibold text-emerald-700">
              {r.type}
            </span>
            <a
              href={r.url}
              target="_blank"
              rel="noreferrer"
              className="group inline-flex min-w-0 items-center gap-1 text-emerald-800 hover:text-emerald-900"
            >
              <span className="truncate underline decoration-emerald-300 underline-offset-2 group-hover:decoration-emerald-500">
                {r.url}
              </span>
              <ExternalLink className="h-3.5 w-3.5 shrink-0 opacity-60" />
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/*  Phase 6 clinical safeguards — UI                                          */
/* -------------------------------------------------------------------------- */

// Prominent, blocking banner for safety-critical content. Filing is disabled
// until the clinician ticks the acknowledgement box.
function SafetyBanner({
  alerts,
  acknowledged,
  onAcknowledge,
}: {
  alerts: SafetyAlert[];
  acknowledged: boolean;
  onAcknowledge: (v: boolean) => void;
}) {
  return (
    <section
      role="alert"
      className="animate-fadeInUp rounded-2xl border-2 border-red-300 bg-red-50 p-5 shadow-card sm:p-6"
    >
      <div className="flex items-center gap-2.5">
        <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-red-600 text-white shadow-sm">
          <ShieldAlert className="h-5 w-5" />
        </span>
        <h2 className="text-sm font-bold uppercase tracking-wide text-red-800">
          Safety alert · {alerts.length} flag{alerts.length === 1 ? "" : "s"}
        </h2>
      </div>
      <p className="mt-2.5 text-xs leading-relaxed text-red-700">
        Potential safety-critical content was detected in this encounter. Review
        and take appropriate clinical action — the note cannot be filed until
        acknowledged.
      </p>
      <ul className="mt-3 space-y-2">
        {alerts.map((a, i) => (
          <li
            key={`${a.category}-${i}`}
            className="rounded-lg border border-red-200 bg-white px-3 py-2"
          >
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded-full bg-red-100 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-red-700">
                {a.label}
              </span>
              <span className="text-[10px] font-medium uppercase tracking-wide text-red-400">
                {a.severity}
              </span>
            </div>
            <p className="mt-1 text-sm italic text-slate-700">“{a.snippet}”</p>
          </li>
        ))}
      </ul>
      <label className="mt-3 flex items-start gap-2 text-sm font-medium text-red-800">
        <input
          type="checkbox"
          checked={acknowledged}
          onChange={(e) => onAcknowledge(e.target.checked)}
          className="mt-0.5 h-4 w-4 rounded border-red-300 text-red-600 focus:ring-red-500"
        />
        I have reviewed these safety alerts and taken appropriate clinical action.
      </label>
    </section>
  );
}

// Coverage line: did every extracted clinical entity make it into the note?
function CompletenessLine({ completeness }: { completeness: Completeness }) {
  const all = completeness.missing.length === 0;
  return (
    <div
      className={`flex flex-wrap items-center gap-x-1.5 gap-y-0.5 rounded-xl border px-4 py-2.5 text-xs ${
        all
          ? "border-emerald-200 bg-emerald-50 text-emerald-800"
          : "border-amber-200 bg-amber-50 text-amber-800"
      }`}
    >
      {all ? (
        <CheckCircle2 className="mr-0.5 h-4 w-4" />
      ) : (
        <AlertTriangle className="mr-0.5 h-4 w-4" />
      )}
      <span className="font-semibold">
        Completeness {completeness.coveredCount}/{completeness.totalCount}
      </span>
      <span className="opacity-80">clinical entities reflected in the note</span>
      {all ? (
        <span className="opacity-80">· all captured</span>
      ) : (
        <span className="opacity-80">
          · not found: {completeness.missing.join(", ")}
        </span>
      )}
    </div>
  );
}

// Inline transcript chips.
function CorrectionChip() {
  return (
    <span
      className="inline-flex items-center gap-1 rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-amber-700"
      title="Self-correction detected — the later, corrected statement is preferred"
    >
      <CornerDownLeft className="h-3 w-3" />
      Correction
    </span>
  );
}

function SafetyChip({ count }: { count: number }) {
  return (
    <span
      className="inline-flex items-center gap-1 rounded-full border border-red-200 bg-red-100 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-red-700"
      title="Safety-critical content detected in this line"
    >
      <ShieldAlert className="h-3 w-3" />
      Safety{count > 1 ? ` ×${count}` : ""}
    </span>
  );
}

/* -------------------------------------------------------------------------- */
/*  Patient identity card                                                     */
/* -------------------------------------------------------------------------- */

function PatientCard({
  patient,
  onChange,
  disabled,
}: {
  patient: PatientInfo;
  onChange: (p: PatientInfo) => void;
  disabled: boolean;
}) {
  const ready = patient.name.trim().length > 0;
  return (
    <Card>
      <CardBody>
        <SectionHeader
          icon={<User className="h-[18px] w-[18px]" />}
          title="Patient"
          description="Identifies the FHIR record this note is filed under — used to create the Patient resource when you approve & push."
          actions={
            ready ? (
              <StatusBadge tone="success">Ready to file</StatusBadge>
            ) : (
              <StatusBadge tone="neutral">Required to file</StatusBadge>
            )
          }
        />
        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          <label className="flex flex-col gap-1.5">
            <FieldLabel>Name</FieldLabel>
            <Input
              value={patient.name}
              disabled={disabled}
              placeholder="e.g. Jane Doe"
              onChange={(e) => onChange({ ...patient, name: e.target.value })}
            />
          </label>
          <label className="flex flex-col gap-1.5">
            <FieldLabel>
              MRN{" "}
              <span className="font-normal normal-case text-slate-400">
                (optional)
              </span>
            </FieldLabel>
            <Input
              value={patient.mrn}
              disabled={disabled}
              placeholder="e.g. demo-0001"
              onChange={(e) => onChange({ ...patient, mrn: e.target.value })}
            />
          </label>
        </div>
      </CardBody>
    </Card>
  );
}

/* -------------------------------------------------------------------------- */
/*  Enrollment panel                                                          */
/* -------------------------------------------------------------------------- */

type EnrollPhase = "idle" | "recording" | "enrolling";

function EnrollmentPanel({
  enrolled,
  onEnrolledChange,
  disabled,
  onError,
  refreshStatus,
}: {
  enrolled: boolean | null;
  onEnrolledChange: (v: boolean) => void;
  disabled: boolean;
  onError: (msg: string) => void;
  refreshStatus: () => Promise<void>;
}) {
  const [phase, setPhase] = useState<EnrollPhase>("idle");
  const [secondsLeft, setSecondsLeft] = useState(ENROLL_SECONDS);
  const [clearing, setClearing] = useState(false);

  // Audio graph refs (separate from the main console so they never coexist).
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const workletRef = useRef<AudioWorkletNode | null>(null);
  const sinkRef = useRef<GainNode | null>(null);
  const chunksRef = useRef<Float32Array[]>([]);
  const stopTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const countdownRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const teardownAudio = useCallback(() => {
    if (stopTimerRef.current !== null) {
      clearTimeout(stopTimerRef.current);
      stopTimerRef.current = null;
    }
    if (countdownRef.current !== null) {
      clearInterval(countdownRef.current);
      countdownRef.current = null;
    }
    if (workletRef.current) {
      workletRef.current.port.onmessage = null;
      try {
        workletRef.current.disconnect();
      } catch {
        /* ignore */
      }
      workletRef.current = null;
    }
    if (sinkRef.current) {
      try {
        sinkRef.current.disconnect();
      } catch {
        /* ignore */
      }
      sinkRef.current = null;
    }
    if (sourceRef.current) {
      try {
        sourceRef.current.disconnect();
      } catch {
        /* ignore */
      }
      sourceRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (audioContextRef.current) {
      const ctx = audioContextRef.current;
      audioContextRef.current = null;
      if (ctx.state !== "closed") {
        ctx.close().catch(() => {
          /* ignore */
        });
      }
    }
  }, []);

  // Clean up on unmount.
  useEffect(() => {
    return () => {
      teardownAudio();
    };
  }, [teardownAudio]);

  // Concatenate the captured Float32 chunks and POST as Int16 PCM bytes.
  const submitEnrollment = useCallback(async () => {
    teardownAudio();
    setPhase("enrolling");

    const chunks = chunksRef.current;
    chunksRef.current = [];
    const total = chunks.reduce((n, c) => n + c.length, 0);
    const merged = new Float32Array(total);
    let off = 0;
    for (const c of chunks) {
      merged.set(c, off);
      off += c.length;
    }
    const pcm = floatTo16BitPCM(merged);

    try {
      const res = await fetch(`${API_URL}/enroll`, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: pcm,
      });
      const data = (await res.json()) as {
        ok: boolean;
        error?: string;
        durationSec?: number;
      };
      if (data.ok) {
        onEnrolledChange(true);
        await refreshStatus();
      } else {
        onError(data.error ?? "Enrollment failed. Please try again.");
      }
    } catch {
      onError("Could not reach the enrollment service.");
    } finally {
      setPhase("idle");
      setSecondsLeft(ENROLL_SECONDS);
    }
  }, [teardownAudio, onEnrolledChange, onError, refreshStatus]);

  const startEnrollment = useCallback(async () => {
    onError("");
    chunksRef.current = [];
    setSecondsLeft(ENROLL_SECONDS);

    try {
      if (typeof window === "undefined" || !navigator.mediaDevices) {
        throw new Error("Microphone access is not available in this browser.");
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      streamRef.current = stream;

      const AudioCtx: typeof AudioContext =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext })
          .webkitAudioContext;
      const audioContext = new AudioCtx({ sampleRate: TARGET_SAMPLE_RATE });
      audioContextRef.current = audioContext;
      if (audioContext.state === "suspended") await audioContext.resume();

      await audioContext.audioWorklet.addModule("/pcm-worklet.js");

      const source = audioContext.createMediaStreamSource(stream);
      sourceRef.current = source;

      const worklet = new AudioWorkletNode(audioContext, "pcm-worklet");
      workletRef.current = worklet;
      worklet.port.onmessage = (event: MessageEvent<Float32Array>) => {
        // Copy out of the transferred buffer for safe accumulation.
        chunksRef.current.push(new Float32Array(event.data));
      };

      const sink = audioContext.createGain();
      sink.gain.value = 0;
      sinkRef.current = sink;

      source.connect(worklet);
      worklet.connect(sink);
      sink.connect(audioContext.destination);

      setPhase("recording");

      // Live countdown.
      countdownRef.current = setInterval(() => {
        setSecondsLeft((s) => (s > 0 ? s - 1 : 0));
      }, 1000);

      // Auto-stop after ~ENROLL_SECONDS and submit.
      stopTimerRef.current = setTimeout(() => {
        void submitEnrollment();
      }, ENROLL_SECONDS * 1000);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to start enrollment.";
      onError(message);
      teardownAudio();
      setPhase("idle");
      setSecondsLeft(ENROLL_SECONDS);
    }
  }, [onError, submitEnrollment, teardownAudio]);

  const clearEnrollment = useCallback(async () => {
    setClearing(true);
    try {
      const res = await fetch(`${API_URL}/enroll`, { method: "DELETE" });
      if (res.ok) {
        onEnrolledChange(false);
        await refreshStatus();
      } else {
        onError("Could not clear the enrolled voice.");
      }
    } catch {
      onError("Could not reach the enrollment service.");
    } finally {
      setClearing(false);
    }
  }, [onEnrolledChange, onError, refreshStatus]);

  const recording = phase === "recording";
  const enrolling = phase === "enrolling";
  const buttonDisabled = disabled || recording || enrolling || clearing;

  return (
    <Card>
      <CardBody>
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div className="flex items-start gap-3">
            <span className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-clinical-accentSoft text-clinical-accent">
              <Mic className="h-[18px] w-[18px]" />
            </span>
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="text-[15px] font-semibold tracking-tight text-slate-900">
                  Nurse voice profile
                </h2>
                <EnrollStatusPill enrolled={enrolled} />
              </div>
              <p className="mt-1 max-w-md text-[13px] leading-relaxed text-slate-500">
                Read a sentence or two in your normal voice so we can tell you
                apart from the patient. Required for live Nurse / Patient labels.
              </p>
              {recording && (
                <p className="mt-2 text-xs font-medium text-clinical-accent">
                  Recording {secondsLeft}s…
                </p>
              )}
              {enrolling && (
                <p className="mt-2 text-xs font-medium text-clinical-accent">
                  Enrolling…
                </p>
              )}
            </div>
          </div>

          <div className="flex shrink-0 items-center gap-2">
            <Button
              onClick={() => void startEnrollment()}
              disabled={buttonDisabled}
              loading={enrolling}
            >
              {!enrolling && <Mic className="h-4 w-4" />}
              {recording
                ? `Recording ${secondsLeft}s…`
                : enrolling
                ? "Enrolling…"
                : enrolled
                ? "Re-enroll voice"
                : "Enroll nurse voice"}
            </Button>

            {enrolled && (
              <Button
                variant="secondary"
                onClick={() => void clearEnrollment()}
                disabled={buttonDisabled}
                title="Clear the enrolled nurse voice"
              >
                Clear
              </Button>
            )}
          </div>
        </div>
      </CardBody>
    </Card>
  );
}

/* -------------------------------------------------------------------------- */
/*  Small presentational pills                                                */
/* -------------------------------------------------------------------------- */

/* ---- clinical entity styling -------------------------------------------- */

// Display order + label + chip styling for each clinical category.
const CATEGORY_META: Record<
  EntityCategory,
  { label: string; chip: string; mark: string }
> = {
  symptom: {
    label: "Symptoms",
    chip: "border-rose-200 bg-rose-50 text-rose-700",
    mark: "bg-rose-100 text-rose-900 decoration-rose-300",
  },
  condition: {
    label: "Conditions",
    chip: "border-orange-200 bg-orange-50 text-orange-700",
    mark: "bg-orange-100 text-orange-900 decoration-orange-300",
  },
  medication: {
    label: "Medications & dosage",
    chip: "border-emerald-200 bg-emerald-50 text-emerald-700",
    mark: "bg-emerald-100 text-emerald-900 decoration-emerald-300",
  },
  vital: {
    label: "Vitals & labs",
    chip: "border-sky-200 bg-sky-50 text-sky-700",
    mark: "bg-sky-100 text-sky-900 decoration-sky-300",
  },
  procedure: {
    label: "Procedures",
    chip: "border-indigo-200 bg-indigo-50 text-indigo-700",
    mark: "bg-indigo-100 text-indigo-900 decoration-indigo-300",
  },
  anatomy: {
    label: "Anatomy",
    chip: "border-teal-200 bg-teal-50 text-teal-700",
    mark: "bg-teal-100 text-teal-900 decoration-teal-300",
  },
  context: {
    label: "Context",
    chip: "border-slate-200 bg-slate-100 text-slate-600",
    mark: "bg-slate-100 text-slate-700 decoration-slate-300",
  },
  other: {
    label: "Other",
    chip: "border-slate-200 bg-slate-100 text-slate-600",
    mark: "bg-slate-100 text-slate-700 decoration-slate-300",
  },
};

const CATEGORY_ORDER: EntityCategory[] = [
  "symptom",
  "condition",
  "medication",
  "vital",
  "procedure",
  "anatomy",
  "context",
  "other",
];

// Render an utterance with its NER spans wrapped in colored highlights, using
// the model's char offsets. Spans are non-overlapping within one utterance.
function renderWithEntities(text: string, entities: Entity[]): ReactNode {
  if (!entities || entities.length === 0) return text;
  const spans = entities
    .filter((e) => e.end > e.start && e.start >= 0 && e.end <= text.length)
    .sort((a, b) => a.start - b.start);

  const out: ReactNode[] = [];
  let cursor = 0;
  spans.forEach((e, i) => {
    if (e.start < cursor) return; // skip any overlap defensively
    if (e.start > cursor) out.push(text.slice(cursor, e.start));
    const meta = CATEGORY_META[e.category] ?? CATEGORY_META.other;
    out.push(
      <mark
        key={`${e.start}-${i}`}
        className={`rounded px-0.5 underline decoration-dotted underline-offset-2 ${meta.mark}`}
        title={`${meta.label} · ${Math.round(e.score * 100)}%`}
      >
        {text.slice(e.start, e.end)}
      </mark>
    );
    cursor = e.end;
  });
  if (cursor < text.length) out.push(text.slice(cursor));
  return out;
}

function speakerBorder(speaker: Speaker): string {
  switch (speaker) {
    case "nurse":
      return "border-l-4 border-cyan-400";
    case "patient":
      return "border-l-4 border-violet-400";
    default:
      return "border-l-4 border-transparent";
  }
}

function SpeakerChip({ speaker, sim }: { speaker: Speaker; sim: number }) {
  const map: Record<Speaker, { label: string; cls: string }> = {
    nurse: {
      label: "Nurse",
      cls: "border-cyan-200 bg-cyan-50 text-cyan-800",
    },
    patient: {
      label: "Patient",
      cls: "border-violet-200 bg-violet-50 text-violet-700",
    },
    unknown: {
      label: "Speaker",
      cls: "border-slate-200 bg-slate-50 text-slate-500",
    },
  };
  const v = map[speaker];
  const title =
    speaker === "unknown"
      ? "Speaker not identified"
      : `Voice match ${Math.round(sim * 100)}%`;
  return (
    <span
      className={`inline-flex w-fit items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${v.cls}`}
      title={title}
    >
      {v.label}
    </span>
  );
}

function EnrollStatusPill({ enrolled }: { enrolled: boolean | null }) {
  if (enrolled === null) return <StatusBadge tone="neutral">Checking…</StatusBadge>;
  return enrolled ? (
    <StatusBadge tone="success">Enrolled</StatusBadge>
  ) : (
    <StatusBadge tone="neutral">Not enrolled</StatusBadge>
  );
}

function FinalizingPill() {
  return (
    <StatusBadge tone="violet" pulse>
      Finalizing speakers…
    </StatusBadge>
  );
}

function ConnectionPill({
  status,
  model,
}: {
  status: ConnectionStatus;
  model: string | null;
}) {
  if (status === "connected") {
    return (
      <Badge tone="success">
        <Wifi className="h-3.5 w-3.5" />
        {model ? `Connected · ${model}` : "Connected"}
      </Badge>
    );
  }
  if (status === "connecting") {
    return (
      <Badge tone="warning">
        <Wifi className="h-3.5 w-3.5 animate-pulse" />
        Connecting…
      </Badge>
    );
  }
  return (
    <Badge tone="neutral">
      <WifiOff className="h-3.5 w-3.5" />
      Disconnected
    </Badge>
  );
}

function VadPill({ state }: { state: VadState }) {
  if (state === "speech") {
    return (
      <Badge tone="accent">
        <Activity className="h-3.5 w-3.5 animate-pulse" />
        Speaking
      </Badge>
    );
  }
  if (state === "silence") {
    return (
      <Badge tone="info">
        <Activity className="h-3.5 w-3.5" />
        Listening
      </Badge>
    );
  }
  return (
    <Badge tone="neutral">
      <Radio className="h-3.5 w-3.5" />
      Idle
    </Badge>
  );
}
