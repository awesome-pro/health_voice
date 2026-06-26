"""Speaker diarization — Phase 2.

Two layers, mirroring the case study:

1. **Voiceprint role-mapping (live):** the nurse enrolls their voice once; we store
   an ECAPA speaker embedding. Each finalized utterance is embedded and compared
   (cosine) to the nurse voiceprint -> labeled "nurse" or "patient" instantly.
   Uses a public model, so it works without any HF token.

2. **Full pyannote pass (on Stop):** pyannote/speaker-diarization-3.1 clusters the
   whole encounter into speaker turns; each cluster is mapped to nurse/patient by
   comparing its mean embedding to the voiceprint. Gated -> needs HF_TOKEN.
"""
import numpy as np
import torch

from .. import config


class SpeakerEmbedder:
    """L2-normalized ECAPA-TDNN speaker embeddings (public speechbrain model).

    We call speechbrain's EncoderClassifier directly rather than via pyannote's
    PretrainedSpeakerEmbedding wrapper, which passes a `token` kwarg incompatible
    with this speechbrain version.
    """

    def __init__(self) -> None:
        from speechbrain.inference.speaker import EncoderClassifier

        # CPU keeps speechbrain stable across torch/MPS versions; embeddings are cheap.
        self._model = EncoderClassifier.from_hparams(
            source=config.SPEAKER_EMBEDDING_MODEL,
            savedir=str(config.BASE_DIR / ".cache" / "ecapa"),
            run_opts={"device": "cpu"},
        )
        self.ready = False

    def warmup(self) -> None:
        self.embed(np.zeros(config.SAMPLE_RATE, dtype=np.float32))
        self.ready = True

    def embed(self, audio: np.ndarray) -> np.ndarray:
        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
        emb = self._model.encode_batch(wav).reshape(-1).cpu().numpy()
        norm = float(np.linalg.norm(emb))
        return emb / norm if norm > 0 else emb


class VoiceprintStore:
    """Persists the enrolled nurse voiceprint (a single normalized embedding)."""

    def __init__(self, path) -> None:
        self.path = path
        self.embedding: np.ndarray | None = None
        if path.exists():
            self.embedding = np.load(path)

    @property
    def enrolled(self) -> bool:
        return self.embedding is not None

    def set(self, emb: np.ndarray) -> None:
        self.embedding = emb
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.path, emb)

    def clear(self) -> None:
        self.embedding = None
        if self.path.exists():
            self.path.unlink()

    def similarity(self, emb: np.ndarray) -> float:
        if self.embedding is None:
            return 0.0
        return float(np.dot(self.embedding, emb))  # both L2-normalized -> cosine

    def label(self, emb: np.ndarray) -> tuple[str, float]:
        """Return (speaker, similarity). 'unknown' if no voiceprint enrolled yet."""
        if self.embedding is None:
            return "unknown", 0.0
        sim = self.similarity(emb)
        return ("nurse" if sim >= config.SPEAKER_THRESHOLD else "patient"), sim


class FullDiarizer:
    """Lazy wrapper over the gated pyannote diarization pipeline (Phase 2b)."""

    def __init__(self) -> None:
        self._pipeline = None
        self.available = config.HF_TOKEN is not None
        self.error: str | None = None

    def _load(self):
        if self._pipeline is None:
            from pyannote.audio import Pipeline

            # pyannote.audio >=3.1 renamed the auth arg `use_auth_token` -> `token`.
            # Try the new name first, fall back to the old one for older installs.
            try:
                self._pipeline = Pipeline.from_pretrained(
                    config.DIARIZATION_MODEL, token=config.HF_TOKEN
                )
            except TypeError:
                self._pipeline = Pipeline.from_pretrained(
                    config.DIARIZATION_MODEL, use_auth_token=config.HF_TOKEN
                )
        return self._pipeline

    def diarize(self, audio: np.ndarray) -> list[dict]:
        """Return [{'start': s, 'end': s, 'speaker': 'SPEAKER_xx'}, ...] over the encounter."""
        pipeline = self._load()
        waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).reshape(1, -1)
        annotation = pipeline({"waveform": waveform, "sample_rate": config.SAMPLE_RATE})
        return [
            {"start": float(turn.start), "end": float(turn.end), "speaker": str(label)}
            for turn, _, label in annotation.itertracks(yield_label=True)
        ]
