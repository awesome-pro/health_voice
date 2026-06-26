"""Clinical entity extraction (medical NER) — Phase 3.

Runs a biomedical NER model (d4data/biomedical-ner-all, BERT-family, public/ungated)
over each finalized utterance and tags clinical concepts — symptoms, conditions,
medications + dosage, vitals/labs, anatomy. These become the highlighted entities in
the transcript and the structured input for the Phase 4 SOAP note.

The model emits ~84 fine-grained labels; we fold them into a few clinical buckets so
the UI stays legible and downstream code has a stable contract.
"""
from .. import config

# Fine-grained model label -> clinical bucket surfaced to the UI / SOAP stage.
_CATEGORY = {
    "Sign_symptom": "symptom",
    "Disease_disorder": "condition",
    "Medication": "medication",
    "Dosage": "medication",
    "Administration": "medication",
    "Frequency": "medication",
    "Lab_value": "vital",
    "Diagnostic_procedure": "vital",
    "Therapeutic_procedure": "procedure",
    "Biological_structure": "anatomy",
    "Duration": "context",
    "Severity": "context",
    "Detailed_description": "context",
    "History": "context",
    "Date": "context",
}
# Only these buckets are surfaced; everything else is dropped as low-signal noise.
_KEEP = {"symptom", "condition", "medication", "vital", "procedure", "anatomy", "context"}

# Common English words the biomedical model occasionally mislabels as clinical
# (often from ASR hallucinations on silence, e.g. "love" -> Lab_value). Dropped
# regardless of the model's confidence, since none are clinical on their own.
_DENYLIST = {
    "love", "grade", "thank", "thanks", "okay", "ok", "yeah", "yes", "no",
    "hello", "hi", "bye", "please", "sorry", "right", "well", "like", "good",
}


class ClinicalNER:
    """Lazy-loaded HuggingFace token-classification pipeline for clinical NER."""

    def __init__(self, model_repo: str) -> None:
        self.model_repo = model_repo
        self._pipe = None
        self.ready = False

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline

            # CPU keeps it stable across torch/MPS versions; spans are cheap.
            # "first" aggregates subwords to whole words (so "ib"+"##uprofen" ->
            # "ibuprofen") and keeps the leading subword's confidence.
            self._pipe = pipeline(
                "ner",
                model=self.model_repo,
                aggregation_strategy="first",
                device=-1,
            )
        return self._pipe

    def warmup(self) -> None:
        self._load()
        self.extract("Patient reports a sore throat and fever since yesterday.")
        self.ready = True

    def extract(self, text: str) -> list[dict]:
        """Return [{text,label,category,start,end,score}, ...] for one utterance."""
        if not text or not text.strip():
            return []
        spans = self._load()(text)
        out = []
        for e in spans:
            score = float(e["score"])
            if score < config.NER_MIN_SCORE:
                continue
            category = _CATEGORY.get(e["entity_group"], "other")
            if category not in _KEEP:
                continue
            start, end = int(e["start"]), int(e["end"])
            span_text = text[start:end]
            if span_text.lower().strip() in _DENYLIST:
                continue
            out.append({
                "text": span_text,
                "label": e["entity_group"],
                "category": category,
                "start": start,
                "end": end,
                "score": round(score, 3),
            })
        return out
