"""Transcript safety + quality analysis — Phase 6 clinical safeguards.

Three deterministic checks the case study calls for, all pure functions over text
so they are fast, transparent, auditable, and never leave the device:

1. scan_safety(text)          -> safety-critical alerts (self-harm / abuse /
                                 mandatory-reporting triggers) for escalation.
2. detect_correction(text)    -> self-correction cue ("actually, I mean…") so the
                                 note prefers the later, corrected statement.
3. check_completeness(...)    -> verify every extracted clinical entity is
                                 reflected in the final note; surface any missing.
"""
import re

# --- 1. Safety-critical detector ----------------------------------------------
# Curated phrase patterns per category. Deterministic (not an LLM) so an alert is
# always explainable by the exact phrase it matched, and it can never be argued
# away or hallucinated. Tuned for recall: a false alert costs a glance; a missed
# one is a safety failure.
_SAFETY_CATEGORIES = [
    {
        "category": "self_harm",
        "label": "Self-harm / suicidal ideation",
        "severity": "critical",
        "patterns": [
            r"\bkill myself\b", r"\bkilling myself\b", r"\bend my life\b",
            r"\btake my own life\b", r"\bwant to die\b", r"\bwanna die\b",
            r"\bwish i (was|were) dead\b", r"\bbetter off dead\b",
            r"\bno (reason|point) (to|in) (living|life)\b", r"\bsuicid(e|al)\b",
            r"\bharm myself\b", r"\bhurt myself\b", r"\bcut myself\b",
            r"\boverdos(e|ing|ed)\b", r"\bself[- ]harm\b",
        ],
    },
    {
        "category": "abuse",
        "label": "Abuse / interpersonal violence",
        "severity": "high",
        "patterns": [
            r"\b(he|she|they|partner|husband|wife|boyfriend|girlfriend) (hit|hits|beat|beats|hurt|hurts|chok(e|es)) me\b",
            r"\bhitting me\b", r"\bbeating me\b", r"\bbeats me\b",
            r"\bthreaten(s|ed)? (to|me)\b", r"\bdomestic violence\b",
            r"\b(sexual(ly)? )?assault(ed|s)?\b", r"\bmolest(ed|ing)?\b",
            r"\bafraid (to go home|of (him|her|them|my))\b",
            r"\bnot safe at home\b", r"\bbeen abused\b", r"\b(he|she|they) abuse(s|d)? me\b",
        ],
    },
    {
        "category": "mandatory_reporting",
        "label": "Possible mandatory-reporting trigger",
        "severity": "high",
        "patterns": [
            r"\bchild abuse\b", r"\belder abuse\b", r"\bneglect(ed|ing)?\b",
            r"\bgun ?shot wound\b", r"\bstab(bed| wound)\b",
            r"\bhurting (the|my|their) (child|kid|baby)\b",
            r"\bunsafe (child|minor|kid)\b",
        ],
    },
]

# Pre-compile for speed (called on every finalized utterance).
for _cat in _SAFETY_CATEGORIES:
    _cat["_compiled"] = [re.compile(p, re.IGNORECASE) for p in _cat["patterns"]]


def _snippet(text: str, start: int, end: int, pad: int = 28) -> str:
    a, b = max(0, start - pad), min(len(text), end + pad)
    s = text[a:b].strip()
    if a > 0:
        s = "…" + s
    if b < len(text):
        s = s + "…"
    return s


def scan_safety(text: str) -> list[dict]:
    """Return safety alerts found in `text`: [{category,label,severity,term,snippet}]."""
    if not text:
        return []
    alerts: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for cat in _SAFETY_CATEGORIES:
        for rx in cat["_compiled"]:
            m = rx.search(text)
            if not m:
                continue
            term = m.group(0)
            key = (cat["category"], term.lower())
            if key in seen:
                continue
            seen.add(key)
            alerts.append({
                "category": cat["category"],
                "label": cat["label"],
                "severity": cat["severity"],
                "term": term,
                "snippet": _snippet(text, m.start(), m.end()),
            })
    return alerts


# --- 2. Correction-pattern detection ------------------------------------------
_CORRECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bactually\b", r"\bi mean\b", r"\bi meant\b", r"\bsorry,? i (mean|meant)\b",
        r"\bscratch that\b", r"\bno,? wait\b", r"\blet me correct\b",
        r"\bcorrection\b", r"\bthat'?s not right\b", r"\bi misspoke\b",
        r"\bstrike that\b",
    ]
]


def detect_correction(text: str) -> dict | None:
    """If `text` contains a self-correction cue, return {marker}; else None.

    The case study's rule: prefer the LATER (corrected) statement. We surface the
    cue so the clinician sees it and so the SOAP prompt is told to honor it.
    """
    if not text:
        return None
    for rx in _CORRECTION_PATTERNS:
        m = rx.search(text)
        if m:
            return {"marker": m.group(0)}
    return None


# --- 3. Completeness check ----------------------------------------------------
# Only clinically meaningful categories must appear in the note; "context" and
# "other" spans (e.g. "yesterday", "feeling") would over-flag.
_RELEVANT_CATEGORIES = {"symptom", "medication", "condition", "vital", "procedure", "anatomy"}


def _note_text(note: dict) -> str:
    parts: list[str] = [note.get("chiefComplaint", "")]
    for key in ("subjective", "objective", "assessment", "plan", "reviewFlags"):
        parts.extend(note.get(key, []) or [])
    for med in note.get("medications", []) or []:
        parts.extend([med.get("name", ""), med.get("dosage", ""), med.get("frequency", "")])
    return " ".join(parts).lower()


def check_completeness(entities: list[dict], note: dict) -> dict:
    """Cross-check extracted clinical entities against the final note.

    Returns {coveredCount, totalCount, missing:[...]}. An entity is "covered" if
    its text (or every word of it) appears somewhere in the note.
    """
    haystack = _note_text(note)
    seen: set[str] = set()
    missing: list[str] = []
    total = 0
    for e in entities or []:
        if e.get("category") not in _RELEVANT_CATEGORIES:
            continue
        term = (e.get("text") or "").strip()
        key = term.lower()
        if not term or key in seen:
            continue
        seen.add(key)
        total += 1
        words = [w for w in re.findall(r"[a-z0-9]+", key) if len(w) > 2]
        covered = key in haystack or (bool(words) and all(w in haystack for w in words))
        if not covered:
            missing.append(term)
    return {"coveredCount": total - len(missing), "totalCount": total, "missing": missing}
