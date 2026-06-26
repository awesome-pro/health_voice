"""Audit trail — Phase 6 compliance safeguard.

Every approve-and-file action is appended as one JSON line to an on-device audit
log (never leaves the machine), capturing WHO approved, WHEN, WHAT they changed
from the AI draft, and which FHIR resources were created. This is the local
counterpart to the FHIR Provenance resource written into the bundle itself.
"""
import json
from datetime import datetime, timezone

from .. import config


def append_event(event: dict) -> dict:
    """Append one timestamped event to the audit log; return it (with timestamp)."""
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
    config.AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.AUDIT_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def recent(limit: int = 50) -> list[dict]:
    """Return the most recent audit events, newest first."""
    if not config.AUDIT_PATH.exists():
        return []
    with config.AUDIT_PATH.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
    out: list[dict] = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(out) >= limit:
            break
    return out
