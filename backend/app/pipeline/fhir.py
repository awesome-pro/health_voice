"""FHIR push — Phase 5.

The clinician-reviewed (and possibly edited) SOAP note is converted to FHIR R4
resources and written to a FHIR server as a single atomic *transaction Bundle*:

    Patient                  <- name + MRN entered by the nurse
    Encounter                <- this visit (status "finished")
    MedicationStatement x N  <- structured meds (name / dosage / frequency)
    Composition              <- the SOAP note as a clinical document: one section
                                per Chief Complaint / S / O / A / P (+ review
                                flags), status "final" because a human approved it.

Resources reference each other by `urn:uuid:` placeholders so the server assigns
real IDs and resolves the links atomically. We target a local HAPI FHIR server
(see docker-compose.yml) so patient data never leaves the machine.

The resource JSON is hand-built (rather than via a heavyweight FHIR SDK) so the
shape is fully transparent for the write-up.
"""
import html
import uuid
from datetime import datetime, timezone

import httpx

from .. import config

# LOINC code for a SOAP-style clinical note.
_NOTE_TYPE = {
    "coding": [{"system": "http://loinc.org", "code": "11488-4", "display": "Consult note"}],
    "text": "SOAP note",
}


def _narrative(lines: list[str]) -> dict:
    """A FHIR Narrative: bullet list of escaped text wrapped in XHTML."""
    items = "".join(f"<li>{html.escape(line)}</li>" for line in lines) or "<li>Not documented.</li>"
    div = f'<div xmlns="http://www.w3.org/1999/xhtml"><ul>{items}</ul></div>'
    return {"status": "generated", "div": div}


def _patient_resource(name: str, mrn: str) -> dict:
    parts = name.split()
    human_name: dict = {"text": name}
    if len(parts) >= 2:
        human_name["family"] = parts[-1]
        human_name["given"] = parts[:-1]
    elif parts:
        human_name["family"] = parts[0]
    resource: dict = {"resourceType": "Patient", "name": [human_name]}
    if mrn:
        resource["identifier"] = [{
            "type": {"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v2-0203",
                "code": "MR", "display": "Medical record number",
            }]},
            "system": "urn:healthvoice:mrn",
            "value": mrn,
        }]
    return resource


def _section(title: str, lines: list[str], entries: list[str] | None = None) -> dict:
    sec = {"title": title, "text": _narrative(lines)}
    if entries:
        sec["entry"] = [{"reference": ref} for ref in entries]
    return sec


def build_bundle(patient: dict, note: dict, clinician: str = "", edits: list | None = None) -> dict:
    """Assemble the transaction Bundle from the patient demographics + SOAP note.

    When `clinician` is given, the Composition records them as the legal attester
    and a FHIR Provenance resource captures the AI-authored / clinician-approved
    chain (with the count of edits made to the AI draft) for the audit trail.
    """
    patient_url = f"urn:uuid:{uuid.uuid4()}"
    encounter_url = f"urn:uuid:{uuid.uuid4()}"
    composition_url = f"urn:uuid:{uuid.uuid4()}"
    now = datetime.now(timezone.utc).isoformat()
    clinician = (clinician or "").strip()
    edit_count = len(edits or [])

    entries: list[dict] = [
        {
            "fullUrl": patient_url,
            "resource": _patient_resource(patient.get("name", "").strip(),
                                          patient.get("mrn", "").strip()),
            "request": {"method": "POST", "url": "Patient"},
        },
        {
            "fullUrl": encounter_url,
            "resource": {
                "resourceType": "Encounter",
                "status": "finished",
                "class": {
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                    "code": "AMB", "display": "ambulatory",
                },
                "subject": {"reference": patient_url},
            },
            "request": {"method": "POST", "url": "Encounter"},
        },
    ]

    # One MedicationStatement per structured medication; collect refs for the
    # Composition's Medications section.
    med_refs: list[str] = []
    for med in note.get("medications", []):
        med_url = f"urn:uuid:{uuid.uuid4()}"
        med_refs.append(med_url)
        dose = " ".join(p for p in (med.get("dosage", ""), med.get("frequency", "")) if p).strip()
        resource = {
            "resourceType": "MedicationStatement",
            "status": "active",
            "medicationCodeableConcept": {"text": med.get("name", "").strip() or "Unspecified"},
            "subject": {"reference": patient_url},
            "context": {"reference": encounter_url},
        }
        if dose:
            resource["dosage"] = [{"text": dose}]
        entries.append({
            "fullUrl": med_url,
            "resource": resource,
            "request": {"method": "POST", "url": "MedicationStatement"},
        })

    # The SOAP note itself, as a clinical document. status "final" = approved by
    # a clinician (the upstream draft was "preliminary").
    chief = (note.get("chiefComplaint") or "").strip()
    sections = [
        _section("Chief Complaint", [chief] if chief else []),
        _section("Subjective", note.get("subjective", [])),
        _section("Objective", note.get("objective", [])),
        _section("Assessment", note.get("assessment", [])),
        _section("Plan", note.get("plan", [])),
    ]
    if med_refs:
        med_lines = [
            " — ".join(p for p in (
                m.get("name", ""),
                " ".join(q for q in (m.get("dosage", ""), m.get("frequency", "")) if q),
            ) if p).strip()
            for m in note.get("medications", [])
        ]
        sections.append(_section("Medications", med_lines, entries=med_refs))
    if note.get("reviewFlags"):
        sections.append(_section("Items flagged for verification", note["reviewFlags"]))

    composition = {
        "resourceType": "Composition",
        "status": "final",
        "type": _NOTE_TYPE,
        "subject": {"reference": patient_url},
        "encounter": {"reference": encounter_url},
        "date": now,
        "author": [{"display": "HealthVoice AI scribe (clinician-reviewed)"}],
        "title": chief or "Clinical encounter note",
        "section": sections,
    }
    # The clinician who approved the note signs it as the legal attester.
    if clinician:
        composition["attester"] = [{
            "mode": "legal",
            "time": now,
            "party": {"display": clinician},
        }]
    entries.append({
        "fullUrl": composition_url,
        "resource": composition,
        "request": {"method": "POST", "url": "Composition"},
    })

    # Provenance: the audit chain for this note — AI authored it, the clinician
    # reviewed + approved it (with N edits), recorded at `now`.
    edit_summary = (
        f"AI-generated draft reviewed and approved by clinician with {edit_count} "
        f"edit(s)." if clinician else
        f"AI-generated draft approved with {edit_count} edit(s)."
    )
    provenance = {
        "resourceType": "Provenance",
        "target": [{"reference": composition_url}],
        "recorded": now,
        "activity": {"coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/v3-DataOperation",
            "code": "CREATE", "display": "create",
        }]},
        "agent": [
            {
                "type": {"coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/provenance-participant-type",
                    "code": "author", "display": "Author",
                }]},
                "who": {"display": clinician or "Clinician (unspecified)"},
            },
            {
                "type": {"coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/provenance-participant-type",
                    "code": "assembler", "display": "Assembler",
                }]},
                "who": {"display": "HealthVoice AI scribe"},
            },
        ],
        "text": {
            "status": "generated",
            "div": f'<div xmlns="http://www.w3.org/1999/xhtml">{html.escape(edit_summary)}</div>',
        },
    }
    entries.append({
        "fullUrl": f"urn:uuid:{uuid.uuid4()}",
        "resource": provenance,
        "request": {"method": "POST", "url": "Provenance"},
    })

    return {"resourceType": "Bundle", "type": "transaction", "entry": entries}


class FhirPublisher:
    """Pushes the SOAP note to a FHIR R4 server as an atomic transaction."""

    def __init__(self) -> None:
        self.base_url = config.FHIR_BASE_URL.rstrip("/")

    async def status(self) -> bool:
        """True if the FHIR server's capability statement is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/metadata",
                                        headers={"Accept": "application/fhir+json"})
            return resp.status_code == 200
        except Exception:
            return False

    async def push(self, patient: dict, note: dict, clinician: str = "",
                   edits: list | None = None) -> dict:
        bundle = build_bundle(patient, note, clinician=clinician, edits=edits)
        async with httpx.AsyncClient(timeout=config.FHIR_TIMEOUT) as client:
            resp = await client.post(
                self.base_url,
                json=bundle,
                headers={"Content-Type": "application/fhir+json",
                         "Accept": "application/fhir+json"},
            )
        if resp.status_code >= 400:
            raise RuntimeError(f"FHIR server returned {resp.status_code}: {resp.text[:500]}")

        result = resp.json()
        created: list[dict] = []
        for entry in result.get("entry", []):
            location = entry.get("response", {}).get("location", "")
            # location looks like "Composition/123/_history/1"
            parts = location.split("/")
            if len(parts) >= 2:
                rtype, rid = parts[0], parts[1]
                created.append({
                    "type": rtype,
                    "id": rid,
                    "url": f"{self.base_url}/{rtype}/{rid}",
                })
        return {"server": self.base_url, "resources": created}
