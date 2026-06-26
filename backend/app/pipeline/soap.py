"""SOAP note generation — Phase 4.

At encounter end, the speaker-labeled transcript + the clinical entities extracted
by NER are sent to GPT-4o, which returns a structured SOAP note (Subjective /
Objective / Assessment / Plan) as strict JSON. The note is always a DRAFT for
clinician review and the model is instructed never to invent clinical facts.

This is the only stage that calls a hosted API — everything upstream is on-device.
"""
import json

from .. import config

_SYSTEM = (
    "You are a clinical documentation assistant that drafts SOAP notes from a "
    "transcribed patient encounter, for review by a licensed clinician.\n\n"
    "Rules:\n"
    "- Use ONLY information explicitly stated in the transcript. Never invent or "
    "infer symptoms, diagnoses, medications, dosages, vital values, or history.\n"
    "- If something is ambiguous, incomplete, or only implied, omit it from the "
    "note body and add a short note to reviewFlags describing what to verify.\n"
    "- Subjective = the patient's reported symptoms/history; Objective = measured "
    "or observed data (vitals, exam findings); Assessment = clinical impression "
    "strictly grounded in the transcript; Plan = stated next steps / treatment.\n"
    "- Each section is a list of concise, single-fact bullet statements.\n"
    "- Do not provide medical advice beyond what was stated in the encounter.\n"
    "- The note is a DRAFT and always requires clinician review.\n"
    "- If the speaker self-corrects (e.g. 'actually', 'I mean', 'sorry, I meant', "
    "'scratch that'), treat the LATER corrected statement as authoritative and "
    "ignore the earlier contradicted one.\n"
    "- If the transcript lacks enough information for a section, return an empty "
    "list for it and explain what is missing in reviewFlags."
)

# Strict JSON schema -> reliable parsing, no fragile text handling.
_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "chiefComplaint": {"type": "string"},
        "subjective": {"type": "array", "items": {"type": "string"}},
        "objective": {"type": "array", "items": {"type": "string"}},
        "assessment": {"type": "array", "items": {"type": "string"}},
        "plan": {"type": "array", "items": {"type": "string"}},
        "medications": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "dosage": {"type": "string"},
                    "frequency": {"type": "string"},
                },
                "required": ["name", "dosage", "frequency"],
            },
        },
        "reviewFlags": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "chiefComplaint", "subjective", "objective", "assessment",
        "plan", "medications", "reviewFlags",
    ],
}


class SoapGenerator:
    """One-shot structured SOAP generation via the OpenAI Chat Completions API."""

    def __init__(self) -> None:
        self._client = None
        self.available = bool(config.OPENAI_API_KEY)

    def _client_or_load(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=config.OPENAI_API_KEY)
        return self._client

    def generate(self, transcript: str, entities_summary: str) -> dict:
        client = self._client_or_load()
        user = (
            "Transcript (speaker-labeled):\n"
            f"{transcript}\n\n"
            "Clinical entities extracted by NER (grounding hints, may be noisy):\n"
            f"{entities_summary or '(none detected)'}\n\n"
            "Produce the SOAP note as JSON following the schema."
        )
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ]
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "soap_note", "strict": True, "schema": _SCHEMA},
        }
        try:
            resp = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                temperature=config.SOAP_TEMPERATURE,
                messages=messages,
                response_format=response_format,
            )
        except Exception as exc:
            # Reasoning models (gpt-5 / o-series) only allow the default temperature.
            if "temperature" not in str(exc).lower():
                raise
            resp = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages,
                response_format=response_format,
            )
        note = json.loads(resp.choices[0].message.content)
        note["requiresReview"] = True  # always a draft for clinician review
        return note
