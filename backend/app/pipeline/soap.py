"""SOAP note generation — Phase 4.

At encounter end, the speaker-labeled transcript + the clinical entities extracted
by NER are sent to a language model, which returns a structured SOAP note
(Subjective / Objective / Assessment / Plan) as strict JSON. The note is always a
DRAFT for clinician review and the model is instructed never to invent clinical
facts.

Two interchangeable backends (config.SOAP_BACKEND):
  - "openai": a hosted OpenAI model. The only stage that touches the network.
  - "ollama": a local model served by Ollama, so the WHOLE pipeline is on-device
    and no patient data ever leaves the machine.
Both are pinned to the same strict JSON schema, so the rest of the app does not
care which one produced the note.
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
    """One-shot structured SOAP generation via OpenAI (hosted) or Ollama (local)."""

    def __init__(self) -> None:
        self.backend = config.SOAP_BACKEND
        self._client = None
        if self.backend == "ollama":
            # Local model. No API key needed; reachability is checked at call time
            # and any failure is surfaced to the UI as a normal generation error.
            self.model = config.OLLAMA_MODEL
            self.available = True
        else:
            self.model = config.OPENAI_MODEL
            self.available = bool(config.OPENAI_API_KEY)

    def generate(self, transcript: str, entities_summary: str) -> dict:
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
        note = (
            self._generate_ollama(messages)
            if self.backend == "ollama"
            else self._generate_openai(messages)
        )
        note["requiresReview"] = True  # always a draft for clinician review
        return note

    # --- OpenAI (hosted) ------------------------------------------------------
    def _client_or_load(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=config.OPENAI_API_KEY)
        return self._client

    def _generate_openai(self, messages: list[dict]) -> dict:
        client = self._client_or_load()
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
        return json.loads(resp.choices[0].message.content)

    # --- Ollama (local, on-device) -------------------------------------------
    def _generate_ollama(self, messages: list[dict]) -> dict:
        """Call the native Ollama API with schema-constrained, no-thinking output.

        `format` is the JSON schema, which makes Ollama constrain decoding to valid
        JSON of that shape. `think=False` disables qwen3-style reasoning so the
        reply is the note and nothing else.
        """
        import httpx

        payload = {
            "model": config.OLLAMA_MODEL,
            "messages": messages,
            "format": _SCHEMA,
            "stream": False,
            "think": False,
            "options": {"temperature": config.SOAP_TEMPERATURE},
        }
        resp = httpx.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=config.SOAP_TIMEOUT,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        return json.loads(content)
