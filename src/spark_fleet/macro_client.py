"""
spark_fleet/macro_client.py

Runs on the Micro Spark (Mac Mini M2).

Responsibilities
----------------
1. Accept raw PDF text chunks.
2. Build a structured prompt that forces the local 122B LLM to return JSON.
3. POST the prompt to the Macro Spark (DGX) via its OpenAI-compatible endpoint.
4. Parse, validate, filter, and deduplicate the returned sponsor list.
5. Raise clear, typed errors on any failure — never let bad LLM output
   silently corrupt downstream stages.

The DGX exposes an OpenAI-compatible chat-completion endpoint.
The client sends a single user message and instructs the model to reply ONLY
with a JSON object matching the schema below.

Expected LLM JSON schema
------------------------
{
  "sponsors": [
    {
      "company_name": "Medtronic",
      "sponsor_tier": "Gold",
      "source_page": 3,
      "evidence_text": "Medtronic – Gold Sponsor",
      "important_people": ["Jane Doe", "Arun Mehta"],
      "confidence": 0.92
    }
  ]
}

Design decisions
----------------
- JSON is extracted with a regex fence stripper before parsing so the model
  can wrap it in ```json ... ``` without breaking the pipeline.
- Confidence filtering (>= 0.5) and deduplication are pure functions so they
  can be tested independently of HTTP calls.
- Timeouts and connection errors surface as MacroSparkError(retryable=True).
- Schema mismatches surface as MacroSparkError(retryable=False).
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from typing import Any

import httpx
from pydantic import ValidationError

from spark_fleet.schemas import ExtractedSponsor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed exception
# ---------------------------------------------------------------------------

class MacroSparkError(RuntimeError):
    """
    Raised for any failure in the Macro Spark extraction stage.

    Attributes
    ----------
    retryable : bool
        True  → transient (network timeout, 5xx).  Caller should back off
                and retry.
        False → permanent (malformed JSON, schema mismatch).  Retrying will
                not help; the job needs human review.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_JSON_SCHEMA_DESCRIPTION = textwrap.dedent("""\
    You are a precise data-extraction assistant.
    Your ONLY task is to read the conference brochure text and extract every
    company listed as a sponsor, exhibitor, or supporter. Also extract named
    people who appear connected to each sponsor and seem useful for outreach.

    You MUST reply with ONLY a single JSON object — no markdown, no explanation,
    no text before or after the JSON. The JSON must be complete and valid.

    Required format:
    {"sponsors": [{"company_name": "<string>", "sponsor_tier": "<Gold|Silver|Platinum|Bronze|Unknown>", "important_people": ["<person name>"], "brochure_emails": ["<email>"], "brochure_phones": ["<phone>"]}]}

    Example of a correct response:
    {"sponsors": [{"company_name": "Pfizer", "sponsor_tier": "Gold", "important_people": ["Jane Doe"], "brochure_emails": ["partnerships@pfizer.com"], "brochure_phones": ["+14155552671"]}, {"company_name": "Novartis", "sponsor_tier": "Silver", "important_people": [], "brochure_emails": [], "brochure_phones": []}, {"company_name": "Medtronic", "sponsor_tier": "Unknown", "important_people": ["Arun Mehta"], "brochure_emails": [], "brochure_phones": []}]}

    Rules:
    - Only include companies explicitly presented as sponsors, exhibitors, or supporters.
    - Do NOT include speakers, organizers, or venue names.
    - For important_people, include only named people directly associated with the sponsor company.
    - Prefer senior or outreach-relevant roles: founder, CEO, CMO, VP, director, marketing, partnerships, sponsorship, business development, commercial, sales, booth/contact person.
    - If no useful person is listed for a sponsor, use [].
    - If any email or phone appears near the sponsor name in the brochure, include it in brochure_emails/brochure_phones.
    - brochure_emails and brochure_phones must always be present (use [] when absent).
    - Use "Unknown" for tier if not stated.
    - Deduplicate: include each company only once.
    - Your ENTIRE response must be valid JSON that can be parsed by json.loads().
""")


def build_extraction_prompt(pdf_text: str, conference_name: str | None = None) -> list[dict[str, str]]:
    """
    Build the OpenAI-compatible chat messages list for the DGX model.

    Parameters
    ----------
    pdf_text        : Raw text extracted from the conference brochure PDF.
    conference_name : Optional hint to improve model focus.

    Returns
    -------
    A ``messages`` list ready to pass as ``json={"messages": ..., "model": ...}``.
    """
    context = f"Conference: {conference_name}\n\n" if conference_name else ""
    user_content = f"{context}--- BROCHURE TEXT START ---\n{pdf_text}\n--- BROCHURE TEXT END ---"
    return [
        {"role": "system", "content": _JSON_SCHEMA_DESCRIPTION},
        {"role": "user",   "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Response parsing helpers (pure, no I/O → easy to unit-test)
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_PHONE_RE = re.compile(r"(?:(?:\+\d{1,3}[\s().-]*)?(?:\d[\s().-]*){8,15})")


def _strip_markdown_fence(text: str) -> str:
    """Remove ```json ... ``` fences that some models wrap around JSON."""
    match = _JSON_FENCE_RE.search(text)
    return match.group(1) if match else text.strip()


_TRUNCATED_JSON_RE = re.compile(r'\{\s*"company_name"\s*:\s*"([^"]+)"[^}]*\}', re.DOTALL)


def _recover_truncated_sponsors(text: str) -> list[ExtractedSponsor] | None:
    """
    Last-resort recovery for truncated JSON.
    Scans for every partial '{"company_name": "..."}' pattern and salvages them.
    Returns a list of sponsors if any could be extracted, else None.
    """
    matches = _TRUNCATED_JSON_RE.findall(text)
    if not matches:
        return None
    sponsors = []
    for name in matches:
        name = name.strip()
        if name:
            # Extract tier from the context around this match if possible
            sponsors.append(ExtractedSponsor(company_name=name, source_page=1))
    return sponsors if sponsors else None


def _normalise_item(item: Any) -> dict[str, Any]:
    """
    Normalise a single raw item from the LLM into a dict that matches
    the ExtractedSponsor schema.

    Handles:
    - Plain strings:        "Pfizer"  →  {"company_name": "Pfizer"}
    - Alt key names:        {"name": "Pfizer", "tier": "Gold"} → normalised
    - Correct format:       {"company_name": "Pfizer", "sponsor_tier": "Gold"}
    """
    if isinstance(item, str):
        return {"company_name": item.strip(), "source_page": 1}

    if not isinstance(item, dict):
        return {}

    # Copy so we don't mutate the original
    d = dict(item)

    # Normalise alternative key names some models use
    for alt, canonical in [
        ("name",         "company_name"),
        ("company",      "company_name"),
        ("tier",         "sponsor_tier"),
        ("level",        "sponsor_tier"),
        ("page",         "source_page"),
        ("page_number",  "source_page"),
        ("evidence",     "evidence_text"),
        ("snippet",      "evidence_text"),
        ("people",       "important_people"),
        ("persons",      "important_people"),
        ("contacts",     "important_people"),
        ("representatives", "important_people"),
        ("emails", "brochure_emails"),
        ("contact_emails", "brochure_emails"),
        ("brochure_email", "brochure_emails"),
        ("phones", "brochure_phones"),
        ("contact_phones", "brochure_phones"),
        ("brochure_phone", "brochure_phones"),
    ]:
        if alt in d and canonical not in d:
            d[canonical] = d.pop(alt)

    if "important_people" not in d:
        d["important_people"] = []
    elif isinstance(d["important_people"], str):
        d["important_people"] = [d["important_people"]]
    elif not isinstance(d["important_people"], list):
        d["important_people"] = []

    d["important_people"] = [
        str(person).strip()
        for person in d["important_people"]
        if str(person).strip()
    ][:5]

    d["brochure_emails"] = _normalise_email_list(d.get("brochure_emails"))
    d["brochure_phones"] = _normalise_phone_list(d.get("brochure_phones"))

    # Ensure source_page is present and is an int
    if "source_page" not in d or not isinstance(d.get("source_page"), int):
        d["source_page"] = 1

    return d


def _normalise_email_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        candidate = str(item).strip().lower()
        if not candidate or "@" not in candidate:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
    return out[:5]


def _normalise_phone_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        normalised = _normalise_phone(str(item).strip())
        if not normalised or normalised in seen:
            continue
        seen.add(normalised)
        out.append(normalised)
    return out[:5]


def _normalise_phone(value: str | None) -> str | None:
    if not value:
        return None
    digits = "".join(ch for ch in value if ch.isdigit())
    if not 8 <= len(digits) <= 15:
        return None
    return f"+{digits}"


def _extract_contacts_near_company(pdf_text: str, company_name: str) -> tuple[list[str], list[str]]:
    text = pdf_text or ""
    company = company_name.strip()
    if not text or not company:
        return [], []

    lower_text = text.lower()
    needle = company.lower()
    indices: list[int] = []
    start = 0
    while True:
        pos = lower_text.find(needle, start)
        if pos == -1:
            break
        indices.append(pos)
        start = pos + len(needle)

    if not indices:
        return [], []

    snippets: list[str] = []
    for pos in indices[:8]:
        begin = max(0, pos - 320)
        end = min(len(text), pos + len(company) + 320)
        snippets.append(text[begin:end])
    joined = "\n".join(snippets)

    emails = _normalise_email_list(_EMAIL_RE.findall(joined))
    phones = _normalise_phone_list(_PHONE_RE.findall(joined))
    return emails, phones


def parse_macro_response(raw: dict[str, Any] | str) -> list[ExtractedSponsor]:
    """
    Parse the Macro Spark's response into a validated list of ExtractedSponsor.

    ``raw`` can be:
    - A dict already decoded from JSON (``{"sponsors": [...]}``).
    - A raw string (model completion text) — fences will be stripped first.

    Raises
    ------
    MacroSparkError(retryable=False)
        If the payload cannot be parsed or does not match the expected schema.
    """
    original_text: str = ""

    # -- Step 1: ensure we have a dict ----------------------------------------
    if isinstance(raw, str):
        original_text = raw
        cleaned = _strip_markdown_fence(raw)
        try:
            raw = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try truncation recovery before giving up
            salvaged = _recover_truncated_sponsors(original_text)
            if salvaged:
                logger.warning(
                    "Partial JSON recovery succeeded: salvaged %d sponsor(s) "
                    "from truncated LLM output.", len(salvaged)
                )
                return salvaged
            raise MacroSparkError(
                "Macro Spark returned malformed JSON that could not be recovered.",
                retryable=False,
            )

    # -- Step 2: extract the sponsors list ------------------------------------
    # Handle multiple response shapes the model might produce
    if isinstance(raw, list):
        # Model returned a bare list instead of wrapping it in {"sponsors": [...]}
        sponsors_raw: list[Any] = raw
    elif isinstance(raw, dict):
        if "sponsors" in raw:
            sponsors_raw = raw["sponsors"]
        else:
            # Maybe it returned a single sponsor object — wrap it
            sponsors_raw = [raw]
    else:
        raise MacroSparkError(
            f"Macro Spark returned unexpected type: {type(raw).__name__}",
            retryable=False,
        )

    if not isinstance(sponsors_raw, list):
        raise MacroSparkError(
            "Macro Spark returned malformed response: 'sponsors' must be a list.",
            retryable=False,
        )

    # -- Step 3: validate each item against the Pydantic schema ---------------
    sponsors: list[ExtractedSponsor] = []
    errors: list[str] = []

    for i, item in enumerate(sponsors_raw):
        try:
            normalised = _normalise_item(item)
            if not normalised.get("company_name"):
                errors.append(f"  item[{i}]: missing company_name")
                continue
            sponsors.append(ExtractedSponsor.model_validate(normalised))
        except ValidationError as exc:
            errors.append(f"  item[{i}]: {exc.error_count()} error(s) — {exc.errors()[0]['msg']}")

    if errors and not sponsors:
        # Every item was invalid — try the regex salvage as a last resort
        if original_text:
            salvaged = _recover_truncated_sponsors(original_text)
            if salvaged:
                logger.warning(
                    "Schema validation failed entirely; regex salvage recovered "
                    "%d sponsor(s).", len(salvaged)
                )
                return salvaged

        detail = "\n".join(errors)
        raise MacroSparkError(
            f"Macro Spark returned malformed sponsor list — no valid items:\n{detail}",
            retryable=False,
        )

    if errors:
        logger.warning(
            "%d item(s) failed schema validation and were skipped:\n%s",
            len(errors), "\n".join(errors),
        )

    return sponsors


# ---------------------------------------------------------------------------
# Pure post-processing functions (independently testable)
# ---------------------------------------------------------------------------

def filter_sponsors(
    sponsors: list[ExtractedSponsor],
    min_confidence: float = 0.5,
) -> list[ExtractedSponsor]:
    """
    Remove sponsors whose confidence is below ``min_confidence``.

    Parameters
    ----------
    sponsors       : Raw list from ``parse_macro_response``.
    min_confidence : Threshold (inclusive).  Default 0.5.

    Returns
    -------
    Filtered list; may be empty if all items are below threshold.
    """
    return [s for s in sponsors if s.confidence >= min_confidence]


def dedupe_sponsors(sponsors: list[ExtractedSponsor]) -> list[ExtractedSponsor]:
    """
    Deduplicate sponsors by normalised company name.

    When the same company appears on multiple pages, keep the entry with the
    highest confidence score.  Case-insensitive, strips trailing punctuation.

    Parameters
    ----------
    sponsors : Raw list, potentially containing duplicates.

    Returns
    -------
    Deduplicated list, one entry per unique company name.
    """
    best: dict[str, ExtractedSponsor] = {}
    for sponsor in sponsors:
        key = _normalise_name(sponsor.company_name)
        current = best.get(key)
        if current is None or sponsor.confidence > current.confidence:
            best[key] = sponsor
    return list(best.values())


def _normalise_name(name: str) -> str:
    """Lowercase + collapse whitespace + strip trailing punctuation."""
    cleaned = re.sub(r"\s+", " ", name).strip().rstrip(".,;:")
    return cleaned.lower()


def enrich_sponsors_with_pdf_contacts(
    sponsors: list[ExtractedSponsor],
    pdf_text: str,
) -> list[ExtractedSponsor]:
    """
    Attach brochure-derived contacts to each sponsor when the model missed them.

    Detection strategy is conservative: it only uses contact details that appear
    in a short text window around the sponsor name to avoid cross-company leaks.
    """
    enriched: list[ExtractedSponsor] = []
    for sponsor in sponsors:
        nearby_emails, nearby_phones = _extract_contacts_near_company(
            pdf_text, sponsor.company_name
        )
        merged_emails = _normalise_email_list(
            sponsor.brochure_emails + nearby_emails
        )
        merged_phones = _normalise_phone_list(
            sponsor.brochure_phones + nearby_phones
        )
        enriched.append(
            sponsor.model_copy(
                update={
                    "brochure_emails": merged_emails,
                    "brochure_phones": merged_phones,
                }
            )
        )
    return enriched


# ---------------------------------------------------------------------------
# MacroSparkClient — the HTTP boundary
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "local-122b"          # whatever alias the DGX vLLM server uses
_DEFAULT_TIMEOUT = 120.0               # LLM inference can be slow on a 122B model


class MacroSparkClient:
    """
    HTTP client that talks to the Macro Spark (DGX) OpenAI-compatible endpoint.

    Usage
    -----
    ::

        client = MacroSparkClient(base_url="http://192.168.1.50:8000")
        sponsors = client.extract_sponsors(
            pdf_text=raw_text,
            conference_name="HIMSS 2025",
            min_confidence=0.5,
        )

    The ``base_url`` is the LAN IP of the DGX box.  The path ``/v1/chat/completions``
    is appended automatically.

    Parameters
    ----------
    base_url        : Base URL of the DGX inference server, e.g.
                      ``"http://192.168.1.50:8000"``.
    model           : Model alias registered with the DGX server.
    timeout_seconds : Request timeout.  120s default accommodates slow
                      first-token latency on a 122B model.
    """

    def __init__(
        self,
        base_url: str,
        model: str = _DEFAULT_MODEL,
        timeout_seconds: float = _DEFAULT_TIMEOUT,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_sponsors(
        self,
        pdf_text: str,
        conference_name: str | None = None,
        min_confidence: float = 0.5,
    ) -> list[ExtractedSponsor]:
        """
        Full pipeline: send PDF text to DGX → parse → filter → deduplicate.

        Parameters
        ----------
        pdf_text        : Raw text from the conference brochure PDF.
        conference_name : Optional hint passed to the prompt.
        min_confidence  : Sponsors below this threshold are dropped.

        Returns
        -------
        A deduplicated, confidence-filtered list of ``ExtractedSponsor`` objects.

        Raises
        ------
        MacroSparkError(retryable=True)
            Network timeout or 5xx from the DGX server.
        MacroSparkError(retryable=False)
            Malformed JSON or schema validation failure.
        """
        raw_payload = self._call_dgx(pdf_text, conference_name)
        sponsors = parse_macro_response(raw_payload)
        sponsors = filter_sponsors(sponsors, min_confidence)
        sponsors = dedupe_sponsors(sponsors)
        sponsors = enrich_sponsors_with_pdf_contacts(sponsors, pdf_text)
        return sponsors

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_dgx(
        self, pdf_text: str, conference_name: str | None
    ) -> dict[str, Any]:
        """
        POST to the DGX ``/v1/chat/completions`` endpoint.

        Returns the raw decoded JSON dict.
        Wraps all httpx errors in MacroSparkError(retryable=True).
        """
        messages = build_extraction_prompt(pdf_text, conference_name)
        request_body: dict[str, Any] = {
            "model":       self.model,
            "messages":    messages,
            "temperature": 0.1,
            "max_tokens":  800,
            "stop":        ["\n\n", "```"],
        }

        # Ollama-specific options (not supported by Groq/OpenAI)
        if not self.api_key:
            request_body["options"] = {
                "repeat_penalty": 1.3,
                "num_predict":    800,
            }

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = httpx.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=request_body,
                timeout=self.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise MacroSparkError(
                f"Macro Spark timed out after {self.timeout_seconds}s: {exc}",
                retryable=True,
            ) from exc
        except httpx.HTTPError as exc:
            raise MacroSparkError(
                f"Macro Spark connection error: {exc}",
                retryable=True,
            ) from exc

        if response.status_code in {429, 500, 502, 503, 504}:
            raise MacroSparkError(
                f"Macro Spark returned retryable HTTP {response.status_code}.",
                retryable=True,
            )
        if response.status_code >= 400:
            raise MacroSparkError(
                f"Macro Spark rejected the request (HTTP {response.status_code}): "
                f"{response.text[:300]}",
                retryable=False,
            )

        # Parse the OpenAI-compatible envelope and extract the text content.
        try:
            envelope = response.json()
            content: str = envelope["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise MacroSparkError(
                f"Macro Spark returned unexpected envelope shape: {exc}",
                retryable=False,
            ) from exc

        # content is a JSON string (or markdown-fenced JSON string).
        # Return it as a dict by delegating to parse_macro_response below.
        # We return the raw string here; parse_macro_response handles decoding.
        return content  # type: ignore[return-value]
        # Note: extract_sponsors calls parse_macro_response(raw_payload) next,
        # and parse_macro_response accepts str | dict, so this is intentional.
