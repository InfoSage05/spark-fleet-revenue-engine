"""
Data schemas for the Spark Fleet pipeline.

Three canonical models move data through the workflow:

  PDF  →  ExtractedSponsor  →  EnrichedLead  →  ZohoPayload  →  Zoho CRM  →  WATI WhatsApp
           (Macro Spark)         (Micro Spark)     (Micro Spark push)

Rules:
- Extra fields are forbidden on all models so bad upstream payloads fail loudly.
- Every field that is optional is explicitly typed as `X | None` with a default.
- Confidence scores are always in [0.0, 1.0].
"""

from __future__ import annotations

from typing import Annotated, Any
import re

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------

class _Strict(BaseModel):
    """All Spark Fleet models reject unknown fields and validate on assignment."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _require_non_empty(value: str, label: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"'{label}' must not be blank")
    return stripped


# ---------------------------------------------------------------------------
# Stage 1 – Macro Spark output
# ---------------------------------------------------------------------------

class ExtractedSponsor(_Strict):
    """
    One sponsor record returned by the Macro Spark (122 B LLM) after reading
    the conference brochure PDF.

    Fields
    ------
    company_name : str
        The exact company name as it appears in the brochure.
    sponsor_tier : str
        Sponsorship level, e.g. 'Gold', 'Silver', 'Platinum', 'Unknown'.
    source_page  : int  (≥ 1)
        Page number the sponsor was found on — aids human QA.
    evidence_text : str | None
        Verbatim snippet from the brochure that proves the sponsorship.
    important_people : list[str]
        Named executives, speakers, booth contacts, or role holders connected
        to the sponsor in the brochure.
    brochure_emails : list[str]
        Email IDs found in brochure text near the sponsor mention.
    brochure_phones : list[str]
        Phone numbers found in brochure text near the sponsor mention.
    confidence   : float  [0.0 – 1.0]
        Model confidence. Leads below 0.5 are filtered during enrichment.
    """

    company_name:  str
    sponsor_tier:  str = "Unknown"
    source_page:   int = Field(ge=1)
    evidence_text: str | None = None
    important_people: list[str] = Field(default_factory=list)
    brochure_emails: list[str] = Field(default_factory=list)
    brochure_phones: list[str] = Field(default_factory=list)
    confidence:    float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("company_name", "sponsor_tier")
    @classmethod
    def _no_blank(cls, v: str) -> str:
        return _require_non_empty(v, "company_name / sponsor_tier")

    @field_validator("brochure_emails", mode="before")
    @classmethod
    def _normalise_brochure_emails(cls, v: Any) -> list[str]:
        if v is None:
            return []
        items = v if isinstance(v, list) else [v]
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            email = str(item).strip().lower()
            if not email or "@" not in email:
                continue
            if email in seen:
                continue
            seen.add(email)
            out.append(email)
        return out[:5]

    @field_validator("brochure_phones", mode="before")
    @classmethod
    def _normalise_brochure_phones(cls, v: Any) -> list[str]:
        if v is None:
            return []
        items = v if isinstance(v, list) else [v]
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            digits = "".join(ch for ch in str(item) if ch.isdigit())
            if not 8 <= len(digits) <= 15:
                continue
            phone = f"+{digits}"
            if phone in seen:
                continue
            seen.add(phone)
            out.append(phone)
        return out[:5]


# ---------------------------------------------------------------------------
# Stage 2 – Micro Spark enrichment output
# ---------------------------------------------------------------------------

class EnrichedLead(_Strict):
    """
    The result of the Micro Spark's LinkedIn / company-data enrichment step.

    Fields
    ------
    company_name        : str
    director_name       : str | None   Full name of the Marketing Director found.
    director_title      : str | None   Exact LinkedIn title string.
    linkedin_url        : HttpUrl | None
    email               : str | None
    phone               : str | None   E.164-like, e.g. '+919876543210'.
    enrichment_confidence : float [0-1]
        0.0 means no contact was found (CONTACT_MISSING path).
    sponsor_tier        : str
    conference_name     : str
    source_page         : int | None
    """

    company_name:           str
    director_name:          str | None  = None
    director_title:         str | None  = None
    linkedin_url:           HttpUrl | None = None
    email:                  str | None  = None
    phone:                  str | None  = None
    enrichment_confidence:  float       = Field(default=0.0, ge=0.0, le=1.0)
    sponsor_tier:           str         = "Unknown"
    conference_name:        str
    source_page:            int | None  = Field(default=None, ge=1)

    @field_validator("company_name", "conference_name", "sponsor_tier")
    @classmethod
    def _no_blank(cls, v: str) -> str:
        return _require_non_empty(v, "text field")

    @field_validator("email")
    @classmethod
    def _valid_email_shape(cls, v: str | None) -> str | None:
        if v is not None and "@" not in v:
            raise ValueError("email must contain '@'")
        return v

    @field_validator("phone")
    @classmethod
    def _valid_phone_shape(cls, v: str | None) -> str | None:
        if v is not None and not re.fullmatch(r"\+?[0-9]{8,15}", v):
            raise ValueError("phone must be E.164-like, e.g. '+919876543210'")
        return v


# ---------------------------------------------------------------------------
# Stage 3 – Zoho CRM payload
# ---------------------------------------------------------------------------

_REQUIRED_ZOHO_FIELDS = frozenset(
    {"Last_Name", "Company", "Lead_Source", "Lead_Status", "Conference_Name", "Sponsor_Tier"}
)


class ZohoPayload(_Strict):
    """
    The final payload pushed directly to Zoho CRM Leads via its v2 REST API.

    The `data` list contains exactly one record dict.  Keeping it as a raw dict
    (rather than a sub-model) means we can pass it directly to
    `httpx.post(..., json=payload.model_dump())` without transformation.

    WATI dispatch fields
    --------------------
    WATI_Status is set to 'Pending' when a phone is present, otherwise
    'Not Sent - Missing Phone'.  A Zoho Workflow Rule watches this field
    and fires a webhook back to the Micro Spark, which then calls WATI.
    That is how we escape Zoho Catalyst's timeout trap.
    """

    data: list[dict[str, Any]]
    duplicate_check_fields: list[str] = Field(
        default_factory=lambda: ["Company", "Conference_Name", "Lead_Source"]
    )

    @model_validator(mode="after")
    def _validate_records(self) -> "ZohoPayload":
        if not self.data:
            raise ValueError("ZohoPayload.data must contain at least one record")
        for record in self.data:
            missing = _REQUIRED_ZOHO_FIELDS - set(record)
            if missing:
                raise ValueError(f"Zoho record missing required fields: {sorted(missing)}")
        return self
