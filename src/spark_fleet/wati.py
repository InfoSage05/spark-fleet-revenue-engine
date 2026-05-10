"""
spark_fleet/wati.py

Runs on the Micro Spark (Mac Mini M2).

Responsibilities
----------------
1. Validate the incoming Zoho webhook payload (Pydantic model).
2. Build the exact JSON structure that WATI's sendTemplateMessage API expects.
3. Render the personalised WhatsApp message text.
4. POST to WATI with retry / back-off for 429 responses.

How this fits the pipeline
--------------------------
Zoho CRM (Workflow Rule)
  ↓  fires webhook when WATI_Status = "Pending"
Micro Spark  /webhook/wati-dispatch   (webhook_server.py)
  ↓  calls build_wati_payload()
  ↓  calls WatiDispatcher.send()
WATI API  → WhatsApp message delivered

WATI API credentials
--------------------
Set environment variables on the Mac Mini before running uvicorn:
  WATI_BASE_URL   – e.g. "https://live-server-XXXXX.wati.io"
  WATI_API_TOKEN  – from WATI dashboard → API → Access Token

WATI template setup (manual, one-time)
---------------------------------------
Create a template named  medical_conference_sponsor_intro_v1  in the WATI
dashboard with the following variable slots:
  {{1}} first_name
  {{2}} company
  {{3}} sponsor_tier
  {{4}} conference_name
The template body must match render_whatsapp_template() exactly.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TEMPLATE = "medical_conference_sponsor_intro_v1"
_DEFAULT_BROADCAST = "spark_conference_sponsor_outreach"
_MAX_RETRIES = 3
_BASE_DELAY_S = 2.0
_MAX_DELAY_S = 120.0
_HTTP_TIMEOUT_S = 15.0


# ===========================================================================
# Typed exception
# ===========================================================================

class WatiApiError(RuntimeError):
    """
    Raised for any failure in the WATI dispatch stage.

    Attributes
    ----------
    retryable   : True for 429/5xx (back off and retry).
                  False for 4xx client errors (fix the payload first).
    status_code : HTTP status code from WATI, if available.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


# ===========================================================================
# Zoho webhook payload schema — validates the incoming Zoho webhook body
# ===========================================================================

class ZohoWebhookPayload(BaseModel):
    """
    The JSON body Zoho sends when its Workflow Rule fires.

    Required fields
    ---------------
    lead_id, company, conference_name, sponsor_tier, phone

    Optional fields
    ---------------
    first_name, last_name, wati_template_key
    """

    model_config = ConfigDict(extra="ignore")   # Zoho may send extra fields

    lead_id:          str
    company:          str
    conference_name:  str
    sponsor_tier:     str
    phone:            str           # required: webhook only fires when phone exists

    first_name:       str | None = None
    last_name:        str | None = None
    wati_template_key: str = Field(default=_DEFAULT_TEMPLATE)

    @field_validator("company", "conference_name", "sponsor_tier", "lead_id", "phone")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("field must not be blank")
        return v.strip()


# ===========================================================================
# Pure functions — no I/O, independently testable
# ===========================================================================

def render_whatsapp_template(
    first_name: str | None,
    company: str,
    sponsor_tier: str,
    conference_name: str,
) -> str:
    """
    Render the exact WhatsApp message text for the WATI template.

    The output is stored on the Zoho lead record as WATI_Personalized_Msg
    AND passed as the rendered preview when calling the WATI API.

    Template text (as specified in Prompt 5)
    -----------------------------------------
    Hi {first_name}, I noticed {company} was a {sponsor_tier} sponsor at
    {conference_name}. We help medical teams use AI to identify and engage
    high-intent healthcare partners automatically. Would it be useful to
    compare notes on how {company} is approaching AI-led growth in
    healthcare this quarter?
    """
    greeting = f"Hi {first_name}," if first_name and first_name.strip() else "Hi,"
    return (
        f"{greeting} I noticed {company} was a {sponsor_tier} sponsor at "
        f"{conference_name}. "
        "We help medical teams use AI to identify and engage high-intent "
        "healthcare partners automatically. "
        f"Would it be useful to compare notes on how {company} is approaching "
        "AI-led growth in healthcare this quarter?"
    )


def build_wati_payload(webhook_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Validate a raw Zoho webhook dict and build the WATI API payload.

    Parameters
    ----------
    webhook_dict : Raw dict from the Zoho Workflow webhook body.

    Returns
    -------
    A dict ready to POST to ``/api/v1/sendTemplateMessage``.

    Raises
    ------
    pydantic.ValidationError
        If required fields (phone, company …) are missing or blank.
    """
    # Validate and parse — raises ValidationError on missing/blank fields.
    validated = ZohoWebhookPayload.model_validate(webhook_dict)

    return {
        "template_name":  validated.wati_template_key,
        "broadcast_name": _DEFAULT_BROADCAST,
        "phone_number":   validated.phone,
        "parameters": [
            {"name": "first_name",      "value": validated.first_name or ""},
            {"name": "company",         "value": validated.company},
            {"name": "sponsor_tier",    "value": validated.sponsor_tier},
            {"name": "conference_name", "value": validated.conference_name},
        ],
    }


# ===========================================================================
# WatiDispatcher — the HTTP boundary
# ===========================================================================

class WatiDispatcher:
    """
    Sends WhatsApp template messages via the WATI API.

    Parameters
    ----------
    base_url      : WATI API base, e.g. "https://live-server-XXXXX.wati.io".
    api_token     : Bearer token from the WATI dashboard.
    max_retries   : Retries on 429 / 5xx (default 3).
    retry_delay_s : Base delay for exponential back-off (seconds).
    timeout_s     : Per-request HTTP timeout.

    Usage
    -----
    ::

        dispatcher = WatiDispatcher(
            base_url=os.environ["WATI_BASE_URL"],
            api_token=os.environ["WATI_API_TOKEN"],
        )
        result = dispatcher.send(build_wati_payload(webhook_dict))
    """

    def __init__(
        self,
        base_url: str,
        api_token: str,
        max_retries: int = _MAX_RETRIES,
        retry_delay_s: float = _BASE_DELAY_S,
        timeout_s: float = _HTTP_TIMEOUT_S,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.max_retries = max_retries
        self.retry_delay_s = retry_delay_s
        self.timeout_s = timeout_s

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, wati_payload: dict[str, Any]) -> dict[str, Any]:
        """
        POST ``wati_payload`` to ``/api/v1/sendTemplateMessage``.

        Retry policy
        ------------
        429 / 5xx → exponential back-off, up to ``max_retries`` retries.
        4xx (not 429) → raise WatiApiError(retryable=False) immediately.

        Parameters
        ----------
        wati_payload : Dict from ``build_wati_payload()``.

        Returns
        -------
        Decoded WATI API response body on success.

        Raises
        ------
        WatiApiError(retryable=True)   after exhausting retries on 429/5xx.
        WatiApiError(retryable=False)  on permanent 4xx errors.
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return self._execute_post(wati_payload)

            except WatiApiError as exc:
                if not exc.retryable:
                    raise   # 4xx — no point retrying
                last_error = exc

            except httpx.HTTPError as exc:
                last_error = exc

            if attempt < self.max_retries:
                delay = min(_MAX_DELAY_S, self.retry_delay_s * (2 ** attempt))
                logger.warning(
                    "WATI send attempt %d/%d failed (%s). Retrying in %.1fs.",
                    attempt + 1,
                    self.max_retries + 1,
                    last_error,
                    delay,
                )
                time.sleep(delay)

        raise WatiApiError(
            f"WATI send failed after {self.max_retries + 1} attempt(s): {last_error}",
            retryable=True,
        ) from last_error

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _execute_post(self, wati_payload: dict[str, Any]) -> dict[str, Any]:
        """Single POST attempt. Raises WatiApiError on HTTP errors."""
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type":  "application/json",
        }
        response = httpx.post(
            f"{self.base_url}/api/v1/sendTemplateMessage",
            headers=headers,
            json=wati_payload,
            timeout=self.timeout_s,
        )

        if response.status_code == 429:
            raise WatiApiError(
                "WATI rate-limited (429). Will retry.",
                retryable=True,
                status_code=429,
            )

        if response.status_code in {500, 502, 503, 504}:
            raise WatiApiError(
                f"WATI server error ({response.status_code}): {response.text[:200]}",
                retryable=True,
                status_code=response.status_code,
            )

        if response.status_code >= 400:
            raise WatiApiError(
                f"WATI rejected payload (HTTP {response.status_code}): "
                f"{response.text[:300]}",
                retryable=False,
                status_code=response.status_code,
            )

        logger.info("WATI message dispatched (HTTP %d).", response.status_code)
        return response.json()
