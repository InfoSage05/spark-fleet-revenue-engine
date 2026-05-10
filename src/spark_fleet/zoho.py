"""
spark_fleet/zoho.py

Runs on the Micro Spark (Mac Mini M2).

Responsibilities
----------------
1. Map an ``EnrichedLead`` into a ``ZohoPayload`` (the schema validated in
   ``schemas.py``).
2. Push that payload to the Zoho CRM Leads API via httpx.
3. Handle OAuth token injection, 429 back-off, and retryable vs.
   non-retryable API errors without ever blocking Zoho Catalyst.

Architecture note: Why this module exists
-----------------------------------------
Zoho Catalyst functions time out in ~10–30 seconds. LinkedIn enrichment
takes 30–120 seconds per company.  Instead of letting Catalyst do any work,
the Micro Spark does ALL heavy lifting and then calls Zoho CRM's REST API
directly from this module.  Zoho sees only a fast, pre-built POST request.

WATI dispatch trigger
---------------------
After a lead is created, a Zoho Workflow Rule watches the ``WATI_Status``
field.  When it equals "Pending", Zoho fires a webhook to the Micro Spark's
``/webhook/wati-dispatch`` endpoint.  The Micro Spark then calls WATI.
That is the mechanism that completely bypasses Zoho Catalyst's timeout.

OAuth tokens
------------
At MVP stage a ``StaticTokenProvider`` (reads from environment variable
``ZOHO_ACCESS_TOKEN``) is used.  Replace it with a ``RefreshingTokenProvider``
once the Zoho OAuth refresh-token flow is wired in.

Retry policy
------------
We implement a simple manual retry loop rather than pulling in Tenacity,
which is not yet in our dependency list.  The loop:
  - Retries up to ``max_retries`` times on HTTP 429 or 5xx.
  - Waits ``retry_delay_seconds * 2^attempt`` between retries (capped at
    300 s) — exponential back-off.
  - Raises ``ZohoApiError(retryable=False)`` immediately on 4xx (except
    429).

Custom Zoho CRM fields required
--------------------------------
These fields must exist in the Zoho CRM Leads module before the first push:
  - Sponsor_Tier          (Single Line)
  - Conference_Name       (Single Line)
  - WATI_Status           (Picklist: Pending | Not Sent - Missing Phone |
                                      Sent | Failed)
  - WATI_Template_Key     (Single Line)
  - WATI_Personalized_Msg (Multi Line)

See docs/zoho_setup.md (to be created) for exact field API names.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any

import httpx

from spark_fleet.schemas import EnrichedLead, ZohoPayload


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Zoho API constants
# ---------------------------------------------------------------------------

def _zoho_api_base() -> str:
    """Return the correct Zoho API base URL for the configured regional domain."""
    domain = os.environ.get("ZOHO_DOMAIN", "zoho.com")
    # Zoho API domains follow the same pattern as account domains
    # e.g. zoho.com -> zohoapis.com, zoho.in -> zohoapis.in, zoho.eu -> zohoapis.eu
    api_domain = domain.replace("zoho.", "zohoapis.")
    return f"https://www.{api_domain}/crm/v6"


_LEADS_ENDPOINT   = f"{_zoho_api_base()}/Leads"
_LEADS_UPSERT_ENDPOINT = f"{_LEADS_ENDPOINT}/upsert"
_LEADS_SEARCH_ENDPOINT = f"{_LEADS_ENDPOINT}/search"
_LEAD_SOURCE      = "Conference Sponsor Extraction"
_LEAD_STATUS      = "New - Spark Enriched"
_WATI_TEMPLATE    = "medical_conference_sponsor_intro_v1"

_MAX_RETRIES      = 3
_BASE_DELAY_S     = 2.0     # seconds; doubles each attempt, capped at 300 s
_MAX_DELAY_S      = 300.0
_HTTP_TIMEOUT_S   = 20.0


# ===========================================================================
# Typed exception
# ===========================================================================

class ZohoApiError(RuntimeError):
    """
    Raised for any failure in the Zoho CRM push stage.

    Attributes
    ----------
    retryable : bool
        True  → transient error (429, 5xx, network).  Worker should retry.
        False → permanent error (400, 401, schema mismatch).  Needs review.
    status_code : int | None
        HTTP status code from Zoho, if available.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable   = retryable
        self.status_code = status_code


# ===========================================================================
# OAuth token providers
# ===========================================================================

class TokenProvider:
    """Base class; subclasses implement ``access_token()``."""

    def access_token(self) -> str:  # pragma: no cover
        raise NotImplementedError


class StaticTokenProvider(TokenProvider):
    """
    Returns a fixed token — useful for local dev and tests.

    In production wire in a ``RefreshingTokenProvider`` that exchanges the
    Zoho refresh-token for a new access-token automatically.

    Environment variable
    --------------------
    ``ZOHO_ACCESS_TOKEN`` — checked first.  Falls back to the ``token``
    constructor argument (useful in tests).
    """

    def __init__(self, token: str = "") -> None:
        self._token = token

    def access_token(self) -> str:
        return os.environ.get("ZOHO_ACCESS_TOKEN", self._token) or "DUMMY_TOKEN_REPLACE_ME"


class RefreshingTokenProvider(TokenProvider):
    """
    Automatically refreshes the Zoho OAuth access token when it expires.

    Zoho access tokens are valid for 3600 seconds (1 hour).  This provider
    caches the token and proactively refreshes it 60 seconds before expiry
    so active push operations are never interrupted mid-batch.

    Usage (production)
    ------------------
    ::

        provider = RefreshingTokenProvider.from_env()
        client   = ZohoCRMClient(token_provider=provider)

    Environment variables
    ---------------------
    ZOHO_CLIENT_ID      Zoho self-client / OAuth app client ID.
    ZOHO_CLIENT_SECRET  Zoho OAuth app client secret.
    ZOHO_REFRESH_TOKEN  Long-lived refresh token from Zoho OAuth flow.

    The token URL defaults to the global Zoho accounts domain.  Adjust for
    regional deployments (e.g. ``accounts.zoho.in``, ``accounts.zoho.eu``).
    """

    _REFRESH_BUFFER_S = 60      # refresh when < 60s remain on the token
    _TOKEN_URL        = "https://accounts.zoho.com/oauth/v2/token"

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        token_url: str = _TOKEN_URL,
    ) -> None:
        self._client_id     = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._token_url     = token_url
        self._access_token: str | None   = None
        self._expires_at:   datetime | None = None

    # ------------------------------------------------------------------
    # TokenProvider interface
    # ------------------------------------------------------------------

    def access_token(self) -> str:
        """Return a valid access token, refreshing if necessary."""
        if self._is_token_valid():
            return self._access_token  # type: ignore[return-value]
        return self._refresh()

    # ------------------------------------------------------------------
    # Class method factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "RefreshingTokenProvider":
        """
        Build a provider from environment variables.

        Raises
        ------
        EnvironmentError
            If any of the three required variables are missing or empty.
        """
        client_id     = os.environ.get("ZOHO_CLIENT_ID", "")
        client_secret = os.environ.get("ZOHO_CLIENT_SECRET", "")
        refresh_token = os.environ.get("ZOHO_REFRESH_TOKEN", "")
        domain        = os.environ.get("ZOHO_DOMAIN", "zoho.com")
        token_url     = f"https://accounts.{domain}/oauth/v2/token"

        missing = [
            name for name, val in {
                "ZOHO_CLIENT_ID":     client_id,
                "ZOHO_CLIENT_SECRET": client_secret,
                "ZOHO_REFRESH_TOKEN": refresh_token,
            }.items()
            if not val
        ]
        if missing:
            raise EnvironmentError(
                f"Missing required Zoho OAuth environment variables: {missing}. "
                "Set ZOHO_CLIENT_ID, ZOHO_CLIENT_SECRET, and ZOHO_REFRESH_TOKEN "
                "on the Micro Spark before starting the webhook server."
            )

        return cls(
            client_id=client_id,
            client_secret=client_secret,
            refresh_token=refresh_token,
            token_url=token_url,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_token_valid(self) -> bool:
        """Return True if the cached token exists and won't expire within 60s."""
        if not self._access_token or not self._expires_at:
            return False
        return datetime.now() < (self._expires_at - timedelta(seconds=self._REFRESH_BUFFER_S))

    def _refresh(self) -> str:
        """
        Exchange the refresh token for a new access token via Zoho OAuth.

        Raises
        ------
        ZohoApiError(retryable=False)
            If Zoho OAuth rejects the credentials.
        """
        logger.info("Refreshing Zoho OAuth access token.")
        try:
            response = httpx.post(
                self._token_url,
                data={
                    "grant_type":    "refresh_token",
                    "client_id":     self._client_id,
                    "client_secret": self._client_secret,
                    "refresh_token": self._refresh_token,
                },
                timeout=10.0,
            )
        except httpx.HTTPError as exc:
            raise ZohoApiError(
                f"Zoho OAuth network error during token refresh: {exc}",
                retryable=True,
            ) from exc

        if response.status_code >= 400:
            raise ZohoApiError(
                f"Zoho OAuth refresh failed (HTTP {response.status_code}): "
                f"{response.text[:300]}",
                retryable=False,
            )

        body = response.json()
        
        if "error" in body:
            raise ZohoApiError(
                f"Zoho OAuth refresh failed: {body.get('error')} - {body}",
                retryable=False,
            )
            
        self._access_token = body["access_token"]
        expires_in         = int(body.get("expires_in", 3600))
        self._expires_at   = datetime.now() + timedelta(seconds=expires_in)

        logger.info("Zoho access token refreshed. Expires in %ds.", expires_in)
        return self._access_token

# ===========================================================================
# Pure mapping function — independently testable with no I/O
# ===========================================================================

def _split_name(full_name: str | None) -> tuple[str | None, str]:
    """
    Split a full name into (first_name, last_name).

    Returns
    -------
    (None, "Unknown")  when full_name is falsy.
    (None, surname)    when only one word is given.
    (first, last)      when two or more words are given.
    """
    if not full_name or not full_name.strip():
        return None, "Unknown"
    parts = full_name.strip().split()
    if len(parts) == 1:
        return None, parts[0]
    return " ".join(parts[:-1]), parts[-1]


def _build_personalized_message(lead: EnrichedLead) -> str:
    """Build the short WhatsApp intro text stored on the CRM record."""
    first_name, _ = _split_name(lead.director_name)
    greeting       = f"Hi {first_name}," if first_name else "Hi,"
    return (
        f"{greeting} I noticed {lead.company_name} was a "
        f"{lead.sponsor_tier} sponsor at {lead.conference_name}. "
        "We help medical teams use AI to identify and engage high-intent "
        "healthcare partners automatically — would love to share how."
    )


def map_lead_to_zoho_payload(lead: EnrichedLead) -> ZohoPayload:
    """
    Convert an ``EnrichedLead`` into a validated ``ZohoPayload``.

    This is a pure function — no I/O, no side effects.  It is the single
    source of truth for how ``EnrichedLead`` fields map to Zoho CRM fields.

    WATI_Status logic
    -----------------
    ``phone`` present  → "Pending"         (Zoho workflow will trigger WATI)
    ``phone`` absent   → "Not Sent - Missing Phone"  (human follow-up needed)

    Parameters
    ----------
    lead : EnrichedLead
        The enriched lead produced by ``EnrichmentOrchestrator.enrich``.

    Returns
    -------
    ZohoPayload
        Validated payload ready to pass to ``ZohoCRMClient.push``.
    """
    first_name, last_name = _split_name(lead.director_name)

    wati_status = "Pending" if lead.phone else "Not Sent - Missing Phone"

    # Build record, omitting None-valued optional fields so Zoho doesn't
    # interpret them as deliberate blanks.
    record: dict[str, Any] = {
        "Last_Name":             last_name,
        "Company":               lead.company_name,
        "Lead_Source":           _LEAD_SOURCE,
        "Lead_Status":           _LEAD_STATUS,
        "Conference_Name":       lead.conference_name,
        "Sponsor_Tier":          lead.sponsor_tier,
        "WATI_Status":           wati_status,
        "WATI_Template_Key":     _WATI_TEMPLATE,
        "WATI_Personalized_Msg": _build_personalized_message(lead),
    }

    # Optional fields — only included when present
    if first_name:
        record["First_Name"] = first_name

    if lead.director_title:
        record["Designation"] = lead.director_title

    if lead.email:
        record["Email"] = lead.email

    if lead.phone:
        record["Mobile"] = lead.phone

    if lead.linkedin_url:
        # HttpUrl must be serialised to str — Zoho API expects a plain string.
        record["LinkedIn_Profile"] = str(lead.linkedin_url)

    if lead.source_page:
        record["Description"] = f"Extracted from brochure page {lead.source_page}."

    return ZohoPayload(data=[record])


# ===========================================================================
# ZohoCRMClient — the HTTP boundary
# ===========================================================================

class ZohoCRMClient:
    """
    HTTP client that pushes ``ZohoPayload`` objects to Zoho CRM.

    Parameters
    ----------
    token_provider  : Supplies the OAuth access token for each request.
    base_url        : Zoho API base (overridable for regional endpoints or
                      tests that point at a mock server).
    max_retries     : How many times to retry on 429 / 5xx before giving up.
    retry_delay_s   : Base delay for exponential back-off (seconds).
    timeout_s       : Per-request HTTP timeout.

    Usage
    -----
    ::

        client = ZohoCRMClient(token_provider=StaticTokenProvider())
        result = client.push(map_lead_to_zoho_payload(enriched_lead))
    """

    def __init__(
        self,
        token_provider: TokenProvider | None = None,
        base_url: str = _LEADS_ENDPOINT,
        max_retries: int = _MAX_RETRIES,
        retry_delay_s: float = _BASE_DELAY_S,
        timeout_s: float = _HTTP_TIMEOUT_S,
    ) -> None:
        self.token_provider = token_provider or StaticTokenProvider()
        self.base_url       = base_url
        self.max_retries    = max_retries
        self.retry_delay_s  = retry_delay_s
        self.timeout_s      = timeout_s

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, payload: ZohoPayload) -> dict[str, Any]:
        """
        POST the payload to Zoho CRM Leads.

        Retries automatically on 429 / 5xx using exponential back-off.

        Parameters
        ----------
        payload : ZohoPayload
            Pre-validated payload from ``map_lead_to_zoho_payload``.

        Returns
        -------
        dict
            Decoded Zoho API response body on success.

        Raises
        ------
        ZohoApiError(retryable=True)
            After exhausting all retries on 429 / 5xx or a network error.
        ZohoApiError(retryable=False)
            On a non-retryable 4xx response (bad request, unauthorised …).
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return self._execute_post(payload)

            except ZohoApiError as exc:
                if not exc.retryable:
                    raise  # Permanent error — no point retrying.
                last_error = exc

            except httpx.HTTPError as exc:
                last_error = exc

            # Compute back-off delay before next attempt.
            if attempt < self.max_retries:
                delay = min(_MAX_DELAY_S, self.retry_delay_s * (2 ** attempt))
                logger.warning(
                    "Zoho push attempt %d/%d failed (%s). Retrying in %.1fs.",
                    attempt + 1,
                    self.max_retries + 1,
                    last_error,
                    delay,
                )
                time.sleep(delay)

        raise ZohoApiError(
            f"Zoho push failed after {self.max_retries + 1} attempts: {last_error}",
            retryable=True,
        ) from last_error

    def update_lead(self, lead_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        """
        Update an existing Zoho lead record by its CRM ID.
        """
        body = {"data": [{"id": lead_id, **fields}]}
        headers = {
            "Authorization": f"Zoho-oauthtoken {self.token_provider.access_token()}",
            "Content-Type":  "application/json",
        }
        response = httpx.put(
            self.base_url,
            headers=headers,
            json=body,
            timeout=self.timeout_s,
        )

        if response.status_code >= 400:
            raise ZohoApiError(
                f"Zoho CRM lead update failed (HTTP {response.status_code}): "
                f"{response.text[:300]}",
                retryable=response.status_code in {429, 500, 502, 503, 504},
                status_code=response.status_code,
            )

        logger.info("Zoho CRM lead %s updated successfully.", lead_id)
        return response.json()

    def find_lead(
        self,
        *,
        company_name: str,
        conference_name: str,
        sponsor_tier: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Search Zoho for the existing lead created by Spark Fleet.
        """
        criteria = [
            f"(Company:equals:{_escape_criteria(company_name)})",
            f"(Conference_Name:equals:{_escape_criteria(conference_name)})",
            f"(Lead_Source:equals:{_escape_criteria(_LEAD_SOURCE)})",
        ]
        if sponsor_tier:
            criteria.append(f"(Sponsor_Tier:equals:{_escape_criteria(sponsor_tier)})")

        query = "and".join(criteria)
        headers = {
            "Authorization": f"Zoho-oauthtoken {self.token_provider.access_token()}",
        }
        response = httpx.get(
            _LEADS_SEARCH_ENDPOINT,
            headers=headers,
            params={"criteria": query},
            timeout=self.timeout_s,
        )

        if response.status_code == 204:
            return None

        if response.status_code >= 400:
            raise ZohoApiError(
                f"Zoho CRM lead search failed (HTTP {response.status_code}): "
                f"{response.text[:300]}",
                retryable=response.status_code in {429, 500, 502, 503, 504},
                status_code=response.status_code,
            )

        data = response.json()
        records = data.get("data", []) if isinstance(data, dict) else []
        if not isinstance(records, list) or not records:
            return None

        preferred = [
            record for record in records
            if record.get("WATI_Status") == "Not Sent - Missing Phone"
        ]
        if preferred:
            return preferred[0]
        return records[0]

    def has_conference_leads(self, conference_name: str) -> bool:
        """
        Return True when Zoho already has at least one Spark Fleet lead for
        this conference/PDF. Used to avoid spending LLM tokens twice.
        """
        headers = {
            "Authorization": f"Zoho-oauthtoken {self.token_provider.access_token()}",
        }
        criteria = "and".join(
            [
                f"(Conference_Name:equals:{_escape_criteria(conference_name)})",
                f"(Lead_Source:equals:{_escape_criteria(_LEAD_SOURCE)})",
            ]
        )
        response = httpx.get(
            _LEADS_SEARCH_ENDPOINT,
            headers=headers,
            params={"criteria": criteria},
            timeout=self.timeout_s,
        )

        if response.status_code == 204:
            return False
        if response.status_code >= 400:
            raise ZohoApiError(
                f"Zoho CRM conference lookup failed (HTTP {response.status_code}): "
                f"{response.text[:300]}",
                retryable=response.status_code in {429, 500, 502, 503, 504},
                status_code=response.status_code,
            )

        data = response.json()
        records = data.get("data", []) if isinstance(data, dict) else []
        return isinstance(records, list) and bool(records)

    def upsert(self, lead: EnrichedLead) -> dict[str, Any]:
        """
        Convenience wrapper: map lead → payload → push.

        Parameters
        ----------
        lead : EnrichedLead
            The enriched lead to upsert into Zoho.

        Returns
        -------
        dict
            Decoded Zoho API response body.
        """
        payload = map_lead_to_zoho_payload(lead)
        return self.push(payload)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _execute_post(self, payload: ZohoPayload) -> dict[str, Any]:
        """
        Execute a single POST to the Zoho Leads endpoint.

        Raises
        ------
        ZohoApiError(retryable=True)   on 429 / 5xx.
        ZohoApiError(retryable=False)  on other 4xx.
        httpx.HTTPError                on connection/timeout (caller retries).
        """
        headers = {
            "Authorization": f"Zoho-oauthtoken {self.token_provider.access_token()}",
            "Content-Type":  "application/json",
        }

        response = httpx.post(
            _LEADS_UPSERT_ENDPOINT,
            headers=headers,
            json=_serialise_payload(payload),
            timeout=self.timeout_s,
        )

        # -- Retryable errors -----------------------------------------------
        if response.status_code == 429:
            raise ZohoApiError(
                f"Zoho CRM rate-limited (429). Will retry.",
                retryable=True,
                status_code=429,
            )

        if response.status_code in {500, 502, 503, 504}:
            raise ZohoApiError(
                f"Zoho CRM server error ({response.status_code}): "
                f"{response.text[:200]}",
                retryable=True,
                status_code=response.status_code,
            )

        # -- Non-retryable 4xx errors ---------------------------------------
        if response.status_code >= 400:
            raise ZohoApiError(
                f"Zoho CRM rejected the request (HTTP {response.status_code}): "
                f"{response.text[:300]}",
                retryable=False,
                status_code=response.status_code,
            )

        # -- Success ---------------------------------------------------------
        logger.info(
            "Zoho CRM push succeeded (HTTP %d).", response.status_code
        )
        return response.json()


def _serialise_payload(payload: ZohoPayload) -> dict[str, Any]:
    """
    Zoho upsert only accepts system-defined or user-defined unique fields in
    duplicate_check_fields. Default to Zoho's native behavior unless Email is
    present and usable for duplicate matching.
    """
    body = payload.model_dump(mode="json")
    record = body["data"][0] if body.get("data") else {}

    duplicate_fields = []
    if record.get("Email"):
        duplicate_fields.append("Email")

    if duplicate_fields:
        body["duplicate_check_fields"] = duplicate_fields
    else:
        body.pop("duplicate_check_fields", None)

    return body


def _escape_criteria(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
