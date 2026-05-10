"""
spark_fleet/adapters/proxycurl_provider.py

Proxycurl-backed implementation of the ``PeopleSearchProvider`` protocol.

Proxycurl is a licensed LinkedIn data API that returns structured JSON
without requiring browser automation. It is compliant with LinkedIn's
terms of service (data is sourced from their public data licensing
programme).

API reference: https://nubela.co/proxycurl/docs

Endpoints used
--------------
GET /proxycurl/api/linkedin/company/employees/search
    Search for employees of a company by job title keyword.
    Returns a paginated list of employee profiles.

Setup (Mac Mini M2 — one time)
-------------------------------
1. Sign up at https://nubela.co/proxycurl
2. Generate an API key from the dashboard.
3. Set the environment variable on the Mac Mini:
       export PROXYCURL_API_KEY="your_key_here"
4. Instantiate via ProxycurlPeopleProvider.from_env().

Rate limits (free tier)
-----------------------
Proxycurl enforces per-second and per-day credits limits.  The
``EnrichmentOrchestrator.RateLimitState`` back-off (already in
``enrichment.py``) handles 429 responses from this provider automatically.

Title ranking
-------------
Proxycurl may return multiple employees per company.  We score each
candidate's title and return the highest-ranked one.  The scoring uses
the same weights already defined in ``enrichment.py``.

Timeout behaviour
-----------------
All httpx calls use a configurable deadline (default 30s).
``httpx.TimeoutException`` is mapped to ``enrichment.TimeoutError`` so the
``EnrichmentOrchestrator`` can catch it and mark the company CONTACT_MISSING
without crashing the batch.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from spark_fleet.enrichment import PersonResult, RateLimitError
from spark_fleet.enrichment import TimeoutError as EnrichmentTimeoutError
from spark_fleet.schemas import ExtractedSponsor

logger = logging.getLogger(__name__)

_BASE_URL = "https://nubela.co"
_EMPLOYEE_SEARCH_PATH = "/proxycurl/api/linkedin/company/employees/search"

# Title keywords sent to Proxycurl's `keyword_regex` filter.
_TITLE_KEYWORDS = "marketing director|head of marketing|vp marketing|growth director|commercial director"

# Title score weights — must stay in sync with enrichment.TITLE_WEIGHTS
_TITLE_WEIGHTS: dict[str, float] = {
    "marketing director":    1.0,
    "head of marketing":     0.92,
    "vp marketing":          0.88,
    "vice president marketing": 0.88,
    "growth director":       0.78,
    "commercial director":   0.72,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _title_score(title: str) -> float:
    """Numeric priority for a LinkedIn title string."""
    lower = title.lower()
    for needle, score in _TITLE_WEIGHTS.items():
        if needle in lower:
            return score
    if "marketing" in lower:
        return 0.55
    if "growth" in lower or "commercial" in lower:
        return 0.45
    return 0.1


def _map_employee(employee: dict[str, Any]) -> PersonResult | None:
    """
    Map a single Proxycurl employee dict to a ``PersonResult``.

    Returns None if the employee dict is missing critical fields.
    """
    profile = employee.get("profile") or {}
    full_name = profile.get("full_name") or ""
    if not full_name.strip():
        return None

    occupation = profile.get("occupation") or profile.get("headline") or ""
    linkedin_url = employee.get("profile_url")

    score = _title_score(occupation)
    return PersonResult(
        full_name=full_name.strip(),
        title=occupation.strip(),
        linkedin_url=linkedin_url,
        confidence=min(1.0, score),
    )


# ---------------------------------------------------------------------------
# ProxycurlPeopleProvider
# ---------------------------------------------------------------------------

class ProxycurlPeopleProvider:
    """
    Implements the ``PeopleSearchProvider`` protocol using Proxycurl.

    Parameters
    ----------
    api_key   : Proxycurl API key.
    timeout_s : Per-request HTTP timeout in seconds (default 30s).
    page_size : Maximum employees to fetch per request (default 10).

    Usage
    -----
    ::

        provider = ProxycurlPeopleProvider.from_env()
        sponsor  = ExtractedSponsor(company_name="Medtronic", source_page=1)
        result   = provider.find_marketing_director(sponsor)
    """

    def __init__(
        self,
        api_key: str,
        timeout_s: float = 30.0,
        page_size: int = 10,
    ) -> None:
        self.api_key   = api_key
        self.timeout_s = timeout_s
        self.page_size = page_size

    @classmethod
    def from_env(cls) -> "ProxycurlPeopleProvider":
        """
        Build from ``PROXYCURL_API_KEY`` environment variable.

        Raises
        ------
        EnvironmentError if the variable is not set.
        """
        key = os.environ.get("PROXYCURL_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "PROXYCURL_API_KEY environment variable is not set. "
                "Sign up at https://nubela.co/proxycurl and set the key "
                "on the Micro Spark before starting the enrichment worker."
            )
        return cls(api_key=key)

    # ------------------------------------------------------------------
    # PeopleSearchProvider interface
    # ------------------------------------------------------------------

    def find_marketing_director(
        self,
        sponsor: ExtractedSponsor,
    ) -> PersonResult | None:
        """
        Search Proxycurl for Marketing Directors at the sponsor company.

        Returns
        -------
        The highest-ranked ``PersonResult``, or ``None`` if no match.

        Raises
        ------
        RateLimitError     on HTTP 429.
        EnrichmentTimeoutError  on httpx timeout.
        """
        employees = self._fetch_employees(sponsor.company_name)

        if not employees:
            logger.info(
                "Proxycurl: no marketing directors found for '%s'.",
                sponsor.company_name,
            )
            return None

        # Map and rank — keep the best match
        candidates: list[PersonResult] = []
        for emp in employees:
            result = _map_employee(emp)
            if result:
                candidates.append(result)

        if not candidates:
            return None

        best = max(candidates, key=lambda r: r.confidence)
        logger.info(
            "Proxycurl: found '%s' (%s) for '%s' — confidence=%.2f.",
            best.full_name, best.title, sponsor.company_name, best.confidence,
        )
        return best

    # ------------------------------------------------------------------
    # Private HTTP layer
    # ------------------------------------------------------------------

    def _fetch_employees(self, company_name: str) -> list[dict[str, Any]]:
        """
        Call the Proxycurl employee-search endpoint.

        Raises
        ------
        RateLimitError        on 429.
        EnrichmentTimeoutError on httpx.TimeoutException.
        """
        params = {
            "company_name":    company_name,
            "keyword_regex":   _TITLE_KEYWORDS,
            "page_size":       self.page_size,
            "employment_status": "current",
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = httpx.get(
                f"{_BASE_URL}{_EMPLOYEE_SEARCH_PATH}",
                params=params,
                headers=headers,
                timeout=self.timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise EnrichmentTimeoutError(
                f"Proxycurl request timed out for '{company_name}' after {self.timeout_s}s."
            ) from exc
        except httpx.HTTPError as exc:
            raise EnrichmentTimeoutError(
                f"Proxycurl connection error for '{company_name}': {exc}"
            ) from exc

        if response.status_code == 429:
            raise RateLimitError(
                f"Proxycurl rate-limited (429) while searching for '{company_name}'."
            )

        if response.status_code >= 400:
            logger.warning(
                "Proxycurl returned HTTP %d for '%s': %s",
                response.status_code, company_name, response.text[:200],
            )
            return []

        return response.json().get("employees", [])
