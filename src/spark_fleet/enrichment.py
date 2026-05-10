"""
spark_fleet/enrichment.py

Runs entirely on the Micro Spark (Mac Mini M2).

Responsibilities
----------------
1. Accept an ``ExtractedSponsor`` from the Macro Spark extraction stage.
2. Call a swappable ``PeopleSearchProvider`` to find the Marketing Director
   of the sponsor company.
3. Return a typed ``EnrichedLead`` in every case — including failures.
4. Surface rate-limit signals as ``EnrichmentPaused`` so the task-queue
   worker can back off without losing the job.
5. Swallow per-company timeouts as a CONTACT_MISSING result so one slow
   company cannot crash the whole batch.

Architecture note: "The Timeout Trap"
--------------------------------------
Zoho Catalyst has strict execution windows (≈ 10–30s).  LinkedIn scraping
routinely takes 30–90s per company.  That is why this entire module lives on
the Micro Spark.  Zoho NEVER calls this code.  The Micro Spark calls Zoho
only at the very end, once an EnrichedLead is fully ready.

Provider contract
-----------------
Any object that exposes ``find_marketing_director(sponsor) -> PersonResult | None``
or ``find_director(company_name) -> dict | None`` qualifies as a
``PeopleSearchProvider``.  In production you will swap in one of:

  - A Playwright/Selenium LinkedIn scraper adapter.
  - A licensed API adapter (Proxycurl, Apollo.io, Hunter.io …).
  - An internal people-data adapter.

In tests, a ``MagicMock`` is used so no real HTTP calls are made.

Exponential back-off
--------------------
``RateLimitState`` tracks consecutive 429s and computes the next pause
duration with capped exponential back-off:

  pause = min(3600, 30 × 2^failures)  seconds

On the first rate-limit hit the worker pauses 60s; by the fifth it is
pausing 960s (≈ 16 minutes), capped at 1 hour.

Per-company timeout
-------------------
``EnrichmentOrchestrator.enrich`` wraps the provider call in a
``concurrent.futures.ThreadPoolExecutor`` with a 45-second deadline.
If the deadline expires, the company is logged as CONTACT_MISSING and
the batch continues.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FuturesTimeout  # py3.11+ alias for builtins.TimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol, runtime_checkable

from spark_fleet.schemas import EnrichedLead, ExtractedSponsor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-enrichment deadline (seconds).  Configurable at orchestrator level.
# ---------------------------------------------------------------------------
DEFAULT_SCRAPE_TIMEOUT: float = 45.0


# ===========================================================================
# Typed exceptions — the caller decides what to do with each signal
# ===========================================================================

class TimeoutError(RuntimeError):                          # noqa: A001
    """
    Raised (or set as side_effect) by a provider when a single company
    lookup exceeds its internal deadline.

    The orchestrator catches this and returns a CONTACT_MISSING lead
    instead of propagating the exception.
    """


class RateLimitError(RuntimeError):
    """
    Raised by a provider when the upstream data source returns HTTP 429
    or an equivalent throttle signal.

    The orchestrator converts this into ``EnrichmentPaused`` and re-raises
    so the task-queue worker can apply a back-off delay before retrying
    the current job.
    """


class EnrichmentPaused(RuntimeError):
    """
    Raised by ``EnrichmentOrchestrator.enrich`` when a rate-limit is hit.

    The message always contains the phrase "rate-limit" so callers can
    match it with ``pytest.raises(EnrichmentPaused, match="rate.?limit")``.
    """


# ===========================================================================
# Data contract for provider results
# ===========================================================================

@dataclass(frozen=True)
class PersonResult:
    """
    The canonical result object returned by any ``PeopleSearchProvider``.

    A ``MagicMock`` with the same attribute names works in tests.
    """
    full_name:    str
    title:        str
    linkedin_url: str | None   = None
    email:        str | None   = None
    phone:        str | None   = None
    confidence:   float        = 0.0


# ===========================================================================
# Provider Protocol — plug in any scraper / API without changing the core
# ===========================================================================

@runtime_checkable
class PeopleSearchProvider(Protocol):
    """
    Any object that implements this method qualifies as a provider.

    The method must return:
    - A ``PersonResult`` (or MagicMock with the same fields) on success.
    - ``None`` if no matching person is found (CONTACT_MISSING path).

    It must raise:
    - ``TimeoutError``    if the lookup exceeds its internal deadline.
    - ``RateLimitError``  if the upstream source throttles the request.

    All other exceptions propagate unchanged to the orchestrator caller.
    """

    def find_marketing_director(
        self,
        sponsor: ExtractedSponsor,
    ) -> PersonResult | None:
        ...


# ===========================================================================
# Rate-limit state tracker
# ===========================================================================

@dataclass
class RateLimitState:
    """
    Tracks consecutive rate-limit failures and computes back-off delays.

    The state is intentionally mutable so a long-running worker process
    can share one instance across many ``enrich`` calls.

    Back-off formula
    ----------------
    ``pause_seconds = min(3600, 30 × 2^failures)``

    failures=1 →   60s
    failures=2 →  120s
    failures=3 →  240s
    failures=4 →  480s
    failures=5 →  960s
    failures=6 → 1920s → capped at 3600s
    """

    failures:     int              = 0
    paused_until: datetime | None  = field(default=None)

    def is_paused(self, now: datetime | None = None) -> bool:
        """Return True if the back-off window has not yet elapsed."""
        current = now or datetime.now()
        return self.paused_until is not None and current < self.paused_until

    def record_rate_limit(self, now: datetime | None = None) -> None:
        """Increment failure counter and compute the next pause window."""
        current = now or datetime.now()
        self.failures += 1
        pause_seconds = min(3600, 30 * (2 ** self.failures))
        self.paused_until = current + timedelta(seconds=pause_seconds)
        logger.warning(
            "Rate-limit recorded (failure #%d). Pausing enrichment for %ds.",
            self.failures,
            pause_seconds,
        )

    def record_success(self) -> None:
        """Reset the failure counter after a successful lookup."""
        self.failures = 0
        self.paused_until = None


# ===========================================================================
# EnrichmentOrchestrator — the main entry point
# ===========================================================================

class EnrichmentOrchestrator:
    """
    Orchestrates the per-sponsor enrichment loop on the Micro Spark.

    Parameters
    ----------
    people_provider   : Any object implementing ``PeopleSearchProvider``.
    rate_limit_state  : Shared ``RateLimitState`` instance.  If omitted a
                        new state is created (suitable for single-sponsor
                        calls in tests).
    scrape_timeout    : Per-company deadline in seconds (default 45s).
    """

    def __init__(
        self,
        people_provider: Any,
        rate_limit_state: RateLimitState | None = None,
        scrape_timeout: float = DEFAULT_SCRAPE_TIMEOUT,
    ) -> None:
        self.people_provider  = people_provider
        self.rate_limit_state = rate_limit_state or RateLimitState()
        self.scrape_timeout   = scrape_timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich(
        self,
        sponsor: ExtractedSponsor,
        conference_name: str,
    ) -> EnrichedLead:
        """
        Enrich a single ``ExtractedSponsor`` into an ``EnrichedLead``.

        Outcome table
        -------------
        Provider raises ``RateLimitError``  → raise ``EnrichmentPaused``
        Provider raises ``TimeoutError``    → return CONTACT_MISSING lead
        Scrape wall-clock > 45s            → return CONTACT_MISSING lead
        Provider returns ``None``           → return CONTACT_MISSING lead
        Provider returns ``PersonResult``   → return full ``EnrichedLead``

        Parameters
        ----------
        sponsor         : The sponsor to enrich.
        conference_name : Passed through to the ``EnrichedLead`` for context.

        Returns
        -------
        ``EnrichedLead`` — never raises except for ``EnrichmentPaused``.
        """
        # -- Rate-limit gate -----------------------------------------------
        if self.rate_limit_state.is_paused():
            raise EnrichmentPaused(
                f"Enrichment paused due to active rate-limit back-off. "
                f"Resume after {self.rate_limit_state.paused_until}."
            )

        # -- Call the provider with a hard wall-clock timeout --------------
        result = self._call_provider_with_timeout(sponsor, conference_name)

        # -- Map outcome → EnrichedLead ------------------------------------
        if result is None:
            logger.info("No director found for '%s'. Marking CONTACT_MISSING.", sponsor.company_name)
            return self._contact_missing(sponsor, conference_name)

        self.rate_limit_state.record_success()
        return self._build_lead(sponsor, conference_name, result)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_provider_with_timeout(
        self,
        sponsor: ExtractedSponsor,
        conference_name: str,
    ) -> Any | None:
        """
        Call ``people_provider.find_marketing_director`` inside a thread
        with a ``self.scrape_timeout`` deadline.

        Returns
        -------
        The provider result, or ``None`` on timeout.

        Raises
        ------
        ``EnrichmentPaused``   if provider signals a rate-limit.
        """
        with ThreadPoolExecutor(max_workers=1) as pool:
            if hasattr(self.people_provider, "set_run_context"):
                self.people_provider.set_run_context(
                    company_name=sponsor.company_name,
                    conference_name=conference_name,
                    sponsor_tier=sponsor.sponsor_tier,
                    source_page=sponsor.source_page,
                )
            if hasattr(self.people_provider, "find_marketing_director"):
                future = pool.submit(
                    self.people_provider.find_marketing_director, sponsor
                )
            else:
                future = pool.submit(
                    self.people_provider.find_director, sponsor.company_name
                )
            try:
                return future.result(timeout=self.scrape_timeout)

            except (_FuturesTimeout, TimeoutError):
                # Wall-clock deadline (futures) OR provider's own TimeoutError —
                # same outcome: log and return None → CONTACT_MISSING.
                logger.warning(
                    "Enrichment for '%s' timed out. Marking CONTACT_MISSING.",
                    sponsor.company_name,
                )
                return None

            except RateLimitError as exc:
                # Rate-limit: record state and surface to the worker.
                self.rate_limit_state.record_rate_limit()
                raise EnrichmentPaused(
                    f"Enrichment paused: rate-limit hit for '{sponsor.company_name}'. "
                    f"Upstream error: {exc}"
                ) from exc

    def _build_lead(
        self,
        sponsor: ExtractedSponsor,
        conference_name: str,
        result: Any,
    ) -> EnrichedLead:
        """Map a successful ``PersonResult`` (or MagicMock) to ``EnrichedLead``."""
        if isinstance(result, dict):
            email = result.get("email") or _first_brochure_email(sponsor)
            phone = result.get("phone") or _first_brochure_phone(sponsor)
            return EnrichedLead(
                company_name          = sponsor.company_name,
                director_name         = result.get("director_name"),
                director_title        = result.get("director_title"),
                linkedin_url          = result.get("linkedin_url"),
                email                 = email,
                phone                 = phone,
                enrichment_confidence = float(result.get("confidence", 0.85)),
                sponsor_tier          = sponsor.sponsor_tier,
                conference_name       = conference_name,
                source_page           = sponsor.source_page,
            )

        email = result.email or _first_brochure_email(sponsor)
        phone = result.phone or _first_brochure_phone(sponsor)
        return EnrichedLead(
            company_name          = sponsor.company_name,
            director_name         = result.full_name,
            director_title        = result.title,
            linkedin_url          = result.linkedin_url,
            email                 = email,
            phone                 = phone,
            enrichment_confidence = float(result.confidence),
            sponsor_tier          = sponsor.sponsor_tier,
            conference_name       = conference_name,
            source_page           = sponsor.source_page,
        )

    def _contact_missing(
        self,
        sponsor: ExtractedSponsor,
        conference_name: str,
    ) -> EnrichedLead:
        """Return a fallback lead when no director can be found."""
        brochure_email = _first_brochure_email(sponsor)
        brochure_phone = _first_brochure_phone(sponsor)
        confidence = 0.55 if (brochure_email or brochure_phone) else 0.0
        return EnrichedLead(
            company_name          = sponsor.company_name,
            email                 = brochure_email,
            phone                 = brochure_phone,
            enrichment_confidence = confidence,
            sponsor_tier          = sponsor.sponsor_tier,
            conference_name       = conference_name,
            source_page           = sponsor.source_page,
        )


def _first_brochure_email(sponsor: ExtractedSponsor) -> str | None:
    return sponsor.brochure_emails[0] if sponsor.brochure_emails else None


def _first_brochure_phone(sponsor: ExtractedSponsor) -> str | None:
    return sponsor.brochure_phones[0] if sponsor.brochure_phones else None
