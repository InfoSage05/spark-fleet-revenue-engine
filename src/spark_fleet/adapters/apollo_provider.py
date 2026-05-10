"""
spark_fleet/adapters/apollo_provider.py

Apollo.io-backed implementation of the Spark Fleet people-search provider.

The provider searches Apollo for a senior marketing or sponsorship decision
maker at a sponsor company, then enriches the best candidate to retrieve any
available email and phone data.
"""

from __future__ import annotations

import logging
import os
import re
import time
import urllib.parse
from typing import Any

import httpx

from spark_fleet.enrichment import PersonResult, RateLimitError
from spark_fleet.enrichment import TimeoutError as EnrichmentTimeoutError
from spark_fleet.schemas import ExtractedSponsor

logger = logging.getLogger(__name__)


APOLLO_BASE_URL = "https://api.apollo.io"
PEOPLE_SEARCH_PATH = "/api/v1/mixed_people/api_search"
PEOPLE_MATCH_PATH = "/api/v1/people/match"

DEFAULT_TARGET_TITLES: tuple[str, ...] = (
    "Chief Marketing Officer",
    "CMO",
    "VP of Marketing",
    "Vice President Marketing",
    "Head of Marketing",
    "Marketing Director",
    "Director of Marketing",
    "Director, Marketing",
    "Sponsorship Director",
    "Director of Sponsorships",
    "Partnerships Director",
)

TITLE_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("chief marketing officer", 1.00),
    ("cmo", 1.00),
    ("vp of marketing", 0.96),
    ("vice president marketing", 0.96),
    ("head of marketing", 0.94),
    ("marketing director", 0.92),
    ("director of marketing", 0.92),
    ("director, marketing", 0.92),
    ("sponsorship", 0.86),
    ("partnership", 0.78),
    ("marketing", 0.60),
)


class ApolloProvider:
    """
    People-search provider backed by Apollo.io REST APIs.

    Public API
    ----------
    ``find_director(company_name) -> dict | None`` is the requested provider
    interface for Apollo.

    ``find_marketing_director(sponsor) -> PersonResult | None`` is retained as
    a compatibility shim for the existing ``EnrichmentOrchestrator`` contract.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = APOLLO_BASE_URL,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        backoff_base_s: float = 1.0,
        per_page: int = 10,
        target_titles: tuple[str, ...] = DEFAULT_TARGET_TITLES,
        phone_webhook_url: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("APOLLO_API_KEY", "")
        if not self.api_key:
            raise EnvironmentError("Missing APOLLO_API_KEY environment variable.")

        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max(0, max_retries)
        self.backoff_base_s = max(0.1, backoff_base_s)
        self.per_page = per_page
        self.target_titles = target_titles
        self.phone_webhook_url = (
            phone_webhook_url or os.environ.get("APOLLO_PHONE_WEBHOOK_URL", "")
        ).strip()
        self._run_context: dict[str, Any] = {}

    def set_run_context(
        self,
        *,
        company_name: str,
        conference_name: str,
        sponsor_tier: str | None = None,
        source_page: int | None = None,
    ) -> None:
        """
        Store the current enrichment context so Apollo webhook callbacks can
        be correlated back to the correct Zoho lead.
        """
        self._run_context = {
            "company_name": company_name,
            "conference_name": conference_name,
            "sponsor_tier": sponsor_tier,
            "source_page": source_page,
        }

    def find_director(self, company_name: str) -> dict[str, str | None] | None:
        """
        Find the best marketing/sponsorship decision-maker for a company.

        Returns a plain dict with Spark Fleet's expected lead fields, or
        ``None`` when Apollo has no suitable match.
        """
        company_name = company_name.strip()
        if not company_name:
            return None

        search_payload = self._search_people(company_name)
        people = self._extract_people(search_payload)
        if not people:
            logger.info("Apollo: no people found for '%s'.", company_name)
            return None

        candidate = self._choose_best_candidate(people, company_name)
        if not candidate:
            logger.info("Apollo: no matching director found for '%s'.", company_name)
            return None

        enriched = self._enrich_person(candidate)
        person = enriched or candidate
        mapped = self._map_person(person)

        if not mapped["director_name"]:
            return None

        logger.info(
            "Apollo: found '%s' (%s) for '%s'.",
            mapped["director_name"],
            mapped["director_title"],
            company_name,
        )
        return mapped

    def find_marketing_director(
        self,
        sponsor: ExtractedSponsor,
    ) -> PersonResult | None:
        """
        Compatibility method for the existing Spark Fleet orchestrator.
        """
        result = self.find_director(sponsor.company_name)
        if result is None:
            return None

        return PersonResult(
            full_name=result.get("director_name") or "",
            title=result.get("director_title") or "",
            linkedin_url=result.get("linkedin_url"),
            email=result.get("email"),
            phone=result.get("phone"),
            confidence=0.85,
        )

    def _search_people(self, company_name: str) -> dict[str, Any]:
        params: list[tuple[str, str | int | bool]] = [
            ("q_organization_name", company_name),
            ("q_keywords", company_name),
            ("person_seniorities[]", "c_suite"),
            ("person_seniorities[]", "vp"),
            ("person_seniorities[]", "head"),
            ("person_seniorities[]", "director"),
            ("include_similar_titles", True),
            ("page", 1),
            ("per_page", self.per_page),
        ]
        for title in self.target_titles:
            params.append(("person_titles[]", title))

        return self._post(PEOPLE_SEARCH_PATH, params=params)

    def _enrich_person(self, candidate: dict[str, Any]) -> dict[str, Any] | None:
        params: list[tuple[str, str | bool]] = [
            ("reveal_personal_emails", "true"),
        ]
        if self.phone_webhook_url:
            params.extend(
                [
                    ("reveal_phone_number", "true"),
                    ("run_waterfall_phone", "true"),
                    ("run_waterfall_email", "true"),
                    ("webhook_url", self._build_callback_url(candidate)),
                ]
            )
        else:
            params.append(("reveal_phone_number", "false"))

        person_id = _as_str(candidate.get("id"))
        if person_id:
            params.append(("id", person_id))
        else:
            name = _as_str(candidate.get("name")) or " ".join(
                part for part in (
                    _as_str(candidate.get("first_name")),
                    _as_str(candidate.get("last_name")),
                )
                if part
            )
            linkedin_url = _as_str(candidate.get("linkedin_url"))
            organization = candidate.get("organization") or {}
            domain = _as_str(organization.get("primary_domain"))

            if name:
                params.append(("name", name))
            if linkedin_url:
                params.append(("linkedin_url", linkedin_url))
            if domain:
                params.append(("domain", domain))

        try:
            payload = self._post(PEOPLE_MATCH_PATH, params=params)
        except Exception as exc:
            logger.warning("Apollo enrichment failed for candidate %r: %s", person_id, exc)
            return None

        person = payload.get("person")
        return person if isinstance(person, dict) else None

    def _post(
        self,
        path: str,
        params: list[tuple[str, str | int | bool]] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {
            "accept": "application/json",
            "Cache-Control": "no-cache",
            "Content-Type": "application/json",
            "X-Api-Key": self.api_key,
        }

        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout_s,
                )
            except httpx.TimeoutException as exc:
                raise EnrichmentTimeoutError(
                    f"Apollo request timed out after {self.timeout_s}s."
                ) from exc

            if response.status_code == 429:
                if attempt >= self.max_retries:
                    raise RateLimitError("Apollo returned HTTP 429 Too Many Requests.")
                time.sleep(self._retry_delay(response, attempt))
                continue

            if response.status_code in (401, 403):
                detail = ""
                try:
                    detail = response.text[:300]
                except Exception:  # noqa: BLE001
                    detail = ""

                if response.status_code == 403 and path == PEOPLE_SEARCH_PATH:
                    raise RuntimeError(
                        "Apollo rejected the People Search request with HTTP 403. "
                        "This endpoint requires a master API key. In Apollo, go to "
                        "Settings > Integrations > Apollo API > API Keys, then either "
                        "create a new key with 'Set as master key' enabled or update "
                        "the current key to include master access. "
                        f"Apollo response: {detail}"
                    )
                raise RuntimeError(
                    "Apollo rejected APOLLO_API_KEY. Check that the key is valid "
                    "and has access to People Search/Enrichment APIs. "
                    f"Apollo response: {detail}"
                )

            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, dict) else {}

        raise RateLimitError("Apollo rate limit retry loop exhausted.")

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        return self.backoff_base_s * (2 ** attempt)

    def _extract_people(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        for key in ("people", "contacts", "mixed_people"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return []

    def _choose_best_candidate(
        self,
        people: list[dict[str, Any]],
        company_name: str,
    ) -> dict[str, Any] | None:
        scored = [
            (self._candidate_score(person, company_name), person)
            for person in people
        ]
        scored = [(score, person) for score, person in scored if score > 0]
        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def _candidate_score(self, person: dict[str, Any], company_name: str) -> float:
        title = _as_str(person.get("title")).lower()
        score = 0.0
        for needle, weight in TITLE_WEIGHTS:
            if needle in title:
                score = max(score, weight)

        organization = person.get("organization") or {}
        org_name = _as_str(organization.get("name")).lower()
        if org_name and _normalized(company_name) in _normalized(org_name):
            score += 0.10

        if _as_str(person.get("linkedin_url")):
            score += 0.03
        if _first_email(person):
            score += 0.03
        if _first_phone(person):
            score += 0.04

        return min(score, 1.0)

    def _map_person(self, person: dict[str, Any]) -> dict[str, str | None]:
        return {
            "director_name": _as_str(person.get("name")) or _join_name(person),
            "director_title": _as_str(person.get("title")),
            "linkedin_url": _as_str(person.get("linkedin_url")),
            "email": _first_email(person),
            "phone": _first_phone(person),
        }

    def _build_callback_url(self, candidate: dict[str, Any]) -> str:
        if not self.phone_webhook_url:
            return ""

        parsed = urllib.parse.urlsplit(self.phone_webhook_url)
        query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        extra_pairs = [
            ("company_name", self._run_context.get("company_name") or ""),
            ("conference_name", self._run_context.get("conference_name") or ""),
            ("sponsor_tier", self._run_context.get("sponsor_tier") or ""),
            ("apollo_person_id", _as_str(candidate.get("id")) or ""),
        ]
        query_pairs.extend((key, value) for key, value in extra_pairs if value)
        return urllib.parse.urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                urllib.parse.urlencode(query_pairs),
                parsed.fragment,
            )
        )


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _join_name(person: dict[str, Any]) -> str | None:
    parts = [
        _as_str(person.get("first_name")),
        _as_str(person.get("last_name")),
    ]
    name = " ".join(part for part in parts if part)
    return name or None


def _first_email(person: dict[str, Any]) -> str | None:
    for key in ("email", "personal_email"):
        email = _as_str(person.get(key))
        if email and "@" in email:
            return email

    for key in ("emails", "email_addresses"):
        values = person.get(key)
        if isinstance(values, list):
            for item in values:
                if isinstance(item, dict):
                    email = _as_str(item.get("email") or item.get("value"))
                else:
                    email = _as_str(item)
                if email and "@" in email:
                    return email

    return None


def _first_phone(person: dict[str, Any]) -> str | None:
    for key in ("mobile_phone", "phone", "sanitized_phone"):
        phone = _normalize_phone(_as_str(person.get(key)))
        if phone:
            return phone

    for key in ("phone_numbers", "phones"):
        values = person.get(key)
        if isinstance(values, list):
            for item in values:
                if isinstance(item, dict):
                    raw = (
                        item.get("sanitized_number")
                        or item.get("raw_number")
                        or item.get("number")
                        or item.get("value")
                    )
                else:
                    raw = item
                phone = _normalize_phone(_as_str(raw))
                if phone:
                    return phone

    return None


def _normalize_phone(value: str | None) -> str | None:
    if not value:
        return None

    has_plus = value.strip().startswith("+")
    digits = re.sub(r"\D", "", value)
    if not 8 <= len(digits) <= 15:
        return None
    return f"+{digits}" if has_plus else digits


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())
