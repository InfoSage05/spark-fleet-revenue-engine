"""
spark_fleet/adapters/free_people_provider.py

Free fallback provider for Spark Fleet.

Strategy:
1. Reuse the existing Playwright LinkedIn discovery to find the likely
   decision-maker.
2. Find the company's official website via public search.
3. Scrape public contact pages for email, phone, or WhatsApp links.
4. Search public LinkedIn company-page snippets for phone/WhatsApp numbers.
5. Optionally use Apollo People Enrichment only after a person and company
   domain are already known. Apollo People Search is not used in this mode.

This path is intentionally conservative: it only uses public web data and an
optional best-effort enrichment call. It cannot guarantee personal mobile
numbers, but it gives Spark Fleet a no-paid-plan fallback path.
"""

from __future__ import annotations

import html
import logging
import os
import re
import urllib.parse
from typing import Any

import httpx

from spark_fleet.adapters.playwright_provider import PlaywrightPeopleProvider
from spark_fleet.enrichment import PersonResult, RateLimitError
from spark_fleet.enrichment import TimeoutError as EnrichmentTimeoutError
from spark_fleet.schemas import ExtractedSponsor

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://html.duckduckgo.com/html/"
_APOLLO_PEOPLE_MATCH_URL = "https://api.apollo.io/api/v1/people/match"

_SKIP_HOST_SUBSTRINGS = (
    "linkedin.com",
    "facebook.com",
    "instagram.com",
    "x.com",
    "twitter.com",
    "youtube.com",
    "zoominfo.com",
    "crunchbase.com",
    "bloomberg.com",
    "rocketreach.co",
)

_EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_PHONE_RE = re.compile(r"(?:(?:\+\d{1,3}[\s().-]*)?(?:\d[\s().-]*){8,15})")
_WHATSAPP_RE = re.compile(r"(?:wa\.me/|api\.whatsapp\.com/send\?phone=)(\+?\d{8,15})", re.I)


class FreePeopleProvider:
    """
    Free provider using Playwright, public web scraping, and optional Apollo
    email enrichment.
    """

    def __init__(
        self,
        people_finder: Any | None = None,
        timeout_s: float = 25.0,
        apollo_api_key: str | None = None,
    ) -> None:
        self.people_finder = people_finder or PlaywrightPeopleProvider(headless=True, timeout_s=30.0)
        self.timeout_s = timeout_s
        self.apollo_api_key = (apollo_api_key or os.environ.get("APOLLO_API_KEY", "")).strip()
        self.last_trace: dict[str, str | None] = {}

    def find_marketing_director(
        self,
        sponsor: ExtractedSponsor,
    ) -> PersonResult | None:
        base_result = self.people_finder.find_marketing_director(sponsor)
        self.last_trace = {
            "linkedin_status": "missing",
            "website_url": None,
            "public_email": None,
            "public_phone": None,
            "linkedin_company_phone": None,
            "apollo_email_status": "skipped",
        }
        if base_result is None:
            return None
        self.last_trace["linkedin_status"] = "found"

        website_url = self._find_company_website(sponsor.company_name)
        self.last_trace["website_url"] = website_url
        email: str | None = None
        phone: str | None = None

        if website_url:
            email, phone = self._scrape_contact_details(website_url)
            self.last_trace["public_email"] = email
            self.last_trace["public_phone"] = phone

        if not phone:
            linkedin_company_phone = self._find_linkedin_company_phone(sponsor.company_name)
            self.last_trace["linkedin_company_phone"] = linkedin_company_phone
            if linkedin_company_phone:
                phone = linkedin_company_phone

        if not email and website_url and self.apollo_api_key:
            apollo_email, status = self._find_email_with_apollo(base_result.full_name, website_url)
            self.last_trace["apollo_email_status"] = status
            if apollo_email:
                email = apollo_email
        elif not self.apollo_api_key:
            self.last_trace["apollo_email_status"] = "skipped_no_key"

        confidence = float(base_result.confidence)
        if email:
            confidence = min(1.0, confidence + 0.07)
        if phone:
            confidence = min(1.0, confidence + 0.12)

        return PersonResult(
            full_name=base_result.full_name,
            title=base_result.title,
            linkedin_url=base_result.linkedin_url,
            email=email,
            phone=phone,
            confidence=confidence,
        )

    def _find_company_website(self, company_name: str) -> str | None:
        html_text = self._search_html(f'"{company_name}" official site')
        for url in _extract_result_urls(html_text):
            if _looks_like_company_site(url):
                return url
        return None

    def _find_linkedin_company_phone(self, company_name: str) -> str | None:
        query = f'site:linkedin.com/company "{company_name}" (phone OR contact OR whatsapp)'
        html_text = self._search_html(query)
        return _extract_phone(html_text)

    def _scrape_contact_details(self, website_url: str) -> tuple[str | None, str | None]:
        pages_to_visit = [website_url]
        homepage = self._get_text(website_url)
        if homepage:
            pages_to_visit.extend(_priority_links(website_url, homepage))

        best_email: str | None = None
        best_phone: str | None = None

        for url in pages_to_visit[:4]:
            page = self._get_text(url)
            if not page:
                continue

            email = _extract_email(page)
            phone = _extract_phone(page)
            if email and not best_email:
                best_email = email
            if phone and not best_phone:
                best_phone = phone
            if best_email and best_phone:
                break

        return best_email, best_phone

    def _find_email_with_apollo(self, full_name: str, website_url: str) -> tuple[str | None, str]:
        domain = _domain_from_url(website_url)
        if not domain:
            return None, "skipped_no_domain"

        parts = full_name.strip().split()
        if len(parts) < 2:
            return None, "skipped_no_name"

        params: list[tuple[str, str | bool]] = [
            ("name", full_name),
            ("domain", domain),
            ("reveal_personal_emails", True),
            ("reveal_phone_number", False),
        ]
        headers = {
            "accept": "application/json",
            "Cache-Control": "no-cache",
            "Content-Type": "application/json",
            "X-Api-Key": self.apollo_api_key,
        }

        try:
            response = httpx.post(
                _APOLLO_PEOPLE_MATCH_URL,
                params=params,
                headers=headers,
                timeout=self.timeout_s,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text.lower()
            if exc.response.status_code == 403 or "api_inaccessible" in body:
                logger.info(
                    "APOLLO_EMAIL: unavailable on current plan, continuing with free public-web data"
                )
                return None, "blocked"
            logger.info("Apollo email lookup failed for %s at %s: %s", full_name, domain, exc)
            return None, "error"
        except httpx.HTTPError as exc:
            logger.info("Apollo email lookup failed for %s at %s: %s", full_name, domain, exc)
            return None, "error"

        body = response.json()
        person = body.get("person", {}) if isinstance(body, dict) else {}
        email = _first_apollo_email(person) if isinstance(person, dict) else None
        return (email, "found") if email else (None, "no_email")

    def _search_html(self, query: str) -> str:
        try:
            response = httpx.get(
                _SEARCH_URL,
                params={"q": query},
                headers={"User-Agent": _desktop_ua()},
                timeout=self.timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise EnrichmentTimeoutError(
                f"FreePeopleProvider search timed out for query '{query}' after {self.timeout_s}s."
            ) from exc
        except httpx.HTTPError as exc:
            raise EnrichmentTimeoutError(
                f"FreePeopleProvider search failed for query '{query}': {exc}"
            ) from exc

        if response.status_code == 429:
            raise RateLimitError(f"DuckDuckGo rate-limited the free fallback search for '{query}'.")
        response.raise_for_status()
        return response.text

    def _get_text(self, url: str) -> str | None:
        try:
            response = httpx.get(
                url,
                headers={"User-Agent": _desktop_ua()},
                follow_redirects=True,
                timeout=self.timeout_s,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            return None
        return response.text


def _desktop_ua() -> str:
    return (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )


def _extract_result_urls(html_text: str) -> list[str]:
    urls: list[str] = []
    for match in re.finditer(r'href="([^"]+)"', html_text, re.I):
        href = html.unescape(match.group(1))
        url = _decode_ddg_redirect(href)
        if url and url not in urls:
            urls.append(url)
    return urls


def _decode_ddg_redirect(href: str) -> str | None:
    if href.startswith("//"):
        href = f"https:{href}"
    if href.startswith("/"):
        href = urllib.parse.urljoin(_SEARCH_URL, href)

    parsed = urllib.parse.urlsplit(href)
    qs = urllib.parse.parse_qs(parsed.query)
    if "uddg" in qs and qs["uddg"]:
        return qs["uddg"][0]

    if parsed.scheme in {"http", "https"}:
        return href
    return None


def _looks_like_company_site(url: str) -> bool:
    host = urllib.parse.urlsplit(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return False
    return not any(bad in host for bad in _SKIP_HOST_SUBSTRINGS)


def _domain_from_url(url: str) -> str | None:
    domain = urllib.parse.urlsplit(url).netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain or None


def _priority_links(base_url: str, html_text: str) -> list[str]:
    base = urllib.parse.urlsplit(base_url)
    links: list[str] = []
    for match in re.finditer(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html_text, re.I | re.S):
        href = html.unescape(match.group(1))
        text = re.sub(r"<[^>]+>", " ", match.group(2))
        label = " ".join(text.split()).lower()
        if not any(keyword in label for keyword in ("contact", "about", "team", "leadership", "company")):
            continue

        absolute = urllib.parse.urljoin(base_url, href)
        target = urllib.parse.urlsplit(absolute)
        if target.scheme not in {"http", "https"}:
            continue
        if target.netloc != base.netloc:
            continue
        if absolute not in links:
            links.append(absolute)
    return links


def _extract_email(html_text: str) -> str | None:
    emails = _EMAIL_RE.findall(html_text)
    for email in emails:
        lowered = email.lower()
        if any(bad in lowered for bad in ("example.com", "wix.com", "sentry.io", ".png")):
            continue
        return email
    return None


def _first_apollo_email(person: dict[str, Any]) -> str | None:
    email = person.get("email")
    if isinstance(email, str) and "@" in email:
        return email.strip()

    for key in ("emails", "contact_emails"):
        values = person.get(key)
        if isinstance(values, list):
            for item in values:
                if isinstance(item, dict):
                    candidate = item.get("email") or item.get("value")
                else:
                    candidate = item
                if isinstance(candidate, str) and "@" in candidate:
                    return candidate.strip()
    return None


def _extract_phone(html_text: str) -> str | None:
    for match in _WHATSAPP_RE.finditer(html_text):
        normalized = _normalize_phone(match.group(1))
        if normalized:
            return normalized

    if "whatsapp" in html_text.lower():
        for candidate in _PHONE_RE.findall(html_text):
            normalized = _normalize_phone(candidate)
            if normalized:
                return normalized

    for candidate in _PHONE_RE.findall(html_text):
        normalized = _normalize_phone(candidate)
        if normalized:
            return normalized
    return None


def _normalize_phone(value: str | None) -> str | None:
    if not value:
        return None
    has_plus = value.strip().startswith("+")
    digits = re.sub(r"\D", "", value)
    if not 8 <= len(digits) <= 15:
        return None
    return f"+{digits}" if has_plus else digits
