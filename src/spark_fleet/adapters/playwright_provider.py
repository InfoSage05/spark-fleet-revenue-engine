"""
spark_fleet/adapters/playwright_provider.py

Playwright-backed implementation of the ``PeopleSearchProvider`` protocol.

Since direct scraping of LinkedIn usually results in IP bans or immediate
login walls, this adapter searches the public web (via DuckDuckGo) for
LinkedIn profiles matching the company and target titles.

Dependencies
------------
Requires Playwright to be installed:
    pip install "playwright>=1.40"
    playwright install chromium
"""

from __future__ import annotations

import logging
import random
import re
import time
import urllib.parse
from typing import Any

from spark_fleet.enrichment import PersonResult, RateLimitError
from spark_fleet.enrichment import TimeoutError as EnrichmentTimeoutError
from spark_fleet.schemas import ExtractedSponsor

logger = logging.getLogger(__name__)

# Title scoring weights (must match Proxycurl / enrichment definitions)
_TITLE_WEIGHTS: dict[str, float] = {
    "marketing director":    1.0,
    "head of marketing":     0.92,
    "vp marketing":          0.88,
    "vice president marketing": 0.88,
    "growth director":       0.78,
    "commercial director":   0.72,
}

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

def _extract_name_from_ddg_title(ddg_title: str) -> str:
    """
    DDG search results usually look like:
    'John Doe - Marketing Director - Medtronic | LinkedIn'
    We want to extract 'John Doe'.
    """
    # Split by common separators
    parts = re.split(r'\s*[-|–—]\s*', ddg_title)
    if not parts:
        return ""
    # The first part is usually the name
    name = parts[0].strip()
    return name


class PlaywrightPeopleProvider:
    """
    Implements the ``PeopleSearchProvider`` protocol using Playwright.
    
    Searches DuckDuckGo HTML Lite (html.duckduckgo.com) for LinkedIn profiles.
    
    Usage
    -----
    ::

        provider = PlaywrightPeopleProvider(headless=True)
        sponsor  = ExtractedSponsor(company_name="Medtronic", source_page=1)
        result   = provider.find_marketing_director(sponsor)
    """

    def __init__(self, headless: bool = True, timeout_s: float = 30.0) -> None:
        self.headless = headless
        self.timeout_s = timeout_s

    def find_marketing_director(
        self,
        sponsor: ExtractedSponsor,
    ) -> PersonResult | None:
        """
        Search DuckDuckGo for Marketing Directors at the sponsor company.

        Returns
        -------
        The highest-ranked ``PersonResult``, or ``None`` if no match.

        Raises
        ------
        EnrichmentTimeoutError on Playwright timeouts.
        RateLimitError on search engine blocking (CAPTCHA/429).
        """
        try:
            # We import playwright here so the module can still be imported
            # without crashing if playwright isn't installed.
            from playwright.sync_api import sync_playwright, TimeoutError as PwTimeoutError  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is not installed. Please run:\n"
                "  pip install playwright\n"
                "  playwright install chromium"
            ) from exc

        search_queries = _build_search_queries(sponsor)

        candidates: list[PersonResult] = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            )
            page = context.new_page()

            try:
                for query in search_queries:
                    encoded_query = urllib.parse.quote_plus(query)
                    search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
                    time.sleep(random.uniform(1.0, 3.0))
                    page.goto(search_url, timeout=self.timeout_s * 1000, wait_until="domcontentloaded")
                    content_text = page.content().lower()
                    if "rate limit" in content_text or "captcha" in content_text or "too many requests" in content_text:
                        browser.close()
                        raise RateLimitError(f"Search engine rate-limited for '{sponsor.company_name}'")

                    results = page.locator(".result").all()
                    candidates.extend(_parse_search_results(results, sponsor.company_name))
                    if candidates:
                        break

            except PwTimeoutError as exc:
                browser.close()
                raise EnrichmentTimeoutError(
                    f"Playwright request timed out for '{sponsor.company_name}' after {self.timeout_s}s."
                ) from exc
            finally:
                browser.close()

        if not candidates:
            logger.info("Playwright: no marketing directors found for '%s'.", sponsor.company_name)
            return None

        # Return the best match
        best = max(candidates, key=lambda r: r.confidence)
        logger.info(
            "Playwright: found '%s' for '%s' — confidence=%.2f.",
            best.full_name, sponsor.company_name, best.confidence,
        )
        return best


def _build_search_queries(sponsor: ExtractedSponsor) -> list[str]:
    queries: list[str] = []
    for person in getattr(sponsor, "important_people", [])[:5]:
        queries.append(f'site:linkedin.com/in "{person}" "{sponsor.company_name}"')
    queries.append(f'site:linkedin.com/in "{sponsor.company_name}" "marketing"')
    queries.append(f'site:linkedin.com/in "{sponsor.company_name}" ("partnerships" OR "sponsorship" OR "business development")')
    return queries


def _parse_search_results(results: list[Any], company_name: str) -> list[PersonResult]:
    candidates: list[PersonResult] = []
    for res in results:
        try:
            title_elem = res.locator(".result__title")
            snippet_elem = res.locator(".result__snippet")
            url_elem = res.locator(".result__url")

            title_text = title_elem.inner_text().strip()
            snippet_text = snippet_elem.inner_text().strip()
            url_text = url_elem.inner_text().strip()

            href = title_elem.locator("a").get_attribute("href")
            if href and "uddg=" in href:
                parsed = urllib.parse.urlparse(href)
                qs = urllib.parse.parse_qs(parsed.query)
                linkedin_url = qs.get("uddg", [url_text])[0]
            else:
                linkedin_url = url_text

            if "linkedin.com/in/" not in linkedin_url.lower():
                continue

            name = _extract_name_from_ddg_title(title_text)
            if not name or name.lower() in ("linkedin", "profiles"):
                continue

            combined_context = f"{title_text} {snippet_text}"
            score = _title_score(combined_context)
            if company_name.lower() in combined_context.lower():
                score = min(1.0, score + 0.12)

            candidates.append(
                PersonResult(
                    full_name=name,
                    title=combined_context[:100],
                    linkedin_url=linkedin_url,
                    confidence=min(1.0, score),
                )
            )
        except Exception as exc:
            logger.debug("Skipped a search result row due to error: %s", exc)
            continue
    return candidates
