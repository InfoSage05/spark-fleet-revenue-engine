"""
spark_fleet/adapters/fallback_people_provider.py

Composite provider that tries a primary people source first and falls back to
secondary providers when the primary is unavailable on the current plan.
"""

from __future__ import annotations

import logging
from typing import Any

from spark_fleet.schemas import ExtractedSponsor

logger = logging.getLogger(__name__)


class FallbackPeopleProvider:
    def __init__(self, *providers: Any) -> None:
        self.providers = [provider for provider in providers if provider is not None]

    def set_run_context(self, **kwargs: Any) -> None:
        for provider in self.providers:
            if hasattr(provider, "set_run_context"):
                provider.set_run_context(**kwargs)

    def find_marketing_director(self, sponsor: ExtractedSponsor):
        last_error: Exception | None = None

        for index, provider in enumerate(self.providers):
            try:
                result = provider.find_marketing_director(sponsor)
                if result is not None:
                    if index > 0:
                        logger.info(
                            "Fallback provider %s returned a result for '%s'.",
                            provider.__class__.__name__,
                            sponsor.company_name,
                        )
                    return result
            except RuntimeError as exc:
                last_error = exc
                message = str(exc).lower()
                if "api_inaccessible" in message or "free plan" in message or "http 403" in message:
                    logger.warning(
                        "Provider %s unavailable for '%s' (%s). Trying fallback.",
                        provider.__class__.__name__,
                        sponsor.company_name,
                        exc,
                    )
                    continue
                raise

        if last_error:
            logger.warning("All providers exhausted for '%s': %s", sponsor.company_name, last_error)
        return None
