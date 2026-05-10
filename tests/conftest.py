"""
Shared pytest fixtures for the Spark Fleet test suite.

All fixtures are lightweight in-memory objects.
No real HTTP calls, no real PDFs, no real LLM inference.
"""

from __future__ import annotations

import pytest
from spark_fleet.schemas import EnrichedLead, ExtractedSponsor, ZohoPayload


# ---------------------------------------------------------------------------
# ExtractedSponsor fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def valid_sponsor() -> ExtractedSponsor:
    """A clean, fully-populated sponsor returned by the Macro Spark."""
    return ExtractedSponsor(
        company_name="Medtronic",
        sponsor_tier="Gold",
        source_page=3,
        evidence_text="Medtronic – Gold Sponsor of HIMSS 2025",
        confidence=0.92,
    )


@pytest.fixture()
def low_confidence_sponsor() -> ExtractedSponsor:
    """A sponsor whose confidence is below the 0.5 enrichment threshold."""
    return ExtractedSponsor(
        company_name="AcmePharma",
        sponsor_tier="Bronze",
        source_page=7,
        confidence=0.3,
    )


@pytest.fixture()
def macro_spark_response(valid_sponsor: ExtractedSponsor) -> list[ExtractedSponsor]:
    """
    Simulates the full list that the Macro Spark would return for a brochure.
    Mix of high- and low-confidence entries.
    """
    return [
        valid_sponsor,
        ExtractedSponsor(
            company_name="Philips Healthcare",
            sponsor_tier="Platinum",
            source_page=1,
            confidence=0.97,
        ),
        ExtractedSponsor(
            company_name="Unknown Corp",
            sponsor_tier="Unknown",
            source_page=12,
            confidence=0.28,          # below threshold – should be filtered
        ),
    ]


# ---------------------------------------------------------------------------
# EnrichedLead fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def enriched_lead_with_phone() -> EnrichedLead:
    """A fully-enriched lead that has a phone number → WATI_Status = Pending."""
    return EnrichedLead(
        company_name="Medtronic",
        director_name="Priya Sharma",
        director_title="Marketing Director",
        linkedin_url="https://www.linkedin.com/in/priya-sharma-medtronic",
        email="priya.sharma@medtronic.com",
        phone="+919876543210",
        enrichment_confidence=0.88,
        sponsor_tier="Gold",
        conference_name="HIMSS 2025",
        source_page=3,
    )


@pytest.fixture()
def enriched_lead_no_phone() -> EnrichedLead:
    """Lead found but no phone → WATI_Status = Not Sent - Missing Phone."""
    return EnrichedLead(
        company_name="Philips Healthcare",
        director_name="James Okafor",
        director_title="Head of Marketing APAC",
        linkedin_url="https://www.linkedin.com/in/james-okafor",
        enrichment_confidence=0.75,
        sponsor_tier="Platinum",
        conference_name="HIMSS 2025",
        source_page=1,
    )


@pytest.fixture()
def enriched_lead_contact_missing() -> EnrichedLead:
    """No Marketing Director found at all → confidence=0, contact fields null."""
    return EnrichedLead(
        company_name="Unknown Corp",
        enrichment_confidence=0.0,
        sponsor_tier="Unknown",
        conference_name="HIMSS 2025",
    )
