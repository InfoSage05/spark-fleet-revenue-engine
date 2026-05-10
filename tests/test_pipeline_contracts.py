"""
tests/test_pipeline_contracts.py

Contract tests for the three pipeline stages.
All external services (Macro Spark, LinkedIn scraper, Zoho API) are mocked.

These tests define the expected BEHAVIOUR of code that does not yet exist.
They will show as ERRORS (ImportError / AttributeError) until the
corresponding implementation modules are written in later prompts.

DO NOT add implementation logic here. The test file is the specification.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from pydantic import ValidationError

from spark_fleet.schemas import EnrichedLead, ExtractedSponsor, ZohoPayload


# ===========================================================================
# Stage 1 – Macro Spark: sponsor extraction from PDF
# ===========================================================================

class TestMacroSparkExtractionContract:
    """
    The Macro Spark client sends a chunk of PDF text and receives a list of
    ExtractedSponsor objects.  These tests verify the contract, not the model.
    """

    def test_returns_list_of_extracted_sponsors(self, macro_spark_response):
        """
        Given a mocked Macro Spark response, the pipeline should return a list
        of ExtractedSponsor objects with at least one item.
        """
        # macro_spark_response fixture contains 3 sponsors
        assert len(macro_spark_response) >= 1
        for sponsor in macro_spark_response:
            assert isinstance(sponsor, ExtractedSponsor)

    def test_filters_low_confidence_sponsors(self, macro_spark_response):
        """
        After filtering, sponsors below the 0.5 confidence threshold must be
        excluded from the enrichment queue.
        (Implementation: spark_fleet.macro_client.filter_sponsors)
        """
        # Import will raise ImportError until implemented – that is expected.
        from spark_fleet.macro_client import filter_sponsors  # noqa: PLC0415

        filtered = filter_sponsors(macro_spark_response, min_confidence=0.5)

        assert all(s.confidence >= 0.5 for s in filtered)
        # The fixture has one sponsor at 0.28 – it must be gone
        company_names = [s.company_name for s in filtered]
        assert "Unknown Corp" not in company_names

    def test_deduplicates_same_company(self):
        """
        If the LLM emits the same company twice (different pages), only the
        entry with the higher confidence should survive.
        (Implementation: spark_fleet.macro_client.dedupe_sponsors)
        """
        from spark_fleet.macro_client import dedupe_sponsors  # noqa: PLC0415

        duplicates = [
            ExtractedSponsor(company_name="Medtronic", source_page=1, confidence=0.7),
            ExtractedSponsor(company_name="Medtronic", source_page=4, confidence=0.95),
            ExtractedSponsor(company_name="Medtronic", source_page=8, confidence=0.5),
        ]
        result = dedupe_sponsors(duplicates)
        assert len(result) == 1
        assert result[0].confidence == 0.95

    def test_malformed_llm_response_raises_clearly(self):
        """
        If the Macro Spark returns JSON that doesn't match ExtractedSponsor,
        the client must raise a descriptive error (not a raw pydantic dump).
        (Implementation: spark_fleet.macro_client.MacroSparkClient.parse_response)
        """
        from spark_fleet.macro_client import MacroSparkError, parse_macro_response  # noqa: PLC0415

        bad_payload = {"sponsors": [{"not_a_field": "surprise"}]}
        with pytest.raises(MacroSparkError, match="malformed"):
            parse_macro_response(bad_payload)

    def test_extracts_important_people_from_llm_response(self):
        """
        Sponsor extraction should carry named sponsor-side people forward so
        enrichment can prioritize those names in LinkedIn search.
        """
        from spark_fleet.macro_client import parse_macro_response  # noqa: PLC0415

        payload = {
            "sponsors": [
                {
                    "company_name": "GSK",
                    "sponsor_tier": "Gold",
                    "source_page": 2,
                    "important_people": ["Jane Doe", "Arun Mehta"],
                }
            ]
        }

        sponsors = parse_macro_response(payload)

        assert sponsors[0].important_people == ["Jane Doe", "Arun Mehta"]

    def test_extracts_brochure_contacts_from_response(self):
        """
        Sponsor extraction should preserve brochure-level emails and phones when
        the LLM returns them so downstream stages can outreach directly.
        """
        from spark_fleet.macro_client import parse_macro_response  # noqa: PLC0415

        payload = {
            "sponsors": [
                {
                    "company_name": "GSK",
                    "sponsor_tier": "Gold",
                    "source_page": 2,
                    "brochure_emails": ["Partnerships@GSK.com"],
                    "brochure_phones": ["+91 98765 43210"],
                }
            ]
        }

        sponsors = parse_macro_response(payload)

        assert sponsors[0].brochure_emails == ["partnerships@gsk.com"]
        assert sponsors[0].brochure_phones == ["+919876543210"]


# ===========================================================================
# Stage 2 – Micro Spark: LinkedIn enrichment resilience
# ===========================================================================

class TestMicroSparkEnrichmentContract:
    """
    The Micro Spark enriches each sponsor by scraping or querying a people-data
    provider.  These tests assert graceful degradation, NOT scraping logic.
    """

    def test_timeout_returns_contact_missing_lead(self, valid_sponsor):
        """
        When the enrichment provider raises a timeout, the orchestrator must NOT
        crash.  It must return an EnrichedLead with enrichment_confidence=0.0
        and a status of CONTACT_MISSING.
        (Implementation: spark_fleet.enrichment.EnrichmentOrchestrator.enrich)
        """
        from spark_fleet.enrichment import (  # noqa: PLC0415
            EnrichmentOrchestrator,
            TimeoutError as EnrichmentTimeout,
        )

        mock_provider = MagicMock()
        mock_provider.find_marketing_director.side_effect = EnrichmentTimeout(
            "LinkedIn request timed out after 30s"
        )

        orchestrator = EnrichmentOrchestrator(people_provider=mock_provider)
        lead = orchestrator.enrich(sponsor=valid_sponsor, conference_name="HIMSS 2025")

        assert isinstance(lead, EnrichedLead)
        assert lead.enrichment_confidence == 0.0
        assert lead.director_name is None
        assert lead.phone is None

    def test_rate_limit_pauses_queue_not_crash(self, valid_sponsor):
        """
        When the provider returns a rate-limit error, the orchestrator must
        signal a pause (raise EnrichmentPaused) so the caller can back off.
        It must NOT silently swallow the error or crash the worker.
        (Implementation: spark_fleet.enrichment.EnrichmentOrchestrator.enrich)
        """
        from spark_fleet.enrichment import (  # noqa: PLC0415
            EnrichmentOrchestrator,
            EnrichmentPaused,
            RateLimitError,
        )

        mock_provider = MagicMock()
        mock_provider.find_marketing_director.side_effect = RateLimitError(
            "429 Too Many Requests"
        )

        orchestrator = EnrichmentOrchestrator(people_provider=mock_provider)
        with pytest.raises(EnrichmentPaused, match="rate.?limit"):
            orchestrator.enrich(sponsor=valid_sponsor, conference_name="HIMSS 2025")

    def test_successful_enrichment_returns_full_lead(self, valid_sponsor):
        """
        Happy path: provider returns a director → lead has all contact fields.
        """
        from spark_fleet.enrichment import EnrichmentOrchestrator  # noqa: PLC0415

        mock_result = MagicMock()
        mock_result.full_name   = "Priya Sharma"
        mock_result.title       = "Marketing Director"
        mock_result.linkedin_url = "https://www.linkedin.com/in/priya-sharma"
        mock_result.email       = "priya@medtronic.com"
        mock_result.phone       = "+919876543210"
        mock_result.confidence  = 0.9

        mock_provider = MagicMock()
        mock_provider.find_marketing_director.return_value = mock_result

        orchestrator = EnrichmentOrchestrator(people_provider=mock_provider)
        lead = orchestrator.enrich(sponsor=valid_sponsor, conference_name="HIMSS 2025")

        assert lead.director_name == "Priya Sharma"
        assert lead.phone == "+919876543210"
        assert lead.enrichment_confidence > 0.5

    def test_no_director_found_returns_contact_missing(self, valid_sponsor):
        """
        Provider succeeds but finds no matching person → CONTACT_MISSING path,
        not a crash.
        """
        from spark_fleet.enrichment import EnrichmentOrchestrator  # noqa: PLC0415

        mock_provider = MagicMock()
        mock_provider.find_marketing_director.return_value = None

        orchestrator = EnrichmentOrchestrator(people_provider=mock_provider)
        lead = orchestrator.enrich(sponsor=valid_sponsor, conference_name="HIMSS 2025")

        assert lead.enrichment_confidence == 0.0
        assert lead.director_name is None

    def test_no_director_uses_brochure_contact_fallback(self):
        """
        If provider finds no director but brochure has direct contact data,
        orchestrator should return that email/phone for direct outreach.
        """
        from spark_fleet.enrichment import EnrichmentOrchestrator  # noqa: PLC0415

        sponsor = ExtractedSponsor(
            company_name="Medtronic",
            sponsor_tier="Gold",
            source_page=1,
            brochure_emails=["contact@medtronic.com"],
            brochure_phones=["+14155552671"],
        )

        mock_provider = MagicMock()
        mock_provider.find_marketing_director.return_value = None

        orchestrator = EnrichmentOrchestrator(people_provider=mock_provider)
        lead = orchestrator.enrich(sponsor=sponsor, conference_name="HIMSS 2025")

        assert lead.director_name is None
        assert lead.email == "contact@medtronic.com"
        assert lead.phone == "+14155552671"
        assert lead.enrichment_confidence > 0.0


# ===========================================================================
# Stage 3 – Zoho CRM: payload mapping
# ===========================================================================

class TestZohoPayloadMapping:
    """
    The Micro Spark converts an EnrichedLead into a ZohoPayload before pushing.
    These tests pin the mapping logic.
    (Implementation: spark_fleet.zoho.map_lead_to_zoho_payload)
    """

    def test_maps_lead_with_phone_to_pending_wati_status(
        self, enriched_lead_with_phone: EnrichedLead
    ):
        from spark_fleet.zoho import map_lead_to_zoho_payload  # noqa: PLC0415

        payload = map_lead_to_zoho_payload(enriched_lead_with_phone)

        assert isinstance(payload, ZohoPayload)
        record = payload.data[0]
        assert record["WATI_Status"] == "Pending"
        assert record["Mobile"] == enriched_lead_with_phone.phone

    def test_maps_lead_without_phone_to_missing_phone_status(
        self, enriched_lead_no_phone: EnrichedLead
    ):
        from spark_fleet.zoho import map_lead_to_zoho_payload  # noqa: PLC0415

        payload = map_lead_to_zoho_payload(enriched_lead_no_phone)
        record = payload.data[0]
        assert "Missing Phone" in record["WATI_Status"]
        assert "Mobile" not in record or record.get("Mobile") is None

    def test_required_zoho_fields_always_present(
        self, enriched_lead_with_phone: EnrichedLead
    ):
        from spark_fleet.zoho import map_lead_to_zoho_payload  # noqa: PLC0415

        payload = map_lead_to_zoho_payload(enriched_lead_with_phone)
        record = payload.data[0]
        for field in ("Last_Name", "Company", "Lead_Source", "Lead_Status",
                      "Conference_Name", "Sponsor_Tier"):
            assert field in record, f"Missing required Zoho field: {field}"

    def test_linkedin_url_serialised_as_string(
        self, enriched_lead_with_phone: EnrichedLead
    ):
        """HttpUrl objects must be cast to str before pushing to Zoho."""
        from spark_fleet.zoho import map_lead_to_zoho_payload  # noqa: PLC0415

        payload = map_lead_to_zoho_payload(enriched_lead_with_phone)
        record = payload.data[0]
        if "LinkedIn_Profile" in record:
            assert isinstance(record["LinkedIn_Profile"], str)

    def test_contact_missing_lead_produces_valid_payload(
        self, enriched_lead_contact_missing: EnrichedLead
    ):
        """Even a contact-missing lead must produce a valid ZohoPayload."""
        from spark_fleet.zoho import map_lead_to_zoho_payload  # noqa: PLC0415

        payload = map_lead_to_zoho_payload(enriched_lead_contact_missing)
        assert isinstance(payload, ZohoPayload)
        record = payload.data[0]
        assert record["Company"] == "Unknown Corp"
        assert "Missing Phone" in record["WATI_Status"]
