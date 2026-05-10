"""
tests/test_schemas.py

Tests for the three Pydantic schemas: ExtractedSponsor, EnrichedLead, ZohoPayload.

Each test is named after what it proves, not how it does it.
Tests are grouped by schema then by valid / invalid scenarios.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spark_fleet.schemas import EnrichedLead, ExtractedSponsor, ZohoPayload


# ===========================================================================
# ExtractedSponsor
# ===========================================================================

class TestExtractedSponsor:

    # --- valid paths ---

    def test_minimal_valid_sponsor(self):
        """Only required fields should be enough to construct a valid record."""
        sponsor = ExtractedSponsor(company_name="Medtronic", source_page=1)
        assert sponsor.company_name == "Medtronic"
        assert sponsor.sponsor_tier == "Unknown"       # default
        assert sponsor.confidence == 1.0               # default
        assert sponsor.evidence_text is None           # optional, not set

    def test_full_valid_sponsor(self, valid_sponsor: ExtractedSponsor):
        assert valid_sponsor.company_name == "Medtronic"
        assert valid_sponsor.sponsor_tier == "Gold"
        assert valid_sponsor.source_page == 3
        assert 0.0 <= valid_sponsor.confidence <= 1.0

    def test_confidence_boundary_values(self):
        """Confidence of exactly 0.0 and 1.0 are both valid."""
        ExtractedSponsor(company_name="A", source_page=1, confidence=0.0)
        ExtractedSponsor(company_name="A", source_page=1, confidence=1.0)

    def test_macro_spark_response_list_length(self, macro_spark_response: list[ExtractedSponsor]):
        """Fixture returns 3 sponsors including one below-threshold entry."""
        assert len(macro_spark_response) == 3

    def test_macro_spark_response_types(self, macro_spark_response: list[ExtractedSponsor]):
        assert all(isinstance(s, ExtractedSponsor) for s in macro_spark_response)

    # --- invalid paths ---

    def test_blank_company_name_raises(self):
        with pytest.raises(ValidationError, match="must not be blank"):
            ExtractedSponsor(company_name="   ", source_page=1)

    def test_empty_company_name_raises(self):
        with pytest.raises(ValidationError):
            ExtractedSponsor(company_name="", source_page=1)

    def test_source_page_zero_raises(self):
        with pytest.raises(ValidationError):
            ExtractedSponsor(company_name="MedCo", source_page=0)

    def test_source_page_negative_raises(self):
        with pytest.raises(ValidationError):
            ExtractedSponsor(company_name="MedCo", source_page=-1)

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValidationError):
            ExtractedSponsor(company_name="MedCo", source_page=1, confidence=1.01)

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValidationError):
            ExtractedSponsor(company_name="MedCo", source_page=1, confidence=-0.1)

    def test_extra_field_raises(self):
        """Extra fields must be rejected (forbid config)."""
        with pytest.raises(ValidationError):
            ExtractedSponsor(company_name="MedCo", source_page=1, rogue_field="oops")


# ===========================================================================
# EnrichedLead
# ===========================================================================

class TestEnrichedLead:

    # --- valid paths ---

    def test_minimal_valid_lead(self):
        """Only company_name and conference_name are truly mandatory."""
        lead = EnrichedLead(company_name="Medtronic", conference_name="HIMSS 2025")
        assert lead.enrichment_confidence == 0.0
        assert lead.phone is None
        assert lead.director_name is None

    def test_full_lead_with_phone(self, enriched_lead_with_phone: EnrichedLead):
        assert enriched_lead_with_phone.phone == "+919876543210"
        assert enriched_lead_with_phone.enrichment_confidence > 0.5

    def test_lead_without_phone(self, enriched_lead_no_phone: EnrichedLead):
        assert enriched_lead_no_phone.phone is None
        assert enriched_lead_no_phone.director_name is not None   # contact found, no phone

    def test_contact_missing_lead(self, enriched_lead_contact_missing: EnrichedLead):
        assert enriched_lead_contact_missing.enrichment_confidence == 0.0
        assert enriched_lead_contact_missing.director_name is None
        assert enriched_lead_contact_missing.phone is None

    def test_linkedin_url_is_valid_http_url(self, enriched_lead_with_phone: EnrichedLead):
        """Pydantic should parse the URL and make it accessible."""
        assert enriched_lead_with_phone.linkedin_url is not None
        assert "linkedin.com" in str(enriched_lead_with_phone.linkedin_url)

    def test_valid_international_phone(self):
        lead = EnrichedLead(
            company_name="X", conference_name="C", phone="+14155552671"
        )
        assert lead.phone == "+14155552671"

    # --- invalid paths ---

    def test_blank_company_name_raises(self):
        with pytest.raises(ValidationError):
            EnrichedLead(company_name="", conference_name="HIMSS 2025")

    def test_bad_email_raises(self):
        with pytest.raises(ValidationError, match="'@'"):
            EnrichedLead(
                company_name="Medtronic",
                conference_name="HIMSS 2025",
                email="not-an-email",
            )

    def test_bad_phone_raises(self):
        with pytest.raises(ValidationError, match="E.164"):
            EnrichedLead(
                company_name="Medtronic",
                conference_name="HIMSS 2025",
                phone="CALL-ME",
            )

    def test_invalid_linkedin_url_raises(self):
        with pytest.raises(ValidationError):
            EnrichedLead(
                company_name="Medtronic",
                conference_name="HIMSS 2025",
                linkedin_url="not-a-url",
            )

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            EnrichedLead(
                company_name="Medtronic",
                conference_name="HIMSS 2025",
                enrichment_confidence=1.5,
            )

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            EnrichedLead(company_name="X", conference_name="C", mystery="field")


# ===========================================================================
# ZohoPayload
# ===========================================================================

def _minimal_record(**overrides) -> dict:
    """Helper: build a record that satisfies all required Zoho fields."""
    base = {
        "Last_Name": "Sharma",
        "Company": "Medtronic",
        "Lead_Source": "Conference Sponsor Extraction",
        "Lead_Status": "New - Spark Enriched",
        "Conference_Name": "HIMSS 2025",
        "Sponsor_Tier": "Gold",
    }
    base.update(overrides)
    return base


class TestZohoPayload:

    # --- valid paths ---

    def test_minimal_valid_payload(self):
        payload = ZohoPayload(data=[_minimal_record()])
        assert len(payload.data) == 1
        assert payload.data[0]["Company"] == "Medtronic"

    def test_default_duplicate_check_fields(self):
        payload = ZohoPayload(data=[_minimal_record()])
        assert "Company" in payload.duplicate_check_fields

    def test_payload_with_wati_pending(self):
        """When phone is present, WATI_Status should be Pending."""
        record = _minimal_record(
            First_Name="Priya",
            Mobile="+919876543210",
            WATI_Status="Pending",
            WATI_Template_Key="medical_conference_sponsor_intro_v1",
        )
        payload = ZohoPayload(data=[record])
        assert payload.data[0]["WATI_Status"] == "Pending"

    def test_payload_with_wati_missing_phone(self):
        """When phone is absent, WATI_Status should indicate missing phone."""
        record = _minimal_record(WATI_Status="Not Sent - Missing Phone")
        payload = ZohoPayload(data=[record])
        assert "Missing Phone" in payload.data[0]["WATI_Status"]

    def test_payload_serialises_to_json_compatible_dict(self):
        """model_dump must produce a plain dict (for httpx json= kwarg)."""
        payload = ZohoPayload(data=[_minimal_record()])
        dumped = payload.model_dump()
        assert isinstance(dumped, dict)
        assert isinstance(dumped["data"], list)
        assert isinstance(dumped["data"][0], dict)

    # --- invalid paths ---

    def test_empty_data_list_raises(self):
        with pytest.raises(ValidationError, match="at least one record"):
            ZohoPayload(data=[])

    def test_missing_required_field_company_raises(self):
        record = _minimal_record()
        del record["Company"]
        with pytest.raises(ValidationError, match="missing required fields"):
            ZohoPayload(data=[record])

    def test_missing_required_field_last_name_raises(self):
        record = _minimal_record()
        del record["Last_Name"]
        with pytest.raises(ValidationError, match="missing required fields"):
            ZohoPayload(data=[record])

    def test_missing_conference_name_raises(self):
        record = _minimal_record()
        del record["Conference_Name"]
        with pytest.raises(ValidationError, match="missing required fields"):
            ZohoPayload(data=[record])

    def test_missing_sponsor_tier_raises(self):
        record = _minimal_record()
        del record["Sponsor_Tier"]
        with pytest.raises(ValidationError, match="missing required fields"):
            ZohoPayload(data=[record])

    def test_extra_field_on_payload_itself_raises(self):
        """ZohoPayload model itself should reject unknown top-level fields."""
        with pytest.raises(ValidationError):
            ZohoPayload(data=[_minimal_record()], surprise="boom")
