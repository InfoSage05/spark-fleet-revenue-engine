"""
tests/test_e2e_pipeline.py

End-to-end integration tests for the Spark Fleet autonomous pipeline.

NO real network calls are made anywhere in this file.
All four HTTP boundaries are mocked at the module level:

  ┌────────────────────────────────────────────────────────────────────┐
  │  HTTP boundary          Mock target                                 │
  ├────────────────────────────────────────────────────────────────────┤
  │  Macro Spark DGX        spark_fleet.macro_client.httpx.post        │
  │  Proxycurl LinkedIn     PeopleSearchProvider (MagicMock object)    │
  │  Zoho CRM API           spark_fleet.zoho.httpx.post               │
  │  WATI WhatsApp API      WatiDispatcher (FastAPI dependency_override)│
  └────────────────────────────────────────────────────────────────────┘

Real code that runs at every stage (nothing skipped):
  pdf_parser.extract_text_from_bytes     (if PyMuPDF installed, else text fixture)
  macro_client.build_extraction_prompt
  macro_client.parse_macro_response
  macro_client.filter_sponsors
  macro_client.dedupe_sponsors
  enrichment.EnrichmentOrchestrator.enrich
  zoho.map_lead_to_zoho_payload          (pure function, no mock needed)
  webhook_server  POST /webhook/wati-dispatch  (FastAPI TestClient)

Data flow validated:
  brochure text
    → ExtractedSponsor(company_name="Medtronic", tier="Gold", …)
    → EnrichedLead(director_name="Priya Sharma", phone="+919876543210", …)
    → ZohoPayload(WATI_Status="Pending", Company="Medtronic", …)
    → HTTP 200 ack  +  WatiDispatcher.send called with correct phone
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from spark_fleet.schemas import EnrichedLead, ExtractedSponsor, ZohoPayload


# ===========================================================================
# Shared fixture data — defines what flows through the pipeline
# ===========================================================================

_CONFERENCE_NAME  = "HIMSS 2025"
_COMPANY_NAME     = "Medtronic"
_SPONSOR_TIER     = "Gold"
_SOURCE_PAGE      = 1
_DIRECTOR_NAME    = "Priya Sharma"
_DIRECTOR_PHONE   = "+919876543210"
_DIRECTOR_EMAIL   = "priya.sharma@medtronic.com"
_LINKEDIN_URL     = "https://www.linkedin.com/in/priya-sharma-medtronic"
_CONFIDENCE_HIGH  = 0.92

# Raw brochure text — the text the pdf_parser would hand to the macro_client.
_BROCHURE_TEXT = (
    "HIMSS 2025 Annual Conference – Exhibitor & Sponsor Guide\n\n"
    "GOLD SPONSORS\n"
    "Medtronic – Advancing medical technology for better patient outcomes.\n\n"
    "PLATINUM SPONSORS\n"
    "Philips Healthcare – Precision diagnostics and patient monitoring.\n\n"
    "SILVER SPONSORS\n"
    "AcmePharma – an exhibitor, not a sponsor (low confidence signal).\n"
)

# The JSON string the DGX LLM returns in choices[0].message.content
_DGX_LLM_JSON = json.dumps({
    "sponsors": [
        {
            "company_name":  _COMPANY_NAME,
            "sponsor_tier":  _SPONSOR_TIER,
            "source_page":   _SOURCE_PAGE,
            "evidence_text": "Medtronic – Gold Sponsor of HIMSS 2025",
            "confidence":    _CONFIDENCE_HIGH,
        },
        {
            "company_name":  "Philips Healthcare",
            "sponsor_tier":  "Platinum",
            "source_page":   1,
            "evidence_text": "Philips Healthcare – Platinum Sponsor",
            "confidence":    0.88,
        },
        {
            "company_name":  "AcmePharma",
            "sponsor_tier":  "Silver",
            "source_page":   3,
            "evidence_text": "AcmePharma – exhibitor",
            "confidence":    0.28,   # ← below 0.5 threshold — must be filtered
        },
    ]
})


# ===========================================================================
# Re-usable mock builders
# ===========================================================================

def _make_dgx_mock_response(llm_json: str = _DGX_LLM_JSON) -> MagicMock:
    """Build a mock httpx Response simulating a successful DGX reply."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": llm_json}}]
    }
    return resp


def _make_zoho_mock_response(lead_id: str = "ZOHO-LEAD-001") -> MagicMock:
    """Build a mock httpx Response simulating a successful Zoho CRM push."""
    resp = MagicMock()
    resp.status_code = 201
    resp.json.return_value = {
        "data": [{"status": "success", "details": {"id": lead_id}}]
    }
    return resp


def _make_proxycurl_provider(
    full_name: str = _DIRECTOR_NAME,
    phone: str = _DIRECTOR_PHONE,
    confidence: float = 0.9,
) -> MagicMock:
    """Build a mock PeopleSearchProvider that returns a successful PersonResult."""
    from spark_fleet.enrichment import PersonResult  # noqa: PLC0415

    result = PersonResult(
        full_name=full_name,
        title="Marketing Director",
        linkedin_url=_LINKEDIN_URL,
        email=_DIRECTOR_EMAIL,
        phone=phone,
        confidence=confidence,
    )
    provider = MagicMock()
    provider.find_marketing_director.return_value = result
    return provider


# ===========================================================================
# Stage 1 – PDF text extraction
# ===========================================================================

class TestE2EStage1PdfExtraction:
    """
    Verify that pdf_parser correctly extracts text from a real in-memory PDF.
    These tests skip gracefully if PyMuPDF is not installed.
    """

    @pytest.fixture(scope="class")
    def real_pdf_bytes(self) -> bytes:
        """Generate a minimal conference brochure PDF using PyMuPDF itself."""
        fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")
        doc  = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 700), _BROCHURE_TEXT[:200])   # page 1
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes

    def test_pdf_extracts_at_least_one_page(self, real_pdf_bytes: bytes) -> None:
        from spark_fleet.pdf_parser import extract_text_from_bytes  # noqa: PLC0415

        pages = extract_text_from_bytes(real_pdf_bytes)

        assert len(pages) >= 1
        assert pages[0].page == 1

    def test_extracted_text_contains_sponsor_keyword(self, real_pdf_bytes: bytes) -> None:
        from spark_fleet.pdf_parser import extract_text_from_bytes  # noqa: PLC0415

        pages = extract_text_from_bytes(real_pdf_bytes)
        combined = " ".join(p.text for p in pages)

        assert "Medtronic" in combined or "GOLD" in combined, (
            f"Expected sponsor keyword in extracted text, got: {combined!r}"
        )


# ===========================================================================
# Stage 2 – Macro Spark LLM extraction
# ===========================================================================

class TestE2EStage2MacroSparkExtraction:
    """
    Verify that MacroSparkClient correctly transforms brochure text into
    validated, filtered, deduplicated ExtractedSponsor objects.
    The DGX HTTP call is mocked.
    """

    @pytest.fixture()
    def sponsors(self) -> list[ExtractedSponsor]:
        from spark_fleet.macro_client import MacroSparkClient  # noqa: PLC0415

        client = MacroSparkClient(base_url="http://dgx.local:8000")
        with patch("spark_fleet.macro_client.httpx.post", return_value=_make_dgx_mock_response()):
            return client.extract_sponsors(
                pdf_text=_BROCHURE_TEXT,
                conference_name=_CONFERENCE_NAME,
                min_confidence=0.5,
            )

    def test_returns_only_high_confidence_sponsors(self, sponsors: list[ExtractedSponsor]) -> None:
        """AcmePharma (confidence=0.28) must be filtered out."""
        company_names = [s.company_name for s in sponsors]
        assert _COMPANY_NAME         in company_names
        assert "Philips Healthcare"   in company_names
        assert "AcmePharma"          not in company_names, (
            "AcmePharma has confidence 0.28 and must be filtered."
        )

    def test_all_returned_sponsors_are_validated_pydantic_models(
        self, sponsors: list[ExtractedSponsor]
    ) -> None:
        for s in sponsors:
            assert isinstance(s, ExtractedSponsor)
            assert s.confidence >= 0.5
            assert s.source_page >= 1

    def test_medtronic_has_correct_tier(self, sponsors: list[ExtractedSponsor]) -> None:
        medtronic = next(s for s in sponsors if s.company_name == _COMPANY_NAME)
        assert medtronic.sponsor_tier == _SPONSOR_TIER

    def test_no_duplicates_in_result(self, sponsors: list[ExtractedSponsor]) -> None:
        names = [s.company_name for s in sponsors]
        assert len(names) == len(set(names)), "Duplicate company names found in result."

    def test_brochure_contact_regex_fallback_attaches_contacts(self) -> None:
        from spark_fleet.macro_client import MacroSparkClient  # noqa: PLC0415

        llm_json = json.dumps(
            {
                "sponsors": [
                    {
                        "company_name": "Medtronic",
                        "sponsor_tier": "Gold",
                        "source_page": 1,
                        "confidence": 0.95,
                    }
                ]
            }
        )
        brochure_text = (
            "Gold Sponsors: Medtronic. Contact us at partnerships@medtronic.com "
            "or +1 (415) 555-2671 for sponsor partnerships."
        )

        client = MacroSparkClient(base_url="http://dgx.local:8000")
        with patch("spark_fleet.macro_client.httpx.post", return_value=_make_dgx_mock_response(llm_json)):
            sponsors = client.extract_sponsors(
                pdf_text=brochure_text,
                conference_name=_CONFERENCE_NAME,
                min_confidence=0.5,
            )

        assert len(sponsors) == 1
        assert sponsors[0].brochure_emails == ["partnerships@medtronic.com"]
        assert sponsors[0].brochure_phones == ["+14155552671"]


# ===========================================================================
# Stage 3 – Enrichment (Proxycurl provider mocked)
# ===========================================================================

class TestE2EStage3Enrichment:
    """
    Verify that EnrichmentOrchestrator correctly converts an ExtractedSponsor
    into a full EnrichedLead when the provider returns a result.
    """

    @pytest.fixture()
    def medtronic_sponsor(self) -> ExtractedSponsor:
        return ExtractedSponsor(
            company_name=_COMPANY_NAME,
            sponsor_tier=_SPONSOR_TIER,
            source_page=_SOURCE_PAGE,
            confidence=_CONFIDENCE_HIGH,
        )

    @pytest.fixture()
    def enriched_lead(self, medtronic_sponsor: ExtractedSponsor) -> EnrichedLead:
        from spark_fleet.enrichment import EnrichmentOrchestrator  # noqa: PLC0415

        provider     = _make_proxycurl_provider()
        orchestrator = EnrichmentOrchestrator(people_provider=provider)
        return orchestrator.enrich(
            sponsor=medtronic_sponsor,
            conference_name=_CONFERENCE_NAME,
        )

    def test_lead_is_valid_enriched_lead(self, enriched_lead: EnrichedLead) -> None:
        assert isinstance(enriched_lead, EnrichedLead)

    def test_lead_carries_director_name(self, enriched_lead: EnrichedLead) -> None:
        assert enriched_lead.director_name == _DIRECTOR_NAME

    def test_lead_carries_phone(self, enriched_lead: EnrichedLead) -> None:
        assert enriched_lead.phone == _DIRECTOR_PHONE

    def test_lead_enrichment_confidence_above_threshold(self, enriched_lead: EnrichedLead) -> None:
        assert enriched_lead.enrichment_confidence > 0.5

    def test_lead_preserves_company_name(self, enriched_lead: EnrichedLead) -> None:
        assert enriched_lead.company_name == _COMPANY_NAME

    def test_lead_preserves_conference_name(self, enriched_lead: EnrichedLead) -> None:
        assert enriched_lead.conference_name == _CONFERENCE_NAME


# ===========================================================================
# Stage 4 – Zoho payload mapping (pure function, no mock needed)
# ===========================================================================

class TestE2EStage4ZohoMapping:
    """
    Verify that map_lead_to_zoho_payload produces a valid, correctly-shaped
    ZohoPayload from the EnrichedLead.  Pure function — no HTTP mocks needed.
    """

    @pytest.fixture()
    def payload(self) -> ZohoPayload:
        from spark_fleet.zoho import map_lead_to_zoho_payload  # noqa: PLC0415

        lead = EnrichedLead(
            company_name         = _COMPANY_NAME,
            director_name        = _DIRECTOR_NAME,
            director_title       = "Marketing Director",
            linkedin_url         = _LINKEDIN_URL,
            email                = _DIRECTOR_EMAIL,
            phone                = _DIRECTOR_PHONE,
            enrichment_confidence= _CONFIDENCE_HIGH,
            sponsor_tier         = _SPONSOR_TIER,
            conference_name      = _CONFERENCE_NAME,
            source_page          = _SOURCE_PAGE,
        )
        return map_lead_to_zoho_payload(lead)

    def test_payload_is_valid_zoho_payload(self, payload: ZohoPayload) -> None:
        assert isinstance(payload, ZohoPayload)

    def test_wati_status_is_pending_when_phone_present(self, payload: ZohoPayload) -> None:
        assert payload.data[0]["WATI_Status"] == "Pending"

    def test_company_and_tier_are_correct(self, payload: ZohoPayload) -> None:
        record = payload.data[0]
        assert record["Company"]      == _COMPANY_NAME
        assert record["Sponsor_Tier"] == _SPONSOR_TIER

    def test_mobile_field_set_correctly(self, payload: ZohoPayload) -> None:
        assert payload.data[0]["Mobile"] == _DIRECTOR_PHONE

    def test_linkedin_url_is_plain_string(self, payload: ZohoPayload) -> None:
        val = payload.data[0].get("LinkedIn_Profile")
        if val is not None:
            assert isinstance(val, str), "LinkedIn_Profile must be a plain string for Zoho."

    def test_all_required_fields_present(self, payload: ZohoPayload) -> None:
        record = payload.data[0]
        required = ("Last_Name", "Company", "Lead_Source", "Lead_Status",
                    "Conference_Name", "Sponsor_Tier", "WATI_Status")
        for field in required:
            assert field in record, f"Required Zoho field missing: {field}"


# ===========================================================================
# Stage 5 – Zoho CRM HTTP push (httpx mocked)
# ===========================================================================

class TestE2EStage5ZohoPush:
    """
    Verify that ZohoCRMClient.push() correctly POSTs the payload and returns
    the decoded Zoho response.  The actual httpx.post is mocked.
    """

    @pytest.fixture()
    def zoho_payload(self) -> ZohoPayload:
        from spark_fleet.zoho import map_lead_to_zoho_payload  # noqa: PLC0415

        lead = EnrichedLead(
            company_name          = _COMPANY_NAME,
            director_name         = _DIRECTOR_NAME,
            phone                 = _DIRECTOR_PHONE,
            enrichment_confidence = _CONFIDENCE_HIGH,
            sponsor_tier          = _SPONSOR_TIER,
            conference_name       = _CONFERENCE_NAME,
            source_page           = _SOURCE_PAGE,
        )
        return map_lead_to_zoho_payload(lead)

    def test_push_returns_success_body(self, zoho_payload: ZohoPayload) -> None:
        from spark_fleet.zoho import ZohoCRMClient, StaticTokenProvider  # noqa: PLC0415

        client = ZohoCRMClient(
            token_provider=StaticTokenProvider("test-token"),
            max_retries=0,
        )
        mock_resp = _make_zoho_mock_response()

        with patch("spark_fleet.zoho.httpx.post", return_value=mock_resp) as mock_post:
            result = client.push(zoho_payload)

        mock_post.assert_called_once()
        assert result["data"][0]["status"] == "success"

    def test_push_sends_correct_authorization_header(self, zoho_payload: ZohoPayload) -> None:
        from spark_fleet.zoho import ZohoCRMClient, StaticTokenProvider  # noqa: PLC0415

        client = ZohoCRMClient(
            token_provider=StaticTokenProvider("my-secret-token"),
            max_retries=0,
        )
        with patch("spark_fleet.zoho.httpx.post", return_value=_make_zoho_mock_response()) as mock_post:
            client.push(zoho_payload)

        _, call_kwargs = mock_post.call_args
        auth_header = call_kwargs["headers"]["Authorization"]
        assert "my-secret-token" in auth_header

    def test_push_json_contains_wati_status_pending(self, zoho_payload: ZohoPayload) -> None:
        from spark_fleet.zoho import ZohoCRMClient, StaticTokenProvider  # noqa: PLC0415

        client = ZohoCRMClient(token_provider=StaticTokenProvider("tok"), max_retries=0)
        with patch("spark_fleet.zoho.httpx.post", return_value=_make_zoho_mock_response()) as mock_post:
            client.push(zoho_payload)

        sent_body = mock_post.call_args[1]["json"]
        assert sent_body["data"][0]["WATI_Status"] == "Pending"


# ===========================================================================
# Stage 6 – Webhook → WATI dispatch (FastAPI TestClient)
# ===========================================================================

class TestE2EStage6WebhookDispatch:
    """
    Verify that the FastAPI /webhook/wati-dispatch endpoint:
      - Returns 200 when WATI dispatches successfully.
      - Calls WatiDispatcher.send() with the correct phone number.
      - Calls ZohoCRMClient.push() to update WATI_Status.
    """

    @pytest.fixture()
    def client_with_mocks(self):
        """TestClient with WatiDispatcher and ZohoCRMClient overridden."""
        from fastapi.testclient import TestClient              # noqa: PLC0415
        from spark_fleet.webhook_server import (               # noqa: PLC0415
            app,
            get_wati_dispatcher,
            get_zoho_client,
        )

        mock_wati          = MagicMock()
        mock_wati.send.return_value = {"id": "wati-msg-e2e", "status": "sent"}

        mock_zoho          = MagicMock()
        mock_zoho.update_lead.return_value = {"data": [{"status": "success"}]}

        app.dependency_overrides[get_wati_dispatcher] = lambda: mock_wati
        app.dependency_overrides[get_zoho_client]     = lambda: mock_zoho

        with TestClient(app) as tc:
            yield tc, mock_wati, mock_zoho

        app.dependency_overrides.clear()

    def _webhook_body(self, phone: str = _DIRECTOR_PHONE) -> dict:
        return {
            "lead_id":        "LEAD-E2E-001",
            "first_name":     "Priya",
            "company":        _COMPANY_NAME,
            "conference_name": _CONFERENCE_NAME,
            "sponsor_tier":   _SPONSOR_TIER,
            "phone":          phone,
        }

    def test_endpoint_returns_200_on_success(self, client_with_mocks) -> None:
        tc, mock_wati, _ = client_with_mocks
        response = tc.post("/webhook/wati-dispatch", json=self._webhook_body())
        assert response.status_code == 200

    def test_response_body_contains_sent_status(self, client_with_mocks) -> None:
        tc, mock_wati, _ = client_with_mocks
        body = tc.post("/webhook/wati-dispatch", json=self._webhook_body()).json()
        assert body["status"] == "sent"
        assert body["lead_id"] == "LEAD-E2E-001"

    def test_wati_dispatcher_called_with_correct_phone(self, client_with_mocks) -> None:
        tc, mock_wati, _ = client_with_mocks
        tc.post("/webhook/wati-dispatch", json=self._webhook_body())

        mock_wati.send.assert_called_once()
        wati_payload = mock_wati.send.call_args[0][0]
        assert wati_payload["phone_number"] == _DIRECTOR_PHONE

    def test_zoho_client_push_called_to_update_status(self, client_with_mocks) -> None:
        tc, _, mock_zoho = client_with_mocks
        tc.post("/webhook/wati-dispatch", json=self._webhook_body())

        mock_zoho.update_lead.assert_called_once_with(
            "LEAD-E2E-001",
            {"WATI_Status": "Sent"},
        )

    def test_endpoint_returns_422_when_phone_missing(self, client_with_mocks) -> None:
        tc, _, _ = client_with_mocks
        bad_body = self._webhook_body()
        del bad_body["phone"]
        assert tc.post("/webhook/wati-dispatch", json=bad_body).status_code == 422


# ===========================================================================
# Full pipeline chain — all stages wired sequentially
# ===========================================================================

class TestE2EFullPipelineFlow:
    """
    The golden-path test: traces a conference brochure through all six stages
    in sequence.  Every stage's output is the next stage's input.

    Assertions are made at the FINAL output only — this is the definitive
    proof that the whole system works end-to-end.
    """

    def test_full_pipeline_from_text_to_wati_dispatch(self) -> None:
        """
        GIVEN  a conference brochure text
        WHEN   the full Spark Fleet pipeline processes it
        THEN   a WhatsApp message is dispatched to the correct phone number
               and Zoho CRM is updated with WATI_Status = 'Sent'.
        """
        from spark_fleet.macro_client import MacroSparkClient         # noqa: PLC0415
        from spark_fleet.enrichment   import EnrichmentOrchestrator   # noqa: PLC0415
        from spark_fleet.zoho         import (                         # noqa: PLC0415
            ZohoCRMClient, StaticTokenProvider, map_lead_to_zoho_payload
        )
        from fastapi.testclient       import TestClient                # noqa: PLC0415
        from spark_fleet.webhook_server import (                       # noqa: PLC0415
            app, get_wati_dispatcher, get_zoho_client,
        )

        # ── Stage 1: brochure text (use the pre-baked fixture directly) ──────
        brochure_text = _BROCHURE_TEXT

        # ── Stage 2: Macro Spark extraction (DGX mocked) ─────────────────────
        macro_client = MacroSparkClient(base_url="http://dgx.local:8000")
        with patch("spark_fleet.macro_client.httpx.post",
                   return_value=_make_dgx_mock_response()):
            sponsors = macro_client.extract_sponsors(
                pdf_text=brochure_text,
                conference_name=_CONFERENCE_NAME,
                min_confidence=0.5,
            )

        assert len(sponsors) == 2, f"Expected 2 sponsors (AcmePharma filtered), got {len(sponsors)}"
        medtronic = next(s for s in sponsors if s.company_name == _COMPANY_NAME)

        # ── Stage 3: Enrichment (Proxycurl provider mocked) ──────────────────
        provider     = _make_proxycurl_provider()
        orchestrator = EnrichmentOrchestrator(people_provider=provider)
        lead         = orchestrator.enrich(
            sponsor=medtronic,
            conference_name=_CONFERENCE_NAME,
        )

        assert lead.director_name        == _DIRECTOR_NAME
        assert lead.phone                == _DIRECTOR_PHONE
        assert lead.enrichment_confidence > 0.5

        # ── Stage 4: Zoho payload mapping (pure function) ─────────────────────
        zoho_payload = map_lead_to_zoho_payload(lead)
        assert isinstance(zoho_payload, ZohoPayload)
        assert zoho_payload.data[0]["WATI_Status"] == "Pending"

        # ── Stage 5: Zoho CRM push (httpx mocked) ────────────────────────────
        zoho_client = ZohoCRMClient(
            token_provider=StaticTokenProvider("e2e-test-token"),
            max_retries=0,
        )
        with patch("spark_fleet.zoho.httpx.post",
                   return_value=_make_zoho_mock_response("LEAD-E2E-FULL")):
            zoho_result = zoho_client.push(zoho_payload)

        assert zoho_result["data"][0]["details"]["id"] == "LEAD-E2E-FULL"

        # ── Stage 6: Webhook → WATI (FastAPI TestClient + mocked dispatcher) ──
        mock_wati = MagicMock()
        mock_wati.send.return_value = {"id": "msg-e2e-final", "status": "sent"}

        mock_zoho_client = MagicMock()
        mock_zoho_client.update_lead.return_value = {"data": [{"status": "success"}]}

        app.dependency_overrides[get_wati_dispatcher] = lambda: mock_wati
        app.dependency_overrides[get_zoho_client]     = lambda: mock_zoho_client

        try:
            with TestClient(app) as tc:
                webhook_body = {
                    "lead_id":         "LEAD-E2E-FULL",
                    "first_name":      "Priya",
                    "company":         lead.company_name,
                    "conference_name": lead.conference_name,
                    "sponsor_tier":    lead.sponsor_tier,
                    "phone":           lead.phone,
                }
                resp = tc.post("/webhook/wati-dispatch", json=webhook_body)
        finally:
            app.dependency_overrides.clear()

        # ── Final assertions ──────────────────────────────────────────────────
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

        resp_body = resp.json()
        assert resp_body["status"] == "sent"
        assert resp_body["lead_id"] == "LEAD-E2E-FULL"

        # The WATI dispatcher must have received the correct phone number.
        mock_wati.send.assert_called_once()
        dispatched_payload = mock_wati.send.call_args[0][0]
        assert dispatched_payload["phone_number"] == _DIRECTOR_PHONE, (
            f"WATI received wrong phone. Expected {_DIRECTOR_PHONE!r}, "
            f"got {dispatched_payload.get('phone_number')!r}"
        )

        # Zoho must have been updated with WATI_Status = Sent.
        mock_zoho_client.update_lead.assert_called_once_with(
            "LEAD-E2E-FULL",
            {"WATI_Status": "Sent"},
        )
