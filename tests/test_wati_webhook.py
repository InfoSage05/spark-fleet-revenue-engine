"""
tests/test_wati_webhook.py

Contract tests for:
  - spark_fleet.wati   (WatiApiError, build_wati_payload, render_whatsapp_template,
                        WatiDispatcher)
  - spark_fleet.webhook_server  (FastAPI POST /webhook/wati-dispatch)

All external I/O (httpx calls to WATI, Zoho) is mocked.
No real HTTP traffic is made anywhere in this file.

Test-first: these tests define the specification for the two modules.
They will show ImportError until the implementations are written.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from pydantic import ValidationError


# ===========================================================================
# Stage 4a – WATI payload builder (pure functions)
# ===========================================================================

class TestWatiPayloadBuilder:

    def test_valid_webhook_dict_maps_to_wati_payload(self):
        """
        A complete webhook dict must produce a dict that matches WATI's
        sendTemplateMessage API shape.
        """
        from spark_fleet.wati import build_wati_payload  # noqa: PLC0415

        webhook = {
            "lead_id":        "LEAD-001",
            "first_name":     "Priya",
            "company":        "Medtronic",
            "conference_name":"HIMSS 2025",
            "sponsor_tier":   "Gold",
            "phone":          "+919876543210",
        }
        payload = build_wati_payload(webhook)

        assert payload["phone_number"] == "+919876543210"
        assert payload["template_name"] == "medical_conference_sponsor_intro_v1"

        params = {p["name"]: p["value"] for p in payload["parameters"]}
        assert params["first_name"]      == "Priya"
        assert params["company"]         == "Medtronic"
        assert params["sponsor_tier"]    == "Gold"
        assert params["conference_name"] == "HIMSS 2025"

    def test_missing_phone_raises_validation_error(self):
        """
        If the webhook body has no 'phone' field, build_wati_payload must
        raise a clear error — not silently dispatch to the wrong number.
        """
        from spark_fleet.wati import build_wati_payload  # noqa: PLC0415

        incomplete = {
            "lead_id":        "LEAD-002",
            "company":        "Medtronic",
            "conference_name":"HIMSS 2025",
            "sponsor_tier":   "Gold",
            # phone deliberately omitted
        }
        with pytest.raises((ValueError, ValidationError)):
            build_wati_payload(incomplete)

    def test_missing_company_raises_validation_error(self):
        """company is required — template text cannot be rendered without it."""
        from spark_fleet.wati import build_wati_payload  # noqa: PLC0415

        with pytest.raises((ValueError, ValidationError)):
            build_wati_payload({
                "lead_id": "X", "conference_name": "C",
                "sponsor_tier": "Gold", "phone": "+1234567890",
            })

    def test_custom_template_key_is_respected(self):
        """If the webhook specifies a custom template_key, use it."""
        from spark_fleet.wati import build_wati_payload  # noqa: PLC0415

        webhook = {
            "lead_id":         "LEAD-003",
            "company":         "Philips",
            "conference_name": "HIMSS 2025",
            "sponsor_tier":    "Platinum",
            "phone":           "+14155552671",
            "wati_template_key": "custom_v2_template",
        }
        payload = build_wati_payload(webhook)
        assert payload["template_name"] == "custom_v2_template"


# ===========================================================================
# Stage 4b – WhatsApp template renderer
# ===========================================================================

class TestWhatsAppTemplateRenderer:

    def test_render_with_first_name(self):
        """Personalised greeting when first_name is provided."""
        from spark_fleet.wati import render_whatsapp_template  # noqa: PLC0415

        text = render_whatsapp_template("Priya", "Medtronic", "Gold", "HIMSS 2025")

        assert text.startswith("Hi Priya,")
        assert "Medtronic" in text
        assert "Gold" in text
        assert "HIMSS 2025" in text
        assert "AI" in text

    def test_render_without_first_name_uses_generic_greeting(self):
        """Graceful fallback when no first name is available."""
        from spark_fleet.wati import render_whatsapp_template  # noqa: PLC0415

        text = render_whatsapp_template(None, "Philips", "Platinum", "MedTech 2025")

        assert text.startswith("Hi,")
        assert "Philips" in text

    def test_render_empty_string_first_name_uses_generic_greeting(self):
        from spark_fleet.wati import render_whatsapp_template  # noqa: PLC0415

        text = render_whatsapp_template("", "Siemens", "Silver", "Arab Health 2025")
        assert text.startswith("Hi,")

    def test_company_appears_twice(self):
        """
        The template contains {company} in two places — the opening sentence
        and the closing question.
        """
        from spark_fleet.wati import render_whatsapp_template  # noqa: PLC0415

        text = render_whatsapp_template("Ana", "Roche", "Gold", "HIMSS 2025")
        assert text.count("Roche") == 2


# ===========================================================================
# Stage 4c – WatiDispatcher network resilience
# ===========================================================================

class TestWatiDispatcherResilience:

    def _make_response(self, status_code: int, body: dict | None = None):
        """Build a minimal mock httpx.Response."""
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = body or {"id": "msg-001", "status": "sent"}
        resp.text = str(body)
        return resp

    def test_successful_send_returns_response_body(self):
        """Happy path: WATI returns 200 → dispatcher returns decoded JSON."""
        from spark_fleet.wati import WatiDispatcher  # noqa: PLC0415

        dispatcher = WatiDispatcher(
            base_url="http://wati.local", api_token="tok", max_retries=0
        )
        wati_payload = {
            "template_name": "medical_conference_sponsor_intro_v1",
            "phone_number":  "+919876543210",
            "parameters":    [{"name": "first_name", "value": "Priya"}],
        }
        with patch("httpx.post", return_value=self._make_response(200)) as mock_post:
            result = dispatcher.send(wati_payload)

        mock_post.assert_called_once()
        assert result["id"] == "msg-001"

    def test_wati_429_triggers_retry_then_succeeds(self):
        """
        First call → 429; second call → 200.
        Dispatcher must retry and return the successful response.
        """
        from spark_fleet.wati import WatiDispatcher  # noqa: PLC0415

        dispatcher = WatiDispatcher(
            base_url="http://wati.local",
            api_token="tok",
            max_retries=2,
            retry_delay_s=0.0,   # no real sleeping in tests
        )
        wati_payload = {"template_name": "t", "phone_number": "+1234567890", "parameters": []}

        responses = [
            self._make_response(429, {}),
            self._make_response(200, {"id": "msg-002", "status": "sent"}),
        ]
        with patch("httpx.post", side_effect=responses):
            with patch("time.sleep"):        # suppress actual back-off delay
                result = dispatcher.send(wati_payload)

        assert result["id"] == "msg-002"

    def test_wati_400_fails_immediately_no_retry(self):
        """
        A 400 Bad Request is a permanent error — dispatcher must raise
        WatiApiError(retryable=False) and must NOT retry.
        """
        from spark_fleet.wati import WatiApiError, WatiDispatcher  # noqa: PLC0415

        dispatcher = WatiDispatcher(
            base_url="http://wati.local", api_token="tok", max_retries=3,
            retry_delay_s=0.0,
        )
        wati_payload = {"template_name": "t", "phone_number": "+1234567890", "parameters": []}

        with patch("httpx.post", return_value=self._make_response(400, {})) as mock_post:
            with pytest.raises(WatiApiError) as exc_info:
                dispatcher.send(wati_payload)

        # Must have given up after exactly 1 attempt, not retried.
        assert mock_post.call_count == 1
        assert exc_info.value.retryable is False

    def test_wati_429_exhausts_retries_raises_retryable_error(self):
        """
        If WATI keeps returning 429 until retries are exhausted, raise
        WatiApiError(retryable=True) so the task queue can back off.
        """
        from spark_fleet.wati import WatiApiError, WatiDispatcher  # noqa: PLC0415

        max_retries = 2
        dispatcher = WatiDispatcher(
            base_url="http://wati.local", api_token="tok",
            max_retries=max_retries, retry_delay_s=0.0,
        )
        wati_payload = {"template_name": "t", "phone_number": "+1234567890", "parameters": []}

        with patch("httpx.post", return_value=self._make_response(429, {})):
            with patch("time.sleep"):
                with pytest.raises(WatiApiError) as exc_info:
                    dispatcher.send(wati_payload)

        assert exc_info.value.retryable is True


# ===========================================================================
# Stage 4d – FastAPI webhook endpoint
# ===========================================================================

class TestWebhookEndpoint:
    """
    Tests for POST /webhook/wati-dispatch using FastAPI's TestClient.

    Dependencies (WatiDispatcher, ZohoCRMClient) are overridden via
    FastAPI's dependency_overrides so no real HTTP calls are made.
    """

    @pytest.fixture()
    def client(self):
        """
        Build a TestClient with both dependencies replaced by mocks.
        The WatiDispatcher mock returns a successful WATI response.
        The ZohoCRMClient mock silently accepts the status update.
        """
        from fastapi.testclient import TestClient                  # noqa: PLC0415
        from spark_fleet.webhook_server import (                   # noqa: PLC0415
            app,
            get_wati_dispatcher,
            get_zoho_client,
        )

        mock_wati = MagicMock()
        mock_wati.send.return_value = {"id": "msg-999", "status": "sent"}

        mock_zoho = MagicMock()
        mock_zoho.update_lead.return_value = {"data": [{"status": "success"}]}

        app.dependency_overrides[get_wati_dispatcher] = lambda: mock_wati
        app.dependency_overrides[get_zoho_client]     = lambda: mock_zoho

        with TestClient(app) as c:
            yield c

        app.dependency_overrides.clear()

    def _valid_body(self) -> dict:
        return {
            "lead_id":        "LEAD-010",
            "first_name":     "James",
            "company":        "Philips Healthcare",
            "conference_name":"HIMSS 2025",
            "sponsor_tier":   "Platinum",
            "phone":          "+14155552671",
        }

    def test_returns_200_on_valid_payload(self, client):
        """Valid webhook body → HTTP 200 and a success body."""
        response = client.post("/webhook/wati-dispatch", json=self._valid_body())

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "sent"
        assert body["lead_id"] == "LEAD-010"

    def test_returns_422_when_phone_is_missing(self, client):
        """Missing required field → FastAPI returns 422 Unprocessable Entity."""
        bad_body = self._valid_body()
        del bad_body["phone"]

        response = client.post("/webhook/wati-dispatch", json=bad_body)
        assert response.status_code == 422

    def test_returns_422_when_body_is_empty(self, client):
        """Empty JSON body → 422."""
        response = client.post("/webhook/wati-dispatch", json={})
        assert response.status_code == 422

    def test_returns_422_when_company_is_missing(self, client):
        """Missing company → 422."""
        bad_body = self._valid_body()
        del bad_body["company"]

        response = client.post("/webhook/wati-dispatch", json=bad_body)
        assert response.status_code == 422

    def test_wati_dispatcher_is_called_with_correct_phone(self, client):
        """
        The WatiDispatcher mock's .send() must be called with a WATI payload
        that contains the correct phone number from the webhook body.
        """
        from spark_fleet.webhook_server import app, get_wati_dispatcher  # noqa: PLC0415
        from fastapi.testclient import TestClient as _TC                  # noqa: PLC0415

        captured_wati = MagicMock()
        captured_wati.send.return_value = {"id": "msg-777", "status": "sent"}

        app.dependency_overrides[get_wati_dispatcher] = lambda: captured_wati

        with _TC(app) as c:
            c.post("/webhook/wati-dispatch", json=self._valid_body())

        app.dependency_overrides.clear()

        captured_wati.send.assert_called_once()
        wati_payload = captured_wati.send.call_args[0][0]   # first positional arg
        assert wati_payload["phone_number"] == "+14155552671"

    def test_zoho_update_is_called_with_sent_status(self, client):
        from spark_fleet.webhook_server import app, get_zoho_client  # noqa: PLC0415
        from fastapi.testclient import TestClient as _TC             # noqa: PLC0415

        captured_zoho = MagicMock()
        captured_zoho.update_lead.return_value = {"data": [{"status": "success"}]}

        app.dependency_overrides[get_zoho_client] = lambda: captured_zoho

        with _TC(app) as c:
            c.post("/webhook/wati-dispatch", json=self._valid_body())

        app.dependency_overrides.clear()

        captured_zoho.update_lead.assert_called_once_with(
            "LEAD-010",
            {"WATI_Status": "Sent"},
        )


class TestApolloWebhookEndpoint:

    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient                  # noqa: PLC0415
        from spark_fleet.webhook_server import (                   # noqa: PLC0415
            app,
            get_zoho_client,
        )

        mock_zoho = MagicMock()
        mock_zoho.find_lead.return_value = {
            "id": "LEAD-APOLLO-001",
            "Company": "Medtronic",
            "Conference_Name": "HIMSS 2025",
            "WATI_Status": "Not Sent - Missing Phone",
        }
        mock_zoho.update_lead.return_value = {"data": [{"status": "success"}]}

        app.dependency_overrides[get_zoho_client] = lambda: mock_zoho

        with TestClient(app) as c:
            yield c, mock_zoho

        app.dependency_overrides.clear()

    def test_apollo_webhook_updates_lead_and_sets_pending(self, client):
        tc, mock_zoho = client
        payload = {
            "status": "success",
            "people": [
                {
                    "first_name": "Priya",
                    "last_name": "Sharma",
                    "title": "Marketing Director",
                    "linkedin_url": "https://www.linkedin.com/in/priya-sharma",
                    "phone_numbers": [{"sanitized_number": "+919876543210"}],
                    "emails": [{"email": "priya@medtronic.com"}],
                }
            ],
        }

        response = tc.post(
            "/webhook/apollo-enrichment?company_name=Medtronic&conference_name=HIMSS%202025&sponsor_tier=Gold",
            json=payload,
        )

        assert response.status_code == 200
        assert response.json()["status"] == "updated"
        mock_zoho.find_lead.assert_called_once()
        mock_zoho.update_lead.assert_called_once_with(
            "LEAD-APOLLO-001",
            {
                "Mobile": "+919876543210",
                "WATI_Status": "Pending",
                "Email": "priya@medtronic.com",
                "First_Name": "Priya",
                "Last_Name": "Sharma",
                "Designation": "Marketing Director",
                "LinkedIn_Profile": "https://www.linkedin.com/in/priya-sharma",
            },
        )
