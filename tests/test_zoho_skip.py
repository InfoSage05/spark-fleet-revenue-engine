from __future__ import annotations

from unittest.mock import MagicMock, patch


def _response(status_code: int, body: dict | None = None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body or {}
    resp.text = str(body or {})
    return resp


def test_has_conference_leads_true_when_zoho_returns_records():
    from spark_fleet.zoho import StaticTokenProvider, ZohoCRMClient

    client = ZohoCRMClient(token_provider=StaticTokenProvider("token"))
    with patch("spark_fleet.zoho.httpx.get", return_value=_response(200, {"data": [{"id": "1"}]})):
        assert client.has_conference_leads("brochure.pdf") is True


def test_has_conference_leads_false_on_204():
    from spark_fleet.zoho import StaticTokenProvider, ZohoCRMClient

    client = ZohoCRMClient(token_provider=StaticTokenProvider("token"))
    with patch("spark_fleet.zoho.httpx.get", return_value=_response(204)):
        assert client.has_conference_leads("brochure.pdf") is False
