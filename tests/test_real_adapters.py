"""
tests/test_real_adapters.py

Contract tests for the three real-world adapters introduced in Prompt 6:

  1. PDF Parser        (spark_fleet.pdf_parser)
  2. Zoho OAuth Refresh (spark_fleet.zoho.RefreshingTokenProvider)
  3. Proxycurl LinkedIn Adapter (spark_fleet.adapters.proxycurl_provider)

All external I/O is mocked.  PyMuPDF tests skip gracefully if the library
is not installed (use `pip install PyMuPDF` on the Mac Mini to run them).

Test-first: these tests are the specification for the implementations.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch


# ===========================================================================
# 1. PDF Parser
# ===========================================================================

class TestPdfParser:

    def test_missing_fitz_raises_missing_pdf_dependency(self):
        """
        If PyMuPDF is not installed, extract_text_from_bytes must raise
        MissingPdfDependency — not a raw ImportError or AttributeError.
        """
        from spark_fleet.pdf_parser import (    # noqa: PLC0415
            MissingPdfDependency,
            extract_text_from_bytes,
        )
        import builtins
        real_import = builtins.__import__

        def patched_import(name, *args, **kwargs):
            if name == "fitz":
                raise ModuleNotFoundError("No module named 'fitz'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=patched_import):
            with pytest.raises(MissingPdfDependency, match="PyMuPDF"):
                extract_text_from_bytes(b"%PDF-1.4 fake")

    def test_extract_returns_list_of_page_text(self):
        """
        extract_text_from_bytes must return a list with one PageText per page.
        Requires PyMuPDF to be installed.
        """
        fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")
        from spark_fleet.pdf_parser import extract_text_from_bytes, PageText  # noqa: PLC0415

        # Build a minimal PDF in memory using fitz itself
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Medtronic – Gold Sponsor of HIMSS 2025")
        pdf_bytes = doc.tobytes()
        doc.close()

        pages = extract_text_from_bytes(pdf_bytes)

        assert len(pages) == 1
        assert isinstance(pages[0], PageText)
        assert pages[0].page == 1
        assert "Medtronic" in pages[0].text

    def test_multi_page_pdf_returns_correct_count(self):
        """Each page in the PDF maps to exactly one PageText entry."""
        fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")
        from spark_fleet.pdf_parser import extract_text_from_bytes  # noqa: PLC0415

        doc = fitz.open()
        for i in range(3):
            p = doc.new_page()
            p.insert_text((72, 72), f"Page {i + 1} content")
        pdf_bytes = doc.tobytes()
        doc.close()

        pages = extract_text_from_bytes(pdf_bytes)
        assert len(pages) == 3
        assert pages[0].page == 1
        assert pages[2].page == 3

    def test_blank_page_returns_empty_text(self):
        """A page with no text content returns PageText with empty string."""
        fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")
        from spark_fleet.pdf_parser import extract_text_from_bytes  # noqa: PLC0415

        doc = fitz.open()
        doc.new_page()      # blank page, no content stream
        pdf_bytes = doc.tobytes()
        doc.close()

        pages = extract_text_from_bytes(pdf_bytes)
        assert len(pages) == 1
        assert pages[0].text == ""

    def test_extract_from_path_matches_extract_from_bytes(self, tmp_path):
        """extract_text_from_path and extract_text_from_bytes must agree."""
        fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")
        from spark_fleet.pdf_parser import (  # noqa: PLC0415
            extract_text_from_bytes,
            extract_text_from_path,
        )

        doc = fitz.open()
        p = doc.new_page()
        p.insert_text((72, 72), "Philips Healthcare – Platinum Sponsor")
        pdf_path = tmp_path / "test.pdf"
        doc.save(str(pdf_path))
        pdf_bytes = doc.tobytes()
        doc.close()

        from_bytes = extract_text_from_bytes(pdf_bytes)
        from_path  = extract_text_from_path(pdf_path)

        assert len(from_bytes) == len(from_path)
        assert from_bytes[0].text == from_path[0].text


# ===========================================================================
# 2. Zoho OAuth — RefreshingTokenProvider
# ===========================================================================

class TestZohoOAuthRefresh:

    def _mock_token_response(self, access_token: str = "fresh_token_abc", expires_in: int = 3600):
        """Return a mock httpx response that looks like a Zoho OAuth response."""
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "access_token": access_token,
            "token_type":   "Bearer",
            "expires_in":   expires_in,
        }
        return resp

    def test_first_call_fetches_and_caches_token(self):
        """
        On the first access_token() call, the provider must POST to Zoho
        OAuth and cache the result.  A second call must NOT hit the network.
        """
        from spark_fleet.zoho import RefreshingTokenProvider  # noqa: PLC0415

        provider = RefreshingTokenProvider(
            client_id="cid", client_secret="csec", refresh_token="rtok"
        )
        with patch("httpx.post", return_value=self._mock_token_response()) as mock_post:
            token1 = provider.access_token()
            token2 = provider.access_token()   # should hit cache

        assert token1 == "fresh_token_abc"
        assert token2 == "fresh_token_abc"
        assert mock_post.call_count == 1        # only one real request

    def test_expired_token_triggers_refresh(self):
        """
        When the cached token has expired, the next access_token() call must
        fetch a new one.
        """
        from spark_fleet.zoho import RefreshingTokenProvider  # noqa: PLC0415

        provider = RefreshingTokenProvider(
            client_id="cid", client_secret="csec", refresh_token="rtok"
        )
        # Manually set an already-expired state
        provider._access_token = "old_token"
        provider._expires_at   = datetime.now() - timedelta(seconds=1)

        with patch("httpx.post", return_value=self._mock_token_response("new_token_xyz")) as mock_post:
            token = provider.access_token()

        assert token == "new_token_xyz"
        assert mock_post.call_count == 1

    def test_token_nearly_expired_triggers_refresh(self):
        """
        Proactive refresh: if the token expires within the next 60 seconds,
        treat it as expired and fetch a new one.
        """
        from spark_fleet.zoho import RefreshingTokenProvider  # noqa: PLC0415

        provider = RefreshingTokenProvider(
            client_id="cid", client_secret="csec", refresh_token="rtok"
        )
        provider._access_token = "nearly_dead"
        provider._expires_at   = datetime.now() + timedelta(seconds=30)   # < 60s buffer

        with patch("httpx.post", return_value=self._mock_token_response("proactive_token")) as mock_post:
            token = provider.access_token()

        assert token == "proactive_token"
        assert mock_post.call_count == 1

    def test_failed_refresh_raises_zoho_api_error(self):
        """
        If Zoho OAuth returns an error response, the provider must raise
        ZohoApiError (not a raw httpx or KeyError).
        """
        from spark_fleet.zoho import RefreshingTokenProvider, ZohoApiError  # noqa: PLC0415

        bad_response = MagicMock()
        bad_response.status_code = 400
        bad_response.json.return_value = {"error": "invalid_client"}
        bad_response.text = '{"error": "invalid_client"}'

        provider = RefreshingTokenProvider(
            client_id="bad_id", client_secret="bad_sec", refresh_token="bad_tok"
        )
        with patch("httpx.post", return_value=bad_response):
            with pytest.raises(ZohoApiError, match="OAuth"):
                provider.access_token()

    def test_from_env_reads_environment_variables(self):
        """
        RefreshingTokenProvider.from_env() must read ZOHO_CLIENT_ID,
        ZOHO_CLIENT_SECRET, and ZOHO_REFRESH_TOKEN from the environment.
        """
        from spark_fleet.zoho import RefreshingTokenProvider  # noqa: PLC0415

        env = {
            "ZOHO_CLIENT_ID":     "env_cid",
            "ZOHO_CLIENT_SECRET": "env_csec",
            "ZOHO_REFRESH_TOKEN": "env_rtok",
        }
        with patch.dict("os.environ", env, clear=False):
            provider = RefreshingTokenProvider.from_env()

        assert provider._client_id     == "env_cid"
        assert provider._client_secret == "env_csec"
        assert provider._refresh_token == "env_rtok"

    def test_from_env_raises_when_vars_missing(self):
        """Missing environment variables must raise EnvironmentError clearly."""
        from spark_fleet.zoho import RefreshingTokenProvider  # noqa: PLC0415

        # Ensure the vars are absent
        safe_env = {}
        with patch.dict("os.environ", safe_env, clear=True):
            with pytest.raises(EnvironmentError, match="ZOHO_"):
                RefreshingTokenProvider.from_env()


# ===========================================================================
# 3. Proxycurl LinkedIn Adapter
# ===========================================================================

class TestProxycurlAdapter:

    def _make_proxycurl_response(
        self,
        employees: list[dict] | None = None,
        status_code: int = 200,
    ):
        """Build a mock httpx response shaped like Proxycurl's API."""
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = {"employees": employees or [], "next_page": None}
        resp.text = ""
        return resp

    def _valid_employee(self) -> dict:
        return {
            "profile_url": "https://www.linkedin.com/in/priya-sharma",
            "profile": {
                "full_name":   "Priya Sharma",
                "occupation":  "Marketing Director at Medtronic",
                "headline":    "Marketing Director | AI in Healthcare",
                "country":     "IN",
            },
            "last_updated": "2025-01-01",
        }

    def test_successful_response_maps_to_person_result(self):
        """
        A valid Proxycurl employees response must be mapped to a PersonResult
        with correct full_name, title, and linkedin_url.
        """
        from spark_fleet.adapters.proxycurl_provider import ProxycurlPeopleProvider  # noqa: PLC0415
        from spark_fleet.enrichment import PersonResult                               # noqa: PLC0415
        from spark_fleet.schemas import ExtractedSponsor                              # noqa: PLC0415

        provider = ProxycurlPeopleProvider(api_key="test_key")
        sponsor  = ExtractedSponsor(company_name="Medtronic", source_page=1)

        with patch("httpx.get", return_value=self._make_proxycurl_response([self._valid_employee()])):
            result = provider.find_marketing_director(sponsor)

        assert result is not None
        assert isinstance(result, PersonResult)
        assert result.full_name == "Priya Sharma"
        assert "linkedin.com" in (result.linkedin_url or "")
        assert result.confidence > 0.5

    def test_no_employees_returns_none(self):
        """Empty employees list → return None (CONTACT_MISSING path)."""
        from spark_fleet.adapters.proxycurl_provider import ProxycurlPeopleProvider  # noqa: PLC0415
        from spark_fleet.schemas import ExtractedSponsor                              # noqa: PLC0415

        provider = ProxycurlPeopleProvider(api_key="test_key")
        sponsor  = ExtractedSponsor(company_name="Medtronic", source_page=1)

        with patch("httpx.get", return_value=self._make_proxycurl_response([])):
            result = provider.find_marketing_director(sponsor)

        assert result is None

    def test_429_raises_rate_limit_error(self):
        """HTTP 429 from Proxycurl must raise enrichment.RateLimitError."""
        from spark_fleet.adapters.proxycurl_provider import ProxycurlPeopleProvider  # noqa: PLC0415
        from spark_fleet.enrichment import RateLimitError                             # noqa: PLC0415
        from spark_fleet.schemas import ExtractedSponsor                              # noqa: PLC0415

        provider = ProxycurlPeopleProvider(api_key="test_key")
        sponsor  = ExtractedSponsor(company_name="Medtronic", source_page=1)

        with patch("httpx.get", return_value=self._make_proxycurl_response(status_code=429)):
            with pytest.raises(RateLimitError):
                provider.find_marketing_director(sponsor)

    def test_timeout_raises_enrichment_timeout_error(self):
        """httpx.TimeoutException from Proxycurl must raise enrichment.TimeoutError."""
        from spark_fleet.adapters.proxycurl_provider import ProxycurlPeopleProvider  # noqa: PLC0415
        from spark_fleet.enrichment import TimeoutError as EnrichmentTimeout          # noqa: PLC0415
        from spark_fleet.schemas import ExtractedSponsor                              # noqa: PLC0415

        provider = ProxycurlPeopleProvider(api_key="test_key")
        sponsor  = ExtractedSponsor(company_name="Medtronic", source_page=1)

        with patch("httpx.get", side_effect=__import__("httpx").TimeoutException("timed out")):
            with pytest.raises(EnrichmentTimeout):
                provider.find_marketing_director(sponsor)

    def test_highest_confidence_director_is_returned_first(self):
        """
        When multiple employees match, the one with the highest title score
        (Marketing Director > Head of Marketing) must come first.
        """
        from spark_fleet.adapters.proxycurl_provider import ProxycurlPeopleProvider  # noqa: PLC0415
        from spark_fleet.schemas import ExtractedSponsor                              # noqa: PLC0415

        head_of_mktg = {
            "profile_url": "https://www.linkedin.com/in/james-head",
            "profile": {"full_name": "James Head", "occupation": "Head of Marketing", "headline": "Head of Marketing"},
        }
        mktg_director = {
            "profile_url": "https://www.linkedin.com/in/priya-director",
            "profile": {"full_name": "Priya Director", "occupation": "Marketing Director", "headline": "Marketing Director"},
        }

        provider = ProxycurlPeopleProvider(api_key="test_key")
        sponsor  = ExtractedSponsor(company_name="Medtronic", source_page=1)

        # Return head first in API response — director should still win
        with patch("httpx.get", return_value=self._make_proxycurl_response([head_of_mktg, mktg_director])):
            result = provider.find_marketing_director(sponsor)

        assert result is not None
        assert result.full_name == "Priya Director"

# ===========================================================================
# 4. Playwright LinkedIn Adapter
# ===========================================================================

class TestPlaywrightAdapter:

    def test_missing_playwright_raises_runtime_error(self):
        """If playwright is not installed, it should raise a clear RuntimeError."""
        from spark_fleet.adapters.playwright_provider import PlaywrightPeopleProvider  # noqa: PLC0415
        from spark_fleet.schemas import ExtractedSponsor  # noqa: PLC0415
        
        provider = PlaywrightPeopleProvider()
        sponsor = ExtractedSponsor(company_name="Medtronic", source_page=1)
        
        import builtins
        real_import = builtins.__import__

        def patched_import(name, *args, **kwargs):
            if name == "playwright.sync_api":
                raise ImportError("No module named 'playwright'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=patched_import):
            with pytest.raises(RuntimeError, match="Playwright is not installed"):
                provider.find_marketing_director(sponsor)

    def test_extract_name_from_ddg_title(self):
        """Test the pure function that extracts names from DuckDuckGo HTML titles."""
        from spark_fleet.adapters.playwright_provider import _extract_name_from_ddg_title  # noqa: PLC0415
        
        assert _extract_name_from_ddg_title("John Doe - Marketing Director | LinkedIn") == "John Doe"
        assert _extract_name_from_ddg_title("Jane Smith – VP Marketing") == "Jane Smith"
        assert _extract_name_from_ddg_title("Bob — Head of Growth") == "Bob"
        assert _extract_name_from_ddg_title("No Separators Here") == "No Separators Here"

    def test_build_search_queries_prioritizes_pdf_people(self):
        """Named people from Macro Spark should be searched before generic company terms."""
        from spark_fleet.adapters.playwright_provider import _build_search_queries  # noqa: PLC0415
        from spark_fleet.schemas import ExtractedSponsor  # noqa: PLC0415

        sponsor = ExtractedSponsor(
            company_name="GSK",
            source_page=1,
            important_people=["Jane Doe"],
        )

        queries = _build_search_queries(sponsor)

        assert queries[0] == 'site:linkedin.com/in "Jane Doe" "GSK"'
        assert any('"GSK" "marketing"' in query for query in queries)
