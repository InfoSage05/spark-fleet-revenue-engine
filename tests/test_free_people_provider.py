from __future__ import annotations

from unittest.mock import MagicMock, patch

from spark_fleet.enrichment import PersonResult
from spark_fleet.schemas import ExtractedSponsor


def test_free_people_provider_combines_person_and_public_contact_data():
    from spark_fleet.adapters.free_people_provider import FreePeopleProvider

    people_finder = MagicMock()
    people_finder.find_marketing_director.return_value = PersonResult(
        full_name="Priya Sharma",
        title="Marketing Director",
        linkedin_url="https://www.linkedin.com/in/priya-sharma",
        confidence=0.8,
    )

    provider = FreePeopleProvider(people_finder=people_finder, timeout_s=5.0)
    sponsor = ExtractedSponsor(company_name="Medtronic", sponsor_tier="Gold", source_page=1)

    ddg_html = """
    <a href="https://www.medtronic.com">Medtronic Official Site</a>
    """
    home_html = """
    <html><body>
      <a href="/contact-us">Contact us</a>
    </body></html>
    """
    contact_html = """
    <html><body>
      <p>Email: hello@medtronic.com</p>
      <p>WhatsApp us: +1 (415) 555-2671</p>
    </body></html>
    """

    responses = []
    for text in (ddg_html, home_html, contact_html):
        response = MagicMock()
        response.text = text
        response.status_code = 200
        response.raise_for_status.return_value = None
        responses.append(response)

    with patch("spark_fleet.adapters.free_people_provider.httpx.get", side_effect=responses):
        result = provider.find_marketing_director(sponsor)

    assert result is not None
    assert result.full_name == "Priya Sharma"
    assert result.email == "hello@medtronic.com"
    assert result.phone == "+14155552671"


def test_fallback_people_provider_uses_free_provider_when_apollo_plan_blocks():
    from spark_fleet.adapters.fallback_people_provider import FallbackPeopleProvider

    apollo = MagicMock()
    apollo.find_marketing_director.side_effect = RuntimeError(
        "API_INACCESSIBLE free plan HTTP 403"
    )

    free = MagicMock()
    free.find_marketing_director.return_value = PersonResult(
        full_name="Priya Sharma",
        title="Marketing Director",
        linkedin_url="https://www.linkedin.com/in/priya-sharma",
        email="hello@medtronic.com",
        phone="+14155552671",
        confidence=0.8,
    )

    provider = FallbackPeopleProvider(apollo, free)
    sponsor = ExtractedSponsor(company_name="Medtronic", sponsor_tier="Gold", source_page=1)

    result = provider.find_marketing_director(sponsor)

    assert result is not None
    assert result.phone == "+14155552671"


def test_free_people_provider_uses_optional_apollo_email_match():
    from spark_fleet.adapters.free_people_provider import FreePeopleProvider

    people_finder = MagicMock()
    people_finder.find_marketing_director.return_value = PersonResult(
        full_name="Priya Sharma",
        title="Marketing Director",
        linkedin_url="https://www.linkedin.com/in/priya-sharma",
        confidence=0.8,
    )

    provider = FreePeopleProvider(
        people_finder=people_finder,
        timeout_s=5.0,
        apollo_api_key="apollo-key",
    )
    sponsor = ExtractedSponsor(company_name="Medtronic", sponsor_tier="Gold", source_page=1)

    search_response = MagicMock()
    search_response.text = '<a href="https://www.medtronic.com">Medtronic Official Site</a>'
    search_response.raise_for_status.return_value = None

    homepage_response = MagicMock()
    homepage_response.text = "<html><body>No public contact here.</body></html>"
    homepage_response.raise_for_status.return_value = None

    apollo_response = MagicMock()
    apollo_response.json.return_value = {"person": {"email": "priya.sharma@medtronic.com"}}
    apollo_response.raise_for_status.return_value = None

    page_scan_response = MagicMock()
    page_scan_response.text = "<html><body>No public contact here.</body></html>"
    page_scan_response.raise_for_status.return_value = None

    linkedin_search_response = MagicMock()
    linkedin_search_response.text = "<html><body>No LinkedIn company phone here.</body></html>"
    linkedin_search_response.raise_for_status.return_value = None

    with patch(
        "spark_fleet.adapters.free_people_provider.httpx.get",
        side_effect=[search_response, homepage_response, page_scan_response, linkedin_search_response],
    ), \
         patch("spark_fleet.adapters.free_people_provider.httpx.post", return_value=apollo_response) as mock_post:
        result = provider.find_marketing_director(sponsor)

    assert result is not None
    assert result.email == "priya.sharma@medtronic.com"
    assert provider.last_trace["apollo_email_status"] == "found"
    mock_post.assert_called_once()


def test_free_people_provider_ignores_blocked_apollo_email_match():
    import httpx
    from spark_fleet.adapters.free_people_provider import FreePeopleProvider

    people_finder = MagicMock()
    people_finder.find_marketing_director.return_value = PersonResult(
        full_name="Priya Sharma",
        title="Marketing Director",
        linkedin_url="https://www.linkedin.com/in/priya-sharma",
        confidence=0.8,
    )

    provider = FreePeopleProvider(
        people_finder=people_finder,
        timeout_s=5.0,
        apollo_api_key="apollo-key",
    )
    sponsor = ExtractedSponsor(company_name="Medtronic", sponsor_tier="Gold", source_page=1)

    search_response = MagicMock()
    search_response.text = '<a href="https://www.medtronic.com">Medtronic Official Site</a>'
    search_response.raise_for_status.return_value = None

    homepage_response = MagicMock()
    homepage_response.text = "<html><body>No public contact here.</body></html>"
    homepage_response.raise_for_status.return_value = None

    blocked_response = httpx.Response(
        403,
        text='{"error_code":"API_INACCESSIBLE"}',
        request=httpx.Request("POST", "https://api.apollo.io/api/v1/people/match"),
    )
    blocked_error = httpx.HTTPStatusError("blocked", request=blocked_response.request, response=blocked_response)

    page_scan_response = MagicMock()
    page_scan_response.text = "<html><body>No public contact here.</body></html>"
    page_scan_response.raise_for_status.return_value = None

    linkedin_search_response = MagicMock()
    linkedin_search_response.text = "<html><body>No LinkedIn company phone here.</body></html>"
    linkedin_search_response.raise_for_status.return_value = None

    with patch(
        "spark_fleet.adapters.free_people_provider.httpx.get",
        side_effect=[search_response, homepage_response, page_scan_response, linkedin_search_response],
    ), \
         patch("spark_fleet.adapters.free_people_provider.httpx.post") as mock_post:
        mock_post.return_value.raise_for_status.side_effect = blocked_error
        result = provider.find_marketing_director(sponsor)

    assert result is not None
    assert result.email is None
    assert provider.last_trace["apollo_email_status"] == "blocked"


def test_free_people_provider_uses_linkedin_company_snippet_phone_when_site_has_no_phone():
    from spark_fleet.adapters.free_people_provider import FreePeopleProvider

    people_finder = MagicMock()
    people_finder.find_marketing_director.return_value = PersonResult(
        full_name="Priya Sharma",
        title="Marketing Director",
        linkedin_url="https://www.linkedin.com/in/priya-sharma",
        confidence=0.8,
    )

    provider = FreePeopleProvider(people_finder=people_finder, timeout_s=5.0)
    sponsor = ExtractedSponsor(company_name="Medtronic", sponsor_tier="Gold", source_page=1)

    official_search = MagicMock()
    official_search.text = '<a href="https://www.medtronic.com">Medtronic Official Site</a>'
    official_search.raise_for_status.return_value = None

    homepage = MagicMock()
    homepage.text = "<html><body>No public phone here.</body></html>"
    homepage.raise_for_status.return_value = None

    page_scan = MagicMock()
    page_scan.text = "<html><body>No public phone here.</body></html>"
    page_scan.raise_for_status.return_value = None

    linkedin_search = MagicMock()
    linkedin_search.text = """
    <a href="https://www.linkedin.com/company/medtronic">Medtronic LinkedIn</a>
    <span>Contact Medtronic on +1 415 555 2671</span>
    """
    linkedin_search.raise_for_status.return_value = None

    with patch(
        "spark_fleet.adapters.free_people_provider.httpx.get",
        side_effect=[official_search, homepage, page_scan, linkedin_search],
    ):
        result = provider.find_marketing_director(sponsor)

    assert result is not None
    assert result.phone == "+14155552671"
    assert provider.last_trace["linkedin_company_phone"] == "+14155552671"
