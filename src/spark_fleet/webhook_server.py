"""
spark_fleet/webhook_server.py

A minimal FastAPI application that runs permanently on the Micro Spark
(Mac Mini M2, port 8080).

Start with:
    uvicorn spark_fleet.webhook_server:app --host 0.0.0.0 --port 8080

The single endpoint  POST /webhook/wati-dispatch  receives the Zoho
Workflow Rule webhook and:
  1. Validates the payload (Pydantic → 422 if invalid).
  2. Builds the WATI message via build_wati_payload().
  3. Dispatches via WatiDispatcher.send().
  4. Updates the Zoho lead record with WATI_Status = "Sent" or "Failed".
  5. Returns a JSON response with the final status.

Why this endpoint exists (the Timeout Trap)
--------------------------------------------
Zoho Catalyst functions time out in ~10–30s.  By offloading all I/O to
this always-on FastAPI server on the Mac Mini, Zoho's workflow rule only
needs to fire a single fast webhook — the heavy work happens here,
outside Zoho entirely.

Environment variables required on the Mac Mini
-----------------------------------------------
  WATI_BASE_URL     e.g. "https://live-server-XXXXX.wati.io"
  WATI_API_TOKEN    from WATI dashboard → API → Access Token
  ZOHO_ACCESS_TOKEN from Zoho OAuth (replace with refresh flow in Prompt 6)

Dependency injection
--------------------
WatiDispatcher and ZohoCRMClient are injected via FastAPI's Depends()
mechanism.  Tests override get_wati_dispatcher() and get_zoho_client()
with mocks, so no real HTTP calls are made during testing.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel

from spark_fleet.wati import (
    WatiApiError,
    WatiDispatcher,
    ZohoWebhookPayload,
    build_wati_payload,
)
from spark_fleet.zoho import ZohoCRMClient, RefreshingTokenProvider

logger = logging.getLogger(__name__)


# ===========================================================================
# FastAPI application
# ===========================================================================

app = FastAPI(
    title="Spark Fleet – Webhook Server",
    description=(
        "Receives Zoho Workflow webhooks and dispatches WhatsApp messages "
        "via WATI.  Runs permanently on the Micro Spark (Mac Mini M2)."
    ),
    version="1.0.0",
)


# ===========================================================================
# Dependency factories — override these in tests via dependency_overrides
# ===========================================================================

def get_wati_dispatcher() -> WatiDispatcher:
    """
    Build a WatiDispatcher from environment variables.

    Override in tests:
    ::

        app.dependency_overrides[get_wati_dispatcher] = lambda: mock_wati
    """
    return WatiDispatcher(
        base_url=os.environ.get("WATI_BASE_URL", "http://wati-not-configured"),
        api_token=os.environ.get("WATI_API_TOKEN", "DUMMY_WATI_TOKEN"),
    )


def get_zoho_client() -> ZohoCRMClient:
    """
    Build a ZohoCRMClient from environment variables.

    Override in tests:
    ::

        app.dependency_overrides[get_zoho_client] = lambda: mock_zoho
    """
    return ZohoCRMClient(token_provider=RefreshingTokenProvider.from_env())


# ===========================================================================
# Response model
# ===========================================================================

class WatiDispatchResponse(BaseModel):
    """Shape of every response from POST /webhook/wati-dispatch."""
    lead_id: str
    status:  str        # "sent" | "failed: <reason>"
    message: str = ""   # optional human-readable detail


class ApolloEnrichmentResponse(BaseModel):
    """Shape of every response from POST /webhook/apollo-enrichment."""
    status: str
    lead_id: str | None = None
    message: str = ""


# ===========================================================================
# Endpoint
# ===========================================================================

@app.post(
    "/webhook/wati-dispatch",
    response_model=WatiDispatchResponse,
    summary="Receive Zoho webhook → dispatch WhatsApp via WATI",
    tags=["Webhook"],
)
async def wati_dispatch(
    payload: ZohoWebhookPayload,
    wati:    WatiDispatcher = Depends(get_wati_dispatcher),
    zoho:    ZohoCRMClient  = Depends(get_zoho_client),
) -> WatiDispatchResponse:
    """
    1. Build the WATI message from the validated Zoho payload.
    2. Send via WatiDispatcher (with built-in retry on 429).
    3. Update the Zoho lead record with the new WATI_Status.
    4. Return the final status to the Zoho webhook caller.

    FastAPI will return 422 automatically if the payload is missing
    required fields — no explicit error handling needed for that path.
    """
    # -- Build WATI payload (pure, no I/O) ----------------------------------
    wati_payload = build_wati_payload(payload.model_dump())

    # -- Dispatch to WATI ----------------------------------------------------
    wati_status: str
    detail: str = ""

    try:
        wati_result = wati.send(wati_payload)
        wati_status = "sent"
        detail = str(wati_result.get("id", ""))
        logger.info(
            "WATI dispatch success for lead %s — message id: %s",
            payload.lead_id, detail,
        )

    except WatiApiError as exc:
        wati_status = f"failed: {exc}"
        logger.error(
            "WATI dispatch failed for lead %s: %s (retryable=%s)",
            payload.lead_id, exc, exc.retryable,
        )

    # -- Update Zoho lead WATI_Status ----------------------------------------
    _update_zoho_wati_status(zoho, payload, wati_status)

    return WatiDispatchResponse(
        lead_id=payload.lead_id,
        status=wati_status,
        message=detail,
    )


@app.post(
    "/webhook/apollo-enrichment",
    response_model=ApolloEnrichmentResponse,
    summary="Receive Apollo enrichment webhook and update Zoho lead",
    tags=["Webhook"],
)
async def apollo_enrichment_webhook(
    request: Request,
    zoho: ZohoCRMClient = Depends(get_zoho_client),
) -> ApolloEnrichmentResponse:
    """
    Accept Apollo's async enrichment webhook and update the existing Zoho lead.

    When a phone number arrives, WATI_Status is flipped to Pending so Zoho's
    workflow rule can trigger the existing WhatsApp dispatch endpoint.
    """
    payload = await request.json()
    if not isinstance(payload, dict):
        return ApolloEnrichmentResponse(
            status="ignored",
            message="Apollo webhook payload must be a JSON object.",
        )

    person = _extract_apollo_person(payload)
    if person is None:
        return ApolloEnrichmentResponse(
            status="ignored",
            message="No Apollo person record found in webhook payload.",
        )

    company_name = request.query_params.get("company_name", "").strip()
    conference_name = request.query_params.get("conference_name", "").strip()
    sponsor_tier = request.query_params.get("sponsor_tier", "").strip() or None

    if not company_name or not conference_name:
        return ApolloEnrichmentResponse(
            status="ignored",
            message="Missing company_name or conference_name in Apollo webhook URL.",
        )

    lead = zoho.find_lead(
        company_name=company_name,
        conference_name=conference_name,
        sponsor_tier=sponsor_tier,
    )
    if lead is None:
        logger.warning(
            "Apollo webhook could not find Zoho lead for company=%s conference=%s sponsor_tier=%s",
            company_name,
            conference_name,
            sponsor_tier,
        )
        return ApolloEnrichmentResponse(
            status="not_found",
            message="No matching Zoho lead found for Apollo callback.",
        )

    update_fields = _build_apollo_lead_update(person)
    if not update_fields:
        return ApolloEnrichmentResponse(
            status="ignored",
            lead_id=str(lead.get("id") or ""),
            message="Apollo callback had no usable phone or email fields.",
        )

    lead_id = str(lead.get("id") or "")
    zoho.update_lead(lead_id, update_fields)
    return ApolloEnrichmentResponse(
        status="updated",
        lead_id=lead_id,
        message="Zoho lead updated from Apollo enrichment callback.",
    )


# ===========================================================================
# Helpers
# ===========================================================================

def _update_zoho_wati_status(
    zoho:       ZohoCRMClient,
    payload:    ZohoWebhookPayload,
    wati_status: str,
) -> None:
    """
    Push a minimal Zoho payload that updates only the WATI_Status field.

    Errors here are logged and swallowed — the WhatsApp message has already
    been dispatched successfully at this point, so we must not return a 500.
    """
    try:
        zoho.update_lead(
            payload.lead_id,
            {"WATI_Status": "Sent" if wati_status == "sent" else "Failed"},
        )
        logger.info("Zoho WATI_Status updated to '%s' for lead %s.", wati_status, payload.lead_id)

    except Exception as exc:  # noqa: BLE001  — log and continue
        logger.error(
            "Failed to update Zoho WATI_Status for lead %s: %s",
            payload.lead_id, exc,
        )


def _extract_apollo_person(payload: dict[str, Any]) -> dict[str, Any] | None:
    people = payload.get("people")
    if isinstance(people, list):
        for person in people:
            if isinstance(person, dict):
                return person

    person = payload.get("person")
    if isinstance(person, dict):
        return person

    return None


def _build_apollo_lead_update(person: dict[str, Any]) -> dict[str, Any]:
    phone = _extract_phone(person)
    email = _extract_email(person)
    first_name = _coerce_text(person.get("first_name"))
    last_name = _coerce_text(person.get("last_name"))
    title = _coerce_text(person.get("title"))
    linkedin_url = _coerce_text(person.get("linkedin_url"))

    update_fields: dict[str, Any] = {}
    if phone:
        update_fields["Mobile"] = phone
        update_fields["WATI_Status"] = "Pending"
    if email:
        update_fields["Email"] = email
    if first_name:
        update_fields["First_Name"] = first_name
    if last_name:
        update_fields["Last_Name"] = last_name
    if title:
        update_fields["Designation"] = title
    if linkedin_url:
        update_fields["LinkedIn_Profile"] = linkedin_url
    return update_fields


def _extract_email(person: dict[str, Any]) -> str | None:
    emails = person.get("emails")
    if isinstance(emails, list):
        for item in emails:
            if isinstance(item, dict):
                email = _coerce_text(item.get("email"))
            else:
                email = _coerce_text(item)
            if email and "@" in email:
                return email

    email = _coerce_text(person.get("email"))
    return email if email and "@" in email else None


def _extract_phone(person: dict[str, Any]) -> str | None:
    phone_numbers = person.get("phone_numbers")
    if isinstance(phone_numbers, list):
        for item in phone_numbers:
            if isinstance(item, dict):
                candidate = _coerce_text(item.get("sanitized_number") or item.get("raw_number"))
            else:
                candidate = _coerce_text(item)
            normalized = _normalize_phone(candidate)
            if normalized:
                return normalized

    return _normalize_phone(_coerce_text(person.get("phone") or person.get("mobile_phone")))


def _coerce_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_phone(value: str | None) -> str | None:
    if not value:
        return None

    digits = "".join(ch for ch in value if ch.isdigit())
    if not 8 <= len(digits) <= 15:
        return None
    return f"+{digits}" if not value.startswith("+") else f"+{digits}"
