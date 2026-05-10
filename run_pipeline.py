"""
run_pipeline.py

Orchestration script to process the local PDF brochures on an HP Victus.

This script demonstrates the end-to-end pipeline:
1. Finds all PDF files in the current directory.
2. Extracts text using PyMuPDF.
3. Calls a local Ollama model (e.g., llama3) via the OpenAI-compatible endpoint.
4. Uses free Playwright/public-web discovery, with optional Apollo email enrichment.
5. Maps the data to Zoho and prints the chosen outreach path.

Requirements
------------
1. pip install -e .
2. pip install PyMuPDF
3. Optional: Add APOLLO_API_KEY for best-effort email enrichment only.
4. Ollama running locally: https://ollama.com/
   Model pulled: `ollama run qwen2.5:14b` or `ollama run llama3.1`
"""

import glob
import logging
import os
import smtplib
import time
from email.message import EmailMessage

from dotenv import load_dotenv

# Load environment variables from .env file BEFORE importing any modules
# that read from os.environ (e.g. RefreshingTokenProvider.from_env())
load_dotenv()

from spark_fleet.pdf_parser import extract_text_from_path
from spark_fleet.macro_client import MacroSparkClient, MacroSparkError
from spark_fleet.enrichment import EnrichmentOrchestrator
from spark_fleet.adapters.free_people_provider import FreePeopleProvider
from spark_fleet.adapters.ocr_provider import TesseractAdapter
from spark_fleet.zoho import ZohoCRMClient, RefreshingTokenProvider, map_lead_to_zoho_payload
from spark_fleet.wati import WatiApiError, WatiDispatcher
from spark_fleet.semantic_chunker import extract_sponsor_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---
# LLM Provider: Groq (free, fast cloud inference)
# Get your free API key at https://console.groq.com/keys
# Add GROQ_API_KEY to your .env file.
LLM_URL = "https://api.groq.com/openai"
LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "qwen-qwq-32b",
]
MIN_CONFIDENCE = 0.5

# Maximum characters sent to the LLM.
MAX_TEXT_CHARS = 8000   # Groq handles large context easily; can go higher

# Timeout in seconds for the Groq API call.
LLM_TIMEOUT_S = 360.0   # High timeout for large PDFs and quality-focused extraction

# Keywords that signal a sponsor section in the brochure.
# Pages containing these words are prioritised over generic pages.
_SPONSOR_KEYWORDS = [
    "sponsor", "gold", "silver", "platinum", "bronze",
    "exhibitor", "partner", "supporter",
]


def _extract_sponsor_text(pages, max_chars: int = MAX_TEXT_CHARS) -> str:
    """
    Return the most sponsor-relevant text from the PDF, capped at max_chars.

    Strategy:
    1. Score each page by how many sponsor keywords it contains.
    2. Sort pages highest-score first.
    3. Concatenate until we hit max_chars.
    """
    def page_score(page_text: str) -> int:
        lower = page_text.lower()
        return sum(lower.count(kw) for kw in _SPONSOR_KEYWORDS)

    scored = sorted(pages, key=lambda p: page_score(p.text), reverse=True)

    collected = []
    total = 0
    for page in scored:
        chunk = page.text.strip()
        if not chunk:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        collected.append(chunk[:remaining])
        total += len(chunk)

    result = "\n\n".join(collected)
    logger.info(
        "Extracted %d chars from %d pages (max %d chars).",
        len(result), len(scored), max_chars,
    )
    return result


def _extract_sponsors_with_model_fallback(
    macro: MacroSparkClient,
    text_to_process: str,
    pdf_path: str,
) -> list:
    """
    Extract sponsors and fallback to alternate models if the configured model
    is not available on the current Groq account.
    """
    candidate_models = [macro.model] + [m for m in LLM_FALLBACK_MODELS if m != macro.model]
    last_err: Exception | None = None

    for model_name in candidate_models:
        macro.model = model_name
        logger.info("Extracting sponsors with model: %s", model_name)
        for attempt in range(3):
            try:
                sponsors = macro.extract_sponsors(text_to_process, conference_name=pdf_path)
                if model_name != LLM_MODEL:
                    print(f"MODEL_FALLBACK: switched to {model_name}")
                return sponsors
            except MacroSparkError as llm_err:
                last_err = llm_err
                msg = str(llm_err)
                model_not_found = "model_not_found" in msg or "does not exist or you do not have access" in msg
                if model_not_found:
                    logger.warning("Model %s unavailable. Trying fallback model.", model_name)
                    break

                if attempt < 2 and getattr(llm_err, "retryable", False):
                    logger.warning("LLM attempt %d failed on %s, retrying in 5s...", attempt + 1, model_name)
                    time.sleep(5)
                    continue
                raise

    if last_err is not None:
        raise last_err
    return []


def run():
    pdfs = glob.glob("*.pdf")
    if not pdfs:
        logger.error("No PDF files found in the current directory.")
        return

    logger.info("Found %d PDFs to process: %s", len(pdfs), pdfs)

    # Initialize LLM client — uses Groq cloud inference
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_api_key:
        logger.error("Missing GROQ_API_KEY in .env file.")
        logger.error("Get your free key at https://console.groq.com/keys")
        return

    macro = MacroSparkClient(
        base_url=LLM_URL,
        model=LLM_MODEL,
        timeout_seconds=LLM_TIMEOUT_S,
        api_key=groq_api_key,
    )
    
    apollo_api_key = os.environ.get("APOLLO_API_KEY", "")
    if apollo_api_key:
        logger.info("APOLLO_EMAIL: enabled for optional people/match email enrichment only.")
    else:
        logger.info("APOLLO_EMAIL: skipped because APOLLO_API_KEY is not configured.")

    provider = FreePeopleProvider(timeout_s=20.0, apollo_api_key=apollo_api_key)
    orchestrator = EnrichmentOrchestrator(people_provider=provider)
    
    # Initialize OCR Fallback
    tesseract_cmd = os.environ.get("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    ocr_adapter = TesseractAdapter(tesseract_cmd=tesseract_cmd)
    
    # Use real Zoho Token Provider (requires Environment Variables)
    try:
        token_provider = RefreshingTokenProvider.from_env()
        zoho = ZohoCRMClient(token_provider=token_provider)
    except EnvironmentError as e:
        logger.error("Missing Zoho Environment Variables. %s", e)
        logger.error("Please set ZOHO_CLIENT_ID, ZOHO_CLIENT_SECRET, and ZOHO_REFRESH_TOKEN to run against the live CRM.")
        return

    # ── Pipeline Results Tracker ──────────────────────────────────────────
    # Each entry records what happened at every stage for the final report.
    results: list[dict] = []

    # ── Direct Email State ─────────────────────────────────────────────────────
    # Tracks whether user chose "send all remaining" to avoid repeated prompts
    email_state = {"send_all_remaining": False}

    for pdf_path in pdfs:
        logger.info("=" * 60)
        logger.info("Processing PDF: %s", pdf_path)
        
        try:
            try:
                if zoho.has_conference_leads(pdf_path):
                    logger.info(
                        "PDF %s already processed in Zoho. Fetching existing leads for display.",
                        pdf_path,
                    )
                    print(f"\n{'='*100}")
                    print(f"ALREADY PROCESSED: {pdf_path}")
                    print(f"{'='*100}")
                    _display_existing_zoho_leads(pdf_path, macro, zoho, results, ocr_adapter)
                    continue
            except Exception as lookup_err:
                logger.warning(
                    "Could not verify whether %s was already processed in Zoho: %s. Continuing.",
                    pdf_path,
                    lookup_err,
                )

            # 1. Read PDF (with OCR fallback for image-only pages)
            pages = extract_text_from_path(pdf_path, ocr_adapter=ocr_adapter)
            if not pages:
                logger.warning("No text extracted from %s", pdf_path)
                results.append({"pdf": pdf_path, "status": "FAILED", "reason": "No text in PDF"})
                continue
                
            text_to_process = extract_sponsor_text(pages, max_chars=MAX_TEXT_CHARS)
            
            # 2. Extract Sponsors (with retry + model fallback on availability errors)
            sponsors = _extract_sponsors_with_model_fallback(macro, text_to_process, pdf_path)
            
            if not sponsors:
                logger.info("No sponsors found in %s", pdf_path)
                results.append({"pdf": pdf_path, "status": "OK", "reason": "No sponsors detected by LLM"})
                continue
                
            logger.info("Found %d sponsors in %s", len(sponsors), pdf_path)
            
            # 3. Enrich & Push
            for sponsor in sponsors:
                important_people = getattr(sponsor, "important_people", []) or []
                entry = {
                    "pdf": pdf_path,
                    "company": sponsor.company_name,
                    "tier": sponsor.sponsor_tier or "Unknown",
                    "important_people": ", ".join(important_people) if important_people else "Not Found",
                    "director": None,
                    "linkedin": None,
                    "email": None,
                    "phone": None,
                    "outreach_decision": None,
                    "zoho_status": None,
                    "wati_status": None,
                    "status": "PROCESSING",
                }
                
                print(f"\nSPONSOR: {sponsor.company_name} | Tier: {sponsor.sponsor_tier or 'Unknown'}")
                if important_people:
                    print("IMPORTANT_PEOPLE_FROM_PDF:")
                    for person in important_people:
                        print(f"- {person}")
                else:
                    print("IMPORTANT_PEOPLE_FROM_PDF: none")
                logger.info("Enriching sponsor: %s (Tier: %s)", sponsor.company_name, sponsor.sponsor_tier)
                
                # Free mode: Playwright + public web + optional Apollo people/match email.
                lead = orchestrator.enrich(sponsor, conference_name=pdf_path)
                
                if lead:
                    entry["director"] = lead.director_name or "Not Found"
                    entry["linkedin"] = str(lead.linkedin_url) if lead.linkedin_url else "Not Found"
                    entry["email"] = lead.email or "Not Found"
                    entry["phone"] = lead.phone or "Not Found"
                    
                    logger.info("Successfully enriched! Director: %s, LinkedIn: %s", lead.director_name, lead.linkedin_url)
                    
                    # 4. Map to Zoho
                    payload = map_lead_to_zoho_payload(lead)
                    wati_status = _try_direct_wati_send(lead)
                    payload.data[0]["WATI_Status"] = wati_status
                    if _env_flag("DIRECT_WATI_SEND"):
                        outreach_decision = f"WATI_{wati_status.upper()}"
                    else:
                        outreach_decision = _outreach_decision(lead)
                    entry["outreach_decision"] = outreach_decision
                    entry["wati_status"] = wati_status
                    _print_sponsor_audit(lead, provider, outreach_decision)
                    logger.info("Mapped to Zoho Payload. WATI_Status: %s", wati_status)

                    if lead.email:
                        _try_direct_email_send(lead, email_state)
                    
                    # 5. Push to LIVE Zoho
                    try:
                        logger.info("Pushing to LIVE Zoho CRM...")
                        zoho.push(payload)
                        entry["zoho_status"] = "Pushed"
                        entry["status"] = "SUCCESS"
                        print("ZOHO: pushed")
                        logger.info("Pushed to LIVE Zoho successfully!")
                    except Exception as zoho_err:
                        entry["zoho_status"] = f"Failed: {zoho_err}"
                        entry["status"] = "ZOHO_FAILED"
                        print(f"ZOHO: failed | {zoho_err}")
                        logger.error("Zoho push failed: %s", zoho_err)
                else:
                    entry["status"] = "NO_LEAD"
                    entry["director"] = "Not Found"
                    entry["linkedin"] = "Not Found"
                    entry["outreach_decision"] = "NO_CONTACT"
                    print("DISCOVERY: LinkedIn missing | Director: Not Found | Title: Not Found")
                    print("CONTACT: Website: missing | Email: missing | Phone: missing")
                    print("APOLLO_EMAIL: skipped")
                    print("OUTREACH: NO_CONTACT")
                    
                results.append(entry)
                    
                # Sleep slightly between companies to avoid hammering Apollo.
                time.sleep(2)
                
        except Exception as e:
            logger.error("Error processing %s: %s", pdf_path, e, exc_info=True)
            results.append({"pdf": pdf_path, "status": "ERROR", "reason": str(e)[:80]})

    # ── Print Pipeline Summary Report ─────────────────────────────────────
    _print_summary(results)


def _print_summary(results: list[dict]) -> None:
    """Print a clear, human-readable summary of the entire pipeline run."""
    print("\n")
    print("=" * 100)
    print("                     SPARK FLEET — PIPELINE SUMMARY REPORT")
    print("=" * 100)
    
    if not results:
        print("  No results to report.")
        print("=" * 100)
        return

    successes = [r for r in results if r.get("status") == "SUCCESS"]
    no_leads = [r for r in results if r.get("status") == "NO_LEAD"]
    failures = [r for r in results if r.get("status") in ("ERROR", "FAILED", "ZOHO_FAILED")]
    wati_ready = [r for r in results if r.get("outreach_decision") == "WATI_PENDING"]
    email_ready = [r for r in results if r.get("outreach_decision") == "EMAIL_DRAFT_READY"]
    linkedin_only = [r for r in results if r.get("outreach_decision") == "MANUAL_LINKEDIN_ONLY"]

    print(f"\n  Total processed: {len(results)}  |  Success: {len(successes)}  |  "
          f"No Lead: {len(no_leads)}  |  Failed: {len(failures)}\n")
    print(f"  Outreach: WATI Pending={len(wati_ready)}  |  Email Draft={len(email_ready)}  |  "
          f"Manual LinkedIn={len(linkedin_only)}\n")

    # Detailed table
    print("-" * 100)
    print(f"  {'Company':<25} {'Director':<22} {'Email':<25} {'Phone':<16} {'Outreach'}")
    print("-" * 100)

    for r in results:
        if "company" in r:
            company = (r.get("company", "?") or "?")[:24]
            director = (r.get("director", "—") or "—")[:24]
            email = (r.get("email", "—") or "—")[:24]
            phone = (r.get("phone", "—") or "—")[:15]
            outreach = r.get("outreach_decision", "—") or "—"
            print(f"  {company:<25} {director:<22} {email:<25} {phone:<16} {outreach}")
        else:
            # PDF-level result (no companies found, or error)
            pdf    = r.get("pdf", "?")[:30]
            reason = r.get("reason", r.get("status", "?"))[:60]
            print(f"  [{pdf}] → {reason}")

    print("-" * 100)
    
    if successes:
        print(f"\n  {len(successes)} lead(s) pushed to Zoho CRM.")
    if no_leads:
        print(f"  {len(no_leads)} sponsor(s) could not be enriched.")
    if failures:
        print(f"  {len(failures)} item(s) failed. Check the logs above for details.")
    
    print("=" * 100)
    print()


def _outreach_decision(lead) -> str:
    if lead.phone:
        return "WATI_PENDING"
    if lead.email:
        return "EMAIL_DRAFT_READY"
    if lead.linkedin_url:
        return "MANUAL_LINKEDIN_ONLY"
    return "NO_CONTACT"


def _print_sponsor_audit(lead, provider: FreePeopleProvider, outreach_decision: str) -> None:
    trace = getattr(provider, "last_trace", {}) or {}
    linkedin_status = "found" if lead.linkedin_url else "missing"
    website_status = "found" if trace.get("website_url") else "missing"
    email_status = lead.email or "missing"
    phone_status = lead.phone or "missing"
    apollo_status = trace.get("apollo_email_status") or "skipped"

    print(
        "DISCOVERY: "
        f"LinkedIn {linkedin_status} | "
        f"Director: {lead.director_name or 'Not Found'} | "
        f"Title: {lead.director_title or 'Not Found'}"
    )
    print(
        "CONTACT: "
        f"Website: {website_status} | "
        f"Email: {email_status} | "
        f"Phone: {phone_status}"
    )
    if trace.get("linkedin_company_phone"):
        print(f"LINKEDIN_COMPANY_PHONE: found | {trace.get('linkedin_company_phone')}")
    else:
        print("LINKEDIN_COMPANY_PHONE: missing")
    if apollo_status == "blocked":
        print("APOLLO_EMAIL: unavailable on current plan, continuing with free public-web data")
    else:
        print(f"APOLLO_EMAIL: {apollo_status}")
    print(f"OUTREACH: {outreach_decision}")

    if lead.email:
        subject, body = _email_draft(lead)
        print("EMAIL_DRAFT_SUBJECT:")
        print(subject)
        print("EMAIL_DRAFT_BODY:")
        print(body)


def _wati_message_payload(lead) -> dict[str, object]:
    first_name = (lead.director_name or "").strip().split()[0] if lead.director_name else ""
    return {
        "template_name": os.environ.get("WATI_TEMPLATE_KEY", "medical_conference_sponsor_intro_v1"),
        "broadcast_name": "spark_conference_sponsor_outreach",
        "phone_number": lead.phone,
        "parameters": [
            {"name": "first_name", "value": first_name},
            {"name": "company", "value": lead.company_name},
            {"name": "sponsor_tier", "value": lead.sponsor_tier},
            {"name": "conference_name", "value": lead.conference_name},
        ],
    }


def _try_direct_wati_send(lead) -> str:
    """
    Best-effort direct WhatsApp send using WATI credentials from environment.

    Set DIRECT_WATI_SEND=true to enable. If disabled or config is incomplete,
    this function exits without changing the CRM flow.

    Returns a Zoho-friendly WATI_Status value: Sent, Failed, Pending, or Not Sent - Missing Phone.
    """
    if not lead.phone:
        return "Not Sent - Missing Phone"

    if not _env_flag("DIRECT_WATI_SEND"):
        return "Pending"

    wati_base_url = os.environ.get("WATI_BASE_URL", "").strip()
    wati_api_token = os.environ.get("WATI_API_TOKEN", "").strip()
    if not (wati_base_url and wati_api_token):
        logger.info("DIRECT_WATI_SEND enabled but WATI config is incomplete; keeping WATI_Status pending.")
        return "Pending"

    dispatcher = WatiDispatcher(
        base_url=wati_base_url,
        api_token=wati_api_token,
        timeout_s=30.0,
    )

    try:
        response = dispatcher.send(_wati_message_payload(lead))
        logger.info("DIRECT_WATI: sent to %s", lead.phone)
        logger.debug("DIRECT_WATI response: %s", response)
        print(f"DIRECT_WATI: sent | {lead.phone}")
        return "Sent"
    except WatiApiError as exc:
        logger.warning("DIRECT_WATI: failed for %s: %s", lead.phone, exc)
        print(f"DIRECT_WATI: failed | {lead.phone} | {exc}")
        return "Failed"
    except Exception as exc:  # noqa: BLE001
        logger.warning("DIRECT_WATI: unexpected failure for %s: %s", lead.phone, exc)
        print(f"DIRECT_WATI: failed | {lead.phone} | {exc}")
        return "Failed"


def _email_draft(lead) -> tuple[str, str]:
    first_name = (lead.director_name or "").strip().split()[0] if lead.director_name else ""
    local_part = lead.email.split("@", 1)[0].lower() if lead.email else ""
    generic_prefixes = {"info", "contact", "sales", "partnerships", "marketing", "hello", "team"}
    greeting = first_name if first_name and local_part not in generic_prefixes else "team"
    subject = f"Quick note after {lead.conference_name}"
    body = (
        f"Hi {greeting},\n\n"
        f"I noticed {lead.company_name} was a {lead.sponsor_tier} sponsor at {lead.conference_name}.\n\n"
        "We help medical teams use AI to identify and engage high-intent healthcare partners "
        f"automatically. Would it be useful to compare notes on how {lead.company_name} is "
        "approaching AI-led growth in healthcare this quarter?\n\n"
        "Best,\n"
        "Ayushman\n"
        "TheRightDoctor"
    )
    return subject, body


def _try_direct_email_send(lead, email_state: dict | None = None) -> None:
    """
    Best-effort direct email send using SMTP credentials from environment.

    Set DIRECT_EMAIL_SEND=true in .env to enable. If disabled or incomplete
    config is detected, this function exits silently.
    
    Parameters
    ----------
    lead : EnrichedLead
        The lead with email to send to.
    email_state : dict | None
        Shared state dict tracking whether user chose "send all remaining".
        Keys: {"send_all_remaining": bool}
    """
    if email_state is None:
        email_state = {"send_all_remaining": False}

    if not _env_flag("DIRECT_EMAIL_SEND"):
        return

    smtp_host = os.environ.get("SMTP_HOST", "").strip()
    smtp_user = os.environ.get("SMTP_USER", "").strip()
    smtp_pass = os.environ.get("SMTP_PASS", "").strip()
    from_email = os.environ.get("SMTP_FROM_EMAIL", smtp_user).strip()

    if not (smtp_host and smtp_user and smtp_pass and from_email):
        logger.info(
            "DIRECT_EMAIL_SEND enabled but SMTP config is incomplete; skipping direct email send."
        )
        return

    subject, body = _email_draft(lead)
    should_send = _confirm_direct_email_send(lead, subject, body, email_state)
    if not should_send:
        logger.info("DIRECT_EMAIL: skipped by user for %s", lead.email)
        print(f"DIRECT_EMAIL: skipped by user | {lead.email}")
        return

    msg = EmailMessage()
    msg["From"] = from_email
    msg["To"] = lead.email
    msg["Subject"] = subject
    msg.set_content(body)

    port = int(os.environ.get("SMTP_PORT", "587"))
    use_tls = _env_flag("SMTP_USE_TLS", default=True)

    try:
        with smtplib.SMTP(smtp_host, port, timeout=15) as server:
            if use_tls:
                server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        logger.info("DIRECT_EMAIL: sent to %s", lead.email)
        print(f"DIRECT_EMAIL: sent | {lead.email}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("DIRECT_EMAIL: failed for %s: %s", lead.email, exc)
        print(f"DIRECT_EMAIL: failed | {lead.email} | {exc}")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _display_existing_zoho_leads(
    pdf_path: str, macro: MacroSparkClient, zoho: ZohoCRMClient, results: list[dict], ocr_adapter: TesseractAdapter
) -> None:
    """
    Extract sponsors from the PDF and query Zoho for all their existing leads.
    Display director, email, phone, company, tier in terminal.
    """
    logger.info("Extracting sponsors from already-processed PDF: %s", pdf_path)
    print("\nExtracting sponsors from PDF...\n")

    # Extract PDF text
    try:
        pages = extract_text_from_path(pdf_path, ocr_adapter=ocr_adapter)
        if not pages:
            logger.warning("No text extracted from %s", pdf_path)
            print("(No text could be extracted from this PDF)")
            return
        text_to_process = extract_sponsor_text(pages, max_chars=MAX_TEXT_CHARS)
    except Exception as e:
        logger.warning("Could not extract text from %s: %s", pdf_path, e)
        print(f"(Error extracting PDF text: {e})")
        return

    # Extract sponsors using LLM (with model fallback)
    try:
        sponsors = _extract_sponsors_with_model_fallback(macro, text_to_process, pdf_path)
    except Exception as e:
        logger.warning("Could not extract sponsors from %s: %s", pdf_path, e)
        print(f"(Error extracting sponsors: {e})")
        return

    if not sponsors:
        logger.info("No sponsors found in already-processed PDF: %s", pdf_path)
        print("(No sponsors detected in this PDF)")
        results.append({"pdf": pdf_path, "status": "SKIPPED", "reason": "Already processed, no sponsors in PDF"})
        return

    logger.info("Found %d sponsors in %s. Querying Zoho for their leads...", len(sponsors), pdf_path)
    print(f"Found {len(sponsors)} sponsors. Fetching leads from Zoho...\n")
    print("-" * 120)
    print(f"  {'Company':<25} {'Tier':<12} {'Director':<22} {'Email':<25} {'Phone':<16} {'Status'}")
    print("-" * 120)

    found_count = 0
    for sponsor in sponsors:
        try:
            lead = zoho.find_lead(
                company_name=sponsor.company_name,
                conference_name=pdf_path,
            )
        except Exception as e:
            logger.warning("Could not query Zoho for %s: %s", sponsor.company_name, e)
            continue

        if lead is None:
            continue

        found_count += 1
        company = lead.get("Company", "N/A")[:24]
        tier = lead.get("Sponsor_Tier", "Unknown")[:11]
        first_name = lead.get("First_Name", "")
        last_name = lead.get("Last_Name", "")
        director = f"{first_name} {last_name}".strip()[:22] if (first_name or last_name) else "N/A"
        email = (lead.get("Email") or "N/A")[:24]
        phone = (lead.get("Mobile") or "N/A")[:15]
        wati_status = lead.get("WATI_Status", "Unknown")
        linkedin = lead.get("LinkedIn_Profile", "N/A")

        print(f"  {company:<25} {tier:<12} {director:<22} {email:<25} {phone:<16} {wati_status}")

        results.append({
            "pdf": pdf_path,
            "company": company,
            "tier": tier,
            "director": director,
            "linkedin": linkedin,
            "email": email,
            "phone": phone,
            "outreach_decision": wati_status,
            "status": "ALREADY_PROCESSED",
        })

    print("-" * 120)
    if found_count == 0:
        print("  (No leads found in Zoho for these sponsors yet.)")
    else:
        print(f"  Total existing leads displayed: {found_count}")
    print()


def _confirm_direct_email_send(lead, subject: str, body: str, email_state: dict | None = None) -> bool:
    """
    Ask for explicit user approval in terminal before sending each direct email.
    
    Supports three options:
    - y: Send this email
    - n: Skip this email
    - a: Send all remaining emails without further prompts
    """
    if email_state is None:
        email_state = {"send_all_remaining": False}

    # If user already chose "send all remaining", skip prompt
    if email_state.get("send_all_remaining", False):
        return True

    print("\n" + "=" * 80)
    print("DIRECT EMAIL PREVIEW")
    print("=" * 80)
    print(f"To      : {lead.email}")
    print(f"Company : {lead.company_name}")
    print(f"Subject : {subject}")
    print("Body:")
    print(body)
    print("=" * 80)

    prompt = "Send this email? [y/n/a (send all)]: "
    while True:
        choice = input(prompt).strip().lower()
        if choice in {"y", "yes"}:
            return True
        if choice in {"n", "no"}:
            return False
        if choice in {"a", "all"}:
            email_state["send_all_remaining"] = True
            return True
        if choice == "":
            return False
        print("Please answer 'y' (send), 'n' (skip), or 'a' (send all remaining).")


if __name__ == "__main__":
    run()
