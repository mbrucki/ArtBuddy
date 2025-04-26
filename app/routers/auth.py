import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

# Import config and state
# from app.config import CORRECT_PIN
from app.config import APP_PIN # Use the renamed variable
from app.state import user_details, chat_histories # Direct state import (simple approach)

logger = logging.getLogger(__name__)
router = APIRouter()

# Assume templates are in a 'templates' directory at the project root
# If you move templates inside 'app', adjust this path
templates = Jinja2Templates(directory="templates")

# --- Step 1: PIN Entry ---
@router.get("/", response_class=HTMLResponse)
async def get_pin_entry(request: Request):
    """Serves the initial PIN entry page and creates a session.
    If APP_PIN is None, redirects directly to terms page.
    """
    session_id = str(uuid.uuid4())
    logger.info(f"New session initiation request, generated ID: {session_id}")

    # Store initial empty state immediately in the global state
    chat_histories[session_id] = []
    user_details[session_id] = {
        "name": None, "city": None,
        "pending_confirmation_type": None,
        "pending_person_name": None,
        "pending_kg_fact": None,
        "pending_original_message": None,
        "pending_original_turn_id": None,
        "pending_original_timestamp": None,
        "confirmed_or_processed_names": set(),
        "session_start_time": datetime.now(timezone.utc)
    }
    logger.info(f"[{session_id}] Initialized empty state for session.")

    # If PIN check is disabled, redirect straight to terms
    if APP_PIN is None:
        logger.info(f"[{session_id}] PIN check disabled. Redirecting directly to terms.")
        terms_url = request.url_for('get_terms').include_query_params(session_id=session_id)
        return RedirectResponse(terms_url, status_code=303)

    # Otherwise, show the PIN entry page
    logger.info(f"[{session_id}] PIN check enabled. Showing PIN entry page.")
    return templates.TemplateResponse(
        "pin_entry.html",
        {"request": request, "session_id": session_id, "error": None}
    )

@router.post("/", response_class=HTMLResponse)
async def post_pin_entry(request: Request, session_id: str = Form(...), pin: str = Form(...)):
    """Handles PIN submission. Redirects to terms on success, re-renders PIN page on failure."""
    logger.info(f"[{session_id}] Received PIN submission.")

    if session_id not in user_details:
        logger.warning(f"[{session_id}] PIN submitted for unknown session. Redirecting to start.")
        return RedirectResponse("/", status_code=303)

    # Check if PIN is enabled and if it matches
    if not APP_PIN or pin == APP_PIN:
        logger.info(f"[{session_id}] PIN correct (or check disabled). Redirecting to terms.")
        # Redirect to GET /terms
        terms_url = request.url_for('get_terms').include_query_params(session_id=session_id)
        return RedirectResponse(terms_url, status_code=303)
    else:
        logger.warning(f"[{session_id}] Incorrect PIN entered.")
        return templates.TemplateResponse(
            "pin_entry.html",
            {
                "request": request,
                "session_id": session_id,
                "error": "Incorrect PIN. Please try again."
            }
        )

# --- Step 2: Terms ---
@router.get("/terms", response_class=HTMLResponse)
async def get_terms(request: Request, session_id: str):
    """Displays the terms and conditions page."""
    logger.info(f"[{session_id}] Displaying terms page.")
    if session_id not in user_details:
        logger.warning(f"[{session_id}] Attempted to access terms for unknown session. Redirecting to start.")
        return RedirectResponse("/", status_code=303)

    return templates.TemplateResponse(
        "terms.html",
        {"request": request, "session_id": session_id}
    )

# The form in terms.html POSTs to /instructions

# --- Step 3: Instructions ---
@router.post("/instructions", response_class=HTMLResponse)
async def post_to_instructions(request: Request, session_id: str = Form(...)):
    """Handles submission from terms page and displays the instructions page."""
    logger.info(f"[{session_id}] Accepted terms. Displaying instructions page.")
    if session_id not in user_details:
        logger.warning(f"[{session_id}] Attempted to access instructions for unknown session. Redirecting to start.")
        return RedirectResponse("/", status_code=303)

    return templates.TemplateResponse(
        "instructions.html",
        {"request": request, "session_id": session_id}
    )

# The form in instructions.html POSTs to /chat (handled in chat.py router) 