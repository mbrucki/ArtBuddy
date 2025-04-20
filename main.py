import asyncio
import logging
import os
import sys
import uuid
import re
import random
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, List, Dict, Any
from contextlib import asynccontextmanager # For lifespan management
import json # Add json import

from dotenv import load_dotenv
# --- FastAPI Imports ---
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
# --- End FastAPI Imports ---
from openai import OpenAI

try:
    # Ensure chatbot_memory.py is importable and correct
    from chatbot_memory import (
        Graphiti, EpisodeType, add_message_episode, search_memory,
        get_person_details, get_person_fact, check_city_exists,
        get_graphiti_instance, close_graphiti_instance # Use instance managers
    )
except ImportError as e:
    print(f"Error importing from chatbot_memory.py: {e}", file=sys.stderr)
    print("Please ensure chatbot_memory.py is in the same directory.", file=sys.stderr)
    sys.exit(1)

# --- Configuration ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# --- Neo4j/OpenAI Config ---
neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')
openai_api_key = os.environ.get('OPENAI_API_KEY')

if not all([neo4j_uri, neo4j_user, neo4j_password]):
    raise ValueError("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY must be set")

# --- Initialize OpenAI Client (Sync) ---
# Can be initialized globally as it's thread-safe
sync_openai_client = OpenAI(api_key=openai_api_key)
logger.info("Sync OpenAI client initialized.")

# --- FastAPI App Setup ---

# Lifespan manager for Graphiti instance
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Actions on startup
    logger.info("Application startup: Initializing Graphiti...")
    await get_graphiti_instance() # Initialize the global instance
    logger.info("Graphiti initialized via lifespan.")
    yield # Application runs here
    # Actions on shutdown
    logger.info("Application shutdown: Closing Graphiti connection...")
    await close_graphiti_instance()
    logger.info("Graphiti connection closed via lifespan.")

app = FastAPI(lifespan=lifespan)

# Mount static files (CSS, JS, images)
# Ensure you have a 'static' directory in your project root
script_dir = os.path.dirname(__file__)
static_dir = os.path.join(script_dir, "static")
if os.path.exists(static_dir):
     app.mount("/static", StaticFiles(directory=static_dir), name="static")
     logger.info(f"Mounted static directory: {static_dir}")
else:
     logger.warning(f"Static directory not found at {static_dir}, skipping mount.")


# Configure Jinja2 templates
# Ensure you have a 'templates' directory
templates_dir = os.path.join(script_dir, "templates")
if os.path.exists(templates_dir):
    templates = Jinja2Templates(directory=templates_dir)
    logger.info(f"Configured Jinja2 templates directory: {templates_dir}")
else:
     logger.error(f"Templates directory not found at {templates_dir}. HTML rendering will fail.")
     # Exit or provide dummy templates object? Let's raise error for clarity.
     raise FileNotFoundError(f"Required 'templates' directory not found at {templates_dir}")


# --- Chat Histories / User Details & State (In-memory) ---
chat_histories = {} # {session_id: [messages]}
user_details = {}   # {session_id: {
                     #    "name": Optional[str],
                     #    "city": Optional[str],
                     #    "pending_confirmation_type": Optional[str], # e.g., "person_disambiguation", "clarification_needed"
                     #    "pending_person_name": Optional[str],
                     #    "pending_kg_fact": Optional[str],
                     #    "pending_original_message": Optional[str],
                     #    "pending_clarification_for": Optional[str] # Which name needs clarifying
                     # }}

# --- REMOVED Regex Entity Extraction Helpers ---
# NAME_PATTERN = ...
# STOP_WORDS = ...
# def extract_entity(...)
# def extract_name(...)
# CITY_PATTERN = ...
# def extract_city(...)

# --- NEW Enhanced LLM Entity Extraction --- 
def extract_entities(user_message: str) -> Dict[str, Any]:
    """Uses OpenAI LLM to extract self-introduced name/city and mentioned persons."""
    logger.info(f"Attempting Enhanced LLM extraction for: '{user_message}'")
    system_prompt = """You are an expert entity extractor. Analyze the user message provided.
Identify the following:
1. SELF_NAME: The user's own name, ONLY if they are explicitly introducing themselves (e.g., 'I am...', 'My name is...').
2. SELF_CITY: The city/location the user explicitly mentions they live in or are from.
3. MENTIONED_PERSONS: A list of *other* people's full names mentioned in the text (NOT the user's own name).
Return the result ONLY as a valid JSON object with keys 'self_name', 'self_city', and 'mentioned_persons'.
Use the actual extracted strings as values. If an entity type is not found, use null for 'self_name'/'self_city' or an empty list [] for 'mentioned_persons'.
Example 1: Input: 'Hi, I am Helena from Gdynia.' -> Output: {"self_name": "Helena", "self_city": "Gdynia", "mentioned_persons": []}
Example 2: Input: 'My name is Bob Smith and I live in New York. I spoke to Alice Jones yesterday.' -> Output: {"self_name": "Bob Smith", "self_city": "New York", "mentioned_persons": ["Alice Jones"]}
Example 3: Input: 'What is the weather like in Paris?' -> Output: {"self_name": null, "self_city": null, "mentioned_persons": []}
Example 4: Input: 'Tell me about Mariusz. He is an artist.' -> Output: {"self_name": null, "self_city": null, "mentioned_persons": ["Mariusz"]}
""" # End triple-quoted string

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_message}
    ]

    default_result = {"self_name": None, "self_city": None, "mentioned_persons": []}

    try:
        response = sync_openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.1,
            max_tokens=100, # Increased slightly for potentially longer lists
            response_format={"type": "json_object"}
        )

        if response.choices and response.choices[0].message and response.choices[0].message.content:
            content = response.choices[0].message.content.strip()
            logger.debug(f"Enhanced LLM Extraction Raw Response: {content}")
            try:
                data = json.loads(content)
                # Validate structure and types
                if isinstance(data, dict):
                    self_name = data.get('self_name') if isinstance(data.get('self_name'), str) else None
                    self_city = data.get('self_city') if isinstance(data.get('self_city'), str) else None
                    mentioned_persons = data.get('mentioned_persons')
                    if isinstance(mentioned_persons, list) and all(isinstance(p, str) for p in mentioned_persons):
                         result = {
                            "self_name": self_name,
                            "self_city": self_city,
                            "mentioned_persons": mentioned_persons
                         }
                         logger.info(f"Enhanced LLM Extraction Successful: {result}")
                         return result
                    else:
                         logger.warning(f"Enhanced LLM Extraction 'mentioned_persons' invalid type: {content}")
                else:
                    logger.warning(f"Enhanced LLM Extraction response not a valid dict: {content}")
            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to decode Enhanced LLM extraction JSON response: {json_err}. Response: {content}")
        else:
            logger.warning("Enhanced LLM Extraction response was empty or malformed.")

    except Exception as e:
        logger.error(f"Error during Enhanced LLM entity extraction API call: {e}", exc_info=True)

    return default_result # Return default on error

# --- End Enhanced LLM Extraction ---

QUESTION_WORDS = ("what", "who", "where", "when", "why", "how", "tell me", "describe", "explain")

def is_direct_question(text: str) -> bool:
    """Check if the text likely starts with a question asking for information."""
    return text.strip().lower().startswith(QUESTION_WORDS)

# --- LLM Response Generation (SIMPLIFIED - No confirmation logic here) ---
def generate_sync_response(conversation_history: list[dict],
                           all_relevant_memory: list[str],
                           user_message: str,
                           user_name: Optional[str],
                           user_city: Optional[str]
                           ) -> str:
    initial_prompt_part = ""
    prompt_focus = "standard" # Default focus - confirmation handled by send_message

    # --- Determine Prompt Focus and Initial Directive ---
    is_question = is_direct_question(user_message)
    has_relevant_memory = bool(all_relevant_memory)

    # Simplified logic: prioritize answering questions, then info gathering
    if is_question:
        prompt_focus = "answering"
        initial_prompt_part = " The user asked a direct question. Prioritize answering it accurately and concisely using the conversation history and relevant memory facts provided below."
    else:
        prompt_focus = "standard" # Default to info gathering/standard interaction
        missing_info = []
        if not user_name: missing_info.append("the user's name")
        if not user_city: missing_info.append("the user's city")

        history_turns_so_far = (len(conversation_history) + 1) // 2
        if (user_name and user_city and history_turns_so_far < 4) or \
           (not user_name and not user_city and history_turns_so_far < 3):
             missing_info.append("the user's connection to the local art world")

        if missing_info:
            # Note: Cannot easily check extracted_city/name here as they aren't passed anymore
            # This info gathering prompt might ask for info just provided by the user
            # A more robust solution might involve passing extracted info flags
            initial_prompt_part = f" Your immediate priority is to politely gather the following missing information: {', '.join(missing_info)}. Ask ONE clear question targeting ONE piece of missing information."
        else:
            # Standard interaction focus
            initial_prompt_part = " Focus on asking specific, curious questions about the art scene (artists, galleries, events, styles, communities) in their city."

    focus_description_text = ""
    if prompt_focus == 'answering':
         focus_description_text = "------ ANSWERING FOCUS -------\nYour task is to directly answer the user's question using the available context and memory. Be factual and concise."
    else: # Default to info gathering / standard
         # Simplified standard focus
         focus_description_text = "------ STANDARD FOCUS -------\nYour PRIMARY GOAL is to learn about the user and the art scene (artists, galleries, events, styles, communities) in their city. Ask relevant questions or respond naturally to the user's statement."

    system_prompt = f"""You are Art Buddy, a cultural archivist.

CURRENT TASK: {prompt_focus.upper()}

{focus_description_text}

{initial_prompt_part}

*Keep your responses concise and, unless answering a question or asking for confirmation, **ask only ONE specific question per response.**
Use relevant memory facts (provided below, if any) for context when answering or asking follow-up questions, but prioritize the current task.

You are mostly interested in:
- Spatial clustering of cultural venues (museums, galleries, theaters, studios, creative quarters)
- Collaborative art-world networks (formal partnerships and informal exchanges among artists, curators, institutions)
- Aestheticization of public space (murals, installations, signage) and image-making through media and events
- Diverse cultural activities (exhibitions, performances, festivals, street-art cruises, participatory workshops)
- Territorial embeddedness (projects co-created with local residents; integration of "fine" and "craft," professional and amateur practices)
- Social bonding & cultural cohesion (arts as spaces for meeting, civic engagement, shared experiences)
"""

    prompt_messages = [{'role': 'system', 'content': system_prompt.strip()}]

    if all_relevant_memory:
        memory_context = "Relevant Memory Facts (Context): " + "; ".join(all_relevant_memory)
        prompt_messages.append({"role": "system", "content": memory_context})

    max_turns_for_context = 10
    history_start_index = max(0, len(conversation_history) - (max_turns_for_context * 2))
    prompt_messages.extend(conversation_history[history_start_index:])

    logger.debug(f"Sending SYNC prompt to OpenAI: {prompt_messages}")

    try:
        response = sync_openai_client.chat.completions.create(
            model="gpt-4o",
            messages=prompt_messages,
            temperature=0.7,
            max_tokens=100,
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            logger.warning("SYNC OpenAI response was empty or malformed.")
            return "Sorry, I couldn't generate a response right now (sync)."
    except Exception as e:
        logger.error(f"Error generating SYNC OpenAI response: {e}", exc_info=True)
        return f"Sorry, I encountered an error trying to respond (sync). ({type(e).__name__})"


# --- FastAPI Routes ---

# Pydantic model for request body
class MessageRequest(BaseModel):
    message: str
    session_id: str

@app.get("/")
async def index(request: Request):
    """Serves the main HTML page."""
    session_id = str(uuid.uuid4())
    chat_histories[session_id] = []
    # Initialize session details including pending state fields
    user_details[session_id] = {
        "name": None, "city": None,
        "pending_confirmation_type": None,
        "pending_person_name": None, # Person name awaiting confirmation
        "pending_kg_fact": None,     # Fact shown for confirmation
        "pending_original_message": None # User msg that triggered confirmation
    }
    logger.info(f"New session started: {session_id}")
    try:
        return templates.TemplateResponse("index.html", {"request": request, "session_id": session_id})
    except NameError:
         logger.critical("Jinja2Templates object 'templates' not initialized. Cannot serve HTML.")
         raise HTTPException(status_code=500, detail="Server configuration error: Template engine not found.")


@app.post("/send_message")
async def send_message(message_req: MessageRequest):
    """Handles incoming chat messages, runs async tasks, generates response."""
    user_message = message_req.message
    session_id = message_req.session_id

    # Basic validation already handled by Pydantic

    if session_id not in chat_histories:
        logger.warning(f"Received message for unknown session ID: {session_id}. Initializing.")
        chat_histories[session_id] = []
        # Initialize full state for new/unknown session
        user_details[session_id] = {
            "name": None, "city": None,
            "pending_confirmation_type": None,
            "pending_person_name": None, # Person name awaiting confirmation
            "pending_kg_fact": None,     # Fact shown for confirmation
            "pending_original_message": None # User msg that triggered confirmation
        }

    # Determine turn ID
    turn_id = len(chat_histories[session_id]) + 1
    conversation_id = f"session_{session_id}"
    request_start_time = datetime.now(timezone.utc) # Record time before processing

    # Append user message to history FIRST
    chat_histories[session_id].append({"role": "user", "content": user_message})
    logger.info(f"[{session_id}] User message received (turn {turn_id}): {user_message}")

    # Get current state
    session_state = user_details.get(session_id, {}).copy()
    pending_type = session_state.get("pending_confirmation_type")

    # --- State Machine Logic ---
    if pending_type:
        logger.info(f"[{session_id}] Handling pending state: {pending_type}")
        # --- Handle Response to Pending Confirmation ---
        bot_response = None
        should_process_original = False
        original_message = session_state.get("pending_original_message")
        pending_name = session_state.get("pending_person_name")

        if pending_type == "person_disambiguation":
            # Simple Yes/No check (can be improved with LLM intent recognition)
            if re.search(r'\b(yes|yeah|correct|true|confirm)\b', user_message, re.IGNORECASE):
                logger.info(f"[{session_id}] User confirmed existing person '{pending_name}'.")
                # Clear state and set flag to process original message normally
                should_process_original = True
                bot_response = f"Got it, thanks for confirming it's the same {pending_name}." # Simple ack
                # Clear pending state
                session_state = { "name": session_state.get("name"), "city": session_state.get("city") }

            elif re.search(r'\b(no|nope|wrong|different)\b', user_message, re.IGNORECASE):
                logger.info(f"[{session_id}] User denied existing person '{pending_name}'. Asking for clarification.")
                # Acknowledge it's different. Graphiti should have created a new node
                # when the original message was added (assuming name was different enough).
                # If Graphiti didn't create a distinct node, we might need manual intervention later.
                bot_response = f"Okay, noted that this is a different person than the {pending_name} I knew. Thanks for clarifying."
                # Clear pending state
                session_state = { "name": session_state.get("name"), "city": session_state.get("city") }
                should_process_original = True # Process original message now
            else:
                # Didn't get clear yes/no, repeat confirmation
                bot_response = f"Sorry, I need a clear confirmation. You mentioned {pending_name}. My records mention: {session_state.get('pending_kg_fact')}. Are you referring to the same person? (Yes/No)"

        # --- Generate standard response if original message should be processed --- 
        if should_process_original:
             logger.info(f"[{session_id}] Processing original message after confirmation: '{original_message}'")
             try:
                 # We already added the episode when the user first sent the original message.
                 # We just need to generate a standard response to it now.
                 memory_results = await search_memory(original_message)
                 extracted_memory_facts = [res.fact for res in memory_results if hasattr(res, 'fact') and res.fact and isinstance(res.fact, str)]
                 
                 # Generate response based on the original message context
                 # Use session_state which is now cleared of pending info
                 # Pass original message to generate_sync_response for context
                 bot_response = generate_sync_response(
                      chat_histories[session_id], # History includes original msg, confirm/clarify, and bot responses
                      extracted_memory_facts,
                      original_message, # Pass original message for context
                      session_state["name"],
                      session_state["city"]
                 )

             except Exception as e_process_orig:
                  logger.error(f"[{session_id}] Error processing original message post-confirmation: {e_process_orig}", exc_info=True)
                  bot_response = "Sorry, I encountered an error while processing that information."
        # --- End processing original message ---

        # --- Finalize response if needed --- 
        if bot_response:
             user_details[session_id] = session_state # Update state (cleared or updated)
             chat_histories[session_id].append({"role": "assistant", "content": bot_response})
             # Add bot's confirmation/clarification response to graphiti
             try:
                 await add_message_episode(conversation_id, turn_id + 1, 'bot', bot_response, datetime.now(timezone.utc))
             except Exception as e_add_bot_confirm:
                  logger.error(f"[{session_id}] Failed to store bot confirmation response in graph: {e_add_bot_confirm}", exc_info=True)
             return {"response": bot_response}
        else:
             # Should ideally not happen if state logic is correct
             logger.error(f"[{session_id}] Reached end of pending state handling with no response generated.")
             raise HTTPException(status_code=500, detail="Internal error handling conversation state.")

    else:
        # --- Standard Flow (No Pending State) ---
        logger.info(f"[{session_id}] No pending state, processing normally.")

        # --- Add User Episode FIRST (Let Graphiti handle entity creation) ---
        try:
            await add_message_episode(
                conversation_id, turn_id, 'user', user_message, request_start_time
            )
            logger.info(f"[{session_id}] Added user message episode (standard flow).")
        except Exception as e_add_user:
            logger.error(f"[{session_id}] Error adding user message episode (standard flow): {e_add_user}", exc_info=True)
            # If adding fails, we probably can't continue meaningfully
            raise HTTPException(status_code=500, detail="Error processing message with knowledge graph.")

        # --- Extract Entities (after adding episode) ---
        extracted_entities = extract_entities(user_message)
        extracted_self_name = extracted_entities.get("self_name")
        extracted_self_city = extracted_entities.get("self_city")
        mentioned_persons = extracted_entities.get("mentioned_persons", [])
        logger.info(f"[{session_id}] LLM Extracted Entities - SelfName: '{extracted_self_name}', SelfCity: '{extracted_self_city}', Mentioned: {mentioned_persons}")

        # --- Update User Details --- 
        # Use .copy() to avoid modifying the original dict during iteration if necessary
        current_session_state = session_state.copy()
        if extracted_self_name and not current_session_state["name"]:
            current_session_state["name"] = extracted_self_name
            logger.info(f"[{session_id}] Updated session name: {extracted_self_name}")
        if extracted_self_city and not current_session_state["city"]:
            current_session_state["city"] = extracted_self_city
            logger.info(f"[{session_id}] Updated session city: {extracted_self_city}")
        # No need to copy back to user_details yet, wait until end of processing

        # --- Check Self Name (for info, not confirmation trigger) ---
        person_exists_from_intro = False
        person_fact_from_intro = None
        if extracted_self_name:
             try:
                 # Check details but don't trigger confirmation based on self-name
                 self_details = await get_person_details(extracted_self_name)
                 if self_details and self_details["exists"]:
                     person_exists_from_intro = True
                     # Log timestamp info for debugging
                     created_at = self_details.get("created_at")
                     logger.info(f"[{session_id}] Self-introduced person '{extracted_self_name}' exists. CreatedAt: {created_at}")
                     # Get fact, might be useful for standard response context
                     person_fact_from_intro = await get_person_fact(extracted_self_name)
                 else:
                     logger.info(f"[{session_id}] Self-introduced person '{extracted_self_name}' not found or details error.")
             except Exception as e_kg_self:
                  logger.error(f"[{session_id}] Error checking self-introduced person '{extracted_self_name}': {e_kg_self}", exc_info=True)

        # --- Check Entities & Trigger Confirmation --- 
        # Combine self-name and mentioned names for checking
        names_to_check = ([extracted_self_name] if extracted_self_name else []) + mentioned_persons
        # Deduplicate while preserving order (important if self_name is also mentioned)
        unique_names_to_check = list(dict.fromkeys(names_to_check))

        triggered_confirmation = False
        if unique_names_to_check:
            logger.info(f"[{session_id}] Checking KG for potential confirmation: {unique_names_to_check}")
            for person_name in unique_names_to_check:
                try:
                    person_details = await get_person_details(person_name)
                    if person_details and person_details["exists"]:
                        created_at = person_details.get("created_at")
                        is_preexisting = False
                        if created_at and isinstance(created_at, datetime) and created_at.tzinfo:
                            if created_at < (request_start_time - timedelta(seconds=10)):
                                is_preexisting = True

                        if is_preexisting: 
                            logger.info(f"[{session_id}] Person '{person_name}' pre-existed.")
                            person_fact_confirm = await get_person_fact(person_name) # Get formatted fact specifically for confirmation
                            
                            # Store the fact retrieved during self-check if it was the self-name
                            if person_name == extracted_self_name:
                                person_fact_from_intro = person_fact_confirm 

                            if person_fact_confirm: # Need a fact to ask for confirmation
                                # --- Trigger Confirmation State --- 
                                current_session_state["pending_confirmation_type"] = "person_disambiguation"
                                current_session_state["pending_person_name"] = person_name
                                current_session_state["pending_kg_fact"] = person_fact_confirm
                                current_session_state["pending_original_message"] = user_message
                                user_details[session_id] = current_session_state # SAVE state NOW

                                # Adjust question slightly if it's self-introduction vs mention
                                if person_name == extracted_self_name:
                                    confirmation_question = f"You introduced yourself as {person_name}. I might know you already - {person_fact_confirm}. Is this correct? Please confirm. (Yes/No)"
                                else:
                                    confirmation_question = f"You mentioned {person_name}. I know someone by that name: {person_fact_confirm}. Are you referring to the same person? (Yes/No)"
                                
                                chat_histories[session_id].append({"role": "assistant", "content": confirmation_question})
                                 # Add bot's confirmation question to graphiti
                                try:
                                    await add_message_episode(conversation_id, turn_id + 1, 'bot', confirmation_question, datetime.now(timezone.utc))
                                except Exception as e_add_bot_q:
                                    logger.error(f"[{session_id}] Failed to store bot confirmation question in graph: {e_add_bot_q}", exc_info=True)
                                logger.info(f"[{session_id}] Triggered confirmation state for '{person_name}'")
                                triggered_confirmation = True
                                return {"response": confirmation_question} # Return immediately
                            else:
                                logger.info(f"[{session_id}] Person '{person_name}' pre-existed but no fact found, skipping confirmation.") 
                        else:
                            # Entity exists but was likely created by the current message's add_episode call
                            logger.info(f"[{session_id}] Person '{person_name}' appears newly created or timestamp invalid/missing, skipping confirmation.")
                    else:
                        # Logged already inside get_person_details if not found
                        pass 
                except Exception as e_kg_mention:
                    logger.error(f"[{session_id}] Error checking mentioned person '{person_name}': {e_kg_mention}", exc_info=True)
                    # Continue to next person if error occurs
                
                if triggered_confirmation: break # Stop checking after first confirmation
        
        # --- Standard Flow Continues if NO confirmation triggered --- 
        if not triggered_confirmation:
            # Search Memory
            extracted_memory_facts = []
            try:
                memory_search_results_raw = await search_memory(user_message)
                logger.info(f"[{session_id}] Graphiti task: Raw memory search returned {len(memory_search_results_raw)} results.")
                extracted_memory_facts = [res.fact for res in memory_search_results_raw if hasattr(res, 'fact') and res.fact and isinstance(res.fact, str)]
                logger.info(f"[{session_id}] Graphiti task: Extracted {len(extracted_memory_facts)} memory facts.")
            except Exception as e_search:
                logger.error(f"[{session_id}] Error searching memory: {e_search}", exc_info=True)

            # Check City (mostly for logging/potential future use)
            city_exists = False 
            if extracted_self_city:
                 try:
                     city_exists = await check_city_exists(extracted_self_city)
                     logger.info(f"[{session_id}] KG Check: City '{extracted_self_city}' exists: {city_exists}")
                 except Exception as e_kg_city:
                      logger.error(f"[{session_id}] Error checking city in KG for '{extracted_self_city}': {e_kg_city}", exc_info=True)

            # Generate Standard Bot Response
            bot_response = generate_sync_response(
                chat_histories[session_id],
                extracted_memory_facts,
                user_message,
                current_session_state["name"],
                current_session_state["city"]
            )

            # Store Bot Response
            bot_turn_id = turn_id + 1
            bot_timestamp = datetime.now(timezone.utc)
            try:
                await add_message_episode(
                    conversation_id, bot_turn_id, 'bot', bot_response, bot_timestamp
                )
                logger.info(f"[{session_id}] Added bot response episode (standard flow).")
            except Exception as e_add_bot:
                logger.error(f"[{session_id}] Failed to store bot response in graph: {e_add_bot}", exc_info=True)

            # Finalize standard flow
            user_details[session_id] = current_session_state # Save final state
            chat_histories[session_id].append({"role": "assistant", "content": bot_response})
            logger.info(f"[{session_id}] Bot response generated (standard flow): {bot_response}")
            return {"response": bot_response}


# --- Removed Flask-specific code ---
# - @app.before_request
# - _run_all_async_tasks helper
# - @app.teardown_appcontext
# - atexit registration
# - app.run() in main block

# --- Main Execution ---
# (No __main__ block needed when running with Uvicorn)
# Uvicorn will import the 'app' object from this file.
# Run with: uvicorn main:app --reload --port 5001 
