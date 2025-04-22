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
    graphiti_instance = await get_graphiti_instance() # Get the initialized instance
    if graphiti_instance:
        logger.info("Graphiti instance obtained successfully.")
        try:
            logger.info("Running Graphiti build_indices_and_constraints()...")
            await graphiti_instance.build_indices_and_constraints()
            logger.info("Graphiti build_indices_and_constraints() completed successfully.")
        except Exception as e_init:
            logger.error(f"Error during Graphiti build_indices_and_constraints(): {e_init}", exc_info=True)
            # Decide how to handle failure: Continue? Raise? Log and continue?
            # For now, log and continue, but the app might be in a bad state.
    else:
        logger.error("Failed to obtain Graphiti instance during startup.")
        # Optionally raise an error here to prevent app startup if Graphiti is critical

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
def extract_entities(user_message: str, last_bot_message_content: Optional[str] = None) -> Dict[str, Any]:
    """Uses OpenAI LLM to extract self-introduced name/city and mentioned persons, considering previous bot context."""
    logger.info(f"Attempting Enhanced LLM extraction for: '{user_message}', Last Bot Msg: '{last_bot_message_content}'")
    
    context_instruction = ""
    if last_bot_message_content:
        context_instruction = f"""CRITICAL CONTEXT: The previous bot message was: "{last_bot_message_content}"
        Use this context to interpret the user's message:
        - If the bot asked for the user's NAME and the user's current message appears to be *only* a name (e.g., 'Jane Doe'), classify that name as SELF_NAME.
        - If the bot asked for the user's CITY/LOCATION and the user's current message appears to be *only* a location (e.g., 'Gdynia'), classify that location as SELF_CITY.
        Apply this context rule *even if* the user message lacks introductory phrases like 'I am' or 'I live in'."""
        
    system_prompt = f"""You are an expert entity extractor. Analyze the user message provided.
{context_instruction}
Identify the following, prioritizing context rules if provided:
1. SELF_NAME: The user's own name. Extract if they use introductory phrases (e.g., 'I am...') OR if they provide *only* a name in direct response to a bot question asking for it (see context rules).
2. SELF_CITY: The city/location. Extract if they mention it explicitly OR if they provide *only* a location in direct response to a bot question asking for it (see context rules).
3. MENTIONED_PERSONS: A list of *other* people's full names mentioned in the text (NOT the user's own name identified as SELF_NAME).
Return the result ONLY as a valid JSON object with keys 'self_name', 'self_city', and 'mentioned_persons'.
Use the actual extracted strings as values. If an entity type is not found, use null for 'self_name'/'self_city' or an empty list [] for 'mentioned_persons'.
Example 1: Input: 'Hi, I am Helena from Gdynia.' -> Output: {{"self_name": "Helena", "self_city": "Gdynia", "mentioned_persons": []}}
Example 2 (with context): Last Bot Msg: "What is your name?" User Input: "Jane Doe" -> Output: {{"self_name": "Jane Doe", "self_city": null, "mentioned_persons": []}}
Example 3 (with context): Last Bot Msg: "Which city?" User Input: "London" -> Output: {{"self_name": null, "self_city": "London", "mentioned_persons": []}}
Example 4: Input: 'Tell me about Mariusz.' -> Output: {{"self_name": null, "self_city": null, "mentioned_persons": ["Mariusz"]}}
Don't confuse a person's name with a name of an organization or a place.
""" 

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_message}
    ]
    
    default_result = {"self_name": None, "self_city": None, "mentioned_persons": []}

    try:
        response = sync_openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,
            temperature=0.1,
            max_tokens=100, 
            response_format={"type": "json_object"}
        )

        if response.choices and response.choices[0].message and response.choices[0].message.content:
            content = response.choices[0].message.content.strip()
            logger.debug(f"Enhanced LLM Extraction Raw Response: {content}")
            try:
                data = json.loads(content)
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

QUESTION_WORDS = (
    "what", "who", "where", "when", "why", "how", 
    "is", "are", "am", "was", "were", 
    "do", "does", "did", 
    "can", "could", "will", "would", "should", "may", "might",
    "tell me", "describe", "explain", "list"
)

def is_direct_question(text: str) -> bool:
    """Check if the text likely starts with a question word or ends with a question mark."""
    cleaned_text = text.strip().lower()
    # Check if it starts with any of the question words
    starts_with_question_word = cleaned_text.startswith(QUESTION_WORDS)
    # Check if it ends with a question mark
    ends_with_question_mark = cleaned_text.endswith('?')
    
    # Consider it a question if either condition is true
    is_q = starts_with_question_word or ends_with_question_mark
    logger.debug(f"is_direct_question check for '{text[:50]}...': StartsWord={starts_with_question_word}, EndsMark={ends_with_question_mark} -> Result: {is_q}")
    return is_q

# --- NEW Function for Acknowledgement ---
def generate_acknowledgement(user_message: str, last_bot_message_content: Optional[str] = None) -> str:
    """Uses LLM to generate a brief acknowledgement/paraphrase of the user message, considering bot context."""
    logger.info(f"Generating acknowledgement for: '{user_message[:100]}...', Last Bot Msg: '{last_bot_message_content}'")
    
    context_hint = ""
    if last_bot_message_content:
        context_hint = f"The bot's previous message was: \"{last_bot_message_content}\"."
        
    system_prompt = f"""You are an AI assistant. The user just sent a message. Your task is to provide a concise but short acknowledgement that specifically reflects the key information provided by the user (e.g., who, what, where, when).
{context_hint}
Examples:
- User: 'I am Jan Adam, a cyclist from Toruń.' -> Bot: 'Got it, you are Jan Adam, a cyclist from Toruń.'
- User: 'I saw the mural festival last weekend.' -> Bot: 'Okay, you saw the mural festival last weekend.'
- User: 'Mariusz is my cousin.' -> Bot: 'Understood, Mariusz is your cousin.'
- User: 'Gdynia' (Context: Bot asked 'Which city?') -> Bot: 'Okay, your city is Gdynia.'
Focus on accuracy and conciseness.""" 
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": f"The user's message was: \"{user_message}\". Generate the specific acknowledgement."} 
    ]
    
    try:
        response = sync_openai_client.chat.completions.create(
            model="gpt-4.1-nano", # Or a faster/cheaper model if preferred
            messages=messages,
            temperature=0.2,
            max_tokens=50, 
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            ack = response.choices[0].message.content.strip()
            logger.info(f"Generated acknowledgement: '{ack}'")
            return ack
        else:
            logger.warning("Acknowledgement generation response was empty.")
            return "Okay, noted." # Fallback
    except Exception as e:
        logger.error(f"Error generating acknowledgement: {e}", exc_info=True)
        return "Okay, got it." # Fallback

# --- LLM Response Generation (MODIFIED - Now ONLY generates the follow-up question) ---
def generate_sync_response(conversation_history: list[dict],
                           all_relevant_memory: list[str],
                           user_message: str, # Still useful for context
                           user_name: Optional[str],
                           user_city: Optional[str]
                           ) -> str:
    """Generates the art-world related follow-up question."""
    logger.info(f"Generating follow-up question based on user: {user_name or 'Unknown'} in {user_city or 'Unknown'}")
    
    # Determine focus: info gathering or specific art question
    prompt_focus = "standard"
    missing_info = []
    if not user_name: missing_info.append("the user's name")
    if not user_city: missing_info.append("the user's city")

    # Simple check: if we have name and city, ask about art, otherwise prioritize missing info.
    if user_name and user_city:
        prompt_focus = "art_question"
        directive = "Your task is to ask ONE specific, curious follow-up question about the art scene (artists, galleries, events, styles, communities) in the user's city, potentially relating to the conversation history or memory facts. The output MUST be a single question ending with a question mark (?)" 
    else:
        prompt_focus = "info_gathering"
        directive = f"Your task is to politely ask ONE clear question to gather ONE piece of missing information: {' or '.join(missing_info)}. Only ask for name or city. The output MUST be a single question ending with a question mark (?)"

    system_prompt = f"""You are Art Buddy, a cultural archivist. Your goal is to learn about the current art scene in the user'scity and the user's connection to it.

CURRENT TASK: {prompt_focus.upper()}

{directive}

*   Keep your questions concise.
*   Use relevant memory facts (provided below, if any) for context ONLY when asking art questions.
*   Do NOT acknowledge the user's previous message here; that was handled separately.
*   Phrase your output ONLY as the question itself.

Your interests related to the art scene:
- Spatial clustering of cultural venues - exact places/areas where art community meets, wher art is made, displayed, etc. - identify the hubs
- Upcomming events, projects, initiatives - exact dates, locations, names
- Important individuals in the scene - names, roles, connections
- Collaborative art-world networks
- Aestheticization of public space
- Diverse cultural activities
- Social bonding & cultural cohesion
"""

    prompt_messages = [{'role': 'system', 'content': system_prompt.strip()}]

    if prompt_focus == "art_question" and all_relevant_memory:
        memory_context = "Relevant Memory Facts (Context): " + "; ".join(all_relevant_memory)
        prompt_messages.append({"role": "system", "content": memory_context})

    # Include conversation history for context
    max_turns_for_context = 10
    history_start_index = max(0, len(conversation_history) - (max_turns_for_context * 2))
    prompt_messages.extend(conversation_history[history_start_index:])
    # Add the latest user message for immediate context, even if history is long
    # Note: History already includes the latest user message from send_message
    # prompt_messages.append({"role": "user", "content": user_message})

    logger.debug(f"Sending Follow-up Question prompt to OpenAI: {prompt_messages}")

    try:
        response = sync_openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=prompt_messages,
            temperature=0.7,
            max_tokens=70, # Reduced slightly as it's just a question
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            question = response.choices[0].message.content.strip()
            if not question.endswith('?'):
                logger.warning(f"Generated follow-up did not end with '?'. Appending one. Original: '{question}'")
                question += ' ?' # Add space for safety
            logger.info(f"Generated follow-up question: '{question}'")
            return question
        else:
            logger.warning("Follow-up question generation response was empty.")
            fallback = f"What else can you tell me about the art scene in {user_city or 'your city'}?"
            logger.info(f"Using fallback question: '{fallback}'")
            return fallback 
    except Exception as e:
        logger.error(f"Error generating follow-up question: {e}", exc_info=True)
        fallback = f"What's interesting in the {user_city or 'local'} art scene right now?"
        logger.info(f"Using fallback question on error: '{fallback}'")
        return fallback 

async def answer_question_from_kg(query: str, session_id: str) -> str:
    """Attempts to answer a user's question using knowledge graph search."""
    logger.info(f"[{session_id}] Attempting KG search to answer query: '{query}'")
    try:
        # Reuse existing search function which calls graphiti.search
        memory_results = await search_memory(query)
        if memory_results:
            # Extract facts - limit the number for conciseness
            facts = [res.fact for res in memory_results if hasattr(res, 'fact') and res.fact and isinstance(res.fact, str)]
            if facts:
                # Simple formatting: join top 3 facts
                answer = "Based on what I know: " + "; ".join(facts[:3])
                logger.info(f"[{session_id}] Generated answer from KG search results: '{answer}'")
                return answer
            else:
                logger.info(f"[{session_id}] KG search returned results but no usable facts extracted.")
                return "I found some related information, but couldn't form a direct answer."
        else:
            logger.info(f"[{session_id}] KG search returned no results for the query.")
            return "I don't have specific information about that in my memory right now."
    except Exception as e:
        logger.error(f"[{session_id}] Error during KG search for answering question '{query}': {e}", exc_info=True)
        return "Sorry, I encountered an error trying to look that up."


# --- FastAPI Routes ---

# Pydantic model for request body
class MessageRequest(BaseModel):
    message: str
    session_id: str

@app.get("/")
async def index(request: Request):
    """Serves the main HTML page and initiates the conversation."""
    session_id = str(uuid.uuid4())
    conversation_id = f"session_{session_id}"
    logger.info(f"New session started: {session_id}")

    # Initialize chat history and user details (as before)
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

    # --- Generate Initial Bot Message --- 
    initial_bot_message = "Hi! I'm Art Buddy. I'd love to learn about the local art scene. To start, could you tell me your name, which city you're in, and how you're connected to the art world there?"
    # (Optional: Replace above with a call to generate_sync_response or a new dedicated function if more dynamic intro is needed)
    logger.info(f"[{session_id}] Generated initial bot message: {initial_bot_message}")

    # Add initial message to history 
    chat_histories[session_id].append({"role": "assistant", "content": initial_bot_message})
    # <<< EDIT: Removed adding initial message episode to graph >>>
    # try:
    #     initial_turn_id = 0 
    #     await add_message_episode(conversation_id, initial_turn_id, 'bot', initial_bot_message, datetime.now(timezone.utc))
    #     logger.info(f"[{session_id}] Added initial bot message episode to graph.")
    # except Exception as e_add_initial:
    #      logger.error(f"[{session_id}] Failed to store initial bot message in graph: {e_add_initial}", exc_info=True)

    try:
        # Pass the initial message to the template
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request, 
                "session_id": session_id, 
                "initial_bot_message": initial_bot_message # Pass the message
            }
        )
    except NameError:
         logger.critical("Jinja2Templates object 'templates' not initialized. Cannot serve HTML.")
         raise HTTPException(status_code=500, detail="Server configuration error: Template engine not found.")


@app.post("/send_message")
async def send_message(message_req: MessageRequest):
    """Handles incoming chat messages, runs async tasks, generates response."""
    user_message = message_req.message
    session_id = message_req.session_id

    # <<< EDIT: Remove old inactivity timeout check block >>>
    # --- Session Initialization Check --- 
    if session_id not in user_details:
        # If session doesn't exist (e.g., after duration end), maybe return error or redirect
        logger.warning(f"Received message for unknown/expired session ID: {session_id}. Denying request.")
        raise HTTPException(status_code=404, detail="Session not found or has ended.")
        
    # --- Proceed with normal processing --- 
    turn_id = len(chat_histories[session_id]) + 1
    conversation_id = f"session_{session_id}"
    request_start_time = datetime.now(timezone.utc) 

    chat_histories[session_id].append({"role": "user", "content": user_message})
    logger.info(f"[{session_id}] User message received (turn {turn_id}): {user_message}")

    # Get current state (Should exist due to check above)
    session_state = user_details.get(session_id, {}).copy() 
    # Log loaded state
    logger.info(f"[{session_id}] Loaded state at request start: Name='{session_state.get('name')}', City='{session_state.get('city')}', StartTime='{session_state.get('session_start_time')}'")
        
    pending_type = session_state.get("pending_confirmation_type")
    confirmed_names_set = session_state.get("confirmed_or_processed_names", set())
    session_start_time = session_state.get("session_start_time") # Get start time
    
    # --- State Machine Logic --- 
    if pending_type:
        logger.info(f"[{session_id}] Handling pending state: {pending_type}")
        # --- Handle Response to Pending Confirmation ---
        bot_response = None
        should_process_original = False
        original_message_to_add = session_state.get("pending_original_message")
        original_turn_id_to_add = session_state.get("pending_original_turn_id")
        original_timestamp_to_add = session_state.get("pending_original_timestamp")
        # pending_name = session_state.get("pending_person_name") # May be string or list now

        # --- SINGLE PERSON CONFIRMATION (Original Logic - Keep for robustness) ---
        if pending_type == "person_disambiguation":
            pending_name = session_state.get("pending_person_name")
            logger.debug(f"[{session_id}] Handling single confirmation for '{pending_name}'")
            # (Keep original handling logic for single name confirmation here)
            if re.search(r'\b(yes|yeah|correct|true|confirm)\b', user_message, re.IGNORECASE):
                logger.info(f"[{session_id}] User confirmed existing person '{pending_name}'.")
                should_process_original = True
                session_state = { "name": session_state.get("name"), "city": session_state.get("city") } # Clear pending state
                confirmed_names_set.add(pending_name) # Add the single confirmed name
                session_state["confirmed_or_processed_names"] = confirmed_names_set
                session_state["session_start_time"] = session_start_time # Ensure start time persists
                bot_response = None # Will proceed to process original message
            elif re.search(r'\b(no|nope|wrong|different)\b', user_message, re.IGNORECASE):
                logger.info(f"[{session_id}] User denied existing person '{pending_name}'.")
                bot_response = f"Okay, noted that this is a different person than the {pending_name} I knew. Thanks for clarifying."
                session_state = { "name": session_state.get("name"), "city": session_state.get("city") } # Clear pending state
                confirmed_names_set.add(pending_name) # Add the single denied name
                session_state["confirmed_or_processed_names"] = confirmed_names_set
                session_state["session_start_time"] = session_start_time # Ensure start time persists
                should_process_original = False
            else:
                # Didn't get clear yes/no, repeat original single confirmation question
                bot_response = f"Sorry, I need a clear confirmation. You mentioned {pending_name}. My records mention: {session_state.get('pending_kg_fact')}. Are you referring to the same person? (Yes/No)"
                should_process_original = False 
        
        # --- MULTI PERSON CONFIRMATION SEQUENCE --- 
        elif pending_type == "person_disambiguation_multi":
            # <<< EDIT: Reworked multi-confirmation handler >>>
            # Retrieve sequence state
            confirmations_list = session_state.get("pending_confirmations_list", []) # List of {name, fact} dicts
            current_index = session_state.get("pending_current_person_index", -1)
            original_message_to_add = session_state.get("pending_original_message")
            original_turn_id_to_add = session_state.get("pending_original_turn_id")
            original_timestamp_to_add = session_state.get("pending_original_timestamp")

            # Basic state validation
            if not confirmations_list or not isinstance(confirmations_list, list) or current_index < 0 or current_index >= len(confirmations_list):
                logger.error(f"[{session_id}] Invalid state for multi-confirmation: List='{confirmations_list}', Index={current_index}. Aborting.")
                # Clear pending state and give error response
                session_state = { "name": session_state.get("name"), "city": session_state.get("city"), "session_start_time": session_start_time, "confirmed_or_processed_names": confirmed_names_set }
                user_details[session_id] = session_state
                bot_response = "Sorry, something went wrong with tracking the confirmation sequence. Let's try again later."
                # Add this error response as an episode? Maybe not.
                return {"response": bot_response} 
            
            current_person_info = confirmations_list[current_index]
            current_person_name = current_person_info["name"]
            current_kg_fact = current_person_info["fact"] # Fact for the person just asked about
            logger.info(f"[{session_id}] Handling multi-confirmation reply for '{current_person_name}' (Index {current_index}/{len(confirmations_list)-1}).")

            # Add current person to processed set regardless of answer
            confirmed_names_set.add(current_person_name)
            session_state["confirmed_or_processed_names"] = confirmed_names_set # Keep set updated

            # --- Handle User Response --- 
            if re.search(r'\b(yes|yeah|correct|true|confirm)\b', user_message, re.IGNORECASE):
                # --- User said YES --- 
                logger.info(f"[{session_id}] User confirmed '{current_person_name}'.")
                next_index = current_index + 1

                if next_index < len(confirmations_list): # More people to confirm?
                    # --- Ask about NEXT person --- 
                    next_person_info = confirmations_list[next_index]
                    next_person_name = next_person_info["name"]
                    next_kg_fact = next_person_info["fact"] # Fact is already stored
                    logger.info(f"[{session_id}] Proceeding to next confirmation: '{next_person_name}'.")

                    # Update state for next question
                    session_state["pending_current_person_index"] = next_index
                    # No need to update fact list, just index
                    user_details[session_id] = session_state # Save updated state
                    
                    # Build question for next person
                    # TODO: Refine check for self-intro case if needed?
                    bot_response = f"Thanks. Now, you also mentioned {next_person_name}. I know someone by that name: {next_kg_fact}. Are you referring to them? (Yes/No)"
                    
                    return {"response": bot_response} 
                else: 
                    # --- LAST person confirmed YES --- 
                    logger.info(f"[{session_id}] Confirmation sequence complete successfully.")
                    # Clear pending state
                    session_state = { 
                        "name": session_state.get("name"), 
                        "city": session_state.get("city"), 
                        "session_start_time": session_start_time,
                        "confirmed_or_processed_names": confirmed_names_set
                    } 
                    user_details[session_id] = session_state
                    should_process_original = True # Signal to process original message outside this block
                    bot_response = None # No immediate response needed

            elif re.search(r'\b(no|nope|wrong|different)\b', user_message, re.IGNORECASE):
                # --- User said NO --- 
                logger.info(f"[{session_id}] User denied '{current_person_name}'. Aborting sequence.")
                # Clear pending state entirely
                session_state = { 
                    "name": session_state.get("name"), 
                    "city": session_state.get("city"), 
                    "session_start_time": session_start_time,
                    "confirmed_or_processed_names": confirmed_names_set
                } 
                user_details[session_id] = session_state
                
                bot_response = f"Okay, noted. To avoid confusion, please refer to {current_person_name} using a different name or a nickname?"
                
                should_process_original = False
                return {"response": bot_response} 
            
            else:
                # --- User response UNCLEAR --- 
                logger.info(f"[{session_id}] Unclear response regarding '{current_person_name}'. Re-asking.")
                # Re-ask for the current person
                bot_response = f"Sorry, I need a clear Yes/No for {current_person_name}. My records mentioned: {current_kg_fact}. Was that the person you meant?"
                
                # State index doesn't change, just save updated confirmed_names_set
                user_details[session_id] = session_state 
                should_process_original = False
                return {"response": bot_response} 
            # <<< END EDIT >>>

        # --- Fallback / Other Pending Types (if any added later) ---
        else:
             logger.warning(f"[{session_id}] Encountered unhandled pending_type: {pending_type}. Clearing state.")
             session_state = { "name": session_state.get("name"), "city": session_state.get("city"), "session_start_time": session_start_time, "confirmed_or_processed_names": confirmed_names_set } 
             user_details[session_id] = session_state
             return {"response": "Sorry, I got a bit confused. Could you please repeat your last message?"}

        # --- Process original message OR finalize bot response --- 
        if should_process_original:
            logger.info(f"[{session_id}] Processing original message '{original_message_to_add}' after successful confirmation sequence.") 
            
            # 1. Add original user message episode
            if original_message_to_add and original_turn_id_to_add and original_timestamp_to_add:
                try:
                    await add_message_episode(
                        conversation_id, original_turn_id_to_add, 'user', 
                        original_message_to_add, original_timestamp_to_add 
                    )
                    logger.info(f"[{session_id}] Added original user message episode post-confirmation sequence.")
                except Exception as e_add_orig_user:
                    logger.error(f"[{session_id}] Error adding original user message episode post-confirmation sequence: {e_add_orig_user}", exc_info=True)
            else:
                logger.error(f"[{session_id}] Failed to add original episode post-sequence: Missing details retrieved from pending state.")

            # 2. Generate and Store Acknowledgement for the original message
            acknowledgement = generate_acknowledgement(original_message_to_add) # Use original message
            chat_histories[session_id].append({"role": "assistant", "content": acknowledgement})
            # Use turn_id + X based on how many confirmation turns happened? Simpler: use current turn_id + 1 for now.
            ack_turn_id = turn_id + 1 
            user_name_context = session_state.get("name") # Use state from *after* sequence clear
            ack_for_graph = f"[User: {user_name_context}] {acknowledgement}" if user_name_context else acknowledgement
            logger.debug(f"Storing acknowledgement in graph post-sequence: '{ack_for_graph}'")
            try:
                await add_message_episode(conversation_id, ack_turn_id, 'bot', ack_for_graph, datetime.now(timezone.utc))
                logger.info(f"[{session_id}] Added acknowledgement episode post-confirmation sequence.")
            except Exception as e_add_ack_post:
                logger.error(f"[{session_id}] Failed to store acknowledgement episode post-confirmation sequence: {e_add_ack_post}", exc_info=True)

            # 3. Check duration BEFORE generating follow-up for original message
            follow_up_content = None
            is_session_over = False
            if session_start_time and isinstance(session_start_time, datetime):
                elapsed_time = datetime.now(timezone.utc) - session_start_time
                max_duration = timedelta(minutes=8)
                logger.info(f"[{session_id}] Duration Check (Post-Sequence): Start='{session_start_time}', Elapsed='{elapsed_time}', Limit='{max_duration}'")
                if elapsed_time > max_duration:
                    is_session_over = True
                    logger.info(f"[{session_id}] Session duration exceeded limit. Ending conversation post-sequence.")
                    follow_up_content = "That's all the time we have for today. Thank you for sharing! Goodbye."
                else:
                     logger.info(f"[{session_id}] Session duration within limit post-sequence.")
            else:
                 logger.warning(f"[{session_id}] Invalid or missing session_start_time post-sequence: '{session_start_time}'. Cannot check duration.")

            if not is_session_over:
                # 4. Search Memory based on original message (for context)
                extracted_memory_facts = []
                try:
                    memory_results = await search_memory(original_message_to_add) # Use original
                    extracted_memory_facts = [res.fact for res in memory_results if hasattr(res, 'fact') and res.fact and isinstance(res.fact, str)]
                except Exception as e_search_orig:
                    logger.error(f"[{session_id}] Error searching memory post-sequence: {e_search_orig}", exc_info=True)
                
                # 5. Generate Follow-up Question (if session not over)
                follow_up_content = generate_sync_response(
                    chat_histories[session_id], 
                    extracted_memory_facts,
                    original_message_to_add, # Use original
                    session_state["name"], # Use state from *after* sequence clear
                    session_state["city"]
                )

            # Append follow-up/goodbye message to history
            chat_histories[session_id].append({"role": "assistant", "content": follow_up_content})

            # Finalize and Return BOTH messages (Ack + Follow-up/Goodbye)
            if not is_session_over:
                 user_details[session_id] = session_state # Save cleared state
            else:
                # Clean up ended session state
                logger.info(f"[{session_id}] Clearing ended session state post-sequence.")
                if session_id in user_details: del user_details[session_id]
                if session_id in chat_histories: del chat_histories[session_id]

            return {"responses": [acknowledgement, follow_up_content]}

        elif bot_response: # Handles denial/re-ask responses generated within the sequence handler
             user_details[session_id] = session_state # Save state (potentially updated index/fact or cleared)
             chat_histories[session_id].append({"role": "assistant", "content": bot_response})
             return {"response": bot_response} 
        else:
             # This path should ideally not be reached if sequence ends successfully (should_process_original=True)
             # or if a response was generated during the sequence (denial/reask)
             logger.error(f"[{session_id}] Reached end of pending state handling with no response logic triggered.")
             raise HTTPException(status_code=500, detail="Internal error handling conversation state.")

    else:
        # --- Standard Flow (No Pending State) ---
        logger.info(f"[{session_id}] No pending state, processing normally.")

        # Get last bot message for context 
        last_bot_message_content = None
        history = chat_histories[session_id]
        if len(history) >= 2: 
            previous_message = history[-2]
            if previous_message.get("role") == "assistant":
                last_bot_message_content = previous_message.get("content")
                logger.debug(f"[{session_id}] Found previous bot message for context: '{last_bot_message_content}'")
            else:
                logger.debug(f"[{session_id}] Previous message was not from assistant (role: {previous_message.get('role')}).")
        else:
            logger.debug(f"[{session_id}] Not enough history (len={len(history)}) to get previous bot message.")
        
        # Extract Entities (with context) 
        extracted_entities = extract_entities(user_message, last_bot_message_content)
        extracted_self_name = extracted_entities.get("self_name")
        extracted_self_city = extracted_entities.get("self_city")
        mentioned_persons = extracted_entities.get("mentioned_persons", [])
        logger.info(f"[{session_id}] LLM Extracted Entities - SelfName: '{extracted_self_name}', SelfCity: '{extracted_self_city}', Mentioned: {mentioned_persons}")

        # --- Update User Details ---
        current_session_state = session_state.copy()
        if extracted_self_name:
            current_session_state["name"] = extracted_self_name
            logger.info(f"[{session_id}] Updated session name: {extracted_self_name}")
        if extracted_self_city:
             # Check if the city actually exists in KG before updating? Optional.
             # city_known = await check_city_exists(extracted_self_city)
             # if city_known:
                 current_session_state["city"] = extracted_self_city
                 logger.info(f"[{session_id}] Updated session city: {extracted_self_city}")
             # else:
             #     logger.info(f"[{session_id}] Extracted city '{extracted_self_city}' not found in KG, not updating state.")

        # --- Check if user message is a direct question --- 
        is_question = is_direct_question(user_message)

        # --- BRANCH: Handle Question vs Standard Flow --- 
        if is_question:
            # --- QUESTION HANDLING FLOW --- 
            logger.info(f"[{session_id}] User message identified as a question. Attempting KG answer.")

            # 1. Get answer from KG
            answer_content = await answer_question_from_kg(user_message, session_id)

            # 2. Check Duration BEFORE finalizing response
            final_response_content = answer_content
            is_session_over = False
            if session_start_time and isinstance(session_start_time, datetime):
                elapsed_time = datetime.now(timezone.utc) - session_start_time
                max_duration = timedelta(minutes=8) # <<< Ensure this is the desired duration
                logger.info(f"[{session_id}] Duration Check (Question Flow): Start='{session_start_time}', Elapsed='{elapsed_time}', Limit='{max_duration}'")
                if elapsed_time > max_duration:
                    is_session_over = True
                    logger.info(f"[{session_id}] Session duration exceeded limit during question answering. Ending conversation.")
                    final_response_content = "That's all the time we have for today. Thank you for sharing! Goodbye."
                else:
                    logger.info(f"[{session_id}] Session duration within limit (Question Flow).")
            else:
                 logger.warning(f"[{session_id}] Invalid or missing session_start_time (Question Flow): '{session_start_time}'. Cannot check duration.")
            
            # 3. Append to History
            chat_histories[session_id].append({"role": "assistant", "content": final_response_content})

            # 4. Save/Clear State
            if not is_session_over:
                # Update processed names just in case entity extraction mentioned someone
                current_session_state["confirmed_or_processed_names"] = confirmed_names_set
                logger.info(f"[{session_id}] Saving final state (Question Flow): {current_session_state}")
                user_details[session_id] = current_session_state
            else:
                logger.info(f"[{session_id}] Clearing ended session state (Question Flow).")
                if session_id in user_details: del user_details[session_id]
                if session_id in chat_histories: del chat_histories[session_id]

            # 5. Return single response
            return {"response": final_response_content}

        else:
            # --- STANDARD NON-QUESTION FLOW --- 
            logger.info(f"[{session_id}] User message is not a question. Proceeding with standard flow.")

            # 1. Check Entities & Trigger Confirmation (if applicable)
            # <<< EDIT: Logic to collect all confirmations and trigger sequence >>>
            confirmations_needed = [] # Collect {name: ..., fact: ...} for sequencing
            triggered_confirmation_sequence = False # Flag if sequence needs to start
            person_fact_from_intro = None

            names_to_check = ([extracted_self_name] if extracted_self_name else []) + mentioned_persons
            unique_names_to_check = list(dict.fromkeys(names_to_check))

            if unique_names_to_check:
                logger.info(f"[{session_id}] Checking KG for potential confirmation sequence: {unique_names_to_check}")
                for person_name in unique_names_to_check:
                    if person_name in confirmed_names_set:
                        logger.info(f"[{session_id}] Skipping check for already processed name: '{person_name}'")
                        continue
                    
                    person_details = None
                    try:
                        person_details = await get_person_details(person_name)
                        if person_details and person_details["exists"]:
                            logger.info(f"[{session_id}] Person '{person_name}' pre-existed.")
                            person_fact_confirm = await get_person_fact(person_name)
                            
                            if person_name == extracted_self_name:
                                person_fact_from_intro = person_fact_confirm 

                            if person_fact_confirm: # Found a fact, add to list for sequence
                                logger.info(f"[{session_id}] Adding '{person_name}' with fact to confirmation sequence list.")
                                confirmations_needed.append({"name": person_name, "fact": person_fact_confirm})
                                # Mark as processed immediately to avoid re-adding if mentioned twice
                                confirmed_names_set.add(person_name) 
                            else: # Exists but no fact
                                if person_name not in confirmed_names_set:
                                    confirmed_names_set.add(person_name)
                                    logger.info(f"[{session_id}] Marked '{person_name}' as processed (found=True, no fact).")
                        else: # Not found in KG
                             if person_name not in confirmed_names_set:
                                 confirmed_names_set.add(person_name)
                                 logger.info(f"[{session_id}] Marked '{person_name}' as processed (found=False).")
                             pass 
                    except Exception as e_kg_mention:
                        logger.error(f"[{session_id}] Error checking mentioned person '{person_name}': {e_kg_mention}", exc_info=True)
                        if person_name not in confirmed_names_set:
                            confirmed_names_set.add(person_name)
                            logger.info(f"[{session_id}] Marked '{person_name}' as processed (Error during check).")

                    # Loop always continues to check all names
            
            # --- AFTER LOOP: Trigger confirmation SEQUENCE if needed --- 
            if confirmations_needed: # Check if the list has items
                logger.info(f"[{session_id}] Starting confirmation sequence for {len(confirmations_needed)} person(s).")
                triggered_confirmation_sequence = True
                
                # Get details for the FIRST person in the sequence
                first_confirmation = confirmations_needed[0]
                first_person_name = first_confirmation["name"]
                first_person_fact = first_confirmation["fact"]

                # Set pending state for multi-confirmation sequence
                current_session_state["pending_confirmation_type"] = "person_disambiguation_multi" # New type
                current_session_state["pending_confirmations_list"] = confirmations_needed # Store list of names and facts
                current_session_state["pending_current_person_index"] = 0 # Start at index 0
                current_session_state["pending_original_message"] = user_message # Original message that triggered this
                current_session_state["pending_original_turn_id"] = turn_id
                current_session_state["pending_original_timestamp"] = request_start_time
                current_session_state["confirmed_or_processed_names"] = confirmed_names_set # Save set including all checked names
                # <<< EDIT: Add flag to track if all confirmations are positive >>>
                current_session_state["pending_all_confirmed"] = True # Initialize as True
                # <<< END EDIT >>>
                user_details[session_id] = current_session_state
                
                # Build the question for the FIRST person
                if first_person_name == extracted_self_name: 
                     confirmation_question = f"You introduced yourself as {first_person_name}. I might know you already. {first_person_fact}. Is this correct? Please confirm. (Yes/No)"
                else:
                     confirmation_question = f"You mentioned {first_person_name}. I know someone by that name. {first_person_fact}. Are you referring to the same person? (Yes/No)"

                chat_histories[session_id].append({"role": "assistant", "content": confirmation_question})
                logger.info(f"[{session_id}] Triggered confirmation sequence, asking about '{first_person_name}'.")
                return {"response": confirmation_question}
            
            # --- Standard Flow Continues if NO confirmation sequence was triggered --- 
            if not triggered_confirmation_sequence:
                
                # <<< EDIT: Add User Episode HERE >>>
                try:
                    await add_message_episode(
                        conversation_id, turn_id, 'user', user_message, request_start_time
                    )
                    logger.info(f"[{session_id}] Added user statement episode (Turn {turn_id}, no confirmation needed).")
                except Exception as e_add_user_s:
                    logger.error(f"[{session_id}] Error adding user statement episode (Turn {turn_id}): {e_add_user_s}", exc_info=True)
                # <<< END EDIT >>>
                
                # 2. Generate and Store Acknowledgement 
                acknowledgement = generate_acknowledgement(user_message, last_bot_message_content)
                chat_histories[session_id].append({"role": "assistant", "content": acknowledgement})
                ack_turn_id = turn_id + 1 # Bot's first response part is next turn
                user_name_context = current_session_state.get("name") 
                ack_for_graph = f"[User: {user_name_context}] {acknowledgement}" if user_name_context else acknowledgement
                logger.debug(f"Storing acknowledgement in graph: '{ack_for_graph}'")
                try:
                    await add_message_episode(conversation_id, ack_turn_id, 'bot', ack_for_graph, datetime.now(timezone.utc))
                    logger.info(f"[{session_id}] Added acknowledgement episode (standard flow).")
                except Exception as e_add_ack:
                    logger.error(f"[{session_id}] Failed to store acknowledgement episode (standard flow): {e_add_ack}", exc_info=True)

                # 3. Check Duration 
                follow_up_content = None
                is_session_over = False
                if session_start_time and isinstance(session_start_time, datetime):
                    elapsed_time = datetime.now(timezone.utc) - session_start_time
                    max_duration = timedelta(minutes=8) # <<< Ensure this is the desired duration
                    logger.info(f"[{session_id}] Duration Check (Std Flow): Start='{session_start_time}', Elapsed='{elapsed_time}', Limit='{max_duration}'")
                    if elapsed_time > max_duration:
                        is_session_over = True
                        logger.info(f"[{session_id}] Session duration exceeded limit. Ending conversation.")
                        follow_up_content = "That's all the time we have for today. Thank you for sharing! Goodbye."
                    else:
                        logger.info(f"[{session_id}] Session duration within limit.")
                else:
                    logger.warning(f"[{session_id}] Invalid or missing session_start_time: '{session_start_time}' (Type: {type(session_start_time)}). Cannot check duration.")

                # 4. Search Memory and Generate Follow-up (if session not over)
                extracted_memory_facts = []
                if not is_session_over:
                    try:
                        memory_search_results_raw = await search_memory(user_message)
                        extracted_memory_facts = [res.fact for res in memory_search_results_raw if hasattr(res, 'fact') and res.fact and isinstance(res.fact, str)]
                    except Exception as e_search:
                        logger.error(f"[{session_id}] Error searching memory: {e_search}", exc_info=True)

                    follow_up_content = generate_sync_response(
                        chat_histories[session_id], # History now includes user msg + acknowledgement
                        extracted_memory_facts,
                        user_message,
                        current_session_state["name"],
                        current_session_state["city"]
                    )
                
                # Append follow-up/goodbye message episode to graph AND history
                # <<< EDIT: REMOVE adding follow_up_content episode >>>
                # follow_up_turn_id = ack_turn_id + 1
                # try:
                #     await add_message_episode(conversation_id, follow_up_turn_id, 'bot', follow_up_content, datetime.now(timezone.utc))
                #     logger.info(f"[{session_id}] Added follow_up/goodbye episode (Turn {follow_up_turn_id}).")
                # except Exception as e_add_follow_up:
                #     logger.error(f"[{session_id}] Failed to store follow_up/goodbye episode: {e_add_follow_up}", exc_info=True)
                # <<< END EDIT >>>
                
                # Append follow-up/goodbye to chat history only
                chat_histories[session_id].append({"role": "assistant", "content": follow_up_content})

                # 6. Save final state (if session isn't over)
                if not is_session_over:
                    current_session_state["confirmed_or_processed_names"] = confirmed_names_set
                    logger.info(f"[{session_id}] Saving final state (standard flow): {current_session_state}")
                    user_details[session_id] = current_session_state
                else:
                    # Clean up ended session state
                    logger.info(f"[{session_id}] Clearing ended session state.")
                    if session_id in user_details: del user_details[session_id]
                    if session_id in chat_histories: del chat_histories[session_id]

                # 7. Return BOTH messages
                return {"responses": [acknowledgement, follow_up_content]}


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
