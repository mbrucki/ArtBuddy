import logging
import json
from typing import Optional, Dict, Any, List

from openai import OpenAI

# Import config values
from app.config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

# --- Initialize OpenAI Client (Sync) ---
try:
    # Can be initialized globally as it's thread-safe
    sync_openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("Sync OpenAI client initialized.")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
    # Depending on criticality, might want to raise or sys.exit here
    sync_openai_client = None # Ensure it's None if init fails


# --- LLM Interaction Functions ---

def extract_entities(user_message: str, last_bot_message_content: Optional[str] = None) -> Dict[str, Any]:
    """Uses OpenAI LLM to extract self-introduced name/city and mentioned persons."""
    if not sync_openai_client:
        logger.error("OpenAI client not initialized. Cannot extract entities.")
        return {"self_name": None, "self_city": None, "mentioned_persons": []}

    logger.info(f"LLM extraction for: '{user_message}', Last Bot Msg: '{last_bot_message_content}'")
    # ... (rest of the function body from original main.py) ...
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

    return default_result

def generate_acknowledgement(user_message: str, last_bot_message_content: Optional[str] = None) -> str:
    """Uses LLM to generate a brief acknowledgement/paraphrase of the user message."""
    if not sync_openai_client:
        logger.error("OpenAI client not initialized. Cannot generate acknowledgement.")
        return "Okay, got it." # Fallback

    logger.info(f"Generating acknowledgement for: '{user_message[:100]}...', Last Bot Msg: '{last_bot_message_content}'")
    # ... (rest of the function body from original main.py) ...
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

def generate_sync_response(
    conversation_history: List[Dict[str, str]],
    all_relevant_memory: List[str],
    user_message: str, # Still useful for context
    user_name: Optional[str],
    user_city: Optional[str]
) -> str:
    """Generates the art-world related follow-up question using LLM."""
    if not sync_openai_client:
        logger.error("OpenAI client not initialized. Cannot generate response.")
        fallback = f"What's interesting in the {user_city or 'local'} art scene right now?"
        logger.info(f"Using fallback question on error: '{fallback}'")
        return fallback

    logger.info(f"Generating follow-up question based on user: {user_name or 'Unknown'} in {user_city or 'Unknown'}")
    # ... (rest of the function body from original main.py) ...
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