import logging
import re
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Request, Form, Depends, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

# Import models, state, services, utils
from app.models.message import MessageRequest
from app.state import user_details, chat_histories # Direct state import
from app.services.graphiti import add_message_episode, search_graph # Added search_graph
from app.services.memory import (
    get_person_details, get_person_fact, answer_question_from_kg
    # check_city_exists # Not currently used in this router
)
from app.services.llm import (
    extract_entities, generate_acknowledgement, generate_sync_response
)
from app.utils.parsing import is_direct_question

logger = logging.getLogger(__name__)
router = APIRouter()

# Assume templates are in a 'templates' directory at the project root
templates = Jinja2Templates(directory="templates")


# --- Step 4: Chat Interface ---
@router.post("/chat", response_class=HTMLResponse)
async def post_to_chat(request: Request, session_id: str = Form(...)):
    """Handles submission from instructions and serves the main chat HTML page."""
    logger.info(f"[{session_id}] Proceeding to chat interface.")

    if session_id not in chat_histories or session_id not in user_details:
        logger.warning(f"[{session_id}] Chat requested for unknown/invalid session. Redirecting to start.")
        if session_id in chat_histories: del chat_histories[session_id]
        if session_id in user_details: del user_details[session_id]
        # Redirect to auth router's root
        auth_root_url = request.url_for('get_pin_entry')
        return RedirectResponse(auth_root_url, status_code=303)

    initial_bot_message = "Hi! I'm Art Buddy. I'd love to learn about the local art scene. To start, could you tell me your name, which city you're in, and how you're connected to the art world there?"
    if not chat_histories[session_id]:
        logger.info(f"[{session_id}] Generating initial bot message for chat: {initial_bot_message}")
        chat_histories[session_id].append({"role": "assistant", "content": initial_bot_message})
    else:
        last_message = chat_histories[session_id][-1]
        if last_message["role"] == "assistant":
             initial_bot_message = last_message["content"]
             logger.info(f"[{session_id}] Chat page reloaded, using last bot message: {initial_bot_message}")
        else:
            initial_bot_message = "Please enter your message."
            logger.info(f"[{session_id}] Chat page reloaded after user message, showing generic prompt.")

    try:
        return templates.TemplateResponse(
            "chat.html",
            {
                "request": request,
                "session_id": session_id,
                "initial_bot_message": initial_bot_message,
                "chat_history": chat_histories[session_id]
            }
        )
    except NameError:
         logger.critical("Jinja2Templates object 'templates' not initialized.")
         raise HTTPException(status_code=500, detail="Server configuration error: Template engine not found.")
    except KeyError:
         logger.error(f"[{session_id}] Session data missing unexpectedly when rendering chat. Redirecting to start.")
         auth_root_url = request.url_for('get_pin_entry')
         return RedirectResponse(auth_root_url, status_code=303)


@router.post("/send_message")
async def send_message(message_req: MessageRequest):
    """Handles incoming chat messages, runs async tasks, generates response."""
    user_message = message_req.message
    session_id = message_req.session_id

    if session_id not in user_details or session_id not in chat_histories:
        logger.warning(f"Received message for unknown/expired session ID: {session_id}. Denying request.")
        raise HTTPException(status_code=404, detail="Session not found or has ended.")

    turn_id = len(chat_histories[session_id])
    conversation_id = f"session_{session_id}"
    request_start_time = datetime.now(timezone.utc)

    chat_histories[session_id].append({"role": "user", "content": user_message})
    logger.info(f"[{session_id}] User message received (turn {turn_id}): {user_message}")

    session_state = user_details.get(session_id, {}).copy()
    logger.info(f"[{session_id}] Loaded state: Name='{session_state.get('name')}', City='{session_state.get('city')}'")

    pending_type = session_state.get("pending_confirmation_type")
    confirmed_names_set = session_state.get("confirmed_or_processed_names", set())
    session_start_time = session_state.get("session_start_time")

    # --- State Machine Logic ---
    if pending_type:
        logger.info(f"[{session_id}] Handling pending state: {pending_type}")
        bot_response = None
        should_process_original = False
        original_message_to_add = session_state.get("pending_original_message")
        original_turn_id_to_add = session_state.get("pending_original_turn_id")
        original_timestamp_to_add = session_state.get("pending_original_timestamp")

        # --- MULTI PERSON CONFIRMATION SEQUENCE --- (Single was removed)
        if pending_type == "person_disambiguation_multi":
            confirmations_list = session_state.get("pending_confirmations_list", [])
            current_index = session_state.get("pending_current_person_index", -1)

            if not confirmations_list or not isinstance(confirmations_list, list) or current_index < 0 or current_index >= len(confirmations_list):
                logger.error(f"[{session_id}] Invalid state for multi-confirmation. Aborting.")
                session_state = { "name": session_state.get("name"), "city": session_state.get("city"), "session_start_time": session_start_time, "confirmed_or_processed_names": confirmed_names_set }
                user_details[session_id] = session_state
                bot_response = "Sorry, something went wrong. Let's try again later."
                return {"response": bot_response}

            current_person_info = confirmations_list[current_index]
            current_person_name = current_person_info["name"]
            current_kg_fact = current_person_info["fact"]
            logger.info(f"[{session_id}] Handling multi-confirmation reply for '{current_person_name}'")

            confirmed_names_set.add(current_person_name)
            session_state["confirmed_or_processed_names"] = confirmed_names_set

            if re.search(r'\b(yes|yeah|correct|true|confirm)\b', user_message, re.IGNORECASE):
                logger.info(f"[{session_id}] User confirmed '{current_person_name}'.")
                next_index = current_index + 1
                if next_index < len(confirmations_list):
                    next_person_info = confirmations_list[next_index]
                    next_person_name = next_person_info["name"]
                    next_kg_fact = next_person_info["fact"]
                    logger.info(f"[{session_id}] Proceeding to next confirmation: '{next_person_name}'.")
                    session_state["pending_current_person_index"] = next_index
                    user_details[session_id] = session_state
                    bot_response = f"Thanks. Now, you also mentioned {next_person_name}. I know someone by that name: {next_kg_fact}. Are you referring to them? (Yes/No)"
                    return {"response": bot_response}
                else:
                    logger.info(f"[{session_id}] Confirmation sequence complete successfully.")
                    session_state = { "name": session_state.get("name"), "city": session_state.get("city"), "session_start_time": session_start_time, "confirmed_or_processed_names": confirmed_names_set }
                    user_details[session_id] = session_state
                    should_process_original = True
                    bot_response = None
            elif re.search(r'\b(no|nope|wrong|different)\b', user_message, re.IGNORECASE):
                logger.info(f"[{session_id}] User denied '{current_person_name}'. Aborting sequence.")
                session_state = { "name": session_state.get("name"), "city": session_state.get("city"), "session_start_time": session_start_time, "confirmed_or_processed_names": confirmed_names_set }
                user_details[session_id] = session_state
                bot_response = f"Okay, noted. To avoid confusion, please refer to {current_person_name} using a different name or a nickname?"
                should_process_original = False
                return {"response": bot_response}
            else:
                logger.info(f"[{session_id}] Unclear response regarding '{current_person_name}'. Re-asking.")
                bot_response = f"Sorry, I need a clear Yes/No for {current_person_name}. My records mentioned: {current_kg_fact}. Was that the person you meant?"
                user_details[session_id] = session_state
                should_process_original = False
                return {"response": bot_response}

        else: # Fallback for unknown pending types
             logger.warning(f"[{session_id}] Encountered unhandled pending_type: {pending_type}. Clearing state.")
             session_state = { "name": session_state.get("name"), "city": session_state.get("city"), "session_start_time": session_start_time, "confirmed_or_processed_names": confirmed_names_set }
             user_details[session_id] = session_state
             return {"response": "Sorry, I got a bit confused. Could you please repeat your last message?"}

        # --- Process original message OR finalize bot response from sequence ---
        if should_process_original:
            logger.info(f"[{session_id}] Processing original message '{original_message_to_add}' post-sequence.")
            if original_message_to_add and original_turn_id_to_add and original_timestamp_to_add:
                try:
                    # Use graphiti service
                    await add_message_episode(
                        conversation_id, original_turn_id_to_add, 'user',
                        original_message_to_add, original_timestamp_to_add
                    )
                    logger.info(f"[{session_id}] Added original user episode post-sequence.")
                except Exception as e_add_orig_user:
                    logger.error(f"Error adding original user episode post-sequence: {e_add_orig_user}", exc_info=True)
            else:
                logger.error(f"Failed to add original episode post-sequence: Missing details.")

            # Use llm service
            acknowledgement = generate_acknowledgement(original_message_to_add)
            chat_histories[session_id].append({"role": "assistant", "content": acknowledgement})
            ack_turn_id = turn_id + 1
            user_name_context = session_state.get("name")
            ack_for_graph = f"[User: {user_name_context}] {acknowledgement}" if user_name_context else acknowledgement
            try:
                # Use graphiti service
                await add_message_episode(conversation_id, ack_turn_id, 'bot', ack_for_graph, datetime.now(timezone.utc))
                logger.info(f"[{session_id}] Added acknowledgement episode post-sequence.")
            except Exception as e_add_ack_post:
                logger.error(f"Failed to store ack episode post-sequence: {e_add_ack_post}", exc_info=True)

            follow_up_content = None
            is_session_over = False
            if session_start_time and isinstance(session_start_time, datetime):
                elapsed_time = datetime.now(timezone.utc) - session_start_time
                max_duration = timedelta(minutes=60)
                logger.info(f"[{session_id}] Duration Check (Post-Sequence): Elapsed='{elapsed_time}'")
                if elapsed_time > max_duration:
                    is_session_over = True
                    logger.info(f"[{session_id}] Session duration exceeded limit post-sequence.")
                    follow_up_content = "That's all the time we have for today. Thank you for sharing! Goodbye."
            else:
                 logger.warning(f"Invalid session_start_time post-sequence: '{session_start_time}'.")

            if not is_session_over:
                extracted_memory_facts = []
                try:
                    # Use memory service (which uses graphiti service)
                    memory_results = await search_graph(original_message_to_add)
                    extracted_memory_facts = [res.fact for res in memory_results if hasattr(res, 'fact') and res.fact and isinstance(res.fact, str)]
                except Exception as e_search_orig:
                    logger.error(f"Error searching memory post-sequence: {e_search_orig}", exc_info=True)

                # Use llm service
                follow_up_content = generate_sync_response(
                    chat_histories[session_id],
                    extracted_memory_facts,
                    original_message_to_add,
                    session_state["name"],
                    session_state["city"]
                )

            chat_histories[session_id].append({"role": "assistant", "content": follow_up_content})

            if not is_session_over:
                 user_details[session_id] = session_state
            else:
                logger.info(f"[{session_id}] Clearing ended session state post-sequence.")
                if session_id in user_details: del user_details[session_id]
                if session_id in chat_histories: del chat_histories[session_id]

            return {"responses": [acknowledgement, follow_up_content]}

        elif bot_response: # Handles denial/re-ask responses generated within sequence handler
             user_details[session_id] = session_state
             chat_histories[session_id].append({"role": "assistant", "content": bot_response})
             return {"response": bot_response}
        else:
             logger.error(f"[{session_id}] Reached end of pending state handling unexpectedly.")
             raise HTTPException(status_code=500, detail="Internal error handling conversation state.")

    else:
        # --- Standard Flow (No Pending State) ---
        logger.info(f"[{session_id}] No pending state, processing normally.")

        last_bot_message_content = None
        history = chat_histories[session_id]
        if len(history) >= 2 and history[-2].get("role") == "assistant":
            last_bot_message_content = history[-2].get("content")

        # Use llm service
        extracted_entities = extract_entities(user_message, last_bot_message_content)
        extracted_self_name = extracted_entities.get("self_name")
        extracted_self_city = extracted_entities.get("self_city")
        mentioned_persons = extracted_entities.get("mentioned_persons", [])
        logger.info(f"[{session_id}] LLM Extracted: SelfName='{extracted_self_name}', SelfCity='{extracted_self_city}', Mentioned={mentioned_persons}")

        current_session_state = session_state.copy()
        if extracted_self_name: current_session_state["name"] = extracted_self_name
        if extracted_self_city: current_session_state["city"] = extracted_self_city

        # Use utils
        is_question = is_direct_question(user_message)

        if is_question:
            # --- QUESTION HANDLING FLOW ---
            logger.info(f"[{session_id}] User message is a question. Answering from KG + LLM.")
            # Use memory service to get facts
            # answer_content = await answer_question_from_kg(user_message, session_id)
            kg_facts = await answer_question_from_kg(user_message, session_id)

            # final_response_content = answer_content
            final_response_content = ""
            if kg_facts is None:
                # Handle error or no results from KG
                logger.warning(f"[{session_id}] No facts found or error during KG search for question.")
                # You could have different messages for error vs. no facts
                final_response_content = "I looked for information about that, but couldn't find anything specific in my current memory. Perhaps you could tell me more?"
            else:
                # Use LLM to generate response based on facts and history
                logger.info(f"[{session_id}] Found {len(kg_facts)} facts. Generating LLM answer.")
                try:
                    final_response_content = generate_sync_response(
                        chat_histories[session_id], # Pass full history
                        kg_facts,                   # Pass facts as context
                        user_message,               # Pass the original question
                        current_session_state.get("name"), # Pass user name if known
                        current_session_state.get("city")  # Pass user city if known
                    )
                except Exception as e_llm_answer:
                    logger.error(f"[{session_id}] Error generating LLM answer for question: {e_llm_answer}", exc_info=True)
                    final_response_content = "I found some information, but had trouble phrasing an answer. Can you ask differently?"

            is_session_over = False
            if session_start_time and isinstance(session_start_time, datetime):
                elapsed_time = datetime.now(timezone.utc) - session_start_time
                max_duration = timedelta(minutes=60) # Keep updated limit
                logger.info(f"[{session_id}] Duration Check (Question): Elapsed='{elapsed_time}'")
                if elapsed_time > max_duration:
                    is_session_over = True
                    logger.info(f"[{session_id}] Session duration exceeded (Question). Ending.")
                    # Override LLM response if session ended
                    final_response_content = "That's all the time we have for today. Thank you for sharing! Goodbye."
            else:
                 logger.warning(f"Invalid session_start_time (Question): '{session_start_time}'.")

            chat_histories[session_id].append({"role": "assistant", "content": final_response_content})

            if not is_session_over:
                current_session_state["confirmed_or_processed_names"] = confirmed_names_set
                user_details[session_id] = current_session_state
            else:
                logger.info(f"[{session_id}] Clearing ended session state (Question).")
                if session_id in user_details: del user_details[session_id]
                if session_id in chat_histories: del chat_histories[session_id]

            return {"response": final_response_content}

        else:
            # --- STANDARD NON-QUESTION FLOW ---
            logger.info(f"[{session_id}] User message is not a question. Standard flow.")

            confirmations_needed = []
            triggered_confirmation_sequence = False

            names_to_check = ([extracted_self_name] if extracted_self_name else []) + mentioned_persons
            unique_names_to_check = list(dict.fromkeys(names_to_check))

            if unique_names_to_check:
                logger.info(f"[{session_id}] Checking KG for confirmations: {unique_names_to_check}")
                for person_name in unique_names_to_check:
                    if person_name in confirmed_names_set:
                        logger.info(f"[{session_id}] Skipping check for processed name: '{person_name}'")
                        continue
                    try:
                        # Use memory service
                        person_details = await get_person_details(person_name)
                        if person_details and person_details["exists"]:
                            logger.info(f"[{session_id}] Person '{person_name}' pre-existed.")
                            # Use memory service
                            person_fact_confirm = await get_person_fact(person_name)
                            if person_fact_confirm:
                                logger.info(f"Adding '{person_name}' to confirmation list.")
                                confirmations_needed.append({"name": person_name, "fact": person_fact_confirm})
                                confirmed_names_set.add(person_name)
                            else:
                                if person_name not in confirmed_names_set: confirmed_names_set.add(person_name)
                                logger.info(f"Marked '{person_name}' as processed (found, no fact).")
                        else:
                             if person_name not in confirmed_names_set: confirmed_names_set.add(person_name)
                             logger.info(f"Marked '{person_name}' as processed (not found).")
                    except Exception as e_kg_mention:
                        logger.error(f"Error checking mentioned person '{person_name}': {e_kg_mention}", exc_info=True)
                        if person_name not in confirmed_names_set: confirmed_names_set.add(person_name)
                        logger.info(f"Marked '{person_name}' as processed (error).")

            if confirmations_needed:
                logger.info(f"[{session_id}] Starting confirmation sequence for {len(confirmations_needed)} person(s).")
                triggered_confirmation_sequence = True
                first_confirmation = confirmations_needed[0]
                first_person_name = first_confirmation["name"]
                first_person_fact = first_confirmation["fact"]

                current_session_state["pending_confirmation_type"] = "person_disambiguation_multi"
                current_session_state["pending_confirmations_list"] = confirmations_needed
                current_session_state["pending_current_person_index"] = 0
                current_session_state["pending_original_message"] = user_message
                current_session_state["pending_original_turn_id"] = turn_id
                current_session_state["pending_original_timestamp"] = request_start_time
                current_session_state["confirmed_or_processed_names"] = confirmed_names_set
                user_details[session_id] = current_session_state

                if first_person_name == extracted_self_name:
                     confirmation_question = f"You introduced yourself as {first_person_name}. I might know you already. {first_person_fact}. Is this correct? Please confirm. (Yes/No)"
                else:
                     confirmation_question = f"You mentioned {first_person_name}. I know someone by that name. {first_person_fact}. Are you referring to the same person? (Yes/No)"

                chat_histories[session_id].append({"role": "assistant", "content": confirmation_question})
                logger.info(f"[{session_id}] Triggered confirmation sequence, asking about '{first_person_name}'.")
                return {"response": confirmation_question}

            if not triggered_confirmation_sequence:
                try:
                    # Use graphiti service
                    await add_message_episode(
                        conversation_id, turn_id, 'user', user_message, request_start_time
                    )
                    logger.info(f"[{session_id}] Added user statement episode (Turn {turn_id}).")
                except Exception as e_add_user_s:
                    logger.error(f"Error adding user statement episode (Turn {turn_id}): {e_add_user_s}", exc_info=True)

                # Use llm service
                acknowledgement = generate_acknowledgement(user_message, last_bot_message_content)
                chat_histories[session_id].append({"role": "assistant", "content": acknowledgement})
                ack_turn_id = turn_id + 1
                user_name_context = current_session_state.get("name")
                ack_for_graph = f"[User: {user_name_context}] {acknowledgement}" if user_name_context else acknowledgement
                try:
                    # Use graphiti service
                    await add_message_episode(conversation_id, ack_turn_id, 'bot', ack_for_graph, datetime.now(timezone.utc))
                    logger.info(f"[{session_id}] Added acknowledgement episode (standard flow).")
                except Exception as e_add_ack:
                    logger.error(f"Failed to store ack episode (standard flow): {e_add_ack}", exc_info=True)

                follow_up_content = None
                is_session_over = False
                if session_start_time and isinstance(session_start_time, datetime):
                    elapsed_time = datetime.now(timezone.utc) - session_start_time
                    max_duration = timedelta(minutes=60)
                    logger.info(f"[{session_id}] Duration Check (Std Flow): Elapsed='{elapsed_time}'")
                    if elapsed_time > max_duration:
                        is_session_over = True
                        logger.info(f"[{session_id}] Session duration exceeded. Ending.")
                        follow_up_content = "That's all the time we have for today. Thank you for sharing! Goodbye."
                else:
                    logger.warning(f"Invalid session_start_time: '{session_start_time}'.")

                extracted_memory_facts = []
                if not is_session_over:
                    try:
                        # Use memory service (which uses graphiti service)
                        memory_search_results_raw = await search_graph(user_message)
                        extracted_memory_facts = [res.fact for res in memory_search_results_raw if hasattr(res, 'fact') and res.fact and isinstance(res.fact, str)]
                    except Exception as e_search:
                        logger.error(f"Error searching memory: {e_search}", exc_info=True)

                    # Use llm service
                    follow_up_content = generate_sync_response(
                        chat_histories[session_id],
                        extracted_memory_facts,
                        user_message,
                        current_session_state["name"],
                        current_session_state["city"]
                    )

                chat_histories[session_id].append({"role": "assistant", "content": follow_up_content})

                if not is_session_over:
                    current_session_state["confirmed_or_processed_names"] = confirmed_names_set
                    user_details[session_id] = current_session_state
                else:
                    logger.info(f"[{session_id}] Clearing ended session state.")
                    if session_id in user_details: del user_details[session_id]
                    if session_id in chat_histories: del chat_histories[session_id]

                return {"responses": [acknowledgement, follow_up_content]} 