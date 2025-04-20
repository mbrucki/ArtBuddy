import asyncio
import logging
import os
import sys
import uuid # For conversation ID
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import AsyncOpenAI # Use async client

# --- Ensure chatbot_memory functions are importable ---
# Assuming chatbot_memory.py is in the same directory
try:
    from chatbot_memory import add_message_episode, search_memory, Graphiti
except ImportError as e:
    print(f"Error importing from chatbot_memory.py: {e}", file=sys.stderr)
    print("Please ensure chatbot_memory.py is in the same directory.", file=sys.stderr)
    sys.exit(1)

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
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


# --- OpenAI Client ---
# Use a context manager potentially if needed for cleanup, but typically not required
# for standard client usage.
try:
    aclient = AsyncOpenAI(api_key=openai_api_key)
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
    raise


async def generate_response(conversation_history: list[dict], relevant_memory: list[str]) -> str:
    """Generates a chatbot response using OpenAI, considering history and memory."""
    prompt_messages = [
        {"role": "system", "content": (
            "You are a policeman. Intorduce yourself. "
            "Ask one question at a time to learn about the user's city. "
            "Wait for an answer before asking the next question. Avoid repeating yourself."
        )
        },
    ]
    if relevant_memory:
         # Add memory context without making it look like user/assistant dialogue
         memory_context = "Consider these potentially relevant facts from past interactions: " + "; ".join(relevant_memory)
         prompt_messages.append({"role": "system", "content": memory_context})

    # Add recent conversation history (limit context window)
    # Keep last N turns (user + bot = 1 turn pair)
    max_turns_for_context = 10
    history_start_index = max(0, len(conversation_history) - (max_turns_for_context * 2))
    prompt_messages.extend(conversation_history[history_start_index:])

    logger.debug(f"Sending prompt to OpenAI: {prompt_messages}")

    try:
        response = await aclient.chat.completions.create(
            model="gpt-4o", # Consider making this configurable
            messages=prompt_messages,
            temperature=0.7, # Adjust for creativity vs consistency
            max_tokens=150, # Limit response length
        )
        # Ensure choice exists and has message content
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            logger.warning("OpenAI response was empty or malformed.")
            return "Sorry, I couldn't generate a response right now."
    except Exception as e:
        logger.error(f"Error generating OpenAI response: {e}", exc_info=True)
        # Provide a more specific error if possible, but avoid leaking sensitive info
        return f"Sorry, I encountered an error trying to respond. ({type(e).__name__})"


async def run_chat():
    """Main chat loop."""
    graphiti = Graphiti(neo4j_uri, neo4j_user, neo4j_password)
    # Generate unique ID for this chat session
    # Could also be passed in or managed differently in a real app
    conversation_id = f"chat_{uuid.uuid4()}"
    turn_id = 0
    # Keep track of messages for OpenAI context (role/content format)
    chat_history = []

    try:
        logger.info('Initializing Graphiti (building indices if needed)...')
        # In a production app, check if indices exist rather than blindly building
        await graphiti.build_indices_and_constraints()
        logger.info(f'Graphiti initialized. Starting conversation ID: {conversation_id}')

        print("\nChatbot ready. Type 'quit' or 'exit' to end the conversation.")
        while True:
            # Use asyncio.to_thread for blocking input in async context
            try:
                user_input = await asyncio.to_thread(input, "You: ")
            except EOFError: # Handle Ctrl+D
                 print("\nBot: Goodbye!")
                 break

            if user_input.lower() in ['quit', 'exit']:
                print("Bot: Goodbye!")
                break

            if not user_input: # Skip empty input
                continue

            turn_id += 1
            timestamp = datetime.now(timezone.utc)

            # 1. Store user message in Graphiti memory
            # We add the raw user input here
            await add_message_episode(
                graphiti, conversation_id, turn_id, 'user', user_input, timestamp
            )
            # Add to chat history for LLM context
            chat_history.append({"role": "user", "content": user_input})

            # 2. Search memory for relevant context based on user's input
            # Graphiti's search result `fact` often contains the core extracted relationship
            memory_results = await search_memory(graphiti, user_input)
            relevant_facts = [res.fact for res in memory_results if hasattr(res, 'fact') and res.fact]
            if relevant_facts:
                 logger.info(f"Found relevant facts in memory: {relevant_facts}")
            else:
                 logger.info("No specific relevant facts found in memory for this input.")


            # 3. Generate bot response using OpenAI, providing history and facts
            bot_response = await generate_response(chat_history, relevant_facts)
            print(f"Bot: {bot_response}")

            # 4. Store bot response in Graphiti memory - COMMENTED OUT
            # turn_id += 1 # Incrementing turn_id here might skew future user message IDs if uncommented
            # Use the same timestamp as the user turn for grouping, or generate a new one
            # Using the same timestamp for now.
            # await add_message_episode(
            #     graphiti, conversation_id, turn_id, 'bot', bot_response, timestamp
            # )

            # Add to chat history for LLM context (Still needed for conversation flow)
            chat_history.append({"role": "assistant", "content": bot_response})

    except Exception as e:
        logger.error(f"An error occurred during the chat loop: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
    finally:
        # Ensure Graphiti connection is closed even if errors occur
        if 'graphiti' in locals() and graphiti.driver:
             await graphiti.close()
             logger.info('Graphiti connection closed.')


if __name__ == '__main__':
    # Add basic check for required environment variables before starting
    if not os.environ.get('OPENAI_API_KEY') or \
       not os.environ.get('NEO4J_URI') or \
       not os.environ.get('NEO4J_USER') or \
       not os.environ.get('NEO4J_PASSWORD'):
        print("Error: Required environment variables (OPENAI_API_KEY, NEO4J_*) are not set.", file=sys.stderr)
        print("Please create a .env file or set them environment variables.", file=sys.stderr)
        sys.exit(1)

    try:
        print("Starting chatbot...")
        asyncio.run(run_chat())
    except KeyboardInterrupt:
        print("\nChat interrupted by user. Exiting.")
    except ValueError as e:
         # Catch config errors raised during setup
         print(f"Configuration Error: {e}", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
         # Catch unexpected errors during startup/shutdown
         logger.error(f"Unhandled error in main execution: {e}", exc_info=True)
         print(f"An unexpected error occurred: {e}", file=sys.stderr)
         sys.exit(1) 