"""
Chatbot Memory Example using Graphiti

This script demonstrates how to use Graphiti to store and retrieve
chatbot conversation history, treating each message as an Episode.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from logging import INFO
from typing import Literal, Optional, Dict, Any

from dotenv import load_dotenv
from neo4j.exceptions import ServiceUnavailable

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

#################################################
# CONFIGURATION
#################################################
# Set up logging and environment variables for
# connecting to Neo4j database and OpenAI
#################################################

# Configure logging
logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

load_dotenv()

# Neo4j connection parameters
neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')

# OpenAI API Key (required by Graphiti for embedding/analysis)
openai_api_key = os.environ.get('OPENAI_API_KEY')

if not neo4j_uri or not neo4j_user or not neo4j_password:
    raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')
if not openai_api_key:
    # Although we don't call OpenAI directly here, Graphiti uses it internally
    raise ValueError('OPENAI_API_KEY must be set for Graphiti embedding')


# --- Global Graphiti instance using imported graphiti_core.Graphiti ---

_graphiti_instance: Optional[Graphiti] = None

async def get_graphiti_instance() -> Graphiti:
    """Gets the global Graphiti instance, initializing if needed."""
    global _graphiti_instance
    if _graphiti_instance is None:
        logger.info("Initializing global Graphiti instance...")
        _graphiti_instance = Graphiti(neo4j_uri, neo4j_user, neo4j_password)
        # It's often better to explicitly call build_indices in a setup script
        # rather than implicitly on first use. Commenting out for now.
        # await _graphiti_instance.build_indices_and_constraints()
    return _graphiti_instance

async def close_graphiti_instance():
    """Closes the global Graphiti instance if it exists."""
    global _graphiti_instance
    if _graphiti_instance:
        logger.info("Closing global Graphiti instance...")
        await _graphiti_instance.close()
        _graphiti_instance = None


# --- Existing helper functions adjusted to use the instance getter ---

async def add_message_episode(
    conversation_id: str,
    turn_id: int,
    speaker: Literal['user', 'bot'],
    message: str,
    timestamp: Optional[datetime] = None,
):
    """Adds a single chat message as an episode to the graph."""
    graphiti = await get_graphiti_instance()
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    episode_name = f'Conv_{conversation_id}_Turn_{turn_id}'
    source_desc = f'{speaker} message'

    await graphiti.add_episode(
        name=episode_name,
        episode_body=message,
        source=EpisodeType.text,
        source_description=source_desc,
        reference_time=timestamp,
    )
    logger.info(f'Added episode: {episode_name} ({source_desc})')


async def search_memory(query: str) -> list:
    """Searches the conversation memory for relevant messages."""
    graphiti = await get_graphiti_instance()
    logger.info(f"Searching memory for: '{query}'")
    # Using the default hybrid search (semantic + keyword)
    results = await graphiti.search(query)
    logger.info(f"Search returned {len(results)} results.")
    return results

# --- REIMPLEMENTED HELPER FUNCTIONS using graphiti._search ---

async def get_person_details(name: str) -> Optional[Dict[str, Any]]:
    """Retrieves Person details (existence, summary, created_at) using a direct Cypher query for exact match."""
    graphiti = await get_graphiti_instance()
    name_param = name # Use original case for parameter, compare lowercased in Cypher

    cypher_query = (
        "MATCH (e:Entity) "
        "WHERE toLower(e.name) = toLower($name) "
        "RETURN e.name AS name, e.summary AS summary, e.created_at AS created_at "
        "LIMIT 1"
    )
    logger.debug(f"Executing exact match Cypher for person details: {cypher_query} with param name='{name_param}'")

    try:
        # result is a tuple: (records, summary, keys)
        result = await graphiti.driver.execute_query(
            cypher_query,
            parameters_={"name": name_param}, # Pass name as parameter
            database_=graphiti.database
        )

        records = result[0] # Get the list of records
        summary_meta = result[1] # Get the summary object
        keys = result[2] # Get the keys

        # Log the raw result for debugging
        logger.info(f"[{name}] Raw driver query result: Records Count={len(records)}, Keys={keys}, Summary Metadata={summary_meta.metadata}")

        if records: # Check if any records were returned
            record = records[0]
            # Extract properties from the record using .get() for safety
            retrieved_name = record.get("name") # Get the name as stored in the DB
            summary_prop = record.get("summary")
            created_at_raw = record.get("created_at")
            created_at_dt = None

            logger.debug(f"Direct Cypher Query found node. Name: '{retrieved_name}', Raw created_at: {created_at_raw} (Type: {type(created_at_raw)})")

            # --- Robust datetime parsing logic ---
            if created_at_raw:
                parsed = False
                if isinstance(created_at_raw, datetime):
                    created_at_dt = created_at_raw
                    parsed = True
                    logger.debug(f"Parsed created_at for '{retrieved_name}' as existing datetime.")
                elif isinstance(created_at_raw, str):
                    try:
                        # Handle potential timezone suffix like [UTC] if present
                        dt_str = created_at_raw.split('[')[0]
                        # Handle Z notation explicitly by replacing with offset
                        if dt_str.endswith('Z'): dt_str = dt_str[:-1] + '+00:00'
                        
                        # <<< EDIT: Truncate fractional seconds to microseconds >>>
                        if '.' in dt_str:
                            main_part, fractional_part_tz = dt_str.split('.', 1)
                            # Separate fractional part from timezone if necessary
                            fractional_part = fractional_part_tz
                            tz_part = ''
                            if '+' in fractional_part_tz:
                                fractional_part, tz_part = fractional_part_tz.split('+', 1)
                                tz_part = '+' + tz_part
                            elif '-' in fractional_part_tz and not fractional_part_tz.startswith('-'): # Ensure it's timezone minus
                                fractional_part, tz_part = fractional_part_tz.split('-', 1)
                                tz_part = '-' + tz_part
                                
                            if len(fractional_part) > 6:
                                fractional_part = fractional_part[:6]
                            dt_str = f"{main_part}.{fractional_part}{tz_part}"
                            logger.debug(f"Truncated/Reformatted dt_str: {dt_str}")
                        # <<< End EDIT >>>
                            
                        created_at_dt = datetime.fromisoformat(dt_str)
                        parsed = True
                        logger.debug(f"Parsed created_at for '{retrieved_name}' using fromisoformat.")
                    except ValueError as e:
                        # Log the error with the specific string that failed
                        logger.warning(f"Could not parse created_at string '{dt_str}' for '{retrieved_name}' using fromisoformat: {e}")
                # Add more parsing attempts here if needed

                if parsed and created_at_dt:
                    # Ensure timezone awareness if parsed successfully but naive
                    if created_at_dt.tzinfo is None or created_at_dt.tzinfo.utcoffset(created_at_dt) is None:
                        created_at_dt = created_at_dt.replace(tzinfo=timezone.utc)
                        logger.debug(f"Made parsed datetime timezone-aware (UTC) for {retrieved_name}")
                elif not parsed:
                    logger.warning(f"Failed all parsing attempts for created_at: {created_at_raw}")
                    created_at_dt = None # Ensure it's None if parsing failed
            # --- End datetime parsing logic ---

            details = {
                "exists": True,
                "summary": summary_prop,
                "created_at": created_at_dt
            }
            logger.info(f"Details retrieved via Cypher for Person '{retrieved_name}' (Input: '{name}'): Exists=True, ParsedCreatedAt={details['created_at']}")
            return details
        else:
            # This is the path that SHOULD be taken based on your Browser query
            logger.info(f"Person '{name}' not found via direct Cypher query.")
            return {"exists": False, "summary": None, "created_at": None}

    except ServiceUnavailable as e:
        logger.error(f"Neo4j connection error checking person '{name}': {e}")
        return None # Indicate connection error
    except Exception as e:
        logger.error(f"Error executing Cypher query for person details '{name}': {e}", exc_info=True)
        return None # Indicate other error



async def get_person_fact(name: str) -> Optional[str]:
    """Helper to get a formatted fact string about a Person entity."""
    details = await get_person_details(name)
    if details and details["exists"]:
        summary = details.get("summary")
        if summary:
            fact_display = f"known for: {summary[:100]}{'...' if len(summary) > 100 else ''}"
            logger.info(f"Retrieved formatted fact for Person '{name.title()}': '{fact_display}'")
            return fact_display
        else:
            logger.info(f"Found Person node '{name.title()}' but no summary/fact property.")
            return f"is a known person" # Fallback fact
    else:
        logger.info(f"Fact retrieval failed for Person '{name.title()}' (not found or error).")
        return None

async def check_city_exists(name: str) -> bool:
    """Helper to check if a City entity exists (by name) using node search."""
    graphiti = await get_graphiti_instance()
    search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    search_config.limit = 1
    try:
        node_search_results = await graphiti._search(
            query=f'"{name.title()}"',
            config=search_config,
        )
        exists = bool(node_search_results and node_search_results.nodes)
        logger.info(f"Node search check for City '{name.title()}'. Exists: {exists}")
        return exists
    except Exception as e:
        logger.error(f"Error during node search check for City '{name.title()}': {e}", exc_info=True)
        return False

async def main():
    #################################################
    # INITIALIZATION
    #################################################
    # Connect to Neo4j and set up Graphiti indices
    #################################################

    graphiti = await get_graphiti_instance()

    try:
        # Initialize the graph database (only needs to be done once typically)
        # In a real app, you might check if indices exist first
        logger.info('Building Graphiti indices and constraints (if they dont exist)...')
        await graphiti.build_indices_and_constraints()
        logger.info('Indices and constraints ready.')

        # --- Example Usage of New Functions ---
        # Test checking entities (use names likely in your test data or add some)
        test_name = "Helena" # Example name
        test_city = "Berlin" # Example city

        # Example: Adding entities if they don't exist (for testing purposes)
        # In a real app, entity creation might be handled differently by Graphiti
        async def ensure_entity(g: Graphiti, type: str, name: str):
            if not await g.check_entity_exists(type, name):
                logger.info(f"Adding test entity {type}: {name}")
                # This is a placeholder for how Graphiti *might* add an entity.
                # Replace with Graphiti's actual entity creation mechanism if available.
                # If Graphiti doesn't have one, you'd use direct Cypher.
                try:
                    await g.driver.execute_query(
                        "MERGE (e:Entity {name: $name, entity_type: $type}) SET e.summary = $summary",
                        parameters={"name": name, "type": type, "summary": f"Test summary for {name}"},
                        database_=g.db_name
                    )
                except Exception as e_add:
                    logger.error(f"Failed to add test entity {type} {name}: {e_add}")

        await ensure_entity(graphiti, "Person", test_name)
        await ensure_entity(graphiti, "City", test_city)
        await ensure_entity(graphiti, "Person", "Alice") # Add another person for fact checking

        # Check existence
        exists_helena = await get_person_details(test_name)
        logger.info(f"Does {test_name} exist? {exists_helena}")
        exists_berlin = await check_city_exists(test_city)
        logger.info(f"Does {test_city} exist? {exists_berlin}")
        exists_bob = await get_person_details("Bob")
        logger.info(f"Does Bob exist? {exists_bob}")


        # Get fact
        fact_helena = await get_person_fact(test_name)
        logger.info(f"Fact about {test_name}: {fact_helena}")

        fact_alice = await get_person_fact("Alice")
        logger.info(f"Fact about Alice: {fact_alice}")

        # --- End Example Usage ---


        #################################################
        # ADDING CONVERSATION MESSAGES (using updated helper)
        #################################################
        conversation_id = 'conv123'
        messages = [
            {'speaker': 'user', 'text': 'Hi, I wanted to ask about flights to London.'},
            {'speaker': 'bot', 'text': 'Sure, which dates are you interested in?'},
            {
                'speaker': 'user',
                'text': 'Around the first week of July, maybe leaving on the 3rd?',
            },
            {
                'speaker': 'bot',
                'text': 'Okay, checking flights around July 3rd to London...',
            },
            {'speaker': 'user', 'text': 'Also, any recommendations for hotels there?'},
        ]

        for i, msg in enumerate(messages):
            await add_message_episode(
                conversation_id, i + 1, msg['speaker'], msg['text']
            )
            # Small delay to simulate time passing and allow processing
            await asyncio.sleep(0.1)

        #################################################
        # SEARCHING CONVERSATION MEMORY (using updated helper)
        #################################################
        search_query = 'What city was the user asking about?'
        memory_results = await search_memory(search_query)

        print(f'\n--- Search Results for "{search_query}" ---')
        if memory_results:
            for result in memory_results:
                print(f'Fact: {result.fact}')
                print(f'Source Node: {result.source_node_label} ({result.source_node_name})')
                print(
                    f'Target Node: {result.target_node_label} ({result.target_node_name})'
                )
                print(f'Timestamp: {result.reference_time}')
                print('---')
        else:
            print('No relevant memories found.')

        # Example: Search for hotels
        search_query_hotels = 'Did the user mention hotels?'
        hotel_results = await search_memory(search_query_hotels)

        print(f'\n--- Search Results for "{search_query_hotels}" ---')
        if hotel_results:
            for result in hotel_results:
                # The 'fact' extracted by Graphiti often contains the relevant info
                print(f'Fact: {result.fact}')
                print(f'Timestamp: {result.reference_time}')
                print('---')
        else:
            print('No relevant memories found about hotels.')

    except Exception as e:
        logger.error(f'An error occurred: {e}', exc_info=True)
    finally:
        #################################################
        # CLEANUP (using new close function)
        #################################################
        # Close the connection to Neo4j
        #################################################
        await close_graphiti_instance()


if __name__ == '__main__':
    # Ensure OpenAI key is set before running
    if not os.environ.get('OPENAI_API_KEY'):
        print(
            'Error: OPENAI_API_KEY environment variable not set.', file=sys.stderr
        )
        # Depending on where Graphiti *needs* the key, this might still fail later
        # Import sys at the top if using this exit
        # import sys
        # sys.exit(1)
    # Added try/finally to ensure cleanup even on script interruption
    try:
        asyncio.run(main())
    finally:
        # Ensure the instance is closed if main() exits early or errors
        # This might run close twice if main completes normally, but close() should be idempotent
        asyncio.run(close_graphiti_instance()) 