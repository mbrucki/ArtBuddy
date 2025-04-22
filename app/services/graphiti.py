import logging
from typing import Optional, Literal
from datetime import datetime, timezone

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

# Import config values
from app.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

logger = logging.getLogger(__name__)

# --- Global Graphiti instance ---
_graphiti_instance: Optional[Graphiti] = None

async def get_graphiti_instance() -> Graphiti:
    """Gets the global Graphiti instance, initializing if needed."""
    global _graphiti_instance
    if _graphiti_instance is None:
        logger.info("Initializing global Graphiti instance...")
        # Use imported config values
        _graphiti_instance = Graphiti(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    return _graphiti_instance

async def close_graphiti_instance():
    """Closes the global Graphiti instance if it exists."""
    global _graphiti_instance
    if _graphiti_instance:
        logger.info("Closing global Graphiti instance...")
        await _graphiti_instance.close()
        _graphiti_instance = None

# --- Core Graphiti Operations ---

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

    try:
        await graphiti.add_episode(
            name=episode_name,
            episode_body=message,
            source=EpisodeType.text,
            source_description=source_desc,
            reference_time=timestamp,
        )
        logger.info(f'Added episode: {episode_name} ({source_desc})')
    except Exception as e:
        logger.error(f"Error adding episode {episode_name}: {e}", exc_info=True)
        # Decide if re-raising is needed

async def search_graph(query: str) -> list:
    """Performs a hybrid search on the graphiti instance."""
    graphiti = await get_graphiti_instance()
    logger.info(f"Performing graph search for: '{query}'")
    try:
        # Using the default hybrid search (semantic + keyword)
        results = await graphiti.search(query)
        logger.info(f"Graph search returned {len(results)} results.")
        return results
    except Exception as e:
        logger.error(f"Error during graph search for query '{query}': {e}", exc_info=True)
        return [] # Return empty list on error

async def execute_cypher_query(query: str, params: Optional[dict] = None) -> tuple:
    """Executes a raw Cypher query using the graphiti driver."""
    graphiti = await get_graphiti_instance()
    logger.debug(f"Executing Cypher: {query} with params: {params}")
    try:
        # result is a tuple: (records, summary, keys)
        result = await graphiti.driver.execute_query(
            query,
            parameters_=params or {},
            database_=graphiti.database
        )
        # Optionally log summary metadata: logger.info(f"Cypher result summary: {result[1].metadata}")
        return result
    except Exception as e:
        logger.error(f"Error executing Cypher query '{query}': {e}", exc_info=True)
        # Return structure indicating error? e.g., ([], None, []) or raise?
        # Returning empty results for now
        return ([], None, [])

async def build_indices_and_constraints():
    """Builds required indices and constraints for Graphiti."""
    graphiti = await get_graphiti_instance()
    try:
        logger.info("Running Graphiti build_indices_and_constraints()...")
        await graphiti.build_indices_and_constraints()
        logger.info("Graphiti build_indices_and_constraints() completed successfully.")
    except Exception as e:
        logger.error(f"Error during Graphiti build_indices_and_constraints(): {e}", exc_info=True)
        # Decide how to handle this - potentially raise to stop app startup? 