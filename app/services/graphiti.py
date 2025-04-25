import logging
from typing import Optional, Literal
from datetime import datetime, timezone

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
# Import specific Neo4j exceptions for better error handling
from neo4j.exceptions import ServiceUnavailable, AuthError

# Import config values
from app.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

logger = logging.getLogger(__name__)

# --- Global Graphiti instance ---
_graphiti_instance: Optional[Graphiti] = None

async def get_graphiti_instance() -> Graphiti:
    """Gets the global Graphiti instance, initializing if needed."""
    global _graphiti_instance
    if _graphiti_instance is None:
        # Log before initialization attempt
        logger.info("Attempting to initialize global Graphiti instance...")
        # Log credentials being used (mask password)
        masked_password = "****" if NEO4J_PASSWORD else "None"
        logger.info(f"Using Neo4j Config: URI={NEO4J_URI}, User={NEO4J_USER}, Password={masked_password}")
        try:
            # Use imported config values
            _graphiti_instance = Graphiti(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            logger.info("Graphiti object created. Verifying Neo4j connection...")
            # Explicitly verify connection
            await _graphiti_instance.driver.verify_connectivity()
            logger.info("Neo4j connection verified successfully.")

        # Catch specific Neo4j connection/auth errors
        except AuthError as e_auth:
            logger.error(f"Neo4j Authentication Error during Graphiti initialization: {e_auth}", exc_info=True)
            _graphiti_instance = None # Ensure instance remains None on error
            raise # Re-raise to stop startup
        except ServiceUnavailable as e_service:
            logger.error(f"Neo4j Service Unavailable Error during Graphiti initialization (check URI/reachability): {e_service}", exc_info=True)
            _graphiti_instance = None # Ensure instance remains None on error
            raise # Re-raise to stop startup
        except Exception as e_init:
            # Log general errors during initialization
            logger.error(f"General Error during Graphiti initialization: {e_init}", exc_info=True)
            _graphiti_instance = None # Ensure instance remains None on error
            raise # Re-raise to stop startup

    if _graphiti_instance is None:
         # Safeguard check
         logger.error("Failed to provide a Graphiti instance after initialization block.")
         raise RuntimeError("Could not initialize Graphiti instance.")

    return _graphiti_instance

async def close_graphiti_instance():
    """Closes the global Graphiti instance if it exists."""
    global _graphiti_instance
    if _graphiti_instance:
        logger.info("Attempting to close global Graphiti instance...")
        try:
            await _graphiti_instance.close()
            logger.info("Graphiti instance closed successfully.")
        except Exception as e_close:
            logger.error(f"Error closing Graphiti instance: {e_close}", exc_info=True)
        finally:
             _graphiti_instance = None # Set to None even if close failed

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
    logger.info("Attempting to run Graphiti build_indices_and_constraints...")
    try:
        graphiti = await get_graphiti_instance() # Get instance (might initialize here if not already)
        logger.info("Obtained Graphiti instance for building indices.")
        await graphiti.build_indices_and_constraints()
        logger.info("Graphiti build_indices_and_constraints() completed successfully.")
    except Exception as e:
        logger.error(f"Error during Graphiti build_indices_and_constraints(): {e}", exc_info=True)
        # Raise the exception to make startup failure explicit if indices fail
        raise 