import asyncio
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
        "WHERE "
        "    toLower(e.name) = toLower($name) OR "
        "    toLower(e.name) STARTS WITH toLower($name) OR "  # Handles "Mira Fernvale" from "Mira"
        "    toLower($name) STARTS WITH toLower(e.name) + ' '"
        "RETURN "
        "    e.name AS name, "
        "    e.summary AS summary, "
        "    CASE "
        "        WHEN toLower(e.name) = toLower($name) THEN 0 "              # Exact match
        "        WHEN toLower(e.name) STARTS WITH toLower($name) THEN 1 " # Entity starts with mention
        "        WHEN toLower($name) STARTS WITH toLower(e.name) THEN 2 " # Mention starts with entity
        "        ELSE 3 "
        "    END AS score "
        "ORDER BY score ASC "
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
        if summary_meta:
             logger.info(f"[{name}] Raw driver query result Summary: {summary_meta.metadata}")

        if records: # Check if any records were returned
            record = records[0]
            # Extract properties from the record using .get() for safety
            retrieved_name = record.get("name") # Get the name as stored in the DB
            summary_prop = record.get("summary")

            logger.debug(f"Direct Cypher Query found node. Name: '{retrieved_name}'")

            details = {
                "exists": True,
                "summary": summary_prop,
            }
            logger.info(f"Details retrieved via Cypher for Person '{retrieved_name}' (Input: '{name}'): Exists=True")
            return details
        else:
            # This is the path that SHOULD be taken based on your Browser query
            logger.info(f"Person '{name}' not found via direct Cypher query.")
            return {"exists": False, "summary": None}

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
            fact_display = f"{summary[:100]}{'...' if len(summary) > 100 else ''}"
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
