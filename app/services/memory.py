import logging
from typing import Optional, Dict, Any

from neo4j.exceptions import ServiceUnavailable
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

# Import graphiti service functions
from app.services.graphiti import execute_cypher_query, search_graph, get_graphiti_instance # Added get_graphiti_instance

logger = logging.getLogger(__name__)

# --- Memory Helper Functions ---

async def get_person_details(name: str) -> Optional[Dict[str, Any]]:
    """Retrieves Person details using a direct Cypher query via the graphiti service."""
    name_param = name
    cypher_query = (
        "MATCH (e:Entity) "
        "WHERE "
        "    toLower(e.name) = toLower($name) OR "
        "    toLower(e.name) STARTS WITH toLower($name) OR "
        "    toLower($name) STARTS WITH toLower(e.name) + ' '"
        "RETURN "
        "    e.name AS name, "
        "    e.summary AS summary, "
        "    CASE "
        "        WHEN toLower(e.name) = toLower($name) THEN 0 "
        "        WHEN toLower(e.name) STARTS WITH toLower($name) THEN 1 "
        "        WHEN toLower($name) STARTS WITH toLower(e.name) THEN 2 "
        "        ELSE 3 "
        "    END AS score "
        "ORDER BY score ASC "
        "LIMIT 1"
    )

    try:
        # Use the graphiti service wrapper
        records, summary_meta, _ = await execute_cypher_query(
            cypher_query,
            params={"name": name_param}
        )

        if records:
            record = records[0]
            retrieved_name = record.get("name")
            summary_prop = record.get("summary")
            details = {"exists": True, "summary": summary_prop}
            logger.info(f"Details retrieved via Cypher for Person '{retrieved_name}': Exists=True")
            return details
        else:
            logger.info(f"Person '{name}' not found via Cypher query.")
            return {"exists": False, "summary": None}

    except ServiceUnavailable as e:
        logger.error(f"Neo4j connection error checking person '{name}': {e}")
        return None # Indicate connection error
    except Exception as e:
        # Catch potential errors from execute_cypher_query if it doesn't handle them fully
        logger.error(f"Error retrieving person details for '{name}': {e}", exc_info=True)
        return None

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
    """DEPRECATED: Helper to check if a City entity exists (by name) using node search."""
    # This function directly used graphiti._search, which might be better handled
    # within the graphiti service or using a more generic entity check.
    # Keeping it for now but marking as potentially needing refactor/removal.
    logger.warning("check_city_exists is using a potentially deprecated pattern accessing graphiti._search")
    graphiti = await get_graphiti_instance() # Need instance for _search
    search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    search_config.limit = 1
    try:
        # ATTENTION: graphiti._search is internal. Prefer using public API if possible.
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

async def answer_question_from_kg(query: str, session_id: str) -> str:
    """Attempts to answer a user's question using knowledge graph search via graphiti service."""
    logger.info(f"[{session_id}] Attempting KG search to answer query: '{query}'")
    try:
        # Use the graphiti service wrapper
        memory_results = await search_graph(query)
        if memory_results:
            facts = [res.fact for res in memory_results if hasattr(res, 'fact') and res.fact and isinstance(res.fact, str)]
            if facts:
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
        # Catch potential errors from search_graph if it doesn't handle them fully
        logger.error(f"[{session_id}] Error during KG search for answering question '{query}': {e}", exc_info=True)
        return "Sorry, I encountered an error trying to look that up." 