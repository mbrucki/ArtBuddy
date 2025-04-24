import logging
import sys # Import sys to force flush

# --- Configure logging VERY FIRST --- 
# Attempt to force logging output even if subsequent imports fail
LOG_LEVEL = "INFO" # Hardcode for initial setup, can be env var later
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)
logger.info("MAIN.PY TOP LEVEL: Logging configured.") 
sys.stdout.flush() # Force flush
# --- End Logging Setup ---

# print("MAIN: Starting imports...") # DEBUG
import os
logger.info("MAIN.PY: Imported os") # DEBUG
sys.stdout.flush() # Force flush
# print("MAIN: Imported os") # DEBUG
from contextlib import asynccontextmanager
logger.info("MAIN.PY: Imported asynccontextmanager") # DEBUG
sys.stdout.flush() # Force flush
# print("MAIN: Imported asynccontextmanager") # DEBUG

from fastapi import FastAPI
logger.info("MAIN.PY: Imported FastAPI") # DEBUG
sys.stdout.flush() # Force flush
# print("MAIN: Imported FastAPI") # DEBUG
from fastapi.staticfiles import StaticFiles
logger.info("MAIN.PY: Imported StaticFiles") # DEBUG
sys.stdout.flush() # Force flush
# print("MAIN: Imported StaticFiles") # DEBUG

# Import config and lifespan manager
logger.info("MAIN.PY: Importing config...") # DEBUG
sys.stdout.flush() # Force flush
# print("MAIN: Importing config...") # DEBUG
# Now config is imported AFTER basic logging is set up
from app.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY, APP_PIN, LOG_LEVEL as CONFIG_LOG_LEVEL # Get actual log level from config too
logger.info(f"MAIN.PY: Imported config OK. Config Log Level: {CONFIG_LOG_LEVEL}") # DEBUG
sys.stdout.flush() # Force flush
# print("MAIN: Imported config OK") # DEBUG

logger.info("MAIN.PY: Importing services.graphiti...") # DEBUG
sys.stdout.flush() # Force flush
# print("MAIN: Importing services.graphiti...") # DEBUG
from app.services.graphiti import (
    get_graphiti_instance, close_graphiti_instance, build_indices_and_constraints
)
logger.info("MAIN.PY: Imported services.graphiti OK") # DEBUG
sys.stdout.flush() # Force flush
# print("MAIN: Imported services.graphiti OK") # DEBUG

# Import routers
logger.info("MAIN.PY: Importing routers.auth...") # DEBUG
sys.stdout.flush() # Force flush
# print("MAIN: Importing routers.auth...") # DEBUG
from app.routers import auth
logger.info("MAIN.PY: Imported routers.auth OK") # DEBUG
sys.stdout.flush() # Force flush
# print("MAIN: Imported routers.auth OK") # DEBUG
logger.info("MAIN.PY: Importing routers.chat...") # DEBUG
sys.stdout.flush() # Force flush
# print("MAIN: Importing routers.chat...") # DEBUG
from app.routers import chat
logger.info("MAIN.PY: Imported routers.chat OK") # DEBUG
sys.stdout.flush() # Force flush
# print("MAIN: Imported routers.chat OK") # DEBUG


# Re-apply log level from config if different
# logger.setLevel(CONFIG_LOG_LEVEL)
# logger.info(f"MAIN.PY: Log level re-applied from config: {CONFIG_LOG_LEVEL}")
# sys.stdout.flush()

# Lifespan manager for Graphiti instance
logger.info("MAIN.PY: Defining lifespan manager...") # DEBUG
sys.stdout.flush()
# print("MAIN: Defining lifespan manager...") # DEBUG
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Actions on startup
    logger.info("Lifespan: START - Initializing Graphiti...") 
    sys.stdout.flush()
    try:
        graphiti_instance = await get_graphiti_instance() # Get/create instance
        logger.info("Lifespan: get_graphiti_instance call successful.")
        sys.stdout.flush()
        if graphiti_instance:
            logger.info("Lifespan: Attempting build_indices_and_constraints...")
            sys.stdout.flush()
            await build_indices_and_constraints() # Build indices
            logger.info("Lifespan: build_indices_and_constraints call successful.")
            sys.stdout.flush()
        else:
             logger.error("Lifespan: Failed to obtain Graphiti instance during startup.")
             sys.stdout.flush()
             # Optionally raise to prevent startup if Graphiti is critical
    except Exception as e_startup:
        logger.error(f"Lifespan: Error during Graphiti startup: {e_startup}", exc_info=True)
        sys.stdout.flush()
        # Optionally raise

    logger.info("Lifespan: Startup phase complete. Yielding control.")
    sys.stdout.flush()
    yield # Application runs here

    # Actions on shutdown
    logger.info("Lifespan: SHUTDOWN - Closing Graphiti connection...")
    sys.stdout.flush()
    await close_graphiti_instance()
    logger.info("Lifespan: Graphiti connection closed via lifespan.")
    sys.stdout.flush()
# logger.info("MAIN.PY: Lifespan manager defined.") # DEBUG
# sys.stdout.flush()


# Create FastAPI app instance
logger.info("MAIN.PY: Creating FastAPI app instance...") # DEBUG
sys.stdout.flush()
# print("MAIN: Creating FastAPI app instance...") # DEBUG
app = FastAPI(lifespan=lifespan, title="ArtBuddy Chatbot")
logger.info("MAIN.PY: FastAPI app instance created.") # DEBUG
sys.stdout.flush()
# print("MAIN: FastAPI app instance created.") # DEBUG

# Mount static files (CSS, JS)
logger.info("MAIN.PY: Setting up static files mount...") # DEBUG
sys.stdout.flush()
# print("MAIN: Setting up static files mount...") # DEBUG
# Assumes static files are in a 'static' directory at the project root
script_dir = os.path.dirname(__file__) # This is app/ directory
project_root = os.path.dirname(script_dir) # Go one level up for project root
static_dir_abs = os.path.join(project_root, "static")

if os.path.exists(static_dir_abs):
    logger.info(f"MAIN.PY: Mounting static directory: {static_dir_abs}") # DEBUG
    sys.stdout.flush()
    # print(f"MAIN: Mounting static directory: {static_dir_abs}") # DEBUG
    app.mount("/static", StaticFiles(directory=static_dir_abs), name="static")
    # logger.info(f"Mounted static directory: {static_dir_abs}") # Restored logger - Now DEBUG
else:
    logger.warning(f"MAIN.PY: Static directory not found at {static_dir_abs}, skipping mount.") # DEBUG
    sys.stdout.flush()
    # print(f"MAIN: Static directory not found at {static_dir_abs}, skipping mount.") # DEBUG
    # logger.warning(f"Static directory not found at {static_dir_abs}, skipping mount.") # Restored logger - Now DEBUG
# logger.info("MAIN.PY: Static files setup complete.") # DEBUG
# sys.stdout.flush()

# Include routers
logger.info("MAIN.PY: Including routers...") # DEBUG
sys.stdout.flush()
# print("MAIN: Including routers...") # DEBUG
app.include_router(auth.router, tags=["Authentication & Setup"])
app.include_router(chat.router, tags=["Chat"])
logger.info("MAIN.PY: Routers included.") # DEBUG
sys.stdout.flush()
# print("MAIN: Routers included.") # DEBUG

# Root endpoint (optional, can be useful for health checks)
logger.info("MAIN.PY: Defining root endpoint...") # DEBUG
sys.stdout.flush()
# print("MAIN: Defining root endpoint...") # DEBUG
@app.get("/")
async def read_root():
    # Redirect to the auth router's root (PIN entry)
    # Note: Can't easily use url_for here as request object isn't available directly
    # Hardcoding or using a known prefix might be needed if this is desired.
    # For now, just return a simple message.
    logger.info("Root endpoint '/' accessed.")
    sys.stdout.flush()
    return {"message": "Welcome to ArtBuddy. Please access the main interface."}
# logger.info("MAIN.PY: Root endpoint defined.") # DEBUG
# sys.stdout.flush()
logger.info("MAIN.PY: Script execution finished. Uvicorn should take over.") # DEBUG
sys.stdout.flush()
# print("MAIN.PY: Script execution finished. Uvicorn should take over.") # DEBUG

# Run with: uvicorn app.main:app --reload --port 5001 