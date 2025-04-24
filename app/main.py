# print("MAIN: Starting imports...") # DEBUG
import logging
# print("MAIN: Imported logging") # DEBUG
import os
# print("MAIN: Imported os") # DEBUG
from contextlib import asynccontextmanager
# print("MAIN: Imported asynccontextmanager") # DEBUG

from fastapi import FastAPI
# print("MAIN: Imported FastAPI") # DEBUG
from fastapi.staticfiles import StaticFiles
# print("MAIN: Imported StaticFiles") # DEBUG

# Import config and lifespan manager
# print("MAIN: Importing config...") # DEBUG
from app.config import LOG_LEVEL, LOG_FORMAT
# print("MAIN: Imported config OK") # DEBUG

# print("MAIN: Importing services.graphiti...") # DEBUG
from app.services.graphiti import (
    get_graphiti_instance, close_graphiti_instance, build_indices_and_constraints
)
# print("MAIN: Imported services.graphiti OK") # DEBUG

# Import routers
# print("MAIN: Importing routers.auth...") # DEBUG
from app.routers import auth
# print("MAIN: Imported routers.auth OK") # DEBUG
# print("MAIN: Importing routers.chat...") # DEBUG
from app.routers import chat
# print("MAIN: Imported routers.chat OK") # DEBUG


# print("MAIN: Configuring logging...") # DEBUG
# Configure logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)
# print("MAIN: Logging configured.") # DEBUG

# Lifespan manager for Graphiti instance
# print("MAIN: Defining lifespan manager...") # DEBUG
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Actions on startup
    logger.info("Lifespan: Initializing Graphiti...") # Changed print to logger
    try:
        graphiti_instance = await get_graphiti_instance() # Get/create instance
        if graphiti_instance:
            await build_indices_and_constraints() # Build indices
        else:
             logger.error("Lifespan: Failed to obtain Graphiti instance during startup.")
             # Optionally raise to prevent startup if Graphiti is critical
    except Exception as e_startup:
        logger.error(f"Lifespan: Error during Graphiti startup: {e_startup}", exc_info=True)
        # Optionally raise

    yield # Application runs here

    # Actions on shutdown
    logger.info("Lifespan: Closing Graphiti connection...")
    await close_graphiti_instance()
    logger.info("Lifespan: Graphiti connection closed via lifespan.")
# print("MAIN: Lifespan manager defined.") # DEBUG


# Create FastAPI app instance
# print("MAIN: Creating FastAPI app instance...") # DEBUG
app = FastAPI(lifespan=lifespan, title="ArtBuddy Chatbot")
# print("MAIN: FastAPI app instance created.") # DEBUG

# Mount static files (CSS, JS)
# print("MAIN: Setting up static files mount...") # DEBUG
# Assumes static files are in a 'static' directory at the project root
script_dir = os.path.dirname(__file__) # This is app/ directory
project_root = os.path.dirname(script_dir) # Go one level up for project root
static_dir_abs = os.path.join(project_root, "static")

if os.path.exists(static_dir_abs):
    # print(f"MAIN: Mounting static directory: {static_dir_abs}") # DEBUG
    app.mount("/static", StaticFiles(directory=static_dir_abs), name="static")
    logger.info(f"Mounted static directory: {static_dir_abs}") # Restored logger
else:
    # print(f"MAIN: Static directory not found at {static_dir_abs}, skipping mount.") # DEBUG
    logger.warning(f"Static directory not found at {static_dir_abs}, skipping mount.") # Restored logger
# print("MAIN: Static files setup complete.") # DEBUG

# Include routers
# print("MAIN: Including routers...") # DEBUG
app.include_router(auth.router, tags=["Authentication & Setup"])
app.include_router(chat.router, tags=["Chat"])
# print("MAIN: Routers included.") # DEBUG

# Root endpoint (optional, can be useful for health checks)
# print("MAIN: Defining root endpoint...") # DEBUG
@app.get("/")
async def read_root():
    # Redirect to the auth router's root (PIN entry)
    # Note: Can't easily use url_for here as request object isn't available directly
    # Hardcoding or using a known prefix might be needed if this is desired.
    # For now, just return a simple message.
    return {"message": "Welcome to ArtBuddy. Please access the main interface."}
# print("MAIN: Root endpoint defined.") # DEBUG
# print("MAIN: Script execution finished. Uvicorn should take over.") # DEBUG

# Run with: uvicorn app.main:app --reload --port 5001 