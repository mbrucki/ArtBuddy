import os
from dotenv import load_dotenv

# Load .env file but don't override existing environment variables
load_dotenv(override=False)

# Neo4j/OpenAI Config
NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'password')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Hardcoded PIN (Consider moving to env var if security is critical)
APP_PIN = os.environ.get("APP_PIN", "6712")

# Basic validation
if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
    raise ValueError("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set in environment or .env file")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in environment or .env file")

# Logging Configuration (Can also be moved here or kept in main)
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' 