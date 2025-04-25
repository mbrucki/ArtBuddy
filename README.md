# ArtBuddy Chatbot

A FastAPI-based chatbot designed to capture and archive local art-scene knowledge through conversation, leveraging Graphiti for dynamic knowledge graph memory.

## Demo 


## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/arcymonka/ArtBuddy.git
    cd ArtBuddy
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The application requires certain environment variables to be set. Create a `.env` file in the project root directory by copying the (to be created) `.env.example` or adding the following:

```dotenv
# Neo4j Connection
NEO4J_URI=bolt://your_neo4j_host:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key

# Optional: Custom PIN for the application
# APP_PIN=1234
```

Replace the placeholder values with your actual Neo4j database credentials and OpenAI API key.

## Running Locally

Ensure your Neo4j database is running and accessible.

Start the FastAPI application using Uvicorn:

```bash
uvicorn app.main:app --reload --port 5001
```

The application will be available at `http://127.0.0.1:5001`.

## Docker

A `Dockerfile` is provided for containerization.

1.  **Build the Docker image:**
    ```bash
    docker build -t artbuddy .
    ```

2.  **Run the Docker container:**
    *   Make sure your `.env` file is present in the project root.
    *   Expose the application port (e.g., 5001) and pass the environment variables from the `.env` file.
    *   Ensure the container can reach your Neo4j instance (you might need to use specific Docker networking options or hostnames depending on your Neo4j setup).

    ```bash
    docker run -p 5001:80 --env-file .env artbuddy
    ```
    *(Note: The Dockerfile exposes port 80 internally, which is mapped to 5001 on the host here. Adjust ports as needed.)*

##  Acknowledgements

This project utilizes the Graphiti library for building knowledge graphs.

- **Graphiti:** [https://github.com/getzep/graphiti](https://github.com/getzep/graphiti)
