# FastAPI LangChain Application

This FastAPI application leverages LangChain to provide chat functionalities powered by HuggingFace embeddings and Ollama language models. It supports initializing a PostgreSQL vector database with text or CSV data.

## Features

- Initialize database with text or CSV data
- Perform chat queries using retrieval-augmented generation (RAG)
- Conversation history management

## Prerequisites

- Python 3.10+
- PostgreSQL database with PGVector extention
  ```sh
  docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 5432:5432 -d pgvector/pgvector:pg16
  ```

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/mrankitvish/rag-chatbot.git
    cd rag-chatbot
    ```

2. Create a virtual environment and activate it:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

4. Create a `.env` file or rename `.env_example` in the root directory and set the following environment variables:

    ```env
    EMBEDDING_MODEL=your_huggingface_embedding_model
    COLLECTION_NAME=your_pgvector_collection_name
    DB_NAME=your_database_name
    DB_IPADDR=your_database_ip
    DB_PORT=your_database_port
    DB_USER=your_database_user
    DB_PASSWD=your_database_password
    DATA=path_to_your_text_data_file
    DATA_CSV=path_to_your_csv_data_file
    LLM_MODEL=your_llama_model
    LLM_BASEURL=your_llama_base_url
    TOP_K=5
    ```

## Running the Application

1. Start the FastAPI application:

    ```sh
    fastapi dev app.py --host 0.0.0.0
    ```

2. The application will be accessible at `http://0.0.0.0:8000`.

## API Endpoints

### Root Endpoint

- **GET /**

    Returns a greeting message.

    **Response:**

    ```json
    {
        "message": "RAG ChatBot built for you to chat with you data."
    }
    ```

### Database Initialization

- **POST /db_init/{txt_or_csv}**

    Initializes the database with text or CSV data.

    **Path Parameters:**

    - `txt_or_csv`: Specifies whether to initialize with text (`txt`) or CSV (`csv`) data.

    **Responses:**

    ```json
    {
        "message": "Database initialized successfully"
    }
    ```

    ```json
    {
        "message": "Invalid input"
    }
    ```

### Chat Endpoint

- **POST /chat**

    Performs a chat query using the RAG chain.

    **Request Body:**

    ```json
    {
        "question": "Your question here"
    }
    ```

    **Response:**

    ```json
    {
        "answer": "Generated response based on the question and context"
    }
    ```

## Example Usage

### Initializing the Database

To initialize the database with text data:

```sh
curl -X POST http://127.0.0.1:8000/db_init/txt
