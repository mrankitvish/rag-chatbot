from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_community.llms import Ollama
from operator import itemgetter
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Create FastAPI app instance
app = FastAPI()

# Initialize HuggingFaceEmbeddings with the specified model
embeddings = HuggingFaceEmbeddings(model_name=os.environ["EMBEDDING_MODEL"])

# Initialize PGVector with the specified configuration
vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=os.environ["COLLECTION_NAME"],
    connection=PGVector.connection_string_from_db_params(
        driver="psycopg",
        database=os.environ["DB_NAME"],
        host=os.environ["DB_IPADDR"],
        port=os.environ["DB_PORT"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWD"]
    ),
    use_jsonb=True,
)

# Initialize CharacterTextSplitter for splitting text into chunks
text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=int(os.environ["CHUNK_SIZE"]),
        chunk_overlap=int(os.environ["CHUNK_OVERLAP"])
    )

# Function to initialize the database with text data
def db_init_txt():
    loader = TextLoader(os.environ["DATA"])
    docs = loader.load_and_split(text_splitter=text_splitter)
    vectorstore.add_documents(docs)
    return {"message": "Database initailized Successfully"}

# Function to initialize the database with CSV data
def db_init_csv():
    file_path = os.environ["DATA_CSV"]
    loader = CSVLoader(file_path=file_path)
    docs = loader.load_and_split(text_splitter=text_splitter)
    vectorstore.add_documents(docs)
    return {"message": "Database initailized Successfully"}

# Root endpoint
@app.get("/")
async def root():
   return {"message": "RAG Chatbot build for you to chat with your data."}

# Endpoint to initialize the database with text or CSV data
@app.post("/db_init/{txt_or_csv}")
async def db_init(txt_or_csv: str):
    if txt_or_csv == "txt":
        return db_init_txt()
    elif txt_or_csv == "csv":
        return db_init_csv()
    else:
        return {"message": "Invalid input"}

# Pydantic model for chat request
class Chat(BaseModel):
    question: str

# Initialize Ollama LLM with the specified configuration
llm = Ollama(
        model = os.environ["LLM_MODEL"],
        base_url = os.environ["LLM_BASEURL"],
        top_p = os.environ["LLM_TOP_P"],
        verbose = True,
        temperature = os.environ["LLM_TEMPERATURE"]
    )

# Initialize retriever and memory
retriever = vectorstore.as_retriever(search_kwargs={"k": int(os.environ["TOP_K"])})
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Chat prompt template
template = os.environ["SYSTEM_MESSAGE"] + " :" + " {context}"

# Chat endpoint
@app.post("/chat")
async def chat(question: Chat):
    if not question:
        return {"error": "Missing question in request body"}
    
    prompt = ChatPromptTemplate.from_template(template)

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", question.question),
    ])

    # # Create RAG (Retrieval-Augmented Generation) chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    # Create final chain
    chain = (
        RunnablePassthrough.assign(
            chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history")
        )
        | final_prompt
        | llm.bind(stop=["Human:"])
    )

    # Get relevant documents based on the question
    docs = retriever.get_relevant_documents(question.question)
    context = "\n".join([doc.page_content for doc in docs])

    # Generate response using the chain
    response = ""
    for chunk in chain.stream({"question": question.question, "context": context}):
        response += chunk

    # Save context to memory
    memory.save_context({"question": question.question}, {"output": response})

    return {"answer": response}