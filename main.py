from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import os
import logging
import tempfile
import json
import uuid
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import chromadb
from policy_agent import PolicyAgent
from user_data_agent import UserDataAgent
from calculation_agent import CalculationAgent
from states import PolicyState, UserDataState, CalculationState
from utils import load_chat_history
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

# File paths
PDF_PATH = os.path.join('public', 'Employee Handbook.pdf')
USER_DATA_PATH = os.path.join('user_details.json')
CHAT_HISTORY_PATH = os.path.join('conversation_history.json')

# Initialize FastAPI app
app = FastAPI(title="XYZ Bank Agents API")

# Initialize persistent directory for Chroma
persist_dir = os.path.join(tempfile.gettempdir(), f"chroma_db_{str(uuid.uuid4())}")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-04-17",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)

# Load user data
def load_user_data():
    logger.info("Loading user data from JSON...")
    if not os.path.exists(USER_DATA_PATH):
        raise ValueError(f"User data JSON file not found at {USER_DATA_PATH}.")
    try:
        with open(USER_DATA_PATH, 'r') as f:
            data = json.load(f)
        logger.info("User data loaded successfully.")
        return data.get("users", {})
    except Exception as e:
        logger.error(f"User data loading error: {str(e)}")
        raise ValueError(f"Error loading user data: {str(e)}")

# Load and split PDF
def load_pdf_and_split():
    logger.info("Loading and splitting PDF...")
    if not os.path.exists(PDF_PATH):
        raise ValueError(f"PDF file not found at {PDF_PATH}.")
    try:
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        logger.info(f"PDF loaded and split into {len(splits)} chunks.")
        return splits
    except Exception as e:
        logger.error(f"PDF loading error: {str(e)}")
        raise ValueError(f"Error loading PDF: {str(e)}")

# Load and index PDF into Chroma
def load_and_index_pdf(splits=None):
    logger.info(f"Using Chroma database at: {persist_dir}")
    class CustomHuggingFaceEmbeddings(HuggingFaceEmbeddings):
        model_config = ConfigDict(protected_namespaces=())
    embedding_function = CustomHuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    client_settings = chromadb.config.Settings(
        persist_directory=persist_dir,
        is_persistent=True,
        anonymized_telemetry=False
    )
    try:
        logger.info("Creating new Chroma database...")
        if splits is None:
            splits = load_pdf_and_split()
        vectorstore = Chroma.from_documents(
            splits,
            embedding_function,
            persist_directory=persist_dir,
            client_settings=client_settings
        )
        logger.info("Chroma database created successfully.")
        return vectorstore
    except Exception as e:
        logger.error(f"Error creating Chroma database: {str(e)}")
        raise e

# Initialize global dependencies
try:
    splits = load_pdf_and_split()
    vectorstore = load_and_index_pdf(splits=splits)
    user_data = load_user_data()
    chat_history = load_chat_history()
except Exception as e:
    logger.error(f"Initialization error: {str(e)}")
    raise ValueError(f"Failed to initialize dependencies: {str(e)}")

# Initialize agent workflows
policy_workflow = PolicyAgent.create_workflow()
user_data_workflow = UserDataAgent.create_workflow()
calculation_workflow = CalculationAgent.create_workflow()

# Pydantic models for request/response
class QueryRequest(BaseModel):
    user_id: str
    query: str

class PolicyResponse(BaseModel):
    response: str
    documents: List[Dict]
    reasoning: Optional[str] = None
    eligibility_criteria: Dict = {}

class UserDataResponse(BaseModel):
    response: str
    user_data: Dict = {}

class CalculationResponse(BaseModel):
    response: str
    calculation_result: Dict = {}
    user_data: Dict = {}

# Save chat history
def save_chat_history(history):
    logger.info(f"Saving chat history to {CHAT_HISTORY_PATH}...")
    try:
        with open(CHAT_HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info("Chat history saved successfully.")
    except Exception as e:
        logger.error(f"Error saving chat history: {str(e)}")

# Policy Agent Endpoint
@app.post("/policy", response_model=PolicyResponse)
async def policy_query(request: QueryRequest):
    logger.info(f"Processing policy query for user {request.user_id}: {request.query}")
    
    chat_history_user = chat_history.get(request.user_id, [])
    
    state = PolicyState(
        query=request.query,
        documents=[],
        reasoning=None,
        response="",
        chat_history=chat_history_user,
        vectorstore=vectorstore,
        llm=llm
    )
    
    try:
        result = policy_workflow.invoke(state)
        
        # Extract documents
        documents = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in result["documents"]
        ]
        
        # Save to chat history
        chat_entry = {
            "query": request.query,
            "response": result["response"],
            "documents": documents,
            "reasoning": result["reasoning"],
            "agent_type": "policy"
        }
        if request.user_id not in chat_history:
            chat_history[request.user_id] = []
        chat_history[request.user_id].append(chat_entry)
        save_chat_history(chat_history)
        
        return PolicyResponse(
            response=result["response"],
            documents=documents,
            reasoning=result["reasoning"],
            eligibility_criteria=result.get("eligibility_criteria", {})
        )
    except Exception as e:
        logger.error(f"Error processing policy query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing policy query: {str(e)}")

# User Data Agent Endpoint
@app.post("/user-data", response_model=UserDataResponse)
async def user_data_query(request: QueryRequest):
    logger.info(f"Processing user data query for user {request.user_id}: {request.query}")
    
    if request.user_id not in user_data:
        raise HTTPException(status_code=404, detail=f"User ID {request.user_id} not found")
    
    chat_history_user = chat_history.get(request.user_id, [])
    
    state = UserDataState(
        user_id=request.user_id,
        query=request.query,
        user_data={},
        response="",
        chat_history=chat_history_user,
        user_data_db=user_data,
        llm=llm
    )
    
    try:
        result = user_data_workflow.invoke(state)
        
        # Save to chat history
        chat_entry = {
            "query": request.query,
            "response": result["response"],
            "user_data": result["user_data"],
            "agent_type": "user_data"
        }
        if request.user_id not in chat_history:
            chat_history[request.user_id] = []
        chat_history[request.user_id].append(chat_entry)
        save_chat_history(chat_history)
        
        return UserDataResponse(
            response=result["response"],
            user_data=result["user_data"]
        )
    except Exception as e:
        logger.error(f"Error processing user data query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing user data query: {str(e)}")

# Calculation Agent Endpoint
@app.post("/calculation", response_model=CalculationResponse)
async def calculation_query(request: QueryRequest):
    logger.info(f"Processing calculation query for user {request.user_id}: {request.query}")
    
    if request.user_id not in user_data:
        raise HTTPException(status_code=404, detail=f"User ID {request.user_id} not found")
    
    chat_history_user = chat_history.get(request.user_id, [])
    
    state = CalculationState(
        user_id=request.user_id,
        query=request.query,
        user_data={},
        user_data_response="",
        required_fields=[],
        missing_fields=[],
        calculation_result={},
        response="",
        chat_history=chat_history_user,
        user_data_db=user_data,
        llm=llm
    )
    
    try:
        result = calculation_workflow.invoke(state)
        
        # Save to chat history
        chat_entry = {
            "query": request.query,
            "response": result["response"],
            "calculation_result": result["calculation_result"],
            "user_data": result["user_data"],
            "agent_type": "calculation"
        }
        if request.user_id not in chat_history:
            chat_history[request.user_id] = []
        chat_history[request.user_id].append(chat_entry)
        save_chat_history(chat_history)
        
        return CalculationResponse(
            response=result["response"],
            calculation_result=result["calculation_result"],
            user_data=result["user_data"]
        )
    except Exception as e:
        logger.error(f"Error processing calculation query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing calculation query: {str(e)}")

# Root endpoint for health check
@app.get("/")
async def root():
    return {"message": "XYZ Bank Agents API is running"}