import streamlit as st
import os
import logging
import tempfile
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from sklearn.metrics.pairwise import cosine_similarity
from typing import TypedDict, List, Optional, Dict
import numpy as np
import chromadb
import uuid
from orchestrator import Orchestrator
from states import PolicyState, UserDataState, CalculationState, CombinedState
from utils import load_chat_history

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in .env file. Please set it to continue.")
    st.stop()

PDF_PATH = os.path.join('public', 'Employee Handbook.pdf')
USER_DATA_PATH = os.path.join('user_details.json')
CHAT_HISTORY_PATH = os.path.join('conversation_history.json')

if "persist_dir" not in st.session_state:
    unique_id = str(uuid.uuid4())
    st.session_state.persist_dir = os.path.join(tempfile.gettempdir(), f"chroma_db_{unique_id}")

PERSIST_DIR = st.session_state.persist_dir

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-04-17",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)

@st.cache_data
def load_user_data():
    logger.info("Loading user data from JSON...")
    if not os.path.exists(USER_DATA_PATH):
        st.error(f"User data JSON file not found at {USER_DATA_PATH}.")
        logger.error(f"JSON not found at {USER_DATA_PATH}")
        st.stop()
    try:
        with open(USER_DATA_PATH, 'r') as f:
            data = json.load(f)
        logger.info("User data loaded successfully.")
        return data.get("users", {})
    except Exception as e:
        st.error(f"Error loading user data: {str(e)}")
        logger.error(f"User data loading error: {str(e)}")
        st.stop()

@st.cache_data
def load_pdf_and_split():
    logger.info("Loading and splitting PDF...")
    if not os.path.exists(PDF_PATH):
        st.error(f"PDF file not found at {PDF_PATH}.")
        logger.error(f"PDF not found at {PDF_PATH}")
        st.stop()
    try:
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        logger.info(f"PDF loaded and split into {len(splits)} chunks.")
        return splits
    except Exception as e:
        st.error(f"Error loading PDF: {str(e)}")
        logger.error(f"PDF loading error: {str(e)}")
        st.stop()

@st.cache_resource
def load_and_index_pdf(_splits=None, persist_dir=None):
    if persist_dir is None:
        persist_dir = PERSIST_DIR
    logger.info(f"Using Chroma database at: {persist_dir}")
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    client_settings = chromadb.config.Settings(
        persist_directory=persist_dir,
        is_persistent=True,
        anonymized_telemetry=False
    )
    try:
        logger.info("Creating new Chroma database...")
        if _splits is None:
            _splits = load_pdf_and_split()
        vectorstore = Chroma.from_documents(
            _splits,
            embedding_function,
            persist_directory=persist_dir,
            client_settings=client_settings
        )
        logger.info("Chroma database created successfully.")
        return vectorstore
    except Exception as e:
        logger.error(f"Error creating Chroma database: {str(e)}")
        raise e

def save_chat_history(history):
    logger.info(f"Saving chat history to {CHAT_HISTORY_PATH}...")
    try:
        with open(CHAT_HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info("Chat history saved successfully.")
    except Exception as e:
        logger.error(f"Error saving chat history: {str(e)}")

def get_relevant_chat_history(query: str, user_id: str, chat_history: Dict, k: int = 3) -> List[Dict]:
    logger.info("Finding relevant chat history...")
    if user_id not in chat_history or not chat_history[user_id]:
        return []
    
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    query_embedding = embedding_function.embed_query(query)
    
    relevant_history = []
    query_lower = query.lower().strip()
    
    # Prioritize calculation results for tenure-related queries
    if "new tenure" in query_lower or "tenure after" in query_lower:
        for entry in reversed(chat_history[user_id]):
            if entry.get("calculation_result") and entry["calculation_result"].get("type") == "Tenure Adjustment":
                relevant_history.append((entry, 1.0))  # Highest priority
                break
    
    # Add other relevant history based on semantic similarity
    for entry in chat_history[user_id]:
        if (entry, 1.0) not in relevant_history:  # Avoid duplicates
            entry_embedding = embedding_function.embed_query(entry["query"])
            similarity = cosine_similarity([query_embedding], [entry_embedding])[0][0]
            relevant_history.append((entry, similarity))
    
    # Sort by similarity, ensuring calculation results stay on top
    relevant_history = sorted(relevant_history, key=lambda x: x[1], reverse=True)[:k]
    
    # Always include the most recent entry if not already included
    most_recent = chat_history[user_id][-1] if chat_history[user_id] else None
    if most_recent and most_recent not in [x[0] for x in relevant_history]:
        relevant_history.append((most_recent, 0))
    
    logger.info(f"Found {len(relevant_history)} relevant chat history entries.")
    return [entry for entry, _ in relevant_history[:k]]

st.title('XYZ Bank Assistant Bot')

if "user_id" not in st.session_state:
    st.session_state.user_id = ""
if "current_chat_display" not in st.session_state:
    st.session_state.current_chat_display = []
if "chat_history_loaded" not in st.session_state:
    st.session_state.chat_history = load_chat_history()
    st.session_state.chat_history_loaded = True

with st.spinner('Loading systems...'):
    try:
        splits = load_pdf_and_split()
        st.session_state.vectorstore = load_and_index_pdf(_splits=splits, persist_dir=PERSIST_DIR)
        st.session_state.user_data = load_user_data()
    except Exception as e:
        st.error(f"Error loading systems: {str(e)}")
        if st.button("Retry Loading", key="retry_loading"):
            st.rerun()
        st.stop()

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = Orchestrator(llm=llm)

col1, col2 = st.columns([3, 1])
with col1:
    user_id = st.text_input('Enter your User ID (e.g., user001):', key="user_id_input", value=st.session_state.user_id)
with col2:
    if user_id:
        if st.button("Logout", key="logout_button"):
            st.session_state.user_id = ""
            st.session_state.current_chat_display = []
            st.session_state.chat_history = load_chat_history()
            st.session_state.orchestrator = Orchestrator(llm=llm)
            st.rerun()

if not user_id:
    st.error("User ID is required to proceed.")
    st.stop()
else:
    st.session_state.user_id = user_id

# Display previous chats without technical details
for chat in st.session_state.current_chat_display:
    with st.chat_message("user"):
        st.markdown(chat['query'])
    with st.chat_message("assistant"):
        st.markdown(chat['response'])

query = st.chat_input('How can I help you today?', key="query_input")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.spinner('Finding information for you...'):
        try:
            chat_history = get_relevant_chat_history(query, user_id, st.session_state.chat_history)

            initial_state = CombinedState(
                user_id=user_id,
                query=query,
                policy_state=PolicyState(
                    query=query,
                    documents=[],
                    reasoning=None,
                    response="",
                    chat_history=chat_history,
                    vectorstore=st.session_state.vectorstore,
                    llm=llm
                ),
                user_data_state=UserDataState(
                    user_id=user_id,
                    query=query,
                    user_data={},
                    response="",
                    chat_history=chat_history,
                    user_data_db=st.session_state.user_data,
                    llm=llm
                ),
                calculation_state=CalculationState(
                    user_id=user_id,
                    query=query,
                    user_data={},
                    user_data_response="",
                    required_fields=[],
                    missing_fields=[],
                    calculation_result={},
                    response="",
                    chat_history=chat_history,
                    user_data_db=st.session_state.user_data,
                    llm=llm
                ),
                final_response="",
                chat_history=chat_history,
                orchestrator_reasoning=""
            )

            result = st.session_state.orchestrator.orchestrate(initial_state)

            policy_documents = [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in result["policy_state"]["documents"]
            ]

            # Determine agent type for this query
            classification = st.session_state.orchestrator.classify_query(query, user_id, chat_history)
            agent_type = classification['agent_type']

            chat_entry = {
                "query": query,
                "response": result["final_response"],
                "policy_documents": policy_documents,
                "policy_reasoning": result["policy_state"]["reasoning"] or "No policy reasoning performed.",
                "user_data": result["user_data_state"]["user_data"],
                "calculation_result": result["calculation_state"]["calculation_result"],
                "orchestrator_reasoning": result["orchestrator_reasoning"],
                "agent_type": agent_type
            }
            if user_id not in st.session_state.chat_history:
                st.session_state.chat_history[user_id] = []
            st.session_state.chat_history[user_id].append(chat_entry)
            save_chat_history(st.session_state.chat_history)
            st.session_state.current_chat_display.append(chat_entry)

            with st.chat_message("assistant"):
                st.markdown(result['final_response'])
                if agent_type != "small_talk":
                    with st.expander(f"🔍 View Technical Details", expanded=False):
                        st.markdown("### Retrieved Policy Excerpts")
                        for i, doc in enumerate(result["policy_state"]["documents"]):
                            st.markdown(f"**Document {i+1}:**")
                            st.markdown(doc.page_content)
                            st.markdown("---")
                        
                        st.markdown("### Policy Reasoning Process")
                        st.markdown(result["policy_state"]["reasoning"] or "No reasoning available.")
                        
                        st.markdown("### User Data")
                        user_data = result["user_data_state"]["user_data"]
                        if user_data:
                            st.markdown("\n".join([f"**{key}**: {value}" for key, value in user_data.items()]))
                        else:
                            st.markdown("No user data available.")
                        
                        st.markdown("### Calculation Results")
                        calculation_result = result["calculation_state"]["calculation_result"]
                        if calculation_result:
                            st.markdown(f"**Type**: {calculation_result.get('type', 'N/A')}")
                            st.markdown(f"**Value**: {calculation_result.get('value', 'N/A')}")
                            st.markdown("**Details**:")
                            st.markdown("\n".join([f"{k}: {v}" for k, v in calculation_result.get("details", {}).items()]))
                        else:
                            st.markdown("No calculation performed.")
                        
                        st.markdown("### Orchestrator Reasoning")
                        st.markdown(result["orchestrator_reasoning"] or "No orchestrator reasoning available.")

        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            logger.error(f"Query processing error: {str(e)}")