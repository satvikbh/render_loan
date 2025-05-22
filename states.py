from typing import TypedDict, List, Optional, Dict
from langchain_core.documents import Document

class PolicyState(TypedDict):
    query: str
    documents: List[Document]
    reasoning: Optional[str]
    response: str
    chat_history: List[Dict]
    vectorstore: Optional[object]  # Chroma vectorstore
    llm: Optional[object]  # ChatGoogleGenerativeAI

class UserDataState(TypedDict):
    user_id: str
    query: str
    user_data: Dict
    response: str
    chat_history: List[Dict]
    user_data_db: Optional[Dict]
    llm: Optional[object]  # ChatGoogleGenerativeAI

class CalculationState(TypedDict):
    user_id: str
    query: str
    user_data: Dict
    user_data_response: str
    required_fields: List[str]
    missing_fields: List[str]
    calculation_result: Optional[Dict]
    response: str
    chat_history: List[Dict]
    user_data_db: Optional[Dict]
    llm: Optional[object]  # ChatGoogleGenerativeAI

class CombinedState(TypedDict):
    user_id: str
    query: str
    policy_state: PolicyState
    user_data_state: UserDataState
    calculation_state: CalculationState
    final_response: str
    chat_history: List[Dict]
    orchestrator_reasoning: Optional[str]