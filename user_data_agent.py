import logging
import re
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from states import UserDataState

logger = logging.getLogger(__name__)

class UserDataAgent:
    @staticmethod
    def create_workflow():
        workflow = StateGraph(UserDataState)
        workflow.add_node("retrieve", UserDataAgent.retrieve_user_data)
        workflow.add_node("generate", UserDataAgent.generate_user_response)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile()

    @staticmethod
    def retrieve_user_data(state: UserDataState) -> UserDataState:
        logger.info(f"Retrieving user data for user_id: {state['user_id']}")
        user_data_db = state.get('user_data_db')
        if state["user_id"] not in user_data_db:
            state["response"] = f"No user data found for user ID: {state['user_id']}"
            logger.warning(f"User ID {state['user_id']} not found in database.")
            return state
        state["user_data"] = user_data_db[state["user_id"]]
        logger.info(f"Retrieved user data for {state['user_id']}.")
        return state

    @staticmethod
    def generate_user_response(state: UserDataState) -> UserDataState:
        logger.info("Generating user data response...")
        
        # Check if user_data contains the requested information
        user_info = ""
        if state.get("user_data"):
            user_info = "\n".join([f"{key}: {value}" for key, value in state["user_data"].items()])
        
        # Initialize history context
        history_context = ""
        if state["chat_history"]:
            history_context = "\n\n*Relevant Previous Conversations*:\n"
            for entry in state["chat_history"]:
                history_context += f"User Query: {entry['query']}\nResponse: {entry['response']}\n---\n"
        
        # Check chat history for user's name if query is about name
        name_from_history = None
        if "name" in state["query"].lower():
            for entry in state["chat_history"]:
                # Look for name in previous queries or responses
                if "my name is" in entry["query"].lower():
                    # Extract name from query like "My name is Satvik Bhardwaj"
                    words = entry["query"].lower().split()
                    try:
                        name_index = words.index("name") + 2  # Skip "my name is"
                        name_from_history = " ".join(words[name_index:]).title()
                    except:
                        pass
                elif "name" in entry["response"].lower():
                    # Extract name from response if mentioned
                    match = re.search(r"(?:your name is|name:)\s*([A-Za-z\s]+)", entry["response"], re.IGNORECASE)
                    if match:
                        name_from_history = match.group(1).title()
        
        # If name is found in history, use it directly
        if name_from_history:
            state["response"] = f"Your name is {name_from_history}."
            logger.info(f"User name retrieved from chat history: {name_from_history}")
            return state
        
        # If no user data and no name in history, provide fallback response
        if not state.get("user_data") and not name_from_history:
            logger.info("No user data or name in history available, using fallback response.")
            state["response"] = "I don't have access to your account information in this conversation. To get your specific details, please log into your account portal or contact our customer service at 1-800-XXX-XXXX with your account details."
            return state
        
        # Use LLM to generate response with user data and history context
        response_prompt = ChatPromptTemplate.from_template(
            """You are a user data assistant. Based on the user's data, query, and previous conversations,
            provide a concise and accurate response to the user-specific part of the query.

            *Instructions:*
            - If you have the EXACT information requested, provide it clearly and directly
            - If you DON'T have the requested information but know the user data exists, explain how they can access it
            - If no relevant user data exists at all, clearly state this fact
            - Format your answer as clear bullet points
            - Be specific about account numbers, balances, and other personal details if available

            *User Data*:
            {user_info}

            *User Query*:
            {query}

            {history_context}

            *User Response (in bullet points):*
            """
        )
        response_chain = response_prompt | state.get('llm')
        response_result = response_chain.invoke({
            "user_info": user_info,
            "query": state["query"],
            "history_context": history_context
        })
        state["response"] = response_result.content
        logger.info("User data response generated.")
        return state