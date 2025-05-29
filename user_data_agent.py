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
        
        # Check if the query explicitly asks for the user's name
        query_lower = state["query"].lower()
        is_name_query = "name" in query_lower and any(word in query_lower for word in ["my name", "what is my name", "who am i"])

        # Check chat history for user's name if query is about name
        name_from_history = None
        if is_name_query and state["chat_history"]:
            for entry in state["chat_history"]:
                if "my name is" in entry["query"].lower():
                    words = entry["query"].lower().split()
                    try:
                        name_index = words.index("name") + 2
                        name_from_history = " ".join(words[name_index:]).title()
                    except:
                        pass
                elif "name" in entry["response"].lower():
                    match = re.search(r"(?:your name is|name:)\s*([A-Za-z\s]+)", entry["response"], re.IGNORECASE)
                    if match:
                        name_from_history = match.group(1).title()
        
        # If the query is explicitly about the name and a name is found
        if is_name_query and (name_from_history or state["user_data"].get("name")):
            name = name_from_history or state["user_data"]["name"]
            state["response"] = f"Your name is {name}."
            logger.info(f"User name retrieved: {name}")
            return state
        
        # If no user data is available, provide fallback response
        if not state.get("user_data"):
            logger.info("No user data available, using fallback response.")
            state["response"] = "I don't have access to your account information. Please log into your bank account portal or contact customer service at 1-800-555-1234 with your account details."
            return state
        
        # Filter user data based on query keywords
        query_keywords = query_lower.split()
        relevant_fields = []
        for field in state["user_data"].keys():
            if any(keyword in field.lower() for keyword in query_keywords) or field in ["outstanding_amount", "loan_amount", "principal", "interest", "penalty", "emi_amount"]:
                if state["user_data"][field] is not None:
                    relevant_fields.append((field, state["user_data"][field]))

        # Prepare user data for response
        user_info = "User Data:\n"
        if relevant_fields:
            user_info += "\n".join([f"- {field.replace('_', ' ').title()}: {value}" for field, value in relevant_fields])
        else:
            user_info += "No relevant user data found for the query."

        # Initialize history context
        history_context = ""
        if state["chat_history"]:
            history_context = "\nPrevious Conversations:\n"
            for entry in state["chat_history"]:
                history_context += f"Query: {entry['query']}\nResponse: {entry['response']}\n---\n"
        
        # Generate response with user data and history context
        response_prompt = ChatPromptTemplate.from_template(
            """You are a bank assistant. Provide a concise, professional response to the user's query based on their data and previous conversations. Use plain text only. Present numerical data in a tabular format. For eligibility queries (e.g., foreclosure), include relevant user data (delinquency period, loan status) to inform the response.
            Output the response in short, concise pointers or a table as needed.

            User Data:
            {user_info}

            User Query:
            {query}

            {history_context}

            Response:
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