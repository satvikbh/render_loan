import logging
import math
from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from states import CalculationState
from user_data_agent import UserDataAgent

logger = logging.getLogger(__name__)

class CalculationAgent:
    @staticmethod
    def create_workflow():
        workflow = StateGraph(CalculationState)
        workflow.add_node("retrieve", CalculationAgent.retrieve_user_data)
        workflow.add_node("validate", CalculationAgent.validate_data)
        workflow.add_node("request_input", CalculationAgent.request_missing_input)
        workflow.add_node("calculate", CalculationAgent.perform_calculation)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "validate")
        workflow.add_conditional_edges(
            "validate",
            CalculationAgent.check_missing_data,
            {
                "complete": "calculate",
                "incomplete": "request_input"
            }
        )
        workflow.add_edge("request_input", "calculate")
        workflow.add_edge("calculate", END)
        return workflow.compile()

    @staticmethod
    def retrieve_user_data(state: CalculationState) -> CalculationState:
        logger.info(f"Retrieving user data for calculation for user_id: {state['user_id']}")
        # Use UserDataAgent to fetch user data
        user_data_state = {
            "user_id": state["user_id"],
            "query": state["query"],
            "user_data": {},
            "response": "",
            "chat_history": state["chat_history"],
            "user_data_db": state["user_data_db"],
            "llm": state["llm"]
        }
        user_data_workflow = UserDataAgent.create_workflow()
        user_data_result = user_data_workflow.invoke(user_data_state)
        state["user_data"] = user_data_result["user_data"]
        state["user_data_response"] = user_data_result["response"]
        logger.info(f"Retrieved user data: {state['user_data']}")
        return state

    @staticmethod
    def validate_data(state: CalculationState) -> CalculationState:
        logger.info("Validating data for calculation...")
        query_lower = state["query"].lower()
        required_fields = []

        # Determine required fields based on query type
        if "emi" in query_lower or "monthly payment" in query_lower:
            required_fields = ["loan_amount", "interest_rate", "loan_tenure"]
        elif "penalty" in query_lower:
            required_fields = ["delinquency_period", "principal"]

        state["required_fields"] = required_fields
        state["missing_fields"] = []

        # Check for missing fields
        for field in required_fields:
            if field not in state["user_data"] or state["user_data"][field] is None:
                state["missing_fields"].append(field)

        logger.info(f"Required fields: {required_fields}, Missing fields: {state['missing_fields']}")
        return state

    @staticmethod
    def check_missing_data(state: CalculationState) -> str:
        return "incomplete" if state["missing_fields"] else "complete"

    @staticmethod
    def request_missing_input(state: CalculationState) -> CalculationState:
        logger.info("Requesting missing input from user...")
        history_context = ""
        if state["chat_history"]:
            history_context = "\n\n*Relevant Previous Conversations*:\n"
            for entry in state["chat_history"]:
                history_context += f"User Query: {entry['query']}\nResponse: {entry['response']}\n---\n"

        missing_fields_str = ", ".join(state["missing_fields"])
        prompt = ChatPromptTemplate.from_template(
            """You are a calculation assistant. The user's query requires specific information that is missing.
            Ask the user to provide the missing details in a clear and friendly manner.

            *User Query*:
            {query}

            *Missing Information*:
            {missing_fields}

            {history_context}

            *Response*:
            To proceed with your calculation, please provide the following details: {missing_fields}.
            For example, if asked for loan tenure, specify the number of months (e.g., 360 for 30 years).
            """
        )
        response_chain = prompt | state["llm"]
        result = response_chain.invoke({
            "query": state["query"],
            "missing_fields": missing_fields_str,
            "history_context": history_context
        })
        state["response"] = result.content
        # Simulate user input for missing fields (in a real app, this would wait for user response)
        # For now, set defaults or prompt user to provide via next query
        for field in state["missing_fields"]:
            if field == "loan_tenure":
                state["user_data"][field] = 360  # Default to 30 years
            elif field == "loan_amount":
                state["user_data"][field] = 100000  # Default loan amount
            elif field == "interest_rate":
                state["user_data"][field] = 5.0  # Default interest rate
            elif field == "delinquency_period":
                state["user_data"][field] = 0  # Default delinquency
            elif field == "principal":
                state["user_data"][field] = state["user_data"].get("loan_amount", 100000)
        logger.info(f"Simulated user input for missing fields: {state['user_data']}")
        return state

    @staticmethod
    def perform_calculation(state: CalculationState) -> CalculationState:
        logger.info("Performing calculation...")
        query_lower = state["query"].lower()
        history_context = ""
        if state["chat_history"]:
            history_context = "\n\n*Relevant Previous Conversations*:\n"
            for entry in state["chat_history"]:
                history_context += f"User Query: {entry['query']}\nResponse: {entry['response']}\n---\n"

        if "emi" in query_lower or "monthly payment" in query_lower:
            # EMI Calculation: EMI = [P x R x (1+R)^N] / [(1+R)^N - 1]
            principal = float(state["user_data"]["loan_amount"])
            annual_rate = float(state["user_data"]["interest_rate"]) / 100
            monthly_rate = annual_rate / 12
            tenure_months = int(state["user_data"]["loan_tenure"])
            try:
                emi = (principal * monthly_rate * (1 + monthly_rate) ** tenure_months) / \
                      ((1 + monthly_rate) ** tenure_months - 1)
                state["calculation_result"] = {
                    "type": "EMI",
                    "value": round(emi, 2),
                    "details": {
                        "loan_amount": principal,
                        "interest_rate": annual_rate * 100,
                        "loan_tenure_months": tenure_months
                    }
                }
            except Exception as e:
                state["response"] = f"Error calculating EMI: {str(e)}"
                logger.error(f"EMI calculation error: {str(e)}")
                return state

        elif "penalty" in query_lower:
            # Penalty Calculation: Assume penalty is $100 per 30 days of delinquency, capped at $2000
            delinquency_period = int(state["user_data"]["delinquency_period"])
            principal = float(state["user_data"]["principal"])
            penalty_per_30_days = 100
            max_penalty = 2000
            penalty = min((delinquency_period // 30) * penalty_per_30_days, max_penalty)
            state["calculation_result"] = {
                "type": "Penalty",
                "value": penalty,
                "details": {
                    "delinquency_period": delinquency_period,
                    "principal": principal
                }
            }

        # Generate response
        response_prompt = ChatPromptTemplate.from_template(
            """You are a calculation assistant. Provide a clear and concise response with the calculation results.

            *Instructions:*
            - Present the calculated value clearly
            - Include relevant details used in the calculation
            - Format the response in bullet points
            - Use history to ensure consistency

            *User Query*:
            {query}

            *Calculation Result*:
            Type: {calc_type}
            Value: {calc_value}
            Details: {calc_details}

            {history_context}

            *Response (in bullet points):*
            """
        )
        response_chain = response_prompt | state["llm"]
        result = response_chain.invoke({
            "query": state["query"],
            "calc_type": state["calculation_result"]["type"],
            "calc_value": state["calculation_result"]["value"],
            "calc_details": "\n".join([f"{k}: {v}" for k, v in state["calculation_result"]["details"].items()]),
            "history_context": history_context
        })
        state["response"] = result.content
        logger.info(f"Calculation response generated: {state['response']}")
        return state