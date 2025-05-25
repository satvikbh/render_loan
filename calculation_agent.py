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

        if "emi" in query_lower or "monthly payment" in query_lower:
            required_fields = ["loan_amount", "interest_rate", "loan_tenure"]
        elif "penalty" in query_lower or "foreclose" in query_lower:
            required_fields = ["delinquency_period", "principal", "interest_rate", "loan_start_date", "foreclosure_status"]

        state["required_fields"] = required_fields
        state["missing_fields"] = []

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
            history_context = "\nPrevious Conversations:\n"
            for entry in state["chat_history"]:
                history_context += f"Query: {entry['query']}\nResponse: {entry['response']}\n---\n"

        missing_fields_str = ", ".join(state["missing_fields"])
        prompt = ChatPromptTemplate.from_template(
            """You are a bank calculation assistant. The query requires missing information. Request the missing details in a concise, professional manner using plain text.

            User Query:
            {query}

            Missing Information:
            {missing_fields}

            {history_context}

            Response:
            Please provide the following details: {missing_fields}. For example, specify loan tenure in months (e.g., 360 for 30 years) or loan start date in YYYY-MM-DD format.
            """
        )
        response_chain = prompt | state["llm"]
        result = response_chain.invoke({
            "query": state["query"],
            "missing_fields": missing_fields_str,
            "history_context": history_context
        })
        state["response"] = result.content
        for field in state["missing_fields"]:
            if field == "loan_tenure":
                state["user_data"][field] = 360
            elif field == "loan_amount":
                state["user_data"][field] = 100000
            elif field == "interest_rate":
                state["user_data"][field] = 5.0
            elif field == "delinquency_period":
                state["user_data"][field] = 0
            elif field == "principal":
                state["user_data"][field] = state["user_data"].get("loan_amount", 100000)
            elif field == "loan_start_date":
                state["user_data"][field] = "2020-01-01"
            elif field == "foreclosure_status":
                state["user_data"][field] = "Not Started"
        logger.info(f"Simulated user input for missing fields: {state['user_data']}")
        return state

    @staticmethod
    def perform_calculation(state: CalculationState) -> CalculationState:
        logger.info("Performing calculation...")
        query_lower = state["query"].lower()
        history_context = ""
        if state["chat_history"]:
            history_context = "\nPrevious Conversations:\n"
            for entry in state["chat_history"]:
                history_context += f"Query: {entry['query']}\nResponse: {entry['response']}\n---\n"

        if "emi" in query_lower or "monthly payment" in query_lower:
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

        elif "penalty" in query_lower or "foreclose" in query_lower:
            from datetime import datetime
            principal = float(state["user_data"]["principal"])
            annual_rate = float(state["user_data"]["interest_rate"]) / 100
            loan_start_date = datetime.strptime(state["user_data"]["loan_start_date"], "%Y-%m-%d")
            current_date = datetime.now()
            tenure_months = state["user_data"].get("loan_tenure", 360)  # Default 30 years
            months_passed = (current_date.year - loan_start_date.year) * 12 + current_date.month - loan_start_date.month
            remaining_months = max(0, tenure_months - months_passed)
            # Penalty: 2% of principal for remaining tenure > 12 months, 1% for <= 12 months
            penalty_rate = 0.02 if remaining_months > 12 else 0.01
            penalty = principal * penalty_rate
            state["calculation_result"] = {
                "type": "Foreclosure Penalty",
                "value": round(penalty, 2),
                "details": {
                    "principal": principal,
                    "remaining_months": remaining_months,
                    "penalty_rate": penalty_rate * 100,
                    "delinquency_period": state["user_data"].get("delinquency_period", 0),
                    "foreclosure_status": state["user_data"].get("foreclosure_status", "Not Started")
                }
            }

        response_prompt = ChatPromptTemplate.from_template(
            """You are a bank calculation assistant. Provide a concise, professional response with the calculation results in a proper **tabular format**. Use plain text only, no bold or italics. Include relevant details used in the calculation.
            **Give the textual output in short and concise pointers**

            User Query:
            {query}

            Calculation Result:
            Type: {calc_type}
            Value: {calc_value}
            Details:
            {calc_details}

            {history_context}

            Response:
            Calculation Results:
            """
        )
        table_details = "\n".join([f"| {k.replace('_', ' ').title()} | {v} |" for k, v in state["calculation_result"]["details"].items()])
        response_chain = response_prompt | state["llm"]
        result = response_chain.invoke({
            "query": state["query"],
            "calc_type": state["calculation_result"]["type"],
            "calc_value": state["calculation_result"]["value"],
            "calc_details": "\n".join([f"{k}: {v}" for k, v in state["calculation_result"]["details"].items()]),
            "history_context": history_context,
            "table_details": table_details
        })
        state["response"] = result.content
        logger.info(f"Calculation response generated: {state['response']}")
        return state