import logging
import math
from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from states import CalculationState
from user_data_agent import UserDataAgent
from datetime import datetime

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
            required_fields = ["loan_amount", "interest_rate", "loan_tenure_months", "emi_amount"]
            if "extra" in query_lower or "tenure" in query_lower:
                required_fields.append("outstanding_amount")
        elif "penalty" in query_lower or "foreclose" in query_lower:
            required_fields = ["delinquency_period", "principal", "interest_rate", "loan_start_date", "foreclosure_status", "outstanding_amount", "loan_tenure_months"]

        state["required_fields"] = required_fields
        state["missing_fields"] = []

        for field in required_fields:
            if field not in state["user_data"] or state["user_data"][field] is None:
                state["missing_fields"].append(field)
            elif field in ["loan_amount", "principal", "interest_rate", "emi_amount", "outstanding_amount"]:
                try:
                    state["user_data"][field] = float(state["user_data"][field])
                    if state["user_data"][field] <= 0:
                        state["missing_fields"].append(field)
                        logger.warning(f"Invalid {field}: {state['user_data'][field]}")
                except (ValueError, TypeError):
                    state["missing_fields"].append(field)
                    logger.warning(f"Invalid {field} format: {state['user_data'][field]}")
            elif field == "loan_tenure_months":
                try:
                    state["user_data"][field] = int(state["user_data"][field])
                    if state["user_data"][field] <= 0:
                        state["missing_fields"].append(field)
                        logger.warning(f"Invalid loan_tenure_months: {state['user_data'][field]}")
                except (ValueError, TypeError):
                    state["missing_fields"].append(field)
                    logger.warning(f"Invalid loan_tenure format: {state['user_data'][field]}")
            elif field == "loan_start_date":
                try:
                    datetime.strptime(state["user_data"][field], "%Y-%m-%d")
                except (ValueError, TypeError):
                    state["missing_fields"].append(field)
                    logger.warning(f"Invalid loan_start_date format: {state['user_data'][field]}")

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
            Please provide the following details: {missing_fields}. For example, specify loan tenure in months (e.g., 360 for 30 years), loan amount in numeric format, or loan start date in YYYY-MM-DD format.
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
            if field == "loan_tenure_months":
                state["user_data"][field] = state["user_data"].get("loan_tenure_months", 360)
            elif field == "loan_amount":
                state["user_data"][field] = state["user_data"].get("loan_amount", 100000)
            elif field == "interest_rate":
                state["user_data"][field] = state["user_data"].get("interest_rate", 5.0)
            elif field == "delinquency_period":
                state["user_data"][field] = state["user_data"].get("delinquency_period", 0)
            elif field == "principal":
                state["user_data"][field] = state["user_data"].get("loan_amount", 100000)
            elif field == "loan_start_date":
                state["user_data"][field] = state["user_data"].get("loan_start_date", "2020-01-01")
            elif field == "foreclosure_status":
                state["user_data"][field] = state["user_data"].get("foreclosure_status", "Not Started")
            elif field == "outstanding_amount":
                state["user_data"][field] = state["user_data"].get("outstanding_amount", state["user_data"].get("loan_amount", 100000))
            elif field == "emi_amount":
                state["user_data"][field] = state["user_data"].get("emi_amount", 0)
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

        user_data = state["user_data"]
        calculation_result = {"type": "", "value": 0, "details": {}}

        def calculate_emi(principal, annual_rate, tenure_months):
            monthly_rate = annual_rate / 100 / 12
            emi = (principal * monthly_rate * (1 + monthly_rate) ** tenure_months) / \
                ((1 + monthly_rate) ** tenure_months - 1)
            return round(emi, 2)

        def calculate_new_tenure(outstanding_amount, monthly_rate, new_emi):
            try:
                if new_emi <= 0 or monthly_rate <= 0:
                    return None
                n = math.log(new_emi / (new_emi - outstanding_amount * monthly_rate)) / math.log(1 + monthly_rate)
                return max(0, math.ceil(n))
            except (ValueError, ZeroDivisionError) as e:
                logger.error(f"Tenure calculation error: {str(e)}")
                return None

        if "emi" in query_lower or "monthly payment" in query_lower:
            principal = float(user_data["loan_amount"])
            annual_rate = float(user_data["interest_rate"])
            tenure_months = int(user_data["loan_tenure_months"])
            stored_emi = float(user_data.get("emi_amount", 0))
            emi = calculate_emi(principal, annual_rate, tenure_months)
            calculation_result = {
                "type": "EMI",
                "value": emi,
                "details": {
                    "loan_amount": principal,
                    "interest_rate": annual_rate,
                    "loan_tenure_months": tenure_months
                }
            }
            if stored_emi and abs(emi - stored_emi) > 0.01:
                logger.warning(f"EMI calculation mismatch: Calculated {emi}, Stored {stored_emi}")
                calculation_result["details"]["note"] = f"Calculated EMI ({emi}) differs from stored EMI ({stored_emi}). Using calculated value."

            # Handle additional EMI payment for tenure recalculation
            if "extra" in query_lower and "tenure" in query_lower:
                extra_amount = None
                try:
                    extra_amount = float(query_lower.split("extra")[1].split()[0])
                except (IndexError, ValueError):
                    extra_amount = 5000  # Default if not found
                outstanding_amount = float(user_data.get("outstanding_amount", principal))
                new_emi = emi + extra_amount
                monthly_rate = annual_rate / 100 / 12
                new_tenure = calculate_new_tenure(outstanding_amount, monthly_rate, new_emi)
                if new_tenure is not None:
                    calculation_result["type"] = "Tenure Adjustment"
                    calculation_result["value"] = new_tenure
                    calculation_result["details"].update({
                        "original_emi": emi,
                        "extra_payment": extra_amount,
                        "new_emi": new_emi,
                        "original_tenure_months": tenure_months,
                        "new_tenure_months": new_tenure,
                        "outstanding_amount": outstanding_amount
                    })
                else:
                    state["response"] = "Error calculating new tenure: Invalid input data."
                    logger.error("Failed to calculate new tenure due to invalid input.")
                    return state

        elif "penalty" in query_lower or "foreclose" in query_lower:
            principal = float(user_data["principal"])
            annual_rate = float(user_data["interest_rate"])
            loan_start_date = datetime.strptime(user_data["loan_start_date"], "%Y-%m-%d")
            current_date = datetime.now()
            tenure_months = int(user_data.get("loan_tenure_months", 360))
            months_passed = (current_date.year - loan_start_date.year) * 12 + current_date.month - loan_start_date.month
            remaining_months = max(0, tenure_months - months_passed)
            penalty_rate = 0.02 if remaining_months > 12 else 0.01
            penalty = principal * penalty_rate
            calculation_result = {
                "type": "Foreclosure Penalty",
                "value": round(penalty, 2),
                "details": {
                    "principal": principal,
                    "remaining_months": remaining_months,
                    "penalty_rate": penalty_rate * 100,
                    "delinquency_period": user_data.get("delinquency_period", 0),
                    "foreclosure_status": user_data.get("foreclosure_status", "Not Started")
                }
            }
            if "penalty" in user_data and user_data["penalty"] is not None:
                stored_penalty = float(user_data["penalty"])
                if abs(penalty - stored_penalty) > 0.01:
                    logger.warning(f"Penalty calculation mismatch: Calculated {penalty}, Stored {stored_penalty}")
                    calculation_result["details"]["note"] = f"Calculated penalty ({penalty}) differs from stored penalty ({stored_penalty}). Using calculated value."

        state["calculation_result"] = calculation_result

        response_prompt = ChatPromptTemplate.from_template(
            """You are a bank calculation assistant. Provide a concise, professional response to the query with calculation results in a plain text Markdown table. Include only validated data used in the calculation. If a mismatch with stored data is noted, include the note in the table. For tenure adjustment queries, clearly state the new tenure and its impact.

            User Query:
            {query}

            Calculation Result (Table):
            | Type | Value |
            |------|-------|
            | {calc_type} | {calc_value} |

            Calculation Details (Table):
            | Field | Value |
            |-------|-------|
            {calc_details}

            {history_context}

            Response:
            Based on your query, here is the calculation result:

            Calculation Result:
            | Type | Value |
            |------|-------|
            | {calc_type} | {calc_value} |

            Calculation Details:
            | Field | Value |
            |-------|-------|
            {calc_details}

            {tenure_impact}
            Please contact the bank to confirm and proceed with any changes to your payment plan.
            """
        )

        table_details = []
        for k, v in calculation_result["details"].items():
            table_details.append(f"| {k.replace('_', ' ').title()} | {v} |")
        table_details_str = "\n".join(table_details)

        tenure_impact = ""
        if calculation_result["type"] == "Tenure Adjustment":
            original_tenure = calculation_result["details"].get("original_tenure_months", 0)
            new_tenure = calculation_result["details"].get("new_tenure_months", 0)
            tenure_reduction = original_tenure - new_tenure
            tenure_impact = f"New Tenure: Your loan tenure will be approximately {new_tenure} months.\nImpact: Adding an extra {calculation_result["details"]["extra_payment"]} to your EMI will reduce your remaining loan tenure by approximately {tenure_reduction} months."

        response_chain = response_prompt | state["llm"]
        result = response_chain.invoke({
            "query": state["query"],
            "calc_type": calculation_result["type"],
            "calc_value": calculation_result["value"],
            "calc_details": table_details_str,
            "history_context": history_context,
            "tenure_impact": tenure_impact
        })
        state["response"] = result.content
        logger.info(f"Calculation response generated: {state['response']}")
        return state