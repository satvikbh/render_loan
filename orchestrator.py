import logging
from typing import Dict, List, Optional
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import pipeline
from states import PolicyState, UserDataState, CalculationState, CombinedState
from policy_agent import PolicyAgent
from user_data_agent import UserDataAgent
from calculation_agent import CalculationAgent
from utils import load_chat_history
import re

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, llm):
        self.llm = llm
        self.policy_workflow = PolicyAgent.create_workflow()
        self.user_data_workflow = UserDataAgent.create_workflow()
        self.calculation_workflow = CalculationAgent.create_workflow()
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Initialize intent classifier (replace with fine-tuned model path if available)
        self.intent_classifier = pipeline("text-classification", model="distilbert-base-uncased")
        try:
            with open('user_details.json', 'r') as f:
                self.user_data_db = json.load(f).get("users", {})
        except Exception as e:
            logger.error(f"Error loading user_details.json: {str(e)}")
            self.user_data_db = {}
        # Reference queries for semantic similarity
        self.reference_queries = {
            "policy": ["What's the foreclosure policy?", "What are the bank rules for loans?"],
            "user_data": ["What's my loan balance?", "Show my account details", "What is my outstanding amount?"],
            "calculation": ["Calculate my EMI", "What's the penalty for foreclosure?", "What is my EMI amount?"],
            "small_talk": ["Hello", "Good morning"],
            "combined": ["Am I eligible to foreclose?", "Can I close my loan early?", "Is my loan eligible for prepayment?"]
        }

    def handle_small_talk(self, query: str, user_id: str, chat_history: List[Dict]) -> str:
        logger.info(f"Handling small talk for query: {query}")
        history_context = ""
        if chat_history:
            history_context = "\nPrevious Conversations:\n"
            for entry in chat_history:
                history_context += f"Query: {entry['query']}\nResponse: {entry['response']}\n---\n"
        
        # Check for user's name in chat history or user data
        user_name = None
        if chat_history:
            for entry in chat_history:
                if "my name is" in entry["query"].lower():
                    words = entry["query"].lower().split()
                    try:
                        name_index = words.index("name") + 2
                        user_name = " ".join(words[name_index:]).title()
                    except:
                        pass
                elif "name" in entry["response"].lower():
                    match = re.search(r"(?:your name is|name:)\s*([A-Za-z\s]+)", entry["response"], re.IGNORECASE)
                    if match:
                        user_name = match.group(1).title()
        if not user_name and user_id in self.user_data_db:
            user_name = self.user_data_db[user_id].get("name")

        # Define the prompt to ensure exact greeting format
        small_talk_prompt = ChatPromptTemplate.from_template(
            """You are a bank assistant. Respond to the user's small talk query (e.g., greetings like 'Hello' or 'Good morning') with exactly the following response:
            - If a user name is provided: "Hello {user_name}, how can I assist you with your banking needs today?"
            - If no user name is provided: "Hello, how can I assist you with your banking needs today?"
            Use plain text only and maintain a professional tone. Do not add any extra text or modify the response format. Refer to the bank instead of the company. Incorporate previous conversation context if relevant, but only to inform continuity, not to change the response.

            User Query:
            {query}

            User Name:
            {user_name}

            {history_context}

            Response:
            """
        )
        response_chain = small_talk_prompt | self.llm
        result = response_chain.invoke({
            "query": query,
            "user_name": user_name if user_name else "N/A",
            "history_context": history_context
        })
        
        # Ensure the response matches the required format
        response = result.content.strip()
        if user_name and f"Hello {user_name}" not in response:
            response = f"Hello {user_name}, how can I assist you with your banking needs today?"
        elif not user_name and response != "Hello, how can I assist you with your banking needs today?":
            response = "Hello, how can I assist you with your banking needs today?"
        
        logger.info(f"Small talk response generated: {response}")
        return response

    def evaluate_eligibility(self, user_data: Dict, eligibility_criteria: Dict) -> Dict:
        logger.info("Evaluating eligibility...")
        result = {"eligible": True, "details": []}
        
        for criterion, value in eligibility_criteria.items():
            if criterion == "delinquency_period" and "days" in value.lower():
                threshold = int(value.split("<")[1].split("days")[0].strip())
                user_value = user_data.get("delinquency_period", float("inf"))
                if user_value >= threshold:
                    result["eligible"] = False
                    result["details"].append(f"Delinquency Period: {user_value} days (must be < {threshold} days)")
                else:
                    result["details"].append(f"Delinquency Period: {user_value} days (meets < {threshold} days)")
            elif criterion == "foreclosure_status":
                allowed_statuses = [s.strip() for s in value.split("or")]
                user_status = user_data.get("foreclosure_status", "Unknown")
                if user_status in allowed_statuses:
                    result["details"].append(f"Foreclosure Status: {user_status} (meets {value})")
                else:
                    result["eligible"] = False
                    result["details"].append(f"Foreclosure Status: {user_status} (must be {value})")
            elif criterion == "lock-in_period":
                from datetime import datetime
                loan_start_date = user_data.get("loan_start_date", "N/A")
                if loan_start_date != "N/A":
                    start_date = datetime.strptime(loan_start_date, "%Y-%m-%d")
                    current_date = datetime.now()
                    months_passed = (current_date.year - start_date.year) * 12 + current_date.month - start_date.month
                    threshold_months = int(value.split("<")[1].split("months")[0].strip()) if "<" in value else 12
                    if months_passed >= threshold_months:
                        result["details"].append(f"Lock-in Period: Completed (loan started {loan_start_date})")
                    else:
                        result["eligible"] = False
                        result["details"].append(f"Lock-in Period: Not completed (loan started {loan_start_date})")
        
        return result

    def classify_query(self, query: str, user_id: str, chat_history: List[Dict]) -> Dict:
        logger.info(f"Classifying query: {query}")

        # Step 1: Intent classification with LLM
        intent_result = self.intent_classifier(query)
        predicted_label = intent_result[0]['label'].lower()  # Assume fine-tuned model returns 'policy', 'user_data', etc.
        intent_score = intent_result[0]['score']
        
        # Step 2: Semantic similarity to confirm classification
        query_embedding = self.embedding_function.embed_query(query)
        category_scores = {}
        for category, ref_queries in self.reference_queries.items():
            similarities = []
            for ref_query in ref_queries:
                ref_embedding = self.embedding_function.embed_query(ref_query)
                similarity = cosine_similarity([query_embedding], [ref_embedding])[0][0]
                similarities.append(similarity)
            category_scores[category] = np.mean(similarities)
        
        # Combine intent classification and semantic similarity
        if intent_score < 0.7:  # If LLM confidence is low, use similarity as fallback
            agent_type = max(category_scores, key=category_scores.get)
        else:
            agent_type = predicted_label

        # Boost specific categories for financial and eligibility queries
        query_lower = query.lower().strip()
        if any(word in query_lower for word in ['emi', 'monthly payment']):
            if category_scores['calculation'] > 0.5:
                agent_type = 'calculation'
        elif any(word in query_lower for word in ['outstanding amount', 'loan balance', 'account details']):
            if category_scores['user_data'] > 0.5:
                agent_type = 'user_data'
        elif any(word in query_lower for word in ['eligible', 'eligibility', 'can i', 'am i', 'qualify']):
            if category_scores['combined'] > 0.5 or any(word in query_lower for word in ['foreclosure', 'foreclose', 'early closure', 'prepayment']):
                agent_type = 'combined'

        # Log classification details
        keyword_context = (
            f"Intent Classification:\n"
            f"- Predicted: {predicted_label} (Score: {intent_score})\n"
            f"Semantic Similarity Scores:\n" +
            "\n".join([f"- {cat}: {score:.3f}" for cat, score in category_scores.items()])
        )
        logger.info(keyword_context)

        history_context = ""
        if chat_history:
            history_context = "\nPrevious Conversations:\n"
            for entry in chat_history:
                history_context += f"Query: {entry['query']}\nResponse: {entry['response']}\n---\n"

        classification_prompt = ChatPromptTemplate.from_template(
            """You are a query classification system for a bank assistant. Validate the predicted category based on the query, semantic similarity, and context. Return the final category. Refer to the bank instead of the company.

            EXACTLY FOLLOW THIS FORMAT:
            1. Reasoning
            2. Agent type: X (where X is policy, user_data, calculation, small_talk, or combined)

            CATEGORY DEFINITIONS:
            - policy: Questions about bank rules or procedures
            - user_data: Questions about personal account details, such as outstanding amount or loan balance
            - calculation: Questions requiring financial calculations, like EMI or penalties
            - small_talk: General conversation or greetings
            - combined: Questions involving multiple types, especially eligibility queries requiring user data and bank policy checks

            User Query:
            {query}

            Predicted Category: {predicted_category} (Score: {intent_score})
            Semantic Similarity Scores: {similarity_scores}

            {history_context}

            Reasoning:
            """
        )
        classification_chain = classification_prompt | self.llm
        similarity_scores_str = "\n".join([f"{cat}: {score:.3f}" for cat, score in category_scores.items()])
        result = classification_chain.invoke({
            "query": query,
            "predicted_category": predicted_label,
            "intent_score": intent_score,
            "similarity_scores": similarity_scores_str,
            "history_context": history_context
        })

        response_text = result.content
        agent_type_match = re.search(r"Agent type:\s*(policy|user_data|calculation|small_talk|combined)", response_text, re.IGNORECASE)
        if agent_type_match:
            agent_type = agent_type_match.group(1).lower()

        reasoning = f"{keyword_context}\n\nLLM Reasoning:\n{response_text}"
        logger.info(f"Query classified as: {agent_type}")
        return {
            "agent_type": agent_type,
            "reasoning": reasoning
        }

    def orchestrate(self, state: CombinedState) -> CombinedState:
        logger.info(f"Orchestrating query for user {state['user_id']}: {state['query']}")
        classification = self.classify_query(state['query'], state['user_id'], state['chat_history'])
        state['orchestrator_reasoning'] = classification['reasoning']
        agent_type = classification['agent_type']
        
        logger.info(f"Agent type determined: {agent_type}")

        if agent_type == 'small_talk':
            state['final_response'] = self.handle_small_talk(state['query'], state['user_id'], state['chat_history'])
            state['policy_state']['response'] = "No bank policy information retrieved."
            state['user_data_state']['response'] = "No user data retrieved."
            state['calculation_state']['response'] = "No calculation performed."
        elif agent_type == 'policy':
            policy_result = self.policy_workflow.invoke(state['policy_state'])
            state['policy_state'] = policy_result
            state['user_data_state']['response'] = "No user-specific data required."
            state['calculation_state']['response'] = "No calculation required."
            state['final_response'] = policy_result['response']
        elif agent_type == 'user_data':
            user_data_result = self.user_data_workflow.invoke(state['user_data_state'])
            state['user_data_state'] = user_data_result
            state['policy_state']['response'] = "No bank policy information required."
            state['calculation_state']['response'] = "No calculation required."
            state['final_response'] = user_data_result['response']
        elif agent_type == 'calculation':
            calculation_result = self.calculation_workflow.invoke(state['calculation_state'])
            state['calculation_state'] = calculation_result
            state['policy_state']['response'] = "No bank policy information required."
            state['user_data_state']['response'] = calculation_result['user_data_response']
            state['final_response'] = calculation_result['response']
        elif agent_type == 'combined':
            logger.info("Initiating combined workflows")
            user_data_result = self.user_data_workflow.invoke(state['user_data_state'])
            policy_result = self.policy_workflow.invoke(state['policy_state'])
            calculation_result = self.calculation_workflow.invoke(state['calculation_state']) if any(k in state['query'].lower() for k in ['emi', 'penalty', 'calculate', 'foreclose']) else state['calculation_state']
            logger.info(f"Bank policy response: {policy_result.get('response', 'No response')[:100]}...")
            logger.info(f"User data response: {user_data_result.get('response', 'No response')[:100]}...")
            logger.info(f"Calculation response: {calculation_result.get('response', 'No response')[:100]}...")
            state['policy_state'] = policy_result
            state['user_data_state'] = user_data_result
            state['calculation_state'] = calculation_result

            # Evaluate eligibility
            eligibility_result = self.evaluate_eligibility(
                user_data_result['user_data'],
                policy_result.get('eligibility_criteria', {})
            )

            combined_prompt = ChatPromptTemplate.from_template(
                """You are a bank assistant. Provide a concise, professional response to determine the user's eligibility for the query based on their data and bank policies. Use plain text only. Provide a clear verdict (e.g., "You are eligible" or "You are not eligible") followed by a table of relevant user data and bank policy criteria. Refer to the bank instead of the company.
                **Give the output in short and concise pointers**

                User Query:
                {query}

                User-Specific Information:
                {user_response}

                Bank Policy Information:
                {policy_response}

                Eligibility Evaluation:
                Eligible: {eligible}
                Details:
                {eligibility_details}

                Response:
                """
            )
            response_chain = combined_prompt | self.llm
            logger.info("Generating combined response")
            combined_result = response_chain.invoke({
                "query": state['query'],
                "user_response": user_data_result['response'],
                "policy_response": policy_result['response'],
                "eligible": eligibility_result['eligible'],
                "eligibility_details": "\n".join(eligibility_result['details'])
            })
            state['final_response'] = combined_result.content
            logger.info(f"Combined response generated: {state['final_response'][:100]}...")

        logger.info(f"Orchestration completed for agent_type: {agent_type}")
        return state