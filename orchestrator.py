import logging
from typing import Dict, List, Optional
import re
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from states import PolicyState, UserDataState, CalculationState, CombinedState
from policy_agent import PolicyAgent
from user_data_agent import UserDataAgent
from calculation_agent import CalculationAgent
from utils import load_chat_history

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, llm):
        self.llm = llm
        self.policy_workflow = PolicyAgent.create_workflow()
        self.user_data_workflow = UserDataAgent.create_workflow()
        self.calculation_workflow = CalculationAgent.create_workflow()
        try:
            with open('user_details.json', 'r') as f:
                self.user_data_db = json.load(f).get("users", {})
        except Exception as e:
            logger.error(f"Error loading user_details.json: {str(e)}")
            self.user_data_db = {}

    def classify_query(self, query: str, user_id: str, chat_history: List[Dict]) -> Dict:
        logger.info(f"Classifying query: {query}")

        policy_keywords = [
            'policy', 'rule', 'procedure', 'guideline', 'regulation', 'process', 'how does',
            'vacation', 'leave', 'benefits', 'insurance', 'foreclosure', 'foreclose', 'foreclosing',
            'terms', 'conditions', 'eligibility', 'requirements', 'bank', 'handbook', 'protocol'
        ]
        user_data_keywords = [
            'my account', 'balance', 'loan status', 'personal', 'details', 'account number',
            'name', 'address', 'payment', 'outstanding', 'due', 'history', 'profile', 'data',
            'statement', 'transaction'
        ]
        calculation_keywords = [
            'emi', 'monthly payment', 'installment', 'penalty', 'calculate', 'calculation',
            'interest calculation', 'loan repayment', 'amount due', 'payoff', 'tenure'
        ]
        small_talk_keywords = [
            'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'thanks',
            'thank you', 'how are you', 'nice to meet', 'bye', 'chat', 'talk', 'greetings'
        ]
        combined_keywords = [
            'affect my', 'impact my', 'how does my', 'my loan and', 'my account and',
            'balance and policy', 'status and procedure', 'personal and bank', ' and ',
            'emi and policy', 'penalty and procedure', 'calculate and policy', 'eligible to foreclose'
        ]

        query_lower = query.lower().strip()

        policy_score = sum(1 for keyword in policy_keywords if keyword in query_lower)
        user_data_score = sum(1 for keyword in user_data_keywords if keyword in query_lower)
        calculation_score = sum(1 for keyword in calculation_keywords if keyword in query_lower)
        small_talk_score = sum(1 for keyword in small_talk_keywords if keyword in query_lower)
        combined_score = sum(1 for keyword in combined_keywords if keyword in query_lower)

        for keyword in combined_keywords:
            if keyword in query_lower:
                combined_score += 2
        for keyword in user_data_keywords:
            if keyword in query_lower and len(keyword.split()) > 1:
                user_data_score += 1
        for keyword in policy_keywords:
            if keyword in query_lower and len(keyword.split()) > 1:
                policy_score += 1
        for keyword in calculation_keywords:
            if keyword in query_lower and len(keyword.split()) > 1:
                calculation_score += 1

        if sum([policy_score > 0, user_data_score > 0, calculation_score > 0]) > 1 or 'eligible to foreclose' in query_lower:
            combined_score += 3
            logger.info("Detected multiple categories or eligibility query, boosting 'combined' score")

        keyword_context = (
            f"Keyword Matching Results:\n"
            f"- Policy score: {policy_score} (Keywords matched: {[k for k in policy_keywords if k in query_lower]})\n"
            f"- User data score: {user_data_score} (Keywords matched: {[k for k in user_data_keywords if k in query_lower]})\n"
            f"- Calculation score: {calculation_score} (Keywords matched: {[k for k in calculation_keywords if k in query_lower]})\n"
            f"- Small talk score: {small_talk_score} (Keywords matched: {[k for k in small_talk_keywords if k in query_lower]})\n"
            f"- Combined score: {combined_score} (Keywords matched: {[k for k in combined_keywords if k in query_lower]})"
        )
        logger.info(keyword_context)

        history_context = ""
        if chat_history:
            history_context = "\nPrevious Conversations:\n"
            for entry in chat_history:
                history_context += f"Query: {entry['query']}\nResponse: {entry['response']}\n---\n"

        classification_prompt = ChatPromptTemplate.from_template(
            """You are a query classification system for a bank assistant. Categorize the query into one category.

            EXACTLY FOLLOW THIS FORMAT:
            1. Reasoning
            2. Agent type: X (where X is policy, user_data, calculation, small_talk, or combined)

            CATEGORY DEFINITIONS:
            - policy: Questions about bank rules or procedures
            - user_data: Questions about personal account details
            - calculation: Questions requiring financial calculations
            - small_talk: General conversation or greetings
            - combined: Questions involving multiple types or eligibility (e.g., foreclosure eligibility)

            EXAMPLES:
            - "What's the foreclosure policy?" -> Agent type: policy
            - "What's my loan balance?" -> Agent type: user_data
            - "Calculate my EMI" -> Agent type: calculation
            - "Hello" -> Agent type: small_talk
            - "Am I eligible to foreclose?" -> Agent type: combined

            User Query:
            {query}

            {history_context}

            Keyword Matching Context:
            {keyword_context}

            Reasoning:
            """
        )
        classification_chain = classification_prompt | self.llm
        result = classification_chain.invoke({
            "query": query,
            "history_context": history_context,
            "keyword_context": keyword_context
        })

        response_text = result.content
        agent_type = 'small_talk'

        agent_type_match = re.search(r"Agent type:\s*(policy|user_data|calculation|small_talk|combined)", response_text, re.IGNORECASE)
        if agent_type_match:
            agent_type_text = agent_type_match.group(1).lower()
            if agent_type_text in ["policy", "user_data", "calculation", "small_talk", "combined"]:
                agent_type = agent_type_text
        else:
            logger.warning("Failed to extract agent type, using fallback")
            if "agent type: policy" in response_text.lower():
                agent_type = 'policy'
            elif "agent type: user_data" in response_text.lower():
                agent_type = 'user_data'
            elif "agent type: calculation" in response_text.lower():
                agent_type = 'calculation'
            elif "agent type: small_talk" in response_text.lower():
                agent_type = 'small_talk'
            elif "agent type: combined" in response_text.lower():
                agent_type = 'combined'

        reasoning = f"{keyword_context}\n\nLLM Reasoning:\n{response_text}"
        logger.info(f"Query classified as: {agent_type}")
        return {
            "agent_type": agent_type,
            "reasoning": reasoning
        }

    def handle_small_talk(self, query: str, user_id: str) -> str:
        logger.info(f"Handling small talk: {query}")
        user_name = self.user_data_db.get(user_id, {}).get("name", None)
        if not user_name:
            chat_history = load_chat_history().get(user_id, [])
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
        
        greeting = f"Hello {user_name}" if user_name else "Hello"
        small_talk_prompt = ChatPromptTemplate.from_template(
            """You are a bank assistant. Respond to the small talk query in a concise, professional tone using plain text. Encourage the user to ask about bank policies, accounts, or calculations.
            **Give the output in short and concise pointers**

            User Query:
            {query}

            Greeting:
            {greeting}

            Response:
            {greeting}. How may I assist you with your banking needs today?
            """
        )
        response_chain = small_talk_prompt | self.llm
        result = response_chain.invoke({"query": query, "greeting": greeting})
        return result.content

    def orchestrate(self, state: CombinedState) -> CombinedState:
        logger.info(f"Orchestrating query for user {state['user_id']}: {state['query']}")
        classification = self.classify_query(state['query'], state['user_id'], state['chat_history'])
        state['orchestrator_reasoning'] = classification['reasoning']
        agent_type = classification['agent_type']
        
        logger.info(f"Agent type determined: {agent_type}")

        if agent_type == 'small_talk':
            state['final_response'] = self.handle_small_talk(state['query'], state['user_id'])
            state['policy_state']['response'] = "No policy information retrieved."
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
            state['policy_state']['response'] = "No policy information required."
            state['calculation_state']['response'] = "No calculation required."
            state['final_response'] = user_data_result['response']
        elif agent_type == 'calculation':
            calculation_result = self.calculation_workflow.invoke(state['calculation_state'])
            state['calculation_state'] = calculation_result
            state['policy_state']['response'] = "No policy information required."
            state['user_data_state']['response'] = calculation_result['user_data_response']
            state['final_response'] = calculation_result['response']
        elif agent_type == 'combined':
            logger.info("Initiating combined workflows")
            policy_result = self.policy_workflow.invoke(state['policy_state']) if any(k in state['query'].lower() for k in ['policy', 'foreclosure', 'procedure', 'eligible']) else state['policy_state']
            user_data_result = self.user_data_workflow.invoke(state['user_data_state']) if any(k in state['query'].lower() for k in ['balance', 'account', 'details', 'eligible']) else state['user_data_state']
            calculation_result = self.calculation_workflow.invoke(state['calculation_state']) if any(k in state['query'].lower() for k in ['emi', 'penalty', 'calculate', 'foreclose']) else state['calculation_state']
            logger.info(f"Policy response: {policy_result.get('response', 'No response')[:100]}...")
            logger.info(f"User data response: {user_data_result.get('response', 'No response')[:100]}...")
            logger.info(f"Calculation response: {calculation_result.get('response', 'No response')[:100]}...")
            state['policy_state'] = policy_result
            state['user_data_state'] = user_data_result
            state['calculation_state'] = calculation_result
            combined_prompt = ChatPromptTemplate.from_template(
                """You are a bank assistant. Create a concise, professional response integrating policy, user data, and calculation results as needed. Use plain text, no bold or italics. Format numerical data in tablular format, not ASCII. For eligibility queries, combine user data (delinquency period, foreclosure status) with policy rules (e.g., delinquency < 180 days, foreclosure not Initiated/Pending).
                **Give the output in short and concise pointers**

                User Query:
                {query}

                User-Specific Information:
                {user_response}

                Calculation Results:
                {calculation_response}

                Bank Policy Information:
                {policy_response}
                
                Response:
                """
            )
            response_chain = combined_prompt | self.llm
            logger.info("Generating combined response")
            combined_result = response_chain.invoke({
                "query": state['query'],
                "user_response": user_data_result['response'],
                "calculation_response": calculation_result['response'],
                "policy_response": policy_result['response']
            })
            state['final_response'] = combined_result.content
            logger.info(f"Combined response generated: {state['final_response'][:100]}...")

        logger.info(f"Orchestration completed for agent_type: {agent_type}")
        return state