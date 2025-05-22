import logging
from typing import Dict, List, Optional
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from states import PolicyState, UserDataState, CalculationState, CombinedState
from policy_agent import PolicyAgent
from user_data_agent import UserDataAgent
from calculation_agent import CalculationAgent

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, llm):
        self.llm = llm
        self.policy_workflow = PolicyAgent.create_workflow()
        self.user_data_workflow = UserDataAgent.create_workflow()
        self.calculation_workflow = CalculationAgent.create_workflow()

    def classify_query(self, query: str, user_id: str, chat_history: List[Dict]) -> Dict:
        logger.info(f"Classifying query: {query}")

        # Step 1: Keyword Matching Logic
        # Define keyword lists for each category
        policy_keywords = [
            'policy', 'rule', 'procedure', 'guideline', 'regulation', 'process', 'how does',
            'vacation', 'leave', 'benefits', 'insurance', 'foreclosure', 'foreclose', 'foreclosing',
            'terms', 'conditions', 'eligibility', 'requirements', 'company', 'handbook', 'protocol'
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
            'balance and policy', 'status and procedure', 'personal and company', ' and ',
            'emi and policy', 'penalty and procedure', 'calculate and policy'
        ]

        # Normalize query for matching
        query_lower = query.lower().strip()

        # Calculate scores for each category based on keyword matches
        policy_score = sum(1 for keyword in policy_keywords if keyword in query_lower)
        user_data_score = sum(1 for keyword in user_data_keywords if keyword in query_lower)
        calculation_score = sum(1 for keyword in calculation_keywords if keyword in query_lower)
        small_talk_score = sum(1 for keyword in small_talk_keywords if keyword in query_lower)
        combined_score = sum(1 for keyword in combined_keywords if keyword in query_lower)

        # Add a boost for exact phrase matches
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

        # Boost 'combined' score if multiple categories are matched
        if sum([policy_score > 0, user_data_score > 0, calculation_score > 0]) > 1:
            combined_score += 3
            logger.info("Detected multiple category keywords, boosting 'combined' score")

        # Prepare keyword matching context for LLM
        keyword_context = (
            f"Keyword Matching Results:\n"
            f"- Policy score: {policy_score} (Keywords matched: {[k for k in policy_keywords if k in query_lower]})\n"
            f"- User data score: {user_data_score} (Keywords matched: {[k for k in user_data_keywords if k in query_lower]})\n"
            f"- Calculation score: {calculation_score} (Keywords matched: {[k for k in calculation_keywords if k in query_lower]})\n"
            f"- Small talk score: {small_talk_score} (Keywords matched: {[k for k in small_talk_keywords if k in query_lower]})\n"
            f"- Combined score: {combined_score} (Keywords matched: {[k for k in combined_keywords if k in query_lower]})"
        )
        logger.info(keyword_context)

        # Step 2: LLM Classification
        history_context = ""
        if chat_history:
            history_context = "\n\n*Relevant Previous Conversations*:\n"
            for entry in chat_history:
                history_context += f"User Query: {entry['query']}\nResponse: {entry['response']}\n---\n"

        # Use LLM with keyword results as context
        classification_prompt = ChatPromptTemplate.from_template(
            """You are a query classification system for a company assistant. Your task is to categorize user queries.

            EXACTLY FOLLOW THIS FORMAT FOR YOUR RESPONSE:
            1. Provide your reasoning
            2. End with EXACTLY "Agent type: X" where X is one of: "policy", "user_data", "calculation", "small_talk", or "combined"
            
            CATEGORY DEFINITIONS:
            - policy: Questions about company rules, processes, or general procedures
            - user_data: Questions about personal account information or user-specific details
            - calculation: Questions requiring financial calculations like EMI or penalties
            - small_talk: General conversation, greetings, thanks
            - combined: Questions requiring multiple types (e.g., policy and calculation, or user data and policy)

            EXAMPLES:
            - "What's the vacation policy?" → Agent type: policy
            - "What's my outstanding balance?" → Agent type: user_data
            - "Calculate my EMI" → Agent type: calculation
            - "Hello there" → Agent type: small_talk
            - "How does my loan status affect my benefits?" → Agent type: combined
            - "What is my EMI and foreclosure policy?" → Agent type: combined

            *User Query*:
            {query}

            {history_context}

            *Keyword Matching Context*:
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

        # Default to small_talk if we can't determine
        agent_type = 'small_talk'

        # Extract agent type from the response using regex
        agent_type_match = re.search(r"Agent type:\s*(policy|user_data|calculation|small_talk|combined)", response_text, re.IGNORECASE)
        
        if agent_type_match:
            agent_type_text = agent_type_match.group(1).lower()
            if agent_type_text in ["policy", "user_data", "calculation", "small_talk", "combined"]:
                agent_type = agent_type_text
        else:
            logger.warning("Failed to extract agent type using regex, falling back to simple text matching")
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
        small_talk_prompt = ChatPromptTemplate.from_template(
            """You are a friendly company assistant chatbot. Respond to the user's small talk query in a warm, conversational tone.
            Keep the response concise and relevant, encouraging the user to ask about company policies, their account, or calculations if needed.

            *User Query*:
            {query}

            *Response*:
            """
        )
        response_chain = small_talk_prompt | self.llm
        result = response_chain.invoke({"query": query})
        return result.content

    def orchestrate(self, state: CombinedState) -> CombinedState:
        logger.info(f"Orchestrating query for user {state['user_id']}: {state['query']}")
        classification = self.classify_query(state['query'], state['user_id'], state['chat_history'])
        state['orchestrator_reasoning'] = classification['reasoning']
        agent_type = classification['agent_type']
        
        logger.info(f"Agent type determined: {agent_type}")

        if agent_type == 'small_talk':
            state['final_response'] = self.handle_small_talk(state['query'], state['user_id'])
            state['policy_state']['response'] = "No policy information retrieved for small talk."
            state['user_data_state']['response'] = "No user data retrieved for small talk."
            state['calculation_state']['response'] = "No calculation performed for small talk."
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
            policy_result = self.policy_workflow.invoke(state['policy_state']) if any(k in state['query'].lower() for k in ['policy', 'foreclosure', 'procedure']) else state['policy_state']
            user_data_result = self.user_data_workflow.invoke(state['user_data_state']) if any(k in state['query'].lower() for k in ['balance', 'account', 'details']) else state['user_data_state']
            calculation_result = self.calculation_workflow.invoke(state['calculation_state']) if any(k in state['query'].lower() for k in ['emi', 'penalty', 'calculate']) else state['calculation_state']
            logger.info(f"Policy response: {policy_result.get('response', 'No response')[:100]}...")
            logger.info(f"User data response: {user_data_result.get('response', 'No response')[:100]}...")
            logger.info(f"Calculation response: {calculation_result.get('response', 'No response')[:100]}...")
            state['policy_state'] = policy_result
            state['user_data_state'] = user_data_result
            state['calculation_state'] = calculation_result
            combined_prompt = ChatPromptTemplate.from_template(
                """You are a friendly company assistant chatbot. Create a single, coherent response that integrates policy information,
                user-specific details, and calculation results as needed.

                *Instructions:*
                - Start by addressing the user's specific data or calculation question directly
                - Then provide the policy information clearly
                - Format your answer as clear sections with bullet points
                - Be direct and easy to read
                - Ensure all relevant information is fully represented
                - Make connections between the user's situation, calculations, and policy where relevant

                *User Query*:
                {query}

                *User-Specific Information*:
                {user_response}

                *Calculation Results*:
                {calculation_response}

                *Policy Information*:
                {policy_response}

                *Final Answer (integrate all relevant information):*
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