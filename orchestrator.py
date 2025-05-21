import logging
from typing import Dict, List, Optional
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from states import PolicyState, UserDataState, CombinedState
from policy_agent import PolicyAgent
from user_data_agent import UserDataAgent

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, llm):
        self.llm = llm
        self.policy_workflow = PolicyAgent.create_workflow()
        self.user_data_workflow = UserDataAgent.create_workflow()

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
        small_talk_keywords = [
            'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'thanks',
            'thank you', 'how are you', 'nice to meet', 'bye', 'chat', 'talk', 'greetings'
        ]
        both_keywords = [
            'affect my', 'impact my', 'how does my', 'my loan and', 'my account and',
            'balance and policy', 'status and procedure', 'personal and company', ' and '
        ]

        # Normalize query for matching
        query_lower = query.lower().strip()

        # Calculate scores for each category based on keyword matches
        policy_score = sum(1 for keyword in policy_keywords if keyword in query_lower)
        user_data_score = sum(1 for keyword in user_data_keywords if keyword in query_lower)
        small_talk_score = sum(1 for keyword in small_talk_keywords if keyword in query_lower)
        both_score = sum(1 for keyword in both_keywords if keyword in query_lower)

        # Add a boost for exact phrase matches
        for keyword in both_keywords:
            if keyword in query_lower:
                both_score += 2
        for keyword in user_data_keywords:
            if keyword in query_lower and len(keyword.split()) > 1:
                user_data_score += 1
        for keyword in policy_keywords:
            if keyword in query_lower and len(keyword.split()) > 1:
                policy_score += 1

        # Boost 'both' score if both user_data and policy keywords are matched
        if policy_score > 0 and user_data_score > 0:
            both_score += 3
            logger.info("Detected both user data and policy keywords, boosting 'both' score")

        # Prepare keyword matching context for LLM
        keyword_context = (
            f"Keyword Matching Results:\n"
            f"- Policy score: {policy_score} (Keywords matched: {[k for k in policy_keywords if k in query_lower]})\n"
            f"- User data score: {user_data_score} (Keywords matched: {[k for k in user_data_keywords if k in query_lower]})\n"
            f"- Small talk score: {small_talk_score} (Keywords matched: {[k for k in small_talk_keywords if k in query_lower]})\n"
            f"- Both score: {both_score} (Keywords matched: {[k for k in both_keywords if k in query_lower]})"
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
            2. End with EXACTLY "Agent type: X" where X is one of: "policy", "user_data", "small_talk", or "both"
            
            CATEGORY DEFINITIONS:
            - policy: Questions about company rules, processes, or general procedures
            - user_data: Questions about personal account information or user-specific details
            - small_talk: General conversation, greetings, thanks
            - both: Questions requiring BOTH policy information AND user-specific data

            EXAMPLES:
            - "What's the vacation policy?" → Agent type: policy
            - "What's my outstanding balance?" → Agent type: user_data
            - "Hello there" → Agent type: small_talk
            - "How does my loan status affect my benefits?" → Agent type: both
            - "What is my outstanding amount and how can I foreclose early?" → Agent type: both

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
        agent_type_match = re.search(r"Agent type:\s*(policy|user_data|small_talk|both)", response_text, re.IGNORECASE)
        
        if agent_type_match:
            agent_type_text = agent_type_match.group(1).lower()
            if agent_type_text in ["policy", "user_data", "both", "small_talk"]:
                agent_type = agent_type_text
        else:
            logger.warning("Failed to extract agent type using regex, falling back to simple text matching")
            if "agent type: policy" in response_text.lower():
                agent_type = 'policy'
            elif "agent type: user_data" in response_text.lower():
                agent_type = 'user_data'
            elif "agent type: both" in response_text.lower():
                agent_type = 'both'
            elif "agent type: small_talk" in response_text.lower():
                agent_type = 'small_talk'

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
            Keep the response concise and relevant, encouraging the user to ask about company policies or their account if needed.

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
        elif agent_type == 'policy':
            policy_result = self.policy_workflow.invoke(state['policy_state'])
            state['policy_state'] = policy_result
            state['user_data_state']['response'] = "No user-specific data required."
            state['final_response'] = policy_result['response']
        elif agent_type == 'user_data':
            user_data_result = self.user_data_workflow.invoke(state['user_data_state'])
            state['user_data_state'] = user_data_result
            state['policy_state']['response'] = "No policy information required."
            state['final_response'] = user_data_result['response']
        elif agent_type == 'both':
            logger.info("Initiating both policy and user data workflows")
            policy_result = self.policy_workflow.invoke(state['policy_state'])
            user_data_result = self.user_data_workflow.invoke(state['user_data_state'])
            logger.info(f"Policy workflow response: {policy_result.get('response', 'No response')[:100]}...")
            logger.info(f"User data workflow response: {user_data_result.get('response', 'No response')[:100]}...")
            state['policy_state'] = policy_result
            state['user_data_state'] = user_data_result
            combined_prompt = ChatPromptTemplate.from_template(
                """You are a friendly company assistant chatbot. Create a single, coherent response that integrates both policy information
                and user-specific details.

                *Instructions:*
                - Start by addressing the user's specific data question directly
                - Then provide the policy information clearly
                - Format your answer as clear sections with bullet points
                - Be direct and easy to read
                - Ensure both sets of information are fully represented in your response
                - Make connections between the user's specific situation and the policy where relevant

                *User Query*:
                {query}

                *User-Specific Information*:
                {user_response}

                *Policy Information*:
                {policy_response}

                *Final Answer (provide BOTH user data and policy information):*
                """
            )
            response_chain = combined_prompt | self.llm
            logger.info("Generating combined response with both user data and policy information")
            combined_result = response_chain.invoke({
                "query": state['query'],
                "user_response": user_data_result['response'],
                "policy_response": policy_result['response']
            })
            state['final_response'] = combined_result.content
            logger.info(f"Combined response generated: {state['final_response'][:100]}...")

        logger.info(f"Orchestration completed for agent_type: {agent_type}")
        return state