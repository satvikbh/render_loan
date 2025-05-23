import logging
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from states import PolicyState

logger = logging.getLogger(__name__)

class PolicyAgent:
    @staticmethod
    def create_workflow():
        workflow = StateGraph(PolicyState)
        workflow.add_node("retrieve", PolicyAgent.retrieve_documents)
        workflow.add_node("reason", PolicyAgent.analyze_and_reason)
        workflow.add_node("generate", PolicyAgent.generate_policy_response)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "reason")
        workflow.add_edge("reason", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile()

    @staticmethod
    def retrieve_documents(state: PolicyState) -> PolicyState:
        logger.info(f"Retrieving documents for query: {state['query']}")
        vectorstore = state.get('vectorstore')
        state["documents"] = vectorstore.similarity_search(state["query"], k=4)
        logger.info(f"Retrieved {len(state['documents'])} documents.")
        return state

   
    @staticmethod
    def analyze_and_reason(state: PolicyState) -> PolicyState:
        logger.info("Analyzing documents and reasoning...")
        context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(state["documents"])])
        history_context = ""
        if state["chat_history"]:
            history_context = "\n\n*Relevant Previous Conversations*:\n"
            for entry in state["chat_history"]:
                history_context += f"User Query: {entry['query']}\nResponse: {entry['response']}\n---\n"
        
        reasoning_prompt = ChatPromptTemplate.from_template(
            """You are a policy analysis system working with a Bank Handbook.
            Analyze the retrieved documents and reason step-by-step about how they relate to the user's query.
            Use relevant previous conversations to ensure continuity.
            Focus on identifying relevant information, bank policy constraints, and accurate answers.

            *Retrieved Documents*:
            {context}

            *User Query*:
            {query}

            {history_context}

            *Step-by-Step Reasoning*:
            Let me think through this carefully:
            1. Understand the bank policy information requested.
            2. Identify relevant documents.
            3. Analyze bank policy rules, exceptions, or processes.
            4. Consider previous conversations for consistency.
            5. Determine conditions or requirements.

            Begin your reasoning now:
            """
        )
        reasoning_chain = reasoning_prompt | state.get('llm')
        reasoning_result = reasoning_chain.invoke({
            "context": context,
            "query": state["query"],
            "history_context": history_context
        })
        state["reasoning"] = reasoning_result.content
        logger.info("Reasoning completed.")
        return state

    @staticmethod
    def generate_policy_response(state: PolicyState) -> PolicyState:
        logger.info("Generating policy response...")
        context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(state["documents"])])
        history_context = ""
        if state["chat_history"]:
            history_context = "\n\n*Relevant Previous Conversations*:\n"
            for entry in state["chat_history"]:
                history_context += f"User Query: {entry['query']}\nResponse: {entry['response']}\n---\n"
        
        response_prompt = ChatPromptTemplate.from_template(
            """You are a policy retrieval assistant for a bank. Based on the bank handbook excerpts,
            reasoning analysis, and previous conversations, provide a concise and accurate response to the
            policy-related part of the query.

            *Instructions:*
            - Answer directly about the bank policy information requested (e.g., early loan foreclosure)
            - Provide specific steps or requirements if available
            - Format your answer as clear bullet points
            - Include any fees, timelines, or important considerations
            - Do not be vague - give specific bank policy details whenever possible

            *Bank Handbook Excerpts*:
            {context}

            *User Query*:
            {query}

            *Your Reasoning Analysis*:
            {reasoning}

            {history_context}

            *Bank Policy Response (in bullet points):*
            """
        )
        response_chain = response_prompt | state.get('llm')
        response_result = response_chain.invoke({
            "context": context,
            "query": state["query"],
            "reasoning": state["reasoning"],
            "history_context": history_context
        })
        state["response"] = response_result.content
        logger.info("Policy response generated.")
        return state