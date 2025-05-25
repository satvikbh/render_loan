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
        context = "\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(state["documents"])])
        history_context = ""
        if state["chat_history"]:
            history_context = "\nPrevious Conversations:\n"
            for entry in state["chat_history"]:
                history_context += f"Query: {entry['query']}\nResponse: {entry['response']}\n---\n"
        
        reasoning_prompt = ChatPromptTemplate.from_template(
            """You are a bank policy analysis system. Analyze the retrieved documents and reason step-by-step about how they relate to the user's query. Use previous conversations for continuity. For eligibility queries (e.g., foreclosure), consider user data like delinquency period and foreclosure status if provided. Focus on bank policy constraints and accurate answers.
            **Give the output in short and concise pointers**
            
            Retrieved Documents:
            {context}

            User Query:
            {query}

            {history_context}

            Reasoning:
            1. Identify the policy information requested.
            2. Match relevant documents to the query.
            3. Analyze policy rules, exceptions, or processes.
            4. For eligibility, check user data constraints (e.g., delinquency < 180 days).
            5. Ensure consistency with previous conversations.
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
        context = "\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(state["documents"])])
        history_context = ""
        if state["chat_history"]:
            history_context = "\nPrevious Conversations:\n"
            for entry in state["chat_history"]:
                history_context += f"Query: {entry['query']}\nResponse: {entry['response']}\n---\n"
        
        response_prompt = ChatPromptTemplate.from_template(
            """You are a bank policy assistant. Provide a concise, professional response to the policy-related query based on the handbook excerpts, reasoning, and previous conversations. Do not use bold, italics, or other markdown formatting. Use plain text only. For numerical data (e.g., fees, timelines), format as tablular format. For eligibility queries (e.g., foreclosure), use user data (delinquency period, foreclosure status) to determine eligibility, assuming policies require delinquency < 180 days and foreclosure status not 'Initiated' or 'Pending'.
            ```Give the output in short and concise pointers like : 
            1. Point 1
            2. Point 2
            3. Point 3
            4. Point 4
            5. Point 5
            6. Point 6```

            Bank Handbook Excerpts:
            {context}

            User Query:
            {query}

            Reasoning Analysis:
            {reasoning}

            {history_context}

            Response:
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