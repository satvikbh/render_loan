import logging
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from states import PolicyState
import json
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
        state["documents"] = vectorstore.similarity_search(state['query'], k=4)
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
            """You are a bank policy analysis system. Analyze the retrieved documents and reason step-by-step about how they relate to the user's query. For eligibility queries (e.g., foreclosure), extract specific criteria (e.g., delinquency period < 180 days, foreclosure status not Initiated/Pending) in a structured JSON format. Use previous conversations for continuity.
            **Return reasoning in short pointers and criteria in JSON**

            Retrieved Documents:
            {context}

            User Query:
            {query}

            {history_context}

            Reasoning:
            1. Identify the policy information requested.
            2. Match relevant documents to the query.
            3. Extract eligibility criteria in JSON format (e.g., {{"delinquency_period": "< 180 days", "foreclosure_status": "Not Initiated or Pending"}}).
            4. Analyze policy rules, exceptions, or processes.
            5. Ensure consistency with previous conversations.

            Eligibility Criteria (JSON):
            """
        )
        reasoning_chain = reasoning_prompt | state.get('llm')
        reasoning_result = reasoning_chain.invoke({
            "context": context,
            "query": state["query"],
            "history_context": history_context
        })
        # Parse reasoning and JSON criteria
        reasoning_text = reasoning_result.content
        try:
            json_start = reasoning_text.rfind("Eligibility Criteria (JSON):") + len("Eligibility Criteria (JSON):\n")
            json_str = reasoning_text[json_start:].strip()
            eligibility_criteria = json.loads(json_str)
            state["eligibility_criteria"] = eligibility_criteria
            state["reasoning"] = reasoning_text[:json_start].strip()
        except:
            state["eligibility_criteria"] = {}
            state["reasoning"] = reasoning_text
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
            """You are a bank policy assistant. Provide a concise, professional response to the policy-related query based on the handbook excerpts, reasoning, and previous conversations. Use plain text only. For numerical data, use tabular format. For eligibility queries, use extracted criteria to inform the response.
            **Give the output in short and concise pointers**

            Bank Handbook Excerpts:
            {context}

            User Query:
            {query}

            Reasoning Analysis:
            {reasoning}

            Eligibility Criteria (JSON):
            {eligibility_criteria}

            {history_context}

            Response:
            """
        )
        response_chain = response_prompt | state.get('llm')
        response_result = response_chain.invoke({
            "context": context,
            "query": state["query"],
            "reasoning": state["reasoning"],
            "eligibility_criteria": json.dumps(state.get("eligibility_criteria", {})),
            "history_context": history_context
        })
        state["response"] = response_result.content
        logger.info("Policy response generated.")
        return state