from typing import Dict, Any, List
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class FarmAIAgents:
    """Multi-agent system for agricultural question processing"""
    
    def __init__(self, rag_system, api_key: str):
        self.rag_system = rag_system
        self.llm = OpenAI(
            temperature=0.1, 
            model="gpt-3.5-turbo", 
            max_tokens=1024, 
            openai_api_key=api_key
        )
        self.query_classifier = self._create_classifier()

    def _create_classifier(self):
        prompt = PromptTemplate(
            input_variables=["query"],
            template="Classify this query into one of the following categories: disease, pest, management, general. Query: {query}\nClassification:"
        )
        return LLMChain(llm=self.llm, prompt=prompt)

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process query through RAG system"""
        # A simplified workflow for robustness. The advanced agent logic can be added back later.
        try:
            return self.rag_system.query(query)
        except Exception as e:
            return {
                "response": f"I encountered an error processing your question: {e}",
                "sources": []
            }
