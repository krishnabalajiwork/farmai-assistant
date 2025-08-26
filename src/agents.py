from typing import Dict, Any, List
import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import re

class FarmAIAgents:
    """Multi-agent system for agricultural question processing"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.llm = OpenAI(temperature=0.1, model_name="gpt-3.5-turbo-instruct", max_tokens=1024, openai_api_base="https://api.chatanywhere.tech/v1")
        
        # Initialize agents
        self.query_classifier = QueryClassifierAgent(self.llm)
        self.diagnostic_agent = DiagnosticAgent(self.llm, self.rag_system)
        self.recommendation_agent = RecommendationAgent(self.llm, self.rag_system)
        self.verification_agent = VerificationAgent(self.llm)
        
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process query through multi-agent workflow"""
        try:
            # Step 1: Classify the query
            query_type = self.query_classifier.classify(query)
            
            # Step 2: Route to appropriate specialist agent
            if query_type in ["disease", "pest", "problem"]:
                # Use diagnostic agent for problem identification
                diagnosis_result = self.diagnostic_agent.diagnose(query)
                
                # Get recommendations based on diagnosis
                recommendation_result = self.recommendation_agent.recommend(
                    query, diagnosis_result.get("diagnosis", "")
                )
                
                # Combine results
                response = self._combine_diagnosis_and_recommendation(
                    diagnosis_result, recommendation_result
                )
                
            elif query_type in ["management", "general", "best_practice"]:
                # Direct recommendation for general queries
                recommendation_result = self.recommendation_agent.recommend(query)
                response = recommendation_result
                
            else:
                # Fallback to basic RAG
                response = self.rag_system.query(query)
            
            # Step 3: Verify and enhance response
            verified_response = self.verification_agent.verify(query, response)
            
            return verified_response
            
        except Exception as e:
            return {
                "response": f"I apologize, but I encountered an error processing your agricultural question: {str(e)}",
                "sources": [],
                "query": query,
                "agent_workflow": ["error"]
            }
    
    def _combine_diagnosis_and_recommendation(self, diagnosis: Dict, recommendation: Dict) -> Dict:
        """Combine diagnosis and recommendation results"""
        combined_response = f"""**Diagnosis:**
{diagnosis.get('response', 'Unable to provide diagnosis')}

**Recommendations:**
{recommendation.get('response', 'Unable to provide recommendations')}"""
        
        # Combine sources from both agents
        all_sources = diagnosis.get('sources', []) + recommendation.get('sources', [])
        # Remove duplicates based on content
        unique_sources = []
        seen_content = set()
        for source in all_sources:
            content_key = source.get('content', '')[:100]
            if content_key not in seen_content:
                unique_sources.append(source)
                seen_content.add(content_key)
        
        return {
            "response": combined_response,
            "sources": unique_sources,
            "query": diagnosis.get('query', ''),
            "agent_workflow": ["classifier", "diagnostic", "recommendation", "verification"]
        }

class QueryClassifierAgent:
    """Agent to classify agricultural queries by type"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            Classify the following agricultural query into one of these categories:
            - disease: Questions about plant diseases, symptoms, identification
            - pest: Questions about pest problems, insect damage
            - management: Questions about crop management, planting, harvesting
            - general: General agricultural questions, best practices
            - problem: General plant problems, environmental issues
            
            Query: {query}
            
            Classification (return only the category):
            """
        )
        
    def classify(self, query: str) -> str:
        """Classify the query type"""
        try:
            chain = LLMChain(llm=self.llm, prompt=self.prompt)
            result = chain.run(query=query)
            classification = result.strip().lower()
            
            # Map to valid categories
            valid_categories = ["disease", "pest", "management", "general", "problem"]
            if classification in valid_categories:
                return classification
            else:
                return "general"  # Default fallback
                
        except Exception as e:
            print(f"Error in query classification: {str(e)}")
            return "general"

class DiagnosticAgent:
    """Agent specialized in diagnosing agricultural problems"""
    
    def __init__(self, llm, rag_system):
        self.llm = llm
        self.rag_system = rag_system
        self.prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""
            You are an expert agricultural diagnostician. Based on the farmer's description and relevant agricultural knowledge, provide a detailed diagnosis.
            
            Farmer's Question: {query}
            
            Relevant Agricultural Knowledge:
            {context}
            
            Please provide:
            1. Most likely diagnosis or problem identification
            2. Key symptoms that support this diagnosis
            3. Possible causes or contributing factors
            4. Confidence level in diagnosis (High/Medium/Low)
            
            Diagnosis:
            """
        )
        
    def diagnose(self, query: str) -> Dict[str, Any]:
        """Diagnose agricultural problem"""
        try:
            # Get relevant context from RAG system
            rag_result = self.rag_system.query(query)
            context = ""
            for source in rag_result.get("sources", []):
                context += f"\n{source.get('content', '')}"
            
            # Generate diagnosis
            chain = LLMChain(llm=self.llm, prompt=self.prompt)
            diagnosis = chain.run(query=query, context=context[:2000])
            
            return {
                "response": diagnosis,
                "sources": rag_result.get("sources", []),
                "diagnosis": diagnosis,
                "query": query
            }
            
        except Exception as e:
            return {
                "response": f"Unable to provide diagnosis: {str(e)}",
                "sources": [],
                "query": query
            }

class RecommendationAgent:
    """Agent specialized in providing agricultural recommendations"""
    
    def __init__(self, llm, rag_system):
        self.llm = llm
        self.rag_system = rag_system
        self.prompt = PromptTemplate(
            input_variables=["query", "context", "diagnosis"],
            template="""
            You are an expert agricultural advisor. Provide actionable recommendations based on the farmer's question and available knowledge.
            
            Farmer's Question: {query}
            {diagnosis}
            
            Relevant Agricultural Knowledge:
            {context}
            
            Please provide:
            1. Immediate action steps
            2. Treatment or management recommendations
            3. Prevention measures for the future
            4. Timeline for expected results
            5. When to seek additional help
            
            Focus on practical, safe, and proven methods. Always prioritize farmer and environmental safety.
            
            Recommendations:
            """
        )
        
    def recommend(self, query: str, diagnosis: str = "") -> Dict[str, Any]:
        """Provide agricultural recommendations"""
        try:
            # Get relevant context from RAG system
            rag_result = self.rag_system.query(query)
            context = ""
            for source in rag_result.get("sources", []):
                context += f"\n{source.get('content', '')}"
            
            # Format diagnosis for prompt
            diagnosis_text = f"Previous Diagnosis: {diagnosis}" if diagnosis else ""
            
            # Generate recommendations
            chain = LLMChain(llm=self.llm, prompt=self.prompt)
            recommendations = chain.run(
                query=query, 
                context=context[:2000], 
                diagnosis=diagnosis_text
            )
            
            return {
                "response": recommendations,
                "sources": rag_result.get("sources", []),
                "query": query
            }
            
        except Exception as e:
            return {
                "response": f"Unable to provide recommendations: {str(e)}",
                "sources": [],
                "query": query
            }

class VerificationAgent:
    """Agent to verify and enhance responses"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["query", "response"],
            template="""
            Review the following agricultural advice for accuracy, safety, and completeness.
            
            Original Question: {query}
            Generated Response: {response}
            
            Please:
            1. Verify the advice is safe and appropriate
            2. Add any important warnings or precautions
            3. Ensure the response is practical for farmers
            4. Add any missing critical information
            
            If the advice looks good, return it as-is. If improvements are needed, provide the enhanced version.
            
            Verified Response:
            """
        )
        
    def verify(self, query: str, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify and enhance the response"""
        try:
            original_response = response_data.get("response", "")
            
            # Skip verification for error responses
            if "error" in original_response.lower() or "apologize" in original_response.lower():
                return response_data
            
            # Verify with LLM
            chain = LLMChain(llm=self.llm, prompt=self.prompt)
            verified_response = chain.run(query=query, response=original_response)
            
            # Update response data
            response_data["response"] = verified_response
            response_data["verified"] = True
            
            return response_data
            
        except Exception as e:
            # Return original if verification fails
            response_data["verified"] = False
            response_data["verification_error"] = str(e)
            return response_data
