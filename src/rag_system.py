import os
from typing import List, Dict, Any
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import numpy as np

class FarmAIRAG:
    """Advanced RAG system for agricultural knowledge"""
    
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    def build_knowledge_base(self, documents: List[Dict[str, Any]]):
        """Build the vector knowledge base from documents"""
        try:
            # Initialize embeddings
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002"
            )
            
            # Convert documents to LangChain Document objects
            langchain_docs = []
            for doc in documents:
                # Split each document into chunks
                chunks = self.text_splitter.split_text(doc['content'])
                for chunk in chunks:
                    langchain_docs.append(Document(
                        page_content=chunk,
                        metadata={
                            'title': doc.get('title', 'Agricultural Guide'),
                            'source': doc.get('source', 'Agricultural Knowledge Base'),
                            'category': doc.get('category', 'general'),
                            'language': doc.get('language', 'en')
                        }
                    ))
            
            # Create vector store
            self.vectorstore = FAISS.from_documents(
                langchain_docs, 
                self.embeddings
            )
            
            # Initialize QA chain
            llm = OpenAI(
                temperature=0.1,
                model_name="gpt-3.5-turbo-instruct",
                max_tokens=512
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={
                        "score_threshold": 0.3,
                        "k": 5
                    }
                ),
                return_source_documents=True
            )
            
            return True
            
        except Exception as e:
            print(f"Error building knowledge base: {str(e)}")
            return False
    
    def query(self, question: str, context: str = None) -> Dict[str, Any]:
        """Query the knowledge base"""
        try:
            if not self.qa_chain:
                raise ValueError("Knowledge base not initialized")
            
            # Enhance question with context if provided
            if context:
                enhanced_question = f"Context: {context}\n\nQuestion: {question}"
            else:
                enhanced_question = question
            
            # Get response from QA chain
            result = self.qa_chain({"query": enhanced_question})
            
            # Process source documents
            sources = []
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    sources.append({
                        "content": doc.page_content,
                        "title": doc.metadata.get("title", "Agricultural Guide"),
                        "source": doc.metadata.get("source", "Knowledge Base"),
                        "category": doc.metadata.get("category", "general"),
                        "score": 0.8  # Placeholder score
                    })
            
            return {
                "response": result["result"],
                "sources": sources,
                "query": question
            }
            
        except Exception as e:
            return {
                "response": f"I apologize, but I encountered an error processing your question: {str(e)}",
                "sources": [],
                "query": question
            }
    
    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform hybrid search (semantic + keyword)"""
        try:
            if not self.vectorstore:
                return []
            
            # Semantic search
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=top_k)
            
            results = []
            for doc, score in docs_with_scores:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(1 - score),  # Convert distance to similarity
                    "search_type": "semantic"
                })
            
            return results
            
        except Exception as e:
            print(f"Error in hybrid search: {str(e)}")
            return []
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        try:
            if not self.vectorstore:
                return {"status": "not_initialized"}
            
            # Get index statistics
            index_size = self.vectorstore.index.ntotal if hasattr(self.vectorstore, 'index') else 0
            
            return {
                "status": "initialized",
                "total_documents": index_size,
                "embedding_model": "text-embedding-ada-002",
                "vector_dimensions": 1536
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}