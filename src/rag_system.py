import os
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

class FarmAIRAG:
    """RAG system for agricultural knowledge"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
    def build_knowledge_base(self, documents: List[Dict[str, Any]]):
        """Build the vector knowledge base from documents"""
        try:
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=self.api_key
            )
            
            langchain_docs = []
            for doc in documents:
                chunks = self.text_splitter.split_text(doc['content'])
                for chunk in chunks:
                    langchain_docs.append(Document(
                        page_content=chunk,
                        metadata={
                            'title': doc.get('title', 'Agricultural Guide'),
                            'category': doc.get('category', 'General')
                        }
                    ))
            
            self.vectorstore = FAISS.from_documents(langchain_docs, self.embeddings)
            
            llm = OpenAI(
                temperature=0.1,
                model="gpt-3.5-turbo",
                max_tokens=512,
                openai_api_key=self.api_key
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(),
                return_source_documents=True
            )
            return True
        except Exception as e:
            print(f"Error building knowledge base: {e}")
            return False
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the knowledge base"""
        try:
            if not self.qa_chain:
                raise ValueError("Knowledge base not initialized")
            
            result = self.qa_chain.invoke(question)
            
            sources = []
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    sources.append({
                        "content": doc.page_content,
                        "title": doc.metadata.get("title", "Agricultural Guide"),
                        "category": doc.metadata.get("category", "General"),
                    })
            
            return {
                "response": result.get("result", "No answer found."),
                "sources": sources
            }
        except Exception as e:
            return {
                "response": f"I apologize, but I encountered an error: {e}",
                "sources": []
            }
