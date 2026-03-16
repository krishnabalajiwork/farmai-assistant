import streamlit as st
import os
from typing import List, Dict, Any
import nest_asyncio

# Apply the patch for the event loop issue
nest_asyncio.apply()

# MODULAR IMPORTS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# FIXED: Absolute direct path for v0.3 compatibility
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# ==============================================================================
# PART 1: DATA LOADER CODE
# ==============================================================================
def load_agricultural_data() -> List[Dict[str, Any]]:
    """Knowledge base containing detailed agricultural data."""
    return [
        {
            "title": "Tomato Disease and Pest Management",
            "content": "Tomato Blight (Early and Late): Early blight shows brown spots; late blight causes dark water-soaked lesions. Control via hygiene and aphid management.",
            "source": "Extension Publications",
            "category": "disease_management",
            "crop": "tomato"
        },
        {
            "title": "Rice Crop Pest and Disease Management",
            "content": "Rice Management: Stem Borer: Use pheromone traps and resistant varieties. Blast Disease: Manage through nitrogen timing and water control.",
            "source": "IRRI",
            "category": "crop_management",
            "crop": "rice"
        }
    ]

# ==============================================================================
# PART 2: RAG SYSTEM
# ==============================================================================
class FarmAISystem:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rag_chain = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    def build_knowledge_base(self, documents: List[Dict[str, Any]]):
        try:
            # Using the stable embedding model
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", 
                google_api_key=self.api_key
            )
            
            langchain_docs = [
                Document(page_content=doc['content'], metadata={'title': doc.get('title', '')}) 
                for doc in documents
            ]
            
            final_docs = self.text_splitter.split_documents(langchain_docs)
            vectorstore = FAISS.from_documents(final_docs, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=self.api_key,
                temperature=0
            )

            system_prompt = (
                "You are an expert agricultural assistant. Use the following context "
                "to answer the question.\n\nContext: {context}"
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            # Building the chain with the new imports
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            self.rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            return True
        except Exception as e:
            st.error(f"Error building knowledge base: {e}")
            return False

    def query(self, question: str):
        if not self.rag_chain:
            return "The system is not ready."
        try:
            response = self.rag_chain.invoke({"input": question})
            return response.get('answer', "I couldn't find an answer.")
        except Exception as e:
            return f"An error occurred: {e}"

# ==============================================================================
# PART 3: MAIN STREAMLIT APP
# ==============================================================================
st.set_page_config(page_title="FarmAI Knowledge Assistant", page_icon="🌾", layout="wide")
st.title("🌾 FarmAI Knowledge Assistant")

@st.cache_resource
def initialize_system():
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        return None, False
    
    documents = load_agricultural_data()
    farm_ai_system = FarmAISystem(api_key=api_key)
    success = farm_ai_system.build_knowledge_base(documents)
    return farm_ai_system, success

farm_ai, success = initialize_system()

if success:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "System initialized. How can I help with your farming queries?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about agriculture..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                response = farm_ai.query(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.error("Please verify that your 'GOOGLE_API_KEY' is active in Streamlit Secrets.")
