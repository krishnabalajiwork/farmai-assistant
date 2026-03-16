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
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# ==============================================================================
# PART 1: DATA LOADER
# ==============================================================================
def load_agricultural_data() -> List[Dict[str, Any]]:
    return [
        {
            "title": "Tomato Disease Management",
            "content": "Tomato Blight shows brown spots. Control via hygiene and aphid management.",
            "crop": "tomato"
        },
        {
            "title": "Rice Crop Management",
            "content": "Stem Borer: Use pheromone traps and resistant varieties. Manage Blast via nitrogen timing.",
            "crop": "rice"
        }
    ]

# ==============================================================================
# PART 2: THE SYSTEM
# ==============================================================================
class FarmAISystem:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rag_chain = None

    def build(self, documents: List[Dict[str, Any]]):
        try:
            # We use the most stable embedding model name
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004", 
                google_api_key=self.api_key
            )
            
            lang_docs = [Document(page_content=d['content']) for d in documents]
            vectorstore = FAISS.from_documents(lang_docs, embeddings)
            retriever = vectorstore.as_retriever()

            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                google_api_key=self.api_key
            )

            prompt = ChatPromptTemplate.from_template("""
            You are a farming expert. Use the context to answer.
            Context: {context}
            Question: {input}
            """)

            combine_docs_chain = create_stuff_documents_chain(llm, prompt)
            self.rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
            return True
        except Exception as e:
            st.error(f"Build Failed: {e}")
            return False

# ==============================================================================
# PART 3: UI
# ==============================================================================
st.set_page_config(page_title="FarmAI Assistant", page_icon="🌾")
st.title("🌾 FarmAI Assistant")

api_key = st.secrets.get("GOOGLE_API_KEY")

if api_key:
    if "farm_ai" not in st.session_state:
        system = FarmAISystem(api_key)
        if system.build(load_agricultural_data()):
            st.session_state.farm_ai = system
            st.success("✅ FarmAI Knowledge Base Loaded!")

    if "farm_ai" in st.session_state:
        if prompt := st.chat_input("Ask a farming question..."):
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                response = st.session_state.farm_ai.rag_chain.invoke({"input": prompt})
                st.markdown(response["answer"])
else:
    st.warning("Please add your GOOGLE_API_KEY to Streamlit Secrets.")
