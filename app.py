import streamlit as st
import os
from typing import List, Dict, Any
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import httpx

# ==============================================================================
# PART 1: DATA LOADER CODE
# All code from src/data_loader.py is now here
# ==============================================================================
def load_agricultural_data() -> List[Dict[str, Any]]:
    """Load agricultural knowledge data from various sources"""
    knowledge_base = [
        {
            "title": "Tomato Disease Management Guide",
            "content": """Common Tomato Diseases and Management: 1. Tomato Blight (Early and Late Blight): - Symptoms: Brown spots on leaves with concentric rings (early blight) or dark lesions with white fuzzy growth (late blight). - Causes: Fungal infections. - Management: Apply copper-based fungicides, ensure good air circulation, avoid overhead watering, rotate crops.""",
            "source": "Agricultural Extension Service", "category": "disease_management", "crop": "tomato"
        },
        {
            "title": "Rice Cultivation Best Practices",
            "content": """Rice Cultivation Guidelines: 1. Land Preparation: - Plow field 2-3 times. - Apply organic matter. 2. Water Management: - Maintain 2-5 cm water level during vegetative stage. - Alternate wetting and drying for water conservation.""",
            "source": "Rice Research Institute", "category": "crop_management", "crop": "rice"
        },
        {
            "title": "Wheat Disease Identification and Control",
            "content": """Major Wheat Diseases: 1. Wheat Rust (Yellow, Brown, Black): - Symptoms: Rust-colored pustules on leaves and stems. - Management: Apply fungicides (Propiconazole), use resistant varieties.""",
            "source": "Wheat Research Institute", "category": "disease_management", "crop": "wheat"
        },
        {
            "title": "Integrated Pest Management Strategies",
            "content": """Integrated Pest Management (IPM) Principles: 1. Prevention: Use resistant crop varieties, practice good field sanitation. 2. Biological Control: Encourage natural enemies like ladybugs. 3. Chemical Control (Last Resort): Use selective pesticides only when necessary.""",
            "source": "IPM Guidelines", "category": "pest_management", "crop": "general"
        }
    ]
    return knowledge_base

# ==============================================================================
# PART 2: RAG SYSTEM CODE
# All code from src/rag_system.py is now here (with fixes)
# ==============================================================================
class FarmAIRAG:
    """Advanced RAG system for agricultural knowledge"""
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
    def build_knowledge_base(self, documents: List[Dict[str, Any]]):
        try:
            http_client = httpx.Client(proxies="")
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_base="https://api.chatanywhere.tech/v1",
                client=http_client
            )
            langchain_docs = [Document(page_content=doc['content'], metadata={'title': doc.get('title', '')}) for doc in documents]
            self.vectorstore = FAISS.from_documents(langchain_docs, self.embeddings)
            llm = OpenAI(
                temperature=0.1,
                model_name="gpt-3.5-turbo",
                max_tokens=512,
                openai_api_base="https://api.chatanywhere.tech/v1",
                client=http_client
            )
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=self.vectorstore.as_retriever(), return_source_documents=True
            )
            return True
        except Exception as e:
            st.error(f"Error in RAG build: {e}") # Show error in UI
            return False
    
    def query(self, question: str):
        try:
            if not self.qa_chain:
                return {"response": "Knowledge base not ready.", "sources": []}
            result = self.qa_chain({"query": question})
            sources = [{"title": doc.metadata.get("title", "Source")} for doc in result.get("source_documents", [])]
            return {"response": result["result"], "sources": sources}
        except Exception as e:
            return {"response": f"Error during query: {e}", "sources": []}

# ==============================================================================
# PART 3: AGENTS CODE
# All code from src/agents.py is now here (with fixes)
# ==============================================================================
class FarmAIAgents:
    """Multi-agent system for agricultural question processing"""
    def __init__(self, rag_system):
        self.rag_system = rag_system
        http_client = httpx.Client(proxies="")
        self.llm = OpenAI(
            temperature=0.1, model_name="gpt-3.5-turbo", max_tokens=1024,
            openai_api_base="https://api.chatanywhere.tech/v1", client=http_client
        )
        self.query_classifier = self._create_classifier()

    def _create_classifier(self):
        prompt = PromptTemplate(
            input_variables=["query"],
            template="Classify the query into: 'disease', 'pest', 'management', or 'general'. Query: {query} Classification:"
        )
        return LLMChain(llm=self.llm, prompt=prompt)

    def process_query(self, query: str):
        try:
            # Simplified workflow: Just use the RAG system directly
            return self.rag_system.query(query)
        except Exception as e:
            return {"response": f"Error in agent processing: {e}", "sources": []}

# ==============================================================================
# PART 4: MAIN STREAMLIT APP
# ==============================================================================
st.set_page_config(page_title="FarmAI Knowledge Assistant", page_icon="🌾", layout="wide")

st.markdown('<h1 style="text-align: center; color: #2E8B57;">🌾 FarmAI Knowledge Assistant</h1>', unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    try:
        documents = load_agricultural_data()
        rag_system = FarmAIRAG()
        success = rag_system.build_knowledge_base(documents)
        if not success:
            return None, False
        agents = FarmAIAgents(rag_system)
        return agents, True
    except Exception as e:
        st.error(f"Critical initialization failed: {e}")
        return None, False

agents, success = initialize_system()

if not success:
    st.error("Failed to initialize the FarmAI system. Please check your API key in Streamlit Secrets and refresh.")
else:
    st.success("✅ FarmAI system initialized successfully!")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your farming questions today?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me about agriculture..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("FarmAI is thinking..."):
                response_data = agents.process_query(prompt)
                response = response_data["response"]
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
