import streamlit as st
import os
from typing import List, Dict, Any

# Import new, specific classes from LangChain
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ==============================================================================
# PART 1: DATA LOADER CODE
# ==============================================================================
def load_agricultural_data() -> List[Dict[str, Any]]:
    """Load agricultural knowledge data"""
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
        }
    ]
    return knowledge_base

# ==============================================================================
# PART 2: RAG AND AGENT SYSTEM (Simplified)
# ==============================================================================
class FarmAISystem:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.qa_chain = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def build_knowledge_base(self, documents: List[Dict[str, Any]]):
        try:
            # The new library versions handle the network environment correctly
            embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=self.api_key,
                openai_api_base=self.base_url
            )

            langchain_docs = [Document(page_content=doc['content'], metadata={'title': doc.get('title', '')}) for doc in documents]
            vectorstore = FAISS.from_documents(langchain_docs, embeddings)

            llm = OpenAI(
                temperature=0.1,
                model="gpt-3.5-turbo", # Note: The parameter is 'model' now, not 'model_name'
                max_tokens=512,
                openai_api_key=self.api_key,
                openai_api_base=self.base_url
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
            )
            return True
        except Exception as e:
            # This will display the specific error on the Streamlit page
            st.error(f"Error building knowledge base: {e}")
            return False

    def query(self, question: str):
        if not self.qa_chain:
            return "The system is not ready. Please check the logs."
        try:
            result = self.qa_chain.invoke(question)
            return result.get('result', "Sorry, I couldn't find an answer.")
        except Exception as e:
            return f"An error occurred: {e}"

# ==============================================================================
# PART 3: MAIN STREAMLIT APP
# ==============================================================================
st.set_page_config(page_title="FarmAI Knowledge Assistant", page_icon="🌾", layout="wide")
st.markdown('<h1 style="text-align: center; color: #2E8B57;">🌾 FarmAI Knowledge Assistant</h1>', unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    """Load data and build the RAG system."""
    api_key = st.secrets.get("OPENAI_API_KEY")
    base_url = "https://api.chatanywhere.org/v1"

    if not api_key:
        return None, False

    documents = load_agricultural_data()
    farm_ai_system = FarmAISystem(api_key=api_key, base_url=base_url)
    success = farm_ai_system.build_knowledge_base(documents)

    return farm_ai_system, success

farm_ai, success = initialize_system()

if not success:
    st.error("Failed to initialize the FarmAI system. Please ensure your API key is correct in Streamlit Secrets and refresh the page.")
else:
    st.success("✅ FarmAI system initialized successfully!")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help with your farming questions today?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about agriculture..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("FarmAI is thinking..."):
                response = farm_ai.query(prompt)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
