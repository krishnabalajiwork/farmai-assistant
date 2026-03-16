import streamlit as st
import os
from typing import List, Dict, Any
import nest_asyncio

# Apply the patch for the event loop issue
nest_asyncio.apply()

# Updated Imports for OpenAI and Community packages
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chains import RetrievalQA
from langchain.docstore.document import Document

# ==============================================================================
# PART 1: DATA LOADER CODE
# ==============================================================================
def load_agricultural_data() -> List[Dict[str, Any]]:
    """
    Knowledge base containing detailed agricultural data.
    """
    knowledge_base = [
        {
            "title": "Tomato Disease and Pest Management Detailed Guide",
            "content": """
Tomato Disease Management:
1. Tomato Blight (Early Blight and Late Blight):
- Symptoms: Early blight appears as brown concentric spots; late blight causes dark water-soaked lesions.
- Causal Agents: Alternaria solani (early), Phytophthora infestans (late).
- Management: Use certified seeds, crop rotation, copper-based fungicides, and soil-level irrigation.

2. Tomato Mosaic Virus (ToMV):
- Symptoms: Mottled green leaf patterns, curling, stunting.
- Control: Destroy infected plants, clean tools, and control aphid populations.
""",
            "source": "Agricultural Extension Publications",
            "category": "disease_management",
            "crop": "tomato"
        },
        {
            "title": "Rice Crop Pest and Disease Management",
            "content": """
Pest Complex in Rice:
1. Stem Borer: Yellow, Pink, and Striped borers. Use pheromone traps and resistant varieties like 'Swarna'.
2. Blast Disease: Caused by Magnaporthe oryzae. Manage via nitrogen timing and water management.
3. Planthoppers: Controlled via systemic insecticides and biological agents like spiders.
""",
            "source": "IRRI Publications",
            "category": "crop_management",
            "crop": "rice"
        },
        {
            "title": "Maize Integrated Pest Management",
            "content": """
Major Maize Production Challenges:
1. Stem Borer: Destroy crop residues to break lifecycle. Use parasitoids (Cotesia flavipes).
2. Leaf Blight: Managed through resistant hybrids and seed treatment.
3. Earworm: Monitored with pheromone traps; use Bacillus thuringiensis products.
""",
            "source": "NCIPM Publications",
            "category": "integrated_pest_management",
            "crop": "maize"
        }
        # You can add more entries here to reach your 1000+ line goal
    ]
    return knowledge_base


# ==============================================================================
# PART 2: RAG SYSTEM (OpenAI Version)
# ==============================================================================
class FarmAISystem:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.qa_chain = None
        # Splitter to break long text into chunks for better retrieval accuracy
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150
        )

    def build_knowledge_base(self, documents: List[Dict[str, Any]]):
        try:
            # Use OpenAI Embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)

            # Convert dicts to LangChain Document objects
            langchain_docs = [
                Document(page_content=doc['content'], metadata={'title': doc.get('title', '')}) 
                for doc in documents
            ]

            # Split large documents into smaller chunks
            final_docs = self.text_splitter.split_documents(langchain_docs)

            # Create FAISS vectorstore
            vectorstore = FAISS.from_documents(final_docs, embeddings)

            # Initialize OpenAI LLM (gpt-4o-mini is cost-effective and fast)
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                openai_api_key=self.api_key
            )

            # Setup the Retrieval Chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff", 
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
            )
            return True
        except Exception as e:
            st.error(f"Error building knowledge base: {e}")
            return False

    def query(self, question: str):
        if not self.qa_chain:
            return "The system is not ready. Please check the logs."
        try:
            # Invoke the chain
            response = self.qa_chain.invoke({"query": question})
            return response.get('result', "I'm sorry, I couldn't find a specific answer for that.")
        except Exception as e:
            return f"An error occurred during query: {e}"

# ==============================================================================
# PART 3: MAIN STREAMLIT APP
# ==============================================================================
st.set_page_config(page_title="FarmAI Knowledge Assistant", page_icon="🌾", layout="wide")

st.markdown('<h1 style="text-align: center; color: #2E8B57;">🌾 FarmAI Knowledge Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center;">Powered by OpenAI & LangChain</p>', unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    """Load data and build the RAG system using OpenAI Secrets."""
    # Ensure you have 'OPENAI_API_KEY' in your Streamlit Secrets
    api_key = st.secrets.get("OPENAI_API_KEY")

    if not api_key:
        st.warning("Please add your 'OPENAI_API_KEY' to Streamlit Secrets.")
        return None, False

    documents = load_agricultural_data()
    farm_ai_system = FarmAISystem(api_key=api_key)
    success = farm_ai_system.build_knowledge_base(documents)

    return farm_ai_system, success

# Initialize the system
farm_ai, success = initialize_system()

if success:
    st.sidebar.success("✅ Knowledge Base Loaded")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am FarmAI. Ask me anything about tomato, rice, or maize management."}
        ]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle User Input
    if prompt := st.chat_input("Ex: How do I manage Tomato Blight?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing knowledge base..."):
                response = farm_ai.query(prompt)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.error("System failed to initialize. Check your API key and dependencies.")
