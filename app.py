import streamlit as st
import os
from typing import List, Dict, Any
import nest_asyncio

# Apply the patch for the event loop issue
nest_asyncio.apply()

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

# ==============================================================================
# PART 1: YOUR EXTENSIVE DATA LOADER
# ==============================================================================
def load_agricultural_data() -> List[Dict[str, Any]]:
    return [
        {
            "title": "Tomato Disease and Pest Management Detailed Guide",
            "content": """
Tomato Disease Management:
1. Tomato Blight (Early Blight and Late Blight):
- Symptoms: Early blight appears as brown concentric spots; late blight causes dark water-soaked lesions.
- Causal Agents: Alternaria solani (early blight), Phytophthora infestans (late blight).
- Management: Use certified seeds, crop rotation, and copper-based fungicides.
2. Tomato Mosaic Virus (ToMV): Mottled green patterns, leaf curling. Control aphids.
3. Fusarium Wilt: Vascular discoloration. Control by soil solarization.
4. Whitefly Management: Use yellow sticky traps and biological agents like Encarsia spp.
""",
            "category": "disease_management",
            "crop": "tomato"
        },
        {
            "title": "Rice Crop Pest and Disease Management Extended Manual",
            "content": """
Rice Management:
1. Stem Borer: Use pheromone traps and resistant varieties like 'Swarna'.
2. Blast Disease (Magnaporthe oryzae): Integrated management includes nitrogen timing and fungicide protocols.
3. Planthopper Management: Use systemic insecticides and predatory bugs.
""",
            "category": "crop_management",
            "crop": "rice"
        }
        # ... Add your other content entries here exactly as in your manual
    ]

# ==============================================================================
# PART 2: THE STRICT RAG SYSTEM
# ==============================================================================
class FarmAISystem:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.qa_chain = None

    def build_knowledge_base(self, documents: List[Dict[str, Any]]):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.api_key
            )

            langchain_docs = [Document(page_content=doc['content'], metadata={'title': doc.get('title', '')}) for doc in documents]
            
            # Split text to ensure the AI gets high-quality chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
            split_docs = text_splitter.split_documents(langchain_docs)
            
            vectorstore = FAISS.from_documents(split_docs, embeddings)

            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest",
                temperature=0, # Set to 0 for maximum factual accuracy
                google_api_key=self.api_key
            )

            # STRICT GROUNDING PROMPT
            template = """You are a strict FarmAI Agricultural Assistant. 
            Use ONLY the following pieces of context to answer the question at the end. 
            If the answer is not contained within the context, politely state that you 
            do not have information on that specific topic in your agricultural manual.
            
            {context}
            
            Question: {question}
            Grounded Answer:"""
            
            QA_PROMPT = PromptTemplate(
                template=template, input_variables=["context", "question"]
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": QA_PROMPT}
            )
            return True
        except Exception as e:
            st.error(f"Error: {e}")
            return False

    def query(self, question: str):
        if not self.qa_chain:
            return "System not ready."
        try:
            # Use invoke instead of run for better compatibility with 2026 standards
            result = self.qa_chain.invoke({"query": question})
            return result.get('result')
        except Exception as e:
            return f"An error occurred: {e}"

# ==============================================================================
# PART 3: UI
# ==============================================================================
st.set_page_config(page_title="FarmAI Knowledge Assistant", page_icon="🌾")
st.title("🌾 FarmAI Grounded Assistant")

@st.cache_resource
def initialize_system():
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key: return None, False
    
    documents = load_agricultural_data()
    system = FarmAISystem(api_key=api_key)
    success = system.build_knowledge_base(documents)
    return system, success

farm_ai, success = initialize_system()

if success:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your manual..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = farm_ai.query(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
