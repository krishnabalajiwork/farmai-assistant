import streamlit as st
import os
from typing import List, Dict, Any
import nest_asyncio

# Apply the patch for the event loop issue
nest_asyncio.apply()

# Updated Imports to fix the ImportError and use modern LCEL chains
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.docstore.document import Document

# ==============================================================================
# PART 1: DATA LOADER CODE
# ==============================================================================
def load_agricultural_data() -> List[Dict[str, Any]]:
    """Knowledge base containing detailed agricultural data."""
    knowledge_base = [
        {
            "title": "Tomato Disease and Pest Management",
            "content": """
Tomato Disease Management:
1. Tomato Blight (Early and Late): Early blight shows brown spots; late blight causes dark water-soaked lesions.
2. Tomato Mosaic Virus: Mottled leaf patterns and stunting. Control via hygiene and aphid management.
""",
            "source": "Extension Publications",
            "category": "disease_management",
            "crop": "tomato"
        },
        {
            "title": "Rice Crop Pest and Disease Management",
            "content": """
Rice Management:
1. Stem Borer: Use pheromone traps and resistant varieties like 'Swarna'.
2. Blast Disease: Manage through nitrogen timing and water control.
""",
            "source": "IRRI",
            "category": "crop_management",
            "crop": "rice"
        }
    ]
    return knowledge_base


# ==============================================================================
# PART 2: RAG SYSTEM (Modern OpenAI Version)
# ==============================================================================
class FarmAISystem:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rag_chain = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    def build_knowledge_base(self, documents: List[Dict[str, Any]]):
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
            
            langchain_docs = [
                Document(page_content=doc['content'], metadata={'title': doc.get('title', '')}) 
                for doc in documents
            ]
            
            final_docs = self.text_splitter.split_documents(langchain_docs)
            vectorstore = FAISS.from_documents(final_docs, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=self.api_key)

            # Modern RAG Chain Construction
            system_prompt = (
                "You are an expert agricultural assistant. Use the following pieces of "
                "retrieved context to answer the question. If you don't know the answer, "
                "say that you don't know. Use three sentences maximum and keep the "
                "answer concise.\n\n{context}"
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            # Create the chain that combines documents and the retrieval chain
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            self.rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            return True
        except Exception as e:
            st.error(f"Error building knowledge base: {e}")
            return False

    def query(self, question: str):
        if not self.rag_chain:
            return "The system is not ready. Please check the logs."
        try:
            # The modern chain uses 'input' as the key and returns 'answer'
            response = self.rag_chain.invoke({"input": question})
            return response.get('answer', "I'm sorry, I couldn't find an answer.")
        except Exception as e:
            return f"An error occurred: {e}"

# ==============================================================================
# PART 3: MAIN STREAMLIT APP
# ==============================================================================
st.set_page_config(page_title="FarmAI Knowledge Assistant", page_icon="🌾", layout="wide")
st.markdown('<h1 style="text-align: center; color: #2E8B57;">🌾 FarmAI Knowledge Assistant</h1>', unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return None, False

    documents = load_agricultural_data()
    farm_ai_system = FarmAISystem(api_key=api_key)
    success = farm_ai_system.build_knowledge_base(documents)
    return farm_ai_system, success

farm_ai, success = initialize_system()

if success:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help with your farming questions?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about agriculture..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = farm_ai.query(prompt)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.error("System failed to initialize. Check your OPENAI_API_KEY in Secrets.")
