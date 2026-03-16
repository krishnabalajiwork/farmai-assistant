import streamlit as st
import os
from typing import List, Dict, Any
import nest_asyncio

# Apply the patch for the event loop issue
nest_asyncio.apply()

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==============================================================================
# PART 1: YOUR AGRICULTURAL DATA
# ==============================================================================
def load_agricultural_data() -> List[Dict[str, Any]]:
    # I am including your data structure here
    return [
        {
            "title": "Tomato Disease and Pest Management",
            "content": "Tomato Blight (Early and Late): Early blight shows brown spots; late blight causes dark water-soaked lesions. Management: Use certified seeds, crop rotation, and copper-based fungicides.",
            "crop": "tomato"
        },
        {
            "title": "Rice Crop Pest and Disease Management",
            "content": "Stem Borer: Use pheromone traps and resistant varieties like 'Swarna'. Blast Disease: Integrated management includes nitrogen timing and fungicide protocols.",
            "crop": "rice"
        },
        # ... paste your other maize/wheat/IPM content here
    ]

# ==============================================================================
# PART 2: THE STABLE RAG SYSTEM (Using LCEL)
# ==============================================================================
class FarmAISystem:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rag_chain = None

    def build_knowledge_base(self, documents: List[Dict[str, Any]]):
        try:
            # 1. Setup Embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.api_key
            )

            # 2. Prepare Documents
            langchain_docs = [Document(page_content=doc['content']) for doc in documents]
            vectorstore = FAISS.from_documents(langchain_docs, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            # 3. Setup LLM (Gemini)
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0, # Factual accuracy
                google_api_key=self.api_key
            )

            # 4. Create the STRICTOR GROUNDING Prompt
            prompt = ChatPromptTemplate.from_template("""
            You are a strict FarmAI Agricultural Assistant. 
            Answer the question ONLY using the provided context.
            If the answer is not in the context, say: "I do not have information on that in my manual."
            
            Context: {context}
            
            Question: {question}
            """)

            # 5. Build the Chain manually (LCEL) - This avoids the ModuleNotFoundError
            self.rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            return True
        except Exception as e:
            st.error(f"Build error: {e}")
            return False

# ==============================================================================
# PART 3: UI LOGIC
# ==============================================================================
st.set_page_config(page_title="FarmAI Grounded Assistant", page_icon="🌾")
st.title("🌾 FarmAI Grounded Assistant")

api_key = st.secrets.get("GOOGLE_API_KEY")

@st.cache_resource
def init():
    if not api_key: return None, False
    data = load_agricultural_data()
    system = FarmAISystem(api_key)
    success = system.build_knowledge_base(data)
    return system, success

farm_ai, success = init()

if success:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # The chain is invoked directly
            response = farm_ai.rag_chain.invoke(user_input)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.error("Please verify your GOOGLE_API_KEY in Streamlit Secrets.")
