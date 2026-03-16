import streamlit as st
import nest_asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA # The simplest, most stable chain
from langchain_core.documents import Document

# Apply the patch for the event loop
nest_asyncio.apply()

st.set_page_config(page_title="FarmAI Assistant", page_icon="🌾")
st.title("🌾 FarmAI Assistant")

# --- Step 1: Setup API Key ---
api_key = st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    st.error("Please add GOOGLE_API_KEY to Streamlit Secrets.")
else:
    try:
        # --- Step 2: Initialize Models ---
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", 
            google_api_key=api_key
        )
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=api_key
        )

        # --- Step 3: Create simple Knowledge Base ---
        data = [
            Document(page_content="Tomato Blight shows brown spots. Control via hygiene."),
            Document(page_content="Rice Stem Borer: Use pheromone traps and resistant varieties.")
        ]
        
        vectorstore = FAISS.from_documents(data, embeddings)
        
        # --- Step 4: Create the Chain (Using the most stable version) ---
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        st.success("✅ System Ready!")

        # --- Step 5: Chat Interface ---
        if prompt := st.chat_input("Ask about tomatoes or rice..."):
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                response = qa_chain.run(prompt)
                st.write(response)

    except Exception as e:
        st.error(f"Initialization Error: {e}")
