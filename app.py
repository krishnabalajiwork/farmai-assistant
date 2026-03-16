import streamlit as st
import nest_asyncio
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

nest_asyncio.apply()

st.set_page_config(page_title="FarmAI Grounded Assistant", page_icon="🌾")
st.title("🌾 FarmAI Grounded Assistant")

# --- 1. Your Agricultural Manual ---
def load_manual():
    return [
        Document(page_content="Tomato Blight: Brown spots with yellow halos. Management: Air circulation and copper-based fungicides."),
        Document(page_content="Rice Stem Borer: Larvae cause 'dead heart'. Management: Pheromone traps."),
        Document(page_content="Tomato Sorting: High-quality tomatoes are firm, uniform, and crack-free."),
        # ... Add your full manual content here
    ]

api_key = st.secrets.get("GOOGLE_API_KEY")

if api_key:
    try:
        # CONFIGURE DIRECT GOOGLE API
        genai.configure(api_key=api_key)
        
        # 2. SET UP SEARCH (EMBEDDINGS)
        # Using the absolute most stable ID to avoid 404
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=api_key
        )
        
        vectorstore = FAISS.from_documents(load_manual(), embeddings)
        st.success("✅ Grounded Knowledge Base Active!")

        if user_query := st.chat_input("Ask a question from the manual..."):
            with st.chat_message("user"):
                st.write(user_query)

            # 3. RETRIEVAL: Find relevant parts of the manual
            relevant_docs = vectorstore.similarity_search(user_query, k=2)
            context = "\n\n".join([d.page_content for d in relevant_docs])

            # 4. GENERATION: Direct call to Gemini (Bypasses LangChain 404 issues)
            with st.chat_message("assistant"):
                with st.spinner("Consulting manual..."):
                    # We use the direct GenerativeModel which defaults to stable paths
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    prompt = f"""You are a specialized Agricultural Assistant. 
                    ONLY use the following context to answer. 
                    If the answer is NOT in the context, say you don't know.
                    
                    CONTEXT:
                    {context}
                    
                    QUESTION:
                    {user_query}"""
                    
                    response = model.generate_content(prompt)
                    st.write(response.text)

    except Exception as e:
        st.error(f"System Error: {e}")
        st.info("If 404 persists, try changing model to 'gemini-pro' on line 54.")
else:
    st.warning("Please add GOOGLE_API_KEY to Streamlit Secrets.")
