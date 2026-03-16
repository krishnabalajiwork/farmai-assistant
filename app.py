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
        # Add your other manual content here...
    ]

api_key = st.secrets.get("GOOGLE_API_KEY")

if api_key:
    try:
        # THE FIX: We use the most stable model names for 2026
        # 'text-embedding-004' is the production standard.
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", 
            google_api_key=api_key,
            task_type="retrieval_query"
        )
        
        vectorstore = FAISS.from_documents(load_manual(), embeddings)
        st.success("✅ Grounded Knowledge Base Active!")

        if user_query := st.chat_input("Ask a question from the manual..."):
            with st.chat_message("user"):
                st.write(user_query)

            # Retrieve info from your manual
            relevant_docs = vectorstore.similarity_search(user_query, k=2)
            context = "\n\n".join([d.page_content for d in relevant_docs])

            # Generation using direct Google library to avoid 404
            with st.chat_message("assistant"):
                genai.configure(api_key=api_key)
                # 'gemini-1.5-flash' is the stable workhorse model
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                prompt = f"""You are a specialized Agricultural Assistant. 
                ONLY use the following context from the manual to answer. 
                If the answer is NOT in the context, say you don't know.
                
                CONTEXT:
                {context}
                
                QUESTION:
                {user_query}"""
                
                response = model.generate_content(prompt)
                st.write(response.text)

    except Exception as e:
        st.error(f"System Error: {e}")
        st.info("If this persists, please REBOOT the app in the Streamlit Dashboard.")
else:
    st.warning("Please add GOOGLE_API_KEY to Streamlit Secrets.")
