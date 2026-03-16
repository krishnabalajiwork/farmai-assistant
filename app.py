import streamlit as st
import nest_asyncio
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Required for Streamlit's async environment
nest_asyncio.apply()

st.set_page_config(page_title="FarmAI Grounded Assistant", page_icon="🌾")
st.title("🌾 FarmAI Grounded Assistant")

# --- 1. Your Agricultural Manual Data ---
def load_manual_data():
    return [
        Document(page_content="Tomato Blight (Early and Late): Early blight shows brown spots; late blight causes dark water-soaked lesions. Management: Use certified seeds, crop rotation, and copper-based fungicides."),
        Document(page_content="Rice Stem Borer: Larvae cause 'dead heart' in young plants. Management: Use pheromone traps and avoid excessive nitrogen."),
        Document(page_content="Rice Blast: Management includes nitrogen timing and fungicide protocols."),
        Document(page_content="Maize Stem Borer: Cultural practices include destruction of crop residues to break lifecycle."),
        Document(page_content="Wheat Rust: Surveillance models help predict epidemics. Use resistant cultivars."),
        Document(page_content="Tomato Sorting: High-quality tomatoes must be firm, uniform in color, and free of cracks.")
    ]

api_key = st.secrets.get("GOOGLE_API_KEY")

if api_key:
    try:
        # CONFIGURE DIRECT GOOGLE SDK (Forces stable production path)
        genai.configure(api_key=api_key)
        
        # 2. SET UP STABLE EMBEDDINGS
        # Using gemini-embedding-001 which is the 2026 production standard
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=api_key,
            task_type="retrieval_query"
        )
        
        vectorstore = FAISS.from_documents(load_manual_data(), embeddings)
        st.success("✅ Grounded Knowledge Base Active!")

        # Chat UI Setup
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if user_query := st.chat_input("Ask a question from the manual..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)
            
            # RETRIEVAL: Find the specific manual paragraphs
            relevant_docs = vectorstore.similarity_search(user_query, k=2)
            context = "\n\n".join([d.page_content for d in relevant_docs])

            with st.chat_message("assistant"):
                with st.spinner("Searching manual..."):
                    # 3. GENERATION: Direct SDK Call (Bypasses the LangChain v1beta issue)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    prompt = f"""You are a specialized Agricultural Assistant. 
                    You MUST ONLY use the provided manual context to answer. 
                    If the answer is NOT in the context, say: "I'm sorry, my manual does not contain information on that topic."
                    
                    CONTEXT FROM MANUAL:
                    {context}
                    
                    QUESTION:
                    {user_query}"""
                    
                    response = model.generate_content(prompt)
                    answer = response.text
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"System Error: {e}")
        st.info("If errors persist, please REBOOT the app in the Streamlit Dashboard.")
else:
    st.warning("Please add GOOGLE_API_KEY to Streamlit Secrets.")
