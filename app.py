import streamlit as st
import nest_asyncio
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Fix for Streamlit event loop
nest_asyncio.apply()

st.set_page_config(page_title="FarmAI Grounded Assistant", page_icon="🌾")
st.title("🌾 FarmAI Grounded Assistant")

# --- 1. Your Agricultural Manual ---
def load_manual():
    # Only the AI answers based on THIS content
    return [
        Document(page_content="Tomato Blight (Early and Late): Early blight shows brown spots; late blight causes dark water-soaked lesions. Management: Use certified seeds, crop rotation, and copper-based fungicides."),
        Document(page_content="Rice Stem Borer: Larvae cause 'dead heart' in young plants. Management: Use pheromone traps and avoid excessive nitrogen."),
        Document(page_content="Rice Blast: Management includes nitrogen timing and fungicide protocols."),
        Document(page_content="Maize Stem Borer: Use Cotesia flavipes parasitoids and destroy crop residues."),
        Document(page_content="Wheat Rust: Surveillance models help predict epidemics. Use resistant cultivars."),
    ]

api_key = st.secrets.get("GOOGLE_API_KEY")

if api_key:
    try:
        # CONFIGURE DIRECT GOOGLE API
        genai.configure(api_key=api_key)
        
        # --- 2. CUSTOM EMBEDDING FUNCTION ---
        # This bypasses LangChain's 404 errors by calling Google directly
        class DirectGoogleEmbeddings:
            def embed_documents(self, texts):
                return [genai.embed_content(model="models/text-embedding-004", content=t, task_type="retrieval_document")["embedding"] for t in texts]
            def embed_query(self, text):
                return genai.embed_content(model="models/text-embedding-004", content=text, task_type="retrieval_query")["embedding"]

        # Initialize the vector store with our direct function
        embeddings = DirectGoogleEmbeddings()
        vectorstore = FAISS.from_documents(load_manual(), embeddings)
        
        st.success("✅ Grounded Knowledge Base Active!")

        # --- 3. CHAT LOGIC ---
        if user_query := st.chat_input("Ask a question from the manual..."):
            with st.chat_message("user"):
                st.write(user_query)
            
            # RETRIEVAL: Search the manual
            relevant_docs = vectorstore.similarity_search(user_query, k=2)
            context = "\n\n".join([d.page_content for d in relevant_docs])

            # GENERATION: Direct Gemini Call
            with st.chat_message("assistant"):
                with st.spinner("Searching manual..."):
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    # STRICT PROMPT: Stops AI from using general knowledge
                    prompt = f"""You are a specialized Agricultural Assistant. 
                    You MUST ONLY use the provided manual context to answer. 
                    If the answer is NOT in the context, say you don't know.
                    
                    CONTEXT FROM MANUAL:
                    {context}
                    
                    QUESTION:
                    {user_query}"""
                    
                    response = model.generate_content(prompt)
                    st.write(response.text)

    except Exception as e:
        st.error(f"System Error: {e}")
        st.info("If you still see 404, please REBOOT the app in the Streamlit Dashboard.")
else:
    st.warning("Please add GOOGLE_API_KEY to Streamlit Secrets.")
