import streamlit as st
import nest_asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Required for Streamlit's async environment
nest_asyncio.apply()

st.set_page_config(page_title="FarmAI Grounded Assistant", page_icon="🌾")
st.title("🌾 FarmAI Grounded Assistant")

# --- 1. Your Agricultural Manual Data ---
def load_manual_data():
    # This is the ONLY source of truth for the AI
    return [
        Document(page_content="Tomato Blight (Early and Late): Early blight shows brown spots; late blight causes dark water-soaked lesions. Management: Use certified seeds, crop rotation, and copper-based fungicides."),
        Document(page_content="Rice Stem Borer: Larvae cause 'dead heart' in young plants. Management: Use pheromone traps and avoid excessive nitrogen."),
        Document(page_content="Rice Blast (Magnaporthe oryzae): Management includes nitrogen timing and fungicide protocols."),
        Document(page_content="Maize Stem Borer: Cultural practices include destruction of crop residues to break lifecycle."),
        Document(page_content="Wheat Rust: Surveillance models help predict epidemics. Use resistant cultivars."),
        Document(page_content="Tomato Sorting: High-quality tomatoes must be firm, uniform in color, and free of cracks.")
    ]

api_key = st.secrets.get("GOOGLE_API_KEY")

if api_key:
    try:
        # THE 2026 STABLE EMBEDDING FIX
        # 'gemini-embedding-001' is the new production standard.
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=api_key,
            task_type="retrieval_query"
        )
        
        # Build Vector Store
        vectorstore = FAISS.from_documents(load_manual_data(), embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        # THE 2026 STABLE LLM FIX
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=api_key,
            temperature=0 # Strict factualness
        )

        # STRICT GROUNDING PROMPT: Prevents general knowledge
        prompt = ChatPromptTemplate.from_template("""
        You are a specialized Agricultural Assistant. 
        You MUST ONLY use the provided manual context to answer. 
        If the answer is NOT in the context, say: "I'm sorry, my manual does not contain information on that topic."
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        """)

        # LCEL Chain - Bypasses 'langchain.chains' ModuleNotFoundError
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        st.success("✅ Grounded Knowledge Base Active!")

        # Chat UI
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if user_query := st.chat_input("Ask a question from the manual..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)
            
            with st.chat_message("assistant"):
                with st.spinner("Searching manual..."):
                    response = rag_chain.invoke(user_query)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"System Error: {e}")
        st.info("If 404 persists, please REBOOT the app in the Streamlit Dashboard.")
else:
    st.warning("Please add GOOGLE_API_KEY to Streamlit Secrets.")
