import streamlit as st
import nest_asyncio
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Fix for Streamlit's async environment
nest_asyncio.apply()

st.set_page_config(page_title="FarmAI OpenAI Assistant", page_icon="🌾")
st.title("🌾 FarmAI Grounded Assistant (OpenAI)")

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

# Get the OpenAI key from secrets
api_key = st.secrets.get("OPENAI_API_KEY")

if api_key:
    try:
        # 2. SET UP OPENAI EMBEDDINGS
        # text-embedding-3-small is the 2026 standard for high-speed RAG
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", 
            openai_api_key=api_key
        )
        
        vectorstore = FAISS.from_documents(load_manual_data(), embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        # 3. SET UP OPENAI LLM
        llm = ChatOpenAI(
            model="gpt-4o", 
            openai_api_key=api_key,
            temperature=0 # Strict factualness
        )

        # STRICT GROUNDING PROMPT
        prompt = ChatPromptTemplate.from_template("""
        You are a specialized Agricultural Assistant. 
        You MUST ONLY use the provided manual context to answer. 
        If the answer is NOT in the context, say: "I'm sorry, my manual does not contain information on that topic."
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        """)

        # LCEL Chain - Modern standard that avoids module errors
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        st.success("✅ OpenAI Knowledge Base Active!")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if user_query := st.chat_input("Ask about your crops..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing manual with GPT-4o..."):
                    response = rag_chain.invoke(user_query)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"System Error: {e}")
else:
    st.warning("Please add OPENAI_API_KEY to your Streamlit Secrets.")
