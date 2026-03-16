import streamlit as st
import nest_asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

nest_asyncio.apply()

st.set_page_config(page_title="FarmAI Grounded Assistant", page_icon="🌾")
st.title("🌾 FarmAI Grounded Assistant")

# --- 1. Load your specific manual data ---
def load_manual():
    return [
        Document(page_content="Tomato Blight (Early and Late): Early blight shows brown spots; late blight causes dark water-soaked lesions. Management: Use certified seeds, crop rotation, and copper-based fungicides."),
        Document(page_content="Rice Stem Borer: Larvae cause 'dead heart' in young plants. Management: Use pheromone traps and avoid excessive nitrogen."),
        # ... paste your full 1000+ line manual here ...
    ]

api_key = st.secrets.get("GOOGLE_API_KEY")

if api_key:
    try:
        # Use the absolute stable embedding model
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=api_key,
            task_type="retrieval_query"
        )
        
        vectorstore = FAISS.from_documents(load_manual(), embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=api_key,
            temperature=0  # Zero temperature makes it more factual
        )

        # STRICT PROMPT: This prevents it from talking about NLP or general knowledge
        prompt = ChatPromptTemplate.from_template("""
        You are a specialized Agricultural Assistant. 
        You MUST ONLY use the provided manual context to answer. 
        If the answer is NOT in the context, say: "I'm sorry, my manual doesn't have information on that."
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        """)

        # Build the chain manually to avoid ModuleNotFoundErrors
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        st.success("✅ Grounded Knowledge Base Active!")

        if user_query := st.chat_input("Ask a question from the manual..."):
            with st.chat_message("user"): st.write(user_query)
            with st.chat_message("assistant"):
                response = rag_chain.invoke(user_query)
                st.write(response)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.warning("Please add GOOGLE_API_KEY to secrets.")
