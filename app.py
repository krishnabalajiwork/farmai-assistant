import streamlit as st
import nest_asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Fix for Streamlit event loop
nest_asyncio.apply()

st.set_page_config(page_title="FarmAI Grounded Assistant", page_icon="🌾")
st.title("🌾 FarmAI Grounded Assistant")

# ==============================================================================
# PART 1: YOUR AGRICULTURAL MANUAL
# ==============================================================================
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
        # EMBEDDING FIX: Using gemini-embedding-001 with explicit task_type
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=api_key,
            task_type="retrieval_query"
        )
        
        vectorstore = FAISS.from_documents(load_manual(), embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        # LLM FIX: Use stable flash ID and temperature 0 for grounding
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=api_key,
            temperature=0
        )

        # THE GROUNDING PROMPT: Forces AI to stay in the manual
        prompt = ChatPromptTemplate.from_template("""
        You are a specialized Agricultural Assistant. 
        You MUST ONLY use the provided context to answer the question. 
        If the answer is NOT in the context, say: "I'm sorry, my manual does not contain information on that topic."
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        """)

        # Chain build without using 'langchain.chains'
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        st.success("✅ Grounded Knowledge Base Active!")

        if user_query := st.chat_input("Ask a question from the manual..."):
            with st.chat_message("user"):
                st.write(user_query)
            
            with st.chat_message("assistant"):
                response = rag_chain.invoke(user_query)
                st.write(response)

    except Exception as e:
        st.error(f"System Error: {e}")
        st.info("If you see 404, please REBOOT the app in the Streamlit Dashboard.")
else:
    st.warning("Please add GOOGLE_API_KEY to Streamlit Secrets.")
