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

# --- Agricultural Manual Data ---
def load_manual():
    return [
        Document(page_content="Tomato Blight (Early and Late): Early blight shows brown spots; late blight causes dark water-soaked lesions. Management: Use certified seeds, crop rotation, and copper-based fungicides."),
        Document(page_content="Rice Stem Borer: Larvae cause 'dead heart' in young plants. Management: Use pheromone traps and avoid excessive nitrogen."),
        Document(page_content="Wheat Rust: Pathogenesis includes stem, leaf, and stripe rust. Surveillance models help predict epidemics."),
    ]

api_key = st.secrets.get("GOOGLE_API_KEY")

if api_key:
    try:
        # THE ABSOLUTE FIX: Switching to the newest 'text-embedding-004' 
        # but EXPLICITLY setting the task_type to 'retrieval_query'.
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", 
            google_api_key=api_key,
            task_type="retrieval_query" 
        )
        
        vectorstore = FAISS.from_documents(load_manual(), embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        # LLM FIX: Use the stable flash ID
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=api_key,
            temperature=0
        )

        prompt = ChatPromptTemplate.from_template("""
        You are a specialized Agricultural Assistant. 
        You MUST ONLY use the provided context to answer the question. 
        If the answer is NOT in the context, say: "I'm sorry, my manual does not contain information on that topic."
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        """)

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
        # We print the full error here to help you debug in case of rate limits
        st.error(f"System Error: {e}")
else:
    st.warning("Please add GOOGLE_API_KEY to Streamlit Secrets.")
