import streamlit as st
import nest_asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

nest_asyncio.apply()

st.set_page_config(page_title="FarmAI Assistant", page_icon="🌾")
st.title("🌾 FarmAI Assistant")

api_key = st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    st.error("Missing API Key! Please add 'GOOGLE_API_KEY' to your Streamlit Secrets.")
else:
    try:
        # THE CRITICAL FIX: Changed to 'gemini-embedding-001'
        # The old 'text-embedding-004' was retired Jan 2026.
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=api_key,
            task_type="retrieval_query" 
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest", 
            google_api_key=api_key,
            temperature=0.3
        )

        
        knowledge_base_data = [
            Document(page_content="Tomato Blight: Identified by brown spots with yellow halos. Management: Ensure air circulation and use copper-based fungicides."),
            Document(page_content="Rice Stem Borer: Larvae cause 'dead heart' in young plants. Management: Use pheromone traps and avoid excessive nitrogen."),
            Document(page_content="Tomato Sorting: High-quality tomatoes should be firm, uniform in color, and free of cracks or blemishes.")
        ]
        
        # Build the searchable index
        vectorstore = FAISS.from_documents(knowledge_base_data, embeddings)
        st.success("✅ FarmAI Knowledge Base is Live!")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if user_query := st.chat_input("Ask a question about your crops..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)

            # Retrieve and Generate
            relevant_docs = vectorstore.similarity_search(user_query, k=2)
            context_text = "\n\n".join([d.page_content for d in relevant_docs])
            full_prompt = f"Context: {context_text}\n\nQuestion: {user_query}"

            with st.chat_message("assistant"):
                ai_response = llm.invoke(full_prompt)
                answer = ai_response.content
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"Initialization Error: {e}")
