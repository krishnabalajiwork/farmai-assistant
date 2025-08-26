import streamlit as st
import os
import sys
from datetime import datetime
import json
import openai  # <--- Add this line

# Use the stored OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import our custom modules
try:
    from src.rag_system import FarmAIRAG
    from src.agents import FarmAIAgents
    from src.data_loader import load_agricultural_data
except ImportError as e:
    st.error(f"Error importing modules: {str(e)}")
    st.info("Please ensure all required files are in the same directory as app.py")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="FarmAI Knowledge Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #2E8B57;
    margin-bottom: 20px;
}
.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    display: flex;
    align-items: flex-start;
}
.user-message {
    background-color: #E8F4FD;
    border-left: 4px solid #1f77b4;
}
.assistant-message {
    background-color: #F0F8E8;
    border-left: 4px solid #2E8B57;
}
.source-box {
    background-color: #F5F5F5;
    padding: 10px;
    border-radius: 5px;
    margin-top: 10px;
    font-size: 0.9em;
}
.stAlert > div {
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    """Initialize the RAG system and agents"""
    try:
        # Load agricultural data
        with st.spinner("Loading agricultural knowledge base..."):
            documents = load_agricultural_data()
        
        # Initialize RAG system
        with st.spinner("Building vector knowledge base..."):
            rag_system = FarmAIRAG()
            success = rag_system.build_knowledge_base(documents)
            
            if not success:
                return None, None, False
        
        # Initialize agents
        with st.spinner("Initializing AI agents..."):
            agents = FarmAIAgents(rag_system)
        
        return rag_system, agents, True
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        return None, None, False

def display_message(role, content, sources=None):
    """Display chat message with styling"""
    css_class = "user-message" if role == "user" else "assistant-message"
    icon = "🧑‍🌾" if role == "user" else "🤖"
    
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <div style="margin-right: 10px; font-size: 1.2em;">{icon}</div>
        <div style="flex: 1;">
            <strong>{"You" if role == "user" else "FarmAI Assistant"}:</strong><br>
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if sources and role == "assistant" and len(sources) > 0:
        with st.expander("📚 Sources & References"):
            for i, source in enumerate(sources, 1):
                st.markdown(f"""
                <div class="source-box">
                    <strong>Source {i}:</strong> {source.get('title', 'Agricultural Knowledge Base')}<br>
                    <em>Category: {source.get('category', 'General')}</em><br>
                    {source.get('content', '')[:200]}...
                </div>
                """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 FarmAI Knowledge Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Democratizing agricultural knowledge through AI-powered conversational assistance</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Project Info")
        st.info("""
        **Tech Stack:**
        - RAG with LangChain
        - Multi-agent workflow
        - Vector search (FAISS)
        - OpenAI GPT-4
        
        **Social Impact:**
        Democratizing agricultural knowledge for small-scale farmers worldwide.
        """)
        
        st.header("💡 Example Queries")
        example_queries = [
            "My tomato plants have yellow spots and wilting leaves. What should I do?",
            "When is the best time to plant rice in monsoon season?",
            "How to identify and treat wheat rust disease?",
            "Organic pest control methods for vegetables",
            "Climate-smart farming practices for drought conditions"
        ]
        
        for query in example_queries:
            if st.button(f"Ask: {query[:30]}...", key=f"example_{query[:10]}"):
                st.session_state.example_query = query
        
        st.header("ℹ️ How It Works")
        st.markdown("""
        1. **Query Classification**: Determines question type
        2. **Diagnostic Agent**: Identifies problems/diseases  
        3. **Recommendation Agent**: Provides solutions
        4. **Verification Agent**: Ensures safety & accuracy
        """)
    
    # Initialize system
    try:
        rag_system, agents, success = initialize_system()
    except Exception as e:
        st.error(f"Failed to initialize the system: {str(e)}")
        st.info("Please check your API key and internet connection, then refresh the page.")
        return
    
    if not success:
        st.error("Failed to initialize the FarmAI system. Please check your API key and try again.")
        return
    
    st.success("✅ FarmAI system initialized successfully!")
    
    # Chat interface
    st.header("💬 Ask Your Agricultural Questions")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! I'm FarmAI, your agricultural knowledge assistant. I can help you with crop management, disease identification, pest control, and farming best practices. What would you like to know?",
            "sources": []
        })
    
    # Handle example query from sidebar
    if hasattr(st.session_state, 'example_query'):
        prompt = st.session_state.example_query
        delattr(st.session_state, 'example_query')
        
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "sources": []
        })
    
    # Display chat history
    for message in st.session_state.messages:
        display_message(message["role"], message["content"], message.get("sources", []))
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about agriculture..."):
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "sources": []
        })
        
        # Display user message
        display_message("user", prompt)
    
    # Process the latest user message
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        latest_query = st.session_state.messages[-1]["content"]
        
        # Generate response
        with st.spinner("🤖 FarmAI is analyzing your question..."):
            try:
                # Use multi-agent system for processing
                response_data = agents.process_query(latest_query)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_data["response"],
                    "sources": response_data.get("sources", [])
                })
                
                # Display assistant response
                display_message("assistant", response_data["response"], response_data.get("sources", []))
                
                # Rerun to update the display
                st.rerun()
                
            except Exception as e:
                error_msg = f"I apologize, but I encountered an error processing your question: {str(e)}"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })
                display_message("assistant", error_msg)
                st.rerun()

# Footer
st.markdown("---")
st.header("API Connection Test")

if st.button("Run API Test"):
    try:
        with st.spinner("Testing connection to chatanywhere..."):
            # Import the necessary libraries
            from openai import OpenAI
            import httpx

            # Manually create a clean http client to bypass proxy issues
            http_client = httpx.Client(proxies="")

            # Pass the custom client to the OpenAI library
            client = OpenAI(
                api_key=st.secrets["OPENAI_API_KEY"],
                base_url="https://api.chatanywhere.tech/v1",
                http_client=http_client
            )

            # Make the API call
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "Hello"}
                ]
            )

            # Display the result
            st.success("✅ API Connection Successful!")
            st.write("Response from model:")
            st.info(completion.choices[0].message.content)

    except Exception as e:
        st.error(f"❌ API Connection Failed. Error: {e}")
