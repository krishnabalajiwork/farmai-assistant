import streamlit as st
import os
import sys

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "src"))

try:
    from rag_system import FarmAIRAG
    from agents import FarmAIAgents
    from data_loader import load_agricultural_data
except ImportError as e:
    st.error(f"Error importing modules: {e}. Please ensure src folder and its files exist.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="FarmAI Knowledge Assistant",
    page_icon="🌾",
    layout="wide"
)

# Initialize system
@st.cache_resource
def initialize_system():
    """Initialize the RAG system and agents"""
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return None, False

    try:
        documents = load_agricultural_data()
        rag_system = FarmAIRAG(api_key=api_key)
        success = rag_system.build_knowledge_base(documents)
        
        if not success:
            return None, False
        
        agents = FarmAIAgents(rag_system, api_key=api_key)
        return agents, True
    except Exception as e:
        st.error(f"Error during initialization: {e}")
        return None, False

agents, success = initialize_system()

# Main app UI
st.markdown('<h1 style="text-align: center; color: #2E8B57;">🌾 FarmAI Knowledge Assistant</h1>', unsafe_allow_html=True)

if not success:
    st.error("Failed to initialize the FarmAI system. Please ensure your OpenAI API key is correctly set in Streamlit Secrets and refresh the page.")
else:
    st.success("✅ FarmAI system initialized successfully!")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm FarmAI. How can I help with your farming questions today?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about agriculture..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("FarmAI is thinking..."):
                response_data = agents.process_query(prompt)
                response = response_data.get("response", "Sorry, I couldn't process that.")
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
