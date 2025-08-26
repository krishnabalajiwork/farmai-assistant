# 🌾 FarmAI Knowledge Assistant

**Democratizing agricultural knowledge through AI-powered conversational assistance**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://farmai-assistant.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-krishnabalajiwork-blue)](https://github.com/krishnabalajiwork/farmai-assistant)

---

## 🎯 Project Overview

FarmAI Knowledge Assistant is an AI-powered conversational system designed to make agricultural knowledge accessible and engaging for students, hobbyists, and young people interested in pursuing farming. Built as a showcase project, it demonstrates a robust RAG (Retrieval-Augmented Generation) pipeline using modern, reliable, and free-to-start AI services. ---
https://farmai-assistant.streamlit.app/

### 🌍 Social Impact
- **Problem**: Young people and aspiring farmers often face a steep learning curve and lack modern, engaging resources to get started in agriculture.
- **Solution**: A RAG-powered chatbot providing instant access to foundational knowledge.
- **Impact**: Inspires and empowers the next generation of agricultural enthusiasts by providing an easy-to-use, AI-powered educational tool.

---

## 🚀 Key Features

### 🤖 AI Capabilities
- **Retrieval-Augmented Generation (RAG)**: Answers questions by finding relevant information from a built-in agricultural knowledge base.
- **Conversational AI**: Provides context-aware responses in a natural, chat-like interface.
- **Source Attribution**: Every answer is based on the original agricultural documents.

---

## 🛠️ Technology Stack

### Core Framework
- **Frontend**: Streamlit
- **Backend**: Python with the LangChain framework
- **LLM**: Google Gemini (`gemini-1.5-flash-latest`)
- **Vector Store**: FAISS for efficient similarity search
- **Embeddings**: Google Generative AI Embeddings (`models/embedding-001`)

### Key Libraries
- **`langchain-google-genai`**: For integration with the Gemini API.
- **`langchain-community`**: Provides access to community components like FAISS.
- **`nest-asyncio`**: Solves asynchronous event loop issues within the Streamlit environment.

---

## 📁 Project Structure

The project has been simplified to a single-file architecture for stability and ease of deployment.

farmai-assistant/
├── app.py                 # Main Streamlit application with all logic -
├── requirements.txt       # Python dependencies -
├── README.md              # This file -
└── .streamlit/ -
└── config.toml        # Streamlit configuration -

---

## 🚀 Quick Start

### Local Development

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/krishnabalajiwork/farmai-assistant.git](https://github.com/krishnabalajiwork/farmai-assistant.git)
    cd farmai-assistant
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Google API Key**
    ```bash
    # Get your free API key from Google AI Studio (aistudio.google.com)
    export GOOGLE_API_KEY="your-google-api-key-here"
    ```

4.  **Run the application**
    ```bash
    streamlit run app.py
    ```

### Streamlit Cloud Deployment

1.  **Fork this repository** to your GitHub account.
2.  **Visit [Google AI Studio](https://aistudio.google.com/)** to create a free API key.
3.  **Visit [Streamlit Cloud](https://share.streamlit.io/)** and deploy your forked repository.
4.  In the app's **Settings -> Secrets**, add your Google API key:
    ```toml
    GOOGLE_API_KEY = "your-google-api-key-here"
    ```

---

## 💡 Usage Examples

### Example Queries
- *"My tomato plants have yellow spots and wilting leaves. What should I do?"*
- *"When is the best time to plant rice in monsoon season?"*
- *"Organic pest control methods for vegetables"*

### RAG Workflow
1.  **User Query**: A student or enthusiast asks a question about a crop disease.
2.  **Document Retrieval**: The system searches its vector knowledge base to find the most relevant agricultural guides.
3.  **Answer Synthesis**: The Gemini model receives the user's question and the retrieved documents, then generates a comprehensive, helpful answer based on the provided context.

---

## 📊 Knowledge Base
The system includes agricultural knowledge covering:
- **Crop Diseases**: Identification, symptoms, and treatment.
- **Pest Management**: Integrated pest management strategies.
- **Best Practices**: Crop-specific cultivation guidelines.

---

## 🎓 Educational Value
This project is an excellent case study in:
- **Pragmatic AI Development**: Demonstrates pivoting from one API (OpenAI) to another (Google Gemini) to solve real-world compatibility and access issues.
- **Resilient RAG Architecture**: The core RAG pipeline is robust and can be adapted to work with different LLMs and embedding models.
- **Environment-Specific Debugging**: Shows how to solve common deployment issues like asynchronous event loops (`nest-asyncio`) and dependency management (`langchain-community`).

---

## 📞 Contact
**Chintha Krishna Balaji**
- 📧 Email: krishnabalajiwork@gmail.com
- 💼 LinkedIn: [chintha-krishna-balaji](https://www.linkedin.com/in/chintha-krishna-balaji)
- 🐱 GitHub: [krishnabalajiwork](https://github.com/krishnabalajiwork)
