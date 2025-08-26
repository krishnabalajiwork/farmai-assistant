# 🌾 FarmAI Knowledge Assistant

**Democratizing agricultural knowledge through AI-powered conversational assistance**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://farmai-assistant.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-krishnabalajiwork-blue)](https://github.com/krishnabalajiwork/farmai-assistant)

## 🎯 Project Overview

FarmAI Knowledge Assistant is an AI-powered conversational system designed to democratize agricultural knowledge for farmers worldwide. Built as a showcase project, it demonstrates a robust RAG (Retrieval-Augmented Generation) pipeline using modern, reliable, and free-to-start AI services.
https://farmai-assistant.streamlit.app/

### 🌍 Social Impact
- **Problem**: Small-scale farmers lack access to timely, localized agricultural advice.
- **Solution**: A RAG-powered chatbot providing instant access to expert knowledge.
- **Impact**: Democratizes agricultural expertise, especially for resource-constrained farmers.

## 🚀 Key Features

### 🤖 AI Capabilities
- **Retrieval-Augmented Generation (RAG)**: Answers questions by finding relevant information from a built-in agricultural knowledge base.
- **Conversational AI**: Provides context-aware responses in a natural, chat-like interface.
- **Source Attribution**: Every answer is based on the original agricultural documents.

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

## 📁 Project Structure

The project has been simplified to a single-file architecture for stability and ease of deployment.
