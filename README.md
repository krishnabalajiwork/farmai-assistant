# 🌾 FarmAI Knowledge Assistant — AgTech RAG System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://farmai-assistant.streamlit.app/)
[![Tech Stack](https://img.shields.io/badge/Stack-LangChain_%7C_Groq_%7C_FAISS-green?style=for-the-badge)](https://github.com/krishnabalajiwork/farmai-assistant)

> **Developer & Architecture Documentation**  
> An interactive Retrieval-Augmented Generation (RAG) assistant designed to democratize domain-specific agricultural knowledge. Built using LangChain, vector similarity search via FAISS, and high-speed LLM inference.

---

## 🏗️ System Architecture & RAG Pipeline Flow

FarmAI Assistant transforms unstructured agricultural documentation into actionable insights through an automated vector search and context-synthesis pipeline:

```text
[ User Query ]
       │
       ▼
[ Query Embedding Engine ] ──> (Semantic Vector Mapping)
                                         │
                                         ▼
                             [ FAISS Vector Database ]
                                         │
                               (Context Retrieval)
                                         ▼
[ High-Speed LLM Inference (Groq / Gemini) ] ──> (Context-Grounded Response)
                                                               │
                                                               ▼
                                                     [ Streamlit UI Output ]

```

---

## ⚡ Key Capabilities

* **Retrieval-Augmented Generation (RAG):** Contextually grounds LLM responses using pre-indexed domain documentation on crop diseases, pest controls, and yield practices.
* **Low-Latency Inference:** Integrated with Groq API / Gemini acceleration for real-time streaming chat interactions.
* **Strict Source Attribution:** Limits hallucinations by forcing answer generation to draw directly from retrieved context chunks.
* **Asynchronous Execution:** Handles Streamlit event-loop constraints using `nest-asyncio` for non-blocking UI updates.

---

## 🌾 Supported Knowledge Base Domains

The assistant currently provides contextual guidance on key crops and common agricultural challenges:

* **🍅 Tomato:** Early Blight, Late Blight, Sorting & Quality Inspection
* **🌾 Rice:** Stem Borer, Blast Disease
* **🌽 Maize:** Stem Borer Control
* **🌾 Wheat:** Rust Identification & Treatment
* **🧪 Soil & Crop Management:** Organic Pest Management & Season-Specific Planting Guidelines

---

## 🔌 Technology Stack

| Layer | Component / Tool | Function |
| --- | --- | --- |
| **Frontend UI** | Streamlit | Lightweight reactive web interface |
| **Orchestration** | Python + LangChain Framework | Document splitting, embedding pipelines, and chain logic |
| **Vector Store** | FAISS (Facebook AI Similarity Search) | High-performance vector index for similarity queries |
| **LLM Inference** | Groq API / Google Gemini | Rapid contextual reasoning and query response synthesis |
| **Async Handling** | `nest-asyncio` | Solves nested event loop conflicts in Streamlit execution |

---

## 📂 Repository Structure

```text
farmai-assistant/
 ├── app.py                # Main Streamlit UI & RAG pipeline execution logic
 ├── requirements.txt      # Python runtime dependencies
 ├── README.md             # Technical documentation
 └── .streamlit/
      └── config.toml      # UI styling & server runtime parameters

```

---

## ⚙️ Environment Configuration

Create a `.env` file in your root project directory (or configure secrets in Streamlit Cloud):

```env
# Groq API Configuration
GROQ_API_KEY="your_groq_api_key"

# Google Gemini API Configuration (Fallback/Alternative)
GOOGLE_API_KEY="your_google_api_key"

```

---

## 🚀 Quickstart & Local Setup

### Prerequisites

* **Python:** v3.10 or higher
* **API Key:** Groq API Key or Google AI Studio API Key

### 1. Clone & Install

```bash
git clone [https://github.com/krishnabalajiwork/farmai-assistant.git](https://github.com/krishnabalajiwork/farmai-assistant.git)
cd farmai-assistant
pip install -r requirements.txt

```

### 2. Export API Key

```bash
# For macOS / Linux:
export GROQ_API_KEY="your-groq-api-key-here"

# For Windows PowerShell:
$env:GROQ_API_KEY="your-groq-api-key-here"

```

### 3. Launch App

```bash
streamlit run app.py

```

Access the local development server at `http://localhost:8501`.

---

## 🐛 Troubleshooting & Known Fixes

#### 1. Asynchronous Event Loop Errors (`RuntimeError: This event loop is already running`)

* **Cause:** Streamlit's execution thread collides with standard async loops inside LangChain.
* **Solution:** Ensure `nest-asyncio` is initialized at the very top of `app.py`:
```python
import nest_asyncio
nest_asyncio.apply()

```



#### 2. Vector Store Memory Limits on Cloud Deployment

* **Cause:** Large document chunk embeddings overloading Streamlit Cloud memory limits.
* **Solution:** Pre-index documents into lightweight FAISS indices and load binary vector indexes statically at boot time.

---

## 👨‍💻 Author & Contact

**Chintha Krishna Balaji**

* **GitHub:** [@krishnabalajiwork](https://github.com/krishnabalajiwork)
* **LinkedIn:** [chintha-krishna-balaji](https://www.linkedin.com/in/chintha-krishna-balaji)
* **Live App:** [farmai-assistant.streamlit.app](https://farmai-assistant.streamlit.app/)

---

## 📝 License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).

```

```
