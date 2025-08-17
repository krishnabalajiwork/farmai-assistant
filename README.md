# 🌾 FarmAI Knowledge Assistant

**Democratizing agricultural knowledge through AI-powered conversational assistance**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://farmai-assistant.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-krishnabalajiwork-blue)](https://github.com/krishnabalajiwork/farmai-assistant)

## 🎯 Project Overview

FarmAI Knowledge Assistant is an AI-powered conversational system designed to democratize agricultural knowledge for farmers worldwide. Built as a showcase project, it demonstrates advanced RAG (Retrieval-Augmented Generation) pipelines, multi-agent workflows, and AI-native development practices.

### 🌍 Social Impact
- **Problem**: Small-scale farmers lack access to timely, localized agricultural advice
- **Solution**: Multilingual RAG-powered chatbot providing instant access to expert knowledge
- **Impact**: Democratizes agricultural expertise, especially for resource-constrained farmers

## 🚀 Key Features

### 🤖 Advanced AI Capabilities
- **Multi-Agent Workflow**: Query classification, diagnostic, recommendation, and verification agents
- **Advanced RAG Pipeline**: Hybrid retrieval with semantic search and query decomposition
- **Conversational AI**: Context-aware responses with follow-up question handling
- **Source Attribution**: Every answer cites original agricultural documents

### 🧠 Technical Highlights (Atlan-Aligned)
- **AI Experimentation**: Cutting-edge LLM techniques and research implementation
- **Conversational Search**: Advanced RAG with HyDE and self-reflective capabilities
- **AI-Native Development**: Built using AI tools throughout the development process
- **Agentic Systems**: Multi-agent orchestration for complex agricultural problem-solving
- **AI Infrastructure**: Observability, evaluation, and production-ready deployment

## 🛠️ Technology Stack

### Core Framework
- **Frontend**: Streamlit (rapid prototyping and deployment)
- **Backend**: Python with LangChain framework
- **LLM**: OpenAI GPT-4/3.5-turbo for reasoning and generation
- **Vector Store**: FAISS for efficient similarity search
- **Embeddings**: OpenAI text-embedding-ada-002

### AI/ML Technologies
- **RAG Architecture**: Advanced retrieval-augmented generation
- **Multi-Agent System**: Specialized agents for different agricultural domains
- **Prompt Engineering**: Optimized prompts for agricultural expertise
- **Vector Search**: Semantic similarity and hybrid retrieval

## 📁 Project Structure

```
farmai-assistant/
├── app.py                 # Main Streamlit application
├── src/
│   ├── rag_system.py      # RAG implementation with FAISS
│   ├── agents.py          # Multi-agent system (classifier, diagnostic, recommendation, verification)
│   └── data_loader.py     # Agricultural knowledge data loading
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .streamlit/
    └── config.toml       # Streamlit configuration
```

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/krishnabalajiwork/farmai-assistant.git
cd farmai-assistant
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up OpenAI API Key**
```bash
# Option 1: Environment variable
export OPENAI_API_KEY="your-openai-api-key-here"

# Option 2: Enter in the Streamlit sidebar when running
```

4. **Run the application**
```bash
streamlit run app.py
```

### Streamlit Cloud Deployment

1. **Fork this repository** to your GitHub account
2. **Visit [Streamlit Cloud](https://share.streamlit.io/)**
3. **Deploy** using your forked repository
4. **Add your OpenAI API key** in the Streamlit Cloud secrets

## 💡 Usage Examples

### Example Queries
- *"My tomato plants have yellow spots and wilting leaves. What should I do?"*
- *"When is the best time to plant rice in monsoon season?"*
- *"How to identify and treat wheat rust disease?"*
- *"Organic pest control methods for vegetables"*
- *"Climate-smart farming practices for drought conditions"*

### Multi-Agent Workflow
1. **Query Classification**: Determines if query is about disease, pest, management, etc.
2. **Diagnostic Agent**: Analyzes symptoms and provides disease/problem identification
3. **Recommendation Agent**: Provides actionable treatment and prevention advice
4. **Verification Agent**: Reviews and enhances response for safety and completeness

## 🧪 Technical Implementation Details

### RAG System Architecture
```python
# Advanced RAG Pipeline
- Document chunking with overlap for better context
- Semantic embeddings using OpenAI text-embedding-ada-002
- FAISS vector store for efficient similarity search
- Hybrid retrieval combining semantic and keyword search
- Query decomposition for complex questions
- Self-reflective RAG for response quality assurance
```

### Multi-Agent System
```python
# Agent Specialization
QueryClassifierAgent    # Routes queries to appropriate specialists
DiagnosticAgent        # Specializes in agricultural problem identification
RecommendationAgent    # Provides actionable farming advice
VerificationAgent      # Ensures safety and accuracy of responses
```

## 📊 Knowledge Base

The system includes comprehensive agricultural knowledge covering:
- **Crop Diseases**: Identification, symptoms, and treatment
- **Pest Management**: Integrated pest management strategies
- **Soil Health**: Fertility management and soil conservation
- **Climate Adaptation**: Climate-smart agriculture practices
- **Best Practices**: Crop-specific cultivation guidelines

## 🎓 Educational Value (For Atlan Internship)

This project demonstrates key competencies sought by Atlan:

### AI-Native Development
- Built using AI tools for coding, debugging, and testing
- Prompt engineering and LLM optimization
- Research paper implementation (RAG, multi-agent systems)

### Technical Proficiency
- LangChain framework expertise
- Vector database implementation (FAISS)
- Production-ready Python application
- API integration and error handling

### Problem-Solving Approach
- Real-world problem identification and solution design
- Experimental mindset with multiple AI techniques
- Scalable architecture for production deployment

### Social Impact Focus
- Addresses UN SDG goals (Zero Hunger, Sustainable Agriculture)
- Designed for accessibility and multilingual support
- Potential for real-world deployment and impact

## 🔮 Future Enhancements

### Technical Roadmap
- [ ] **Multi-modal Support**: Image-based crop disease identification
- [ ] **Multilingual Capabilities**: Support for local languages
- [ ] **Real-time Updates**: Integration with agricultural research databases
- [ ] **Mobile App**: React Native or Flutter mobile interface
- [ ] **Offline Mode**: Edge deployment for areas with poor connectivity

### AI/ML Improvements
- [ ] **Fine-tuned Models**: Domain-specific LLM fine-tuning
- [ ] **Advanced RAG**: Graph RAG and hierarchical retrieval
- [ ] **Agent Improvements**: More specialized agricultural agents
- [ ] **Evaluation Framework**: Automated response quality assessment

## 🤝 Contributing

Contributions are welcome! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Agricultural Experts**: Knowledge base compiled from various agricultural institutions
- **Open Source Community**: LangChain, Streamlit, and other amazing tools
- **Farmers Worldwide**: The ultimate inspiration for democratizing agricultural knowledge

## 📞 Contact

**Chintha Krishna Balaji**
- 📧 Email: krishnabalajiwork@gmail.com
- 💼 LinkedIn: [chintha-krishna-balaji](https://www.linkedin.com/in/chintha-krishna-balaji)
- 🐱 GitHub: [krishnabalajiwork](https://github.com/krishnabalajiwork)

---

*Built with ❤️ for farmers and the Atlan AI Engineering Internship*
