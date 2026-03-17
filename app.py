import streamlit as st
import nest_asyncio
from groq import Groq

nest_asyncio.apply()

st.set_page_config(page_title="FarmAI Assistant", page_icon="🌾")

# --- Header ---
st.title("🌾 FarmAI Assistant")
st.markdown("""
Welcome to **FarmAI** — your smart farming companion!

I can help you with:
- 🍅 **Tomato** — Blight (Early & Late), Sorting
- 🌾 **Rice** — Stem Borer, Blast
- 🌽 **Maize** — Stem Borer
- 🌿 **Wheat** — Rust

> 💬 Just type your crop problem below and I'll guide you!
""")
st.divider()

# --- Manual Data ---
MANUAL_DOCS = [
    "Tomato Blight (Early and Late): Early blight shows brown spots; late blight causes dark water-soaked lesions. Management: Use certified seeds, crop rotation, and copper-based fungicides.",
    "Rice Stem Borer: Larvae cause 'dead heart' in young plants. Management: Use pheromone traps and avoid excessive nitrogen.",
    "Rice Blast: Management includes nitrogen timing and fungicide protocols.",
    "Maize Stem Borer: Cultural practices include destruction of crop residues to break lifecycle.",
    "Wheat Rust: Surveillance models help predict epidemics. Use resistant cultivars.",
    "Tomato Sorting: High-quality tomatoes must be firm, uniform in color, and free of cracks."
]

def simple_retrieve(query: str, docs: list, k: int = 2) -> str:
    query_words = set(query.lower().split())
    scored = [(len(set(doc.lower().split()) & query_words), doc) for doc in docs]
    scored.sort(reverse=True)
    top = [doc for score, doc in scored[:k] if score > 0]
    return "\n\n".join(top) if top else ""

api_key = st.secrets.get("GROQ_API_KEY")

SYSTEM_PROMPT = """You are FarmAI, a friendly and knowledgeable agricultural assistant.

Your ONLY purpose is to help farmers with crop-related questions such as:
- Crop diseases and pests
- Crop management and treatment
- Harvesting and sorting of farm produce
- General farming advice

RULES:
1. For greetings like "hi", "hello", "hey" — respond warmly and introduce yourself, then ask what crop problem they need help with.
2. If the user asks something NOT related to farming, crops, or agriculture — politely say:
   "I'm FarmAI, designed only to assist with farming and crop-related questions. I'm not built for that topic! 🌾 Ask me about your crops instead."
3. For farming questions — use the provided CONTEXT from the manual if available. If context is empty but the question is farm-related, answer from your general agricultural knowledge.
4. Always be warm, simple, and easy to understand — many users are farmers who need practical advice.
5. Keep answers concise and actionable.
"""

if api_key:
    try:
        client = Groq(api_key=api_key)
        st.success("✅ FarmAI is ready to help you!")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Suggested questions buttons (only show if no chat yet)
        if not st.session_state.messages:
            st.markdown("**Quick questions to get started:**")
            cols = st.columns(2)
            suggestions = [
                "How to treat tomato blight?",
                "What causes rice stem borer?",
                "How to manage wheat rust?",
                "How to sort tomatoes?"
            ]
            for i, suggestion in enumerate(suggestions):
                if cols[i % 2].button(suggestion, use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": suggestion})
                    st.rerun()

        if user_query := st.chat_input("Ask me about your crops..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)

            with st.chat_message("assistant"):
                with st.spinner("FarmAI is thinking..."):
                    context = simple_retrieve(user_query, MANUAL_DOCS)

                    if context:
                        grounded_query = f"CONTEXT FROM MANUAL:\n{context}\n\nFARMER'S QUESTION:\n{user_query}"
                    else:
                        grounded_query = f"FARMER'S QUESTION:\n{user_query}"

                    history = [{"role": "system", "content": SYSTEM_PROMPT}]
                    for m in st.session_state.messages[:-1]:
                        history.append({"role": m["role"], "content": m["content"]})
                    history.append({"role": "user", "content": grounded_query})

                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=history,
                        max_tokens=1024,
                        temperature=0.3
                    )
                    answer = response.choices[0].message.content
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"System Error: {e}")
else:
    st.warning("⚠️ Please add GROQ_API_KEY to your Streamlit Secrets.")
```

**`requirements.txt`** (unchanged):
```
streamlit
groq
nest-asyncio
