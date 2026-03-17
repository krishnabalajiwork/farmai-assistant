import streamlit as st
import nest_asyncio
from anthropic import Anthropic
import numpy as np

nest_asyncio.apply()

st.set_page_config(page_title="FarmAI Assistant", page_icon="🌾")
st.title("🌾 FarmAI Grounded Assistant (Claude)")

# --- Agricultural Manual Data ---
MANUAL_DOCS = [
    "Tomato Blight (Early and Late): Early blight shows brown spots; late blight causes dark water-soaked lesions. Management: Use certified seeds, crop rotation, and copper-based fungicides.",
    "Rice Stem Borer: Larvae cause 'dead heart' in young plants. Management: Use pheromone traps and avoid excessive nitrogen.",
    "Rice Blast: Management includes nitrogen timing and fungicide protocols.",
    "Maize Stem Borer: Cultural practices include destruction of crop residues to break lifecycle.",
    "Wheat Rust: Surveillance models help predict epidemics. Use resistant cultivars.",
    "Tomato Sorting: High-quality tomatoes must be firm, uniform in color, and free of cracks."
]

def simple_retrieve(query: str, docs: list, k: int = 2) -> str:
    """Keyword-based retrieval (no embeddings needed)."""
    query_words = set(query.lower().split())
    scored = []
    for doc in docs:
        doc_words = set(doc.lower().split())
        score = len(query_words & doc_words)
        scored.append((score, doc))
    scored.sort(reverse=True)
    top = [doc for _, doc in scored[:k] if _ > 0]
    return "\n\n".join(top) if top else "No relevant manual entries found."

api_key = st.secrets.get("ANTHROPIC_API_KEY")

if api_key:
    try:
        client = Anthropic(api_key=api_key)
        st.success("✅ Claude Knowledge Base Active!")

        SYSTEM_PROMPT = """You are a specialized Agricultural Assistant.
You MUST ONLY use the provided manual context to answer.
If the answer is NOT in the context, say: "I'm sorry, my manual does not contain information on that topic."
"""

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if user_query := st.chat_input("Ask about your crops..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing manual with Claude..."):
                    context = simple_retrieve(user_query, MANUAL_DOCS)
                    
                    # Build messages with context injected into latest user message
                    history = st.session_state.messages[:-1]  # all but last
                    grounded_query = f"CONTEXT:\n{context}\n\nQUESTION:\n{user_query}"
                    
                    api_messages = [
                        *[{"role": m["role"], "content": m["content"]} for m in history],
                        {"role": "user", "content": grounded_query}
                    ]

                    response = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=1024,
                        system=SYSTEM_PROMPT,
                        messages=api_messages
                    )
                    answer = response.content[0].text
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"System Error: {e}")
else:
    st.warning("Please add ANTHROPIC_API_KEY to your Streamlit Secrets.")
