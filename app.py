import streamlit as st
import nest_asyncio
from groq import Groq

nest_asyncio.apply()

st.set_page_config(page_title="FarmAI Assistant", page_icon="🌾")
st.title("🌾 FarmAI Grounded Assistant (Groq)")

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
    return "\n\n".join(top) if top else "No relevant manual entries found."

api_key = st.secrets.get("GROQ_API_KEY")

SYSTEM_PROMPT = """You are a specialized Agricultural Assistant.
You MUST ONLY use the provided manual context to answer.
If the answer is NOT in the context, say: "I'm sorry, my manual does not contain information on that topic."
"""

if api_key:
    try:
        client = Groq(api_key=api_key)
        st.success("✅ Groq Knowledge Base Active!")

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
                with st.spinner("Analyzing manual with Groq..."):
                    context = simple_retrieve(user_query, MANUAL_DOCS)
                    grounded_query = f"CONTEXT:\n{context}\n\nQUESTION:\n{user_query}"

                    # Build full message history for context
                    history = [{"role": "system", "content": SYSTEM_PROMPT}]
                    for m in st.session_state.messages[:-1]:
                        history.append({"role": m["role"], "content": m["content"]})
                    history.append({"role": "user", "content": grounded_query})

                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",  # Free, fast, powerful
                        messages=history,
                        max_tokens=1024,
                        temperature=0
                    )
                    answer = response.choices[0].message.content
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"System Error: {e}")
else:
    st.warning("Please add GROQ_API_KEY to your Streamlit Secrets.")
