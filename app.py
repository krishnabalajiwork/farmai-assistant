import streamlit as st
import nest_asyncio
import google.generativeai as genai

nest_asyncio.apply()

st.set_page_config(page_title="FarmAI Assistant", page_icon="🌾")
st.title("🌾 FarmAI Grounded Assistant (Gemini)")

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
    """Keyword-based retrieval — no embeddings needed."""
    query_words = set(query.lower().split())
    scored = [(len(set(doc.lower().split()) & query_words), doc) for doc in docs]
    scored.sort(reverse=True)
    top = [doc for score, doc in scored[:k] if score > 0]
    return "\n\n".join(top) if top else "No relevant manual entries found."

api_key = st.secrets.get("GEMINI_API_KEY")

SYSTEM_PROMPT = """You are a specialized Agricultural Assistant.
You MUST ONLY use the provided manual context to answer.
If the answer is NOT in the context, say: "I'm sorry, my manual does not contain information on that topic."
"""

if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="models/gemini-2.0-flash",   # full path format
            system_instruction=SYSTEM_PROMPT
)
        )

        st.success("✅ Gemini Knowledge Base Active!")

        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "chat" not in st.session_state:
            st.session_state.chat = model.start_chat(history=[])

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if user_query := st.chat_input("Ask about your crops..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing manual with Gemini..."):
                    context = simple_retrieve(user_query, MANUAL_DOCS)
                    grounded_query = f"CONTEXT:\n{context}\n\nQUESTION:\n{user_query}"

                    response = st.session_state.chat.send_message(grounded_query)
                    answer = response.text

                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"System Error: {e}")
else:
    st.warning("Please add GEMINI_API_KEY to your Streamlit Secrets.")
