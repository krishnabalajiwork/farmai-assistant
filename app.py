import streamlit as st
import nest_asyncio
from groq import Groq

nest_asyncio.apply()

st.set_page_config(page_title="FarmAI Assistant", page_icon="🌾")

st.title("🌾 FarmAI Assistant")
st.markdown("""
Welcome to **FarmAI** — your smart farming companion!

I can only help you with:
- 🍅 **Tomato** — Blight (Early & Late), Sorting
- 🌾 **Rice** — Stem Borer, Blast
- 🌽 **Maize** — Stem Borer
- 🌿 **Wheat** — Rust

> 💬 Ask me anything about the above crops!
""")
st.divider()

# --- YOUR MANUAL DOCS (only these will be used to answer) ---
MANUAL_DOCS = [
    "Tomato Blight (Early and Late): Early blight shows brown spots; late blight causes dark water-soaked lesions. Management: Use certified seeds, crop rotation, and copper-based fungicides.",
    "Rice Stem Borer: Larvae cause 'dead heart' in young plants. Management: Use pheromone traps and avoid excessive nitrogen.",
    "Rice Blast: Management includes nitrogen timing and fungicide protocols.",
    "Maize Stem Borer: Cultural practices include destruction of crop residues to break lifecycle.",
    "Wheat Rust: Surveillance models help predict epidemics. Use resistant cultivars.",
    "Tomato Sorting: High-quality tomatoes must be firm, uniform in color, and free of cracks."
]

GREETINGS = ["hi", "hello", "hey", "hii", "helo", "sup", "whats up", "what's up", "howdy"]

def simple_retrieve(query: str, docs: list, k: int = 2) -> str:
    query_words = set(query.lower().split())
    scored = [(len(set(doc.lower().split()) & query_words), doc) for doc in docs]
    scored.sort(reverse=True)
    top = [doc for score, doc in scored[:k] if score > 0]
    return "\n\n".join(top) if top else ""

api_key = st.secrets.get("GROQ_API_KEY")

SYSTEM_PROMPT = """You are FarmAI, a strict agricultural assistant.

STRICT RULES — follow these exactly, no exceptions:

1. GREETING: If the user says hi/hello/hey or any greeting, reply warmly:
   "Hello! 👋 I'm FarmAI, your farming assistant. I can help you with:
   - 🍅 Tomato Blight and Sorting
   - 🌾 Rice Stem Borer and Blast
   - 🌽 Maize Stem Borer
   - 🌿 Wheat Rust
   What would you like to know?"

2. CONTEXT PROVIDED: If CONTEXT is provided, answer ONLY using that context word-for-word. Do not add any extra information, tips, or knowledge outside the context.

3. NO CONTEXT: If no CONTEXT is provided, it means the topic is not in the manual. Reply exactly:
   "I'm sorry, that topic is not in my manual. I can only help with Tomato Blight, Tomato Sorting, Rice Stem Borer, Rice Blast, Maize Stem Borer, and Wheat Rust. 🌾"

4. OFF-TOPIC: If the question is not about farming at all, reply exactly:
   "I'm FarmAI, built only for farming questions. I'm not designed for that topic! 🌾 Ask me about your crops instead."

NEVER use your own knowledge. ONLY use what is in the CONTEXT provided.
"""

if api_key:
    try:
        client = Groq(api_key=api_key)
        st.success("✅ FarmAI is ready!")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Quick suggestion buttons
        if not st.session_state.messages:
            st.markdown("**Try asking:**")
            cols = st.columns(2)
            suggestions = [
                "How to treat tomato blight?",
                "What is rice stem borer?",
                "How to manage wheat rust?",
                "How to sort tomatoes?"
            ]
            for i, s in enumerate(suggestions):
                if cols[i % 2].button(s, use_container_width=True, key=f"btn_{i}"):
                    st.session_state.messages.append({"role": "user", "content": s})
                    st.rerun()

        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Auto-answer if last message is from user with no reply yet
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            last_query = st.session_state.messages[-1]["content"]
            
            with st.chat_message("assistant"):
                with st.spinner("FarmAI is thinking..."):

                    # Check if greeting
                    if last_query.strip().lower() in GREETINGS:
                        answer = """Hello! 👋 I'm FarmAI, your farming assistant. I can help you with:
- 🍅 Tomato Blight and Sorting
- 🌾 Rice Stem Borer and Blast
- 🌽 Maize Stem Borer
- 🌿 Wheat Rust

What would you like to know?"""
                    else:
                        context = simple_retrieve(last_query, MANUAL_DOCS)

                        if context:
                            user_msg = f"CONTEXT:\n{context}\n\nQUESTION:\n{last_query}"
                        else:
                            user_msg = f"NO CONTEXT AVAILABLE.\n\nQUESTION:\n{last_query}"

                        history = [{"role": "system", "content": SYSTEM_PROMPT}]
                        for m in st.session_state.messages[:-1]:
                            history.append({"role": m["role"], "content": m["content"]})
                        history.append({"role": "user", "content": user_msg})

                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=history,
                            max_tokens=512,
                            temperature=0  # Zero temperature = strict, no creativity
                        )
                        answer = response.choices[0].message.content

                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.rerun()

        if user_query := st.chat_input("Ask me about your crops..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.rerun()

    except Exception as e:
        st.error(f"System Error: {e}")
else:
    st.warning("⚠️ Please add GROQ_API_KEY to your Streamlit Secrets.")
