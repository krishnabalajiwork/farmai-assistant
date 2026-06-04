import streamlit as st
from groq import Groq

st.set_page_config(
    page_title="FarmAI Assistant",
    page_icon="🌾"
)

st.title("🌾 FarmAI Assistant")

st.markdown("""
Welcome to **FarmAI** — your smart farming companion!

I can only help you with:

- 🍅 Tomato — Blight (Early & Late), Sorting
- 🌾 Rice — Stem Borer, Blast
- 🌽 Maize — Stem Borer
- 🌿 Wheat — Rust

💬 Ask me anything about the above crops!
""")

st.divider()

MANUAL_DOCS = [
    "Tomato Blight (Early and Late): Early blight shows brown spots; late blight causes dark water-soaked lesions. Management: Use certified seeds, crop rotation, and copper-based fungicides.",
    "Rice Stem Borer: Larvae cause dead heart in young plants. Management: Use pheromone traps and avoid excessive nitrogen.",
    "Rice Blast: Management includes nitrogen timing and fungicide protocols.",
    "Maize Stem Borer: Cultural practices include destruction of crop residues to break lifecycle.",
    "Wheat Rust: Surveillance models help predict epidemics. Use resistant cultivars.",
    "Tomato Sorting: High-quality tomatoes must be firm, uniform in color, and free of cracks."
]

GREETINGS = [
    "hi",
    "hello",
    "hey",
    "hii",
    "helo",
    "howdy",
    "sup",
    "whats up",
    "what's up"
]


def simple_retrieve(query, docs, k=2):
    query_words = set(query.lower().split())

    scored = []

    for doc in docs:
        score = len(set(doc.lower().split()) & query_words)
        scored.append((score, doc))

    scored.sort(reverse=True)

    top_docs = [doc for score, doc in scored[:k] if score > 0]

    return "\n\n".join(top_docs) if top_docs else ""


SYSTEM_PROMPT = """
You are FarmAI.

RULES:

1. If the user greets you, greet them.

2. If CONTEXT is provided, answer ONLY from that context.

3. If no context is available, reply exactly:

I'm sorry, that topic is not in my manual. I can only help with Tomato Blight, Tomato Sorting, Rice Stem Borer, Rice Blast, Maize Stem Borer, and Wheat Rust. 🌾

4. If off-topic, reply exactly:

I'm FarmAI, built only for farming questions. I'm not designed for that topic! 🌾 Ask me about your crops instead.
"""

try:

    api_key = st.secrets["GROQ_API_KEY"]

    client = Groq(api_key=api_key)

    st.success("✅ FarmAI is ready!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_query = st.chat_input("Ask me about your crops...")

    if user_query:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_query
            }
        )

        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    if user_query.lower().strip() in GREETINGS:

                        answer = """
Hello! 👋 I'm FarmAI.

I can help with:

🍅 Tomato Blight & Sorting
🌾 Rice Stem Borer & Blast
🌽 Maize Stem Borer
🌿 Wheat Rust

What would you like to know?
"""

                    else:

                        context = simple_retrieve(
                            user_query,
                            MANUAL_DOCS
                        )

                        if context:

                            prompt = f"""
CONTEXT:
{context}

QUESTION:
{user_query}
"""

                        else:

                            prompt = f"""
NO CONTEXT AVAILABLE

QUESTION:
{user_query}
"""

                        st.info("Calling Groq API...")

                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {
                                    "role": "system",
                                    "content": SYSTEM_PROMPT
                                },
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],
                            temperature=0,
                            max_tokens=512
                        )

                        st.success("Groq API replied!")

                        answer = response.choices[0].message.content

                    st.write(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as e:
                    st.error(f"Groq Error: {str(e)}")

except KeyError:

    st.error("""
GROQ_API_KEY not found.

Add this in Streamlit Secrets:

GROQ_API_KEY = "your_groq_api_key"
""")

except Exception as e:

    st.error(f"Startup Error: {str(e)}")
```
