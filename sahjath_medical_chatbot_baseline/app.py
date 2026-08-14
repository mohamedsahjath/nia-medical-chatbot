from pathlib import Path

import streamlit as st

from src.chatbot import MedicalChatbot


BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="AI Medical Chatbot", page_icon="🩺", layout="centered")


@st.cache_resource
def load_chatbot() -> MedicalChatbot:
    return MedicalChatbot(BASE_DIR / "data" / "medical_knowledge_base.csv")


bot = load_chatbot()

st.title("🩺 AI-Based Medical Chatbot")
st.caption("NLP-powered general healthcare information assistant")
st.warning(
    "Educational information only. This chatbot does not diagnose illness, prescribe treatment, "
    "or replace a doctor. For an emergency, contact local emergency services."
)

with st.sidebar:
    st.header("About")
    st.write("The system uses text preprocessing, TF-IDF, intent classification and knowledge retrieval.")
    st.subheader("Example questions")
    st.markdown(
        "- What are the symptoms of diabetes?\n"
        "- How can I prevent dengue?\n"
        "- What is high blood pressure?\n"
        "- Can I take antibiotics for a cold?"
    )
    st.info("Sri Lanka ambulance service: **1990 Suwa Seriya**")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me a general healthcare question."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask a healthcare question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    result = bot.respond(question)
    content = result.answer
    if result.source and not result.emergency:
        content += f"\n\n[Trusted information source]({result.source})"
    with st.chat_message("assistant"):
        st.markdown(content)
        if not result.emergency:
            st.caption(f"Detected topic: {result.intent.replace('_', ' ').title()} · Confidence: {result.confidence:.0%}")
    st.session_state.messages.append({"role": "assistant", "content": content})

