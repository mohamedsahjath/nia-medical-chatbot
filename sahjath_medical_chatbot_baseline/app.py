from pathlib import Path

import streamlit as st

from src.chatbot import MedicalChatbot


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "medical_knowledge_base.csv"


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Medical Chatbot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       MAIN BACKGROUND
       ===================================================== */

    .stApp {
        background: linear-gradient(
            135deg,
            #03152B 0%,
            #064B73 55%,
            #0B88B5 100%
        );
        min-height: 100vh;
    }


    /* =====================================================
       MAIN CONTENT
       ===================================================== */

    .main .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }


    /* =====================================================
       ALL MAIN TEXT = WHITE
       ===================================================== */

    .main .block-container p,
    .main .block-container span,
    .main .block-container label,
    .main .block-container li,
    .main .block-container div {
        color: #FFFFFF !important;
    }


    /* =====================================================
       HEADINGS
       ===================================================== */

    h1,
    h2,
    h3,
    h4,
    h5,
    h6 {
        color: #FFFFFF !important;
    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #021329 0%,
            #064A70 100%
        ) !important;
    }


    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }


    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] label {
        color: #FFFFFF !important;
        line-height: 1.7 !important;
    }


    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }


    /* =====================================================
       SIDEBAR INFO
       ===================================================== */

    section[data-testid="stSidebar"]
    [data-testid="stAlert"] {

        background: rgba(255,255,255,0.10) !important;
        border: 1px solid rgba(255,255,255,0.20) !important;
    }


    section[data-testid="stSidebar"]
    [data-testid="stAlert"] * {
        color: #FFFFFF !important;
    }


    /* =====================================================
       CHAT MESSAGES
       ===================================================== */

    [data-testid="stChatMessage"] {
        border-radius: 16px !important;
        margin-bottom: 12px !important;
    }


    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div {
        color: #FFFFFF !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
    }


    /* =====================================================
       CHAT INPUT
       ===================================================== */

    [data-testid="stChatInput"] {
        background: #FFFFFF !important;
        border: 2px solid #168ACB !important;
        border-radius: 16px !important;
    }


    [data-testid="stChatInput"] textarea {
        background: #FFFFFF !important;
        color: #062B4C !important;
        font-size: 16px !important;
        font-weight: 500 !important;
    }


    [data-testid="stChatInput"] textarea::placeholder {
        color: #31566B !important;
        opacity: 1 !important;
    }


    /* =====================================================
       BUTTON
       ===================================================== */

    .stButton > button {
        width: 100% !important;
        background: #168ACB !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px !important;
        font-weight: 700 !important;
    }


    .stButton > button * {
        color: #FFFFFF !important;
    }


    .stButton > button:hover {
        background: #0B6F9E !important;
        color: #FFFFFF !important;
    }


    /* =====================================================
       ALERT / WARNING
       ===================================================== */

    [data-testid="stAlert"] {
        border-radius: 15px !important;
    }


    [data-testid="stAlert"] * {
        color: #FFFFFF !important;
    }


    /* =====================================================
       CAPTION
       ===================================================== */

    [data-testid="stCaptionContainer"] {
        color: #FFFFFF !important;
    }


    [data-testid="stCaptionContainer"] * {
        color: #FFFFFF !important;
    }


    /* =====================================================
       DIVIDER
       ===================================================== */

    hr {
        border-color: rgba(255,255,255,0.35) !important;
    }


    /* =====================================================
       MARKDOWN
       ===================================================== */

    .main .block-container strong,
    .main .block-container b {
        color: #FFFFFF !important;
    }


    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD CHATBOT
# =========================================================

@st.cache_resource
def load_chatbot():
    return MedicalChatbot(DATA_PATH)


bot = load_chatbot()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🩺 Medical Assistant")

    st.subheader("📖 About")

    st.write(
        "This AI-based medical chatbot provides "
        "general healthcare information using:"
    )

    st.markdown(
        """
        • Natural Language Processing  
        • TF-IDF  
        • Logistic Regression  
        • Genetic Algorithm  
        • Medical Knowledge Base
        """
    )

    st.divider()

    st.subheader("💡 Example Questions")

    st.markdown(
        """
        • What are the symptoms of diabetes?

        • What are the symptoms of malaria?

        • What is high blood pressure?

        • How can I prevent flu?

        • What causes a sore throat?

        • What should I do for a headache?
        """
    )

    st.divider()

    st.subheader("🚨 Emergency")

    st.write(
        "If you have severe or life-threatening "
        "symptoms, seek emergency medical help."
    )

    st.info("🇱🇰 Sri Lanka Ambulance: 1990")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! 👋 I'm your AI Medical Assistant.\n\n"
                    "You can ask me about symptoms, prevention, "
                    "and general healthcare information."
                ),
            }
        ]

        st.rerun()


# =========================================================
# HEADER
# =========================================================

st.title("🩺 AI-Based Medical Chatbot")

st.subheader(
    "General Healthcare Information Assistant"
)

st.write(
    "NLP-powered general healthcare information assistant"
)


# =========================================================
# WELCOME
# =========================================================

st.info(
    """
    👋 **Welcome to your AI Medical Assistant**

    Ask questions about common symptoms,
    prevention, and general healthcare information.

    💬 Type your healthcare question in the chat box below
    to get started.
    """
)


# =========================================================
# MEDICAL DISCLAIMER
# =========================================================

st.warning(
    """
    ⚠️ **Medical Disclaimer**

    This chatbot provides general educational information only.

    It does not diagnose diseases, prescribe medicines,
    or replace professional medical advice.

    🚨 For emergencies, contact emergency medical services
    immediately.
    """
)


# =========================================================
# CHAT SESSION
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 I'm your AI Medical Assistant.\n\n"
                "You can ask me about symptoms, prevention, "
                "and general healthcare information.\n\n"
                "💬 Please type your question below."
            ),
        }
    ]


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "💬 Type your healthcare question here..."
)


# =========================================================
# PROCESS QUESTION
# =========================================================

if question:

    # -----------------------------------------------------
    # USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.write(question)


    # -----------------------------------------------------
    # CHATBOT RESPONSE
    # -----------------------------------------------------

    result = bot.respond(question)

    content = result.answer


    # -----------------------------------------------------
    # TRUSTED SOURCE
    # -----------------------------------------------------

    if result.source and not result.emergency:

        content += (
            "\n\n🔗 **Trusted information source:** "
            f"{result.source}"
        )


    # -----------------------------------------------------
    # ASSISTANT RESPONSE
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        st.write(content)


        if result.emergency:

            st.error(
                "🚨 Emergency detected. "
                "Please seek emergency medical help immediately."
            )

        else:

            topic = result.intent.replace(
                "_",
                " "
            ).title()

            confidence = result.confidence

            st.write(
                f"📌 Detected Topic: {topic}  |  "
                f"🎯 Confidence: {confidence:.0%}"
            )


    # -----------------------------------------------------
    # SAVE RESPONSE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": content,
        }
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.write("🩺 AI-Based Medical Chatbot")

st.write(
    "NLP • TF-IDF • Logistic Regression • Genetic Algorithm"
)

st.write("For educational purposes only")