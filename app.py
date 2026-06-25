import streamlit as st
from rag_core import RagEngine
import time

st.set_page_config(
    page_title="Course Notes Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM STYLING ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(180deg, #0F1117 0%, #161925 100%);
    }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #0F1117 0%, #161925 100%);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .hero {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }

    .hero h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #7C5CFF 0%, #46D6C5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }

    .hero p {
        color: #9098B1;
        font-size: 0.95rem;
        font-weight: 400;
    }

    .stat-row {
        display: flex;
        justify-content: center;
        gap: 12px;
        margin: 1rem 0 1.5rem 0;
        flex-wrap: wrap;
    }

    .stat-pill {
        background: rgba(124, 92, 255, 0.12);
        border: 1px solid rgba(124, 92, 255, 0.3);
        border-radius: 20px;
        padding: 6px 16px;
        font-size: 0.8rem;
        color: #B8AEFF;
        font-weight: 500;
    }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 4px;
        margin-bottom: 8px;
    }

    /* User message accent */
    [data-testid="stChatMessageContent"] p {
        color: #E4E6F0;
    }

    /* Input box */
    [data-testid="stChatInput"] textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(124, 92, 255, 0.4) !important;
        border-radius: 14px !important;
        color: #fff !important;
    }

    [data-testid="stChatInput"] {
        background: transparent !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #11131C;
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #E4E6F0 !important;
    }

    [data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
        color: #9098B1 !important;
    }

    /* Source expander */
    [data-testid="stExpander"] {
        background: rgba(70, 214, 197, 0.06);
        border: 1px solid rgba(70, 214, 197, 0.2);
        border-radius: 12px;
    }

    [data-testid="stExpander"] summary {
        color: #46D6C5 !important;
        font-weight: 500;
        font-size: 0.85rem;
    }

    /* Suggested question chips */
    .stButton button {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: #C4C8DA;
        font-size: 0.85rem;
        padding: 10px 14px;
        text-align: left;
        width: 100%;
        transition: all 0.2s;
    }

    .stButton button:hover {
        background: rgba(124, 92, 255, 0.15);
        border-color: rgba(124, 92, 255, 0.5);
        color: #fff;
    }

    .empty-state {
        text-align: center;
        padding: 2rem 1rem;
        color: #5A6178;
        font-size: 0.9rem;
    }

    .source-tag {
        display: inline-block;
        background: rgba(70, 214, 197, 0.1);
        color: #46D6C5;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        margin: 2px 4px 2px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown("""
<div class="hero">
    <h1>📚 Course notes assistant</h1>
    <p>Ask anything about your uploaded papers — answers are grounded in the text, with sources cited.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_engine():
    return RagEngine()


# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### ⚙️ How this works")
    st.markdown("""
    1. Your PDFs are chunked and embedded
    2. Stored in a local Chroma vector database
    3. Your question retrieves the most relevant chunks
    4. Groq's Llama-3-8B generates a grounded answer
    """)
    st.divider()
    st.markdown("### 🧰 Stack")
    st.markdown("""
    - LangChain
    - Chroma (vector store)
    - HuggingFace embeddings (local)
    - Groq API (Llama-3-8B generation)
    - Streamlit
    """)
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- LOAD ENGINE ----------
if "engine_loaded" not in st.session_state:
    with st.spinner("Loading models for the first time... this takes about a minute"):
        st.session_state.engine = load_engine()
        st.session_state.engine_loaded = True
else:
    st.session_state.engine = load_engine()

engine = st.session_state.engine

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- STAT PILLS ----------
st.markdown("""
<div class="stat-row">
    <span class="stat-pill">🔒 Fully local</span>
    <span class="stat-pill">💸 Zero API cost</span>
    <span class="stat-pill">📄 Grounded with citations</span>
</div>
""", unsafe_allow_html=True)

# ---------- EMPTY STATE WITH SUGGESTED QUESTIONS ----------
if not st.session_state.messages:
    st.markdown('<div class="empty-state">Try one of these, or type your own question below 👇</div>', unsafe_allow_html=True)

    suggestions = [
        "What problems exist with offline evaluation methods for recommenders?",
        "What datasets were used in this research?",
        "What evaluation metrics are discussed?",
    ]

    cols = st.columns(1)
    for s in suggestions:
        if st.button(s, key=s, use_container_width=True):
            st.session_state.pending_question = s
            st.rerun()

# ---------- CHAT HISTORY ----------
for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 View sources"):
                for s in msg["sources"]:
                    st.markdown(f'<span class="source-tag">{s["file"]} · p.{s["page"]}</span>', unsafe_allow_html=True)

# ---------- HANDLE PENDING SUGGESTED QUESTION ----------
pending = st.session_state.pop("pending_question", None)

# ---------- CHAT INPUT ----------
question = st.chat_input("Ask a question about your course documents...")
question = question or pending

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(question)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching documents and generating answer..."):
            start = time.time()
            answer, sources = engine.ask(question)
            elapsed = time.time() - start
        st.write(answer)
        st.caption(f"⏱️ {elapsed:.1f}s")
        if sources:
            with st.expander("📎 View sources"):
                for s in sources:
                    st.markdown(f'<span class="source-tag">{s["file"]} · p.{s["page"]}</span>', unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
    st.rerun()