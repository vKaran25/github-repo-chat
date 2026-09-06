# =============================================================================
# FILE: app.py
# PURPOSE: Streamlit UI — production ready for deployment
#
# WHAT CHANGED FROM LAST VERSION:
#   - Removed subtitle caption below title
#   - Removed octopus emoji from title
#   - Fixed placeholder text on GitHub URL input
#   - Rewrote sidebar "How it works" to user-facing steps
#   - Added Groq API key input in sidebar
#   - api_key passed to build_rag_chain()
#   - Removed debug st.code(traceback) — not for production
#   - Removed python-dotenv dependency
#
# IMPORTS FROM:
#   ingest.py → ingest_github_repo()
#   chain.py  → build_rag_chain(vectorstore, api_key), reset_memory()
# =============================================================================

import requests
import streamlit as st
from ingest import ingest_github_repo
from chain import build_rag_chain, reset_memory, get_llm

st.set_page_config(page_title="GitHub RAG", page_icon="💬", layout="centered")

# Title — no emoji, no subtitle caption
st.title("GitHub RAG")

# ── Session state ─────────────────────────────────────────────────────────────
if "chain" not in st.session_state:
    st.session_state.chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Retriever stored so we can rebuild the chain on model switch without re-indexing
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Model picker state
if "available_models" not in st.session_state:
    st.session_state.available_models = []       # list fetched from Groq /models

if "active_model" not in st.session_state:
    st.session_state.active_model = None         # currently selected model id

if "last_fetched_key" not in st.session_state:
    st.session_state.last_fetched_key = ""       # key we last fetched models for

# ── Memory config ─────────────────────────────────────────────────────────────
SESSION_ID = "default"
MEMORY_CONFIG = {"configurable": {"session_id": SESSION_ID}}


# ── Groq model filtering ──────────────────────────────────────────────────────
# Groq's /models endpoint returns ALL model types — chat, transcription, TTS,
# safety classifiers. We only want chat/text models in the dropdown.
# Groq doesn't return a clean "type" field, so we filter by name patterns.
EXCLUDED_PATTERNS = [
    "whisper",    # Whisper audio transcription family
    "distil",     # distil-whisper
    "guard",      # Llama Guard safety classifiers
    "tts",        # Text-to-speech (playai-tts, orpheus-tts-*)
    "orpheus",    # Orpheus TTS
    "playai",     # PlayAI TTS
]

def is_chat_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return not any(pat in lowered for pat in EXCLUDED_PATTERNS)


def fetch_groq_models(api_key: str) -> list[str]:
    """
    Calls GET https://api.groq.com/openai/v1/models with the user's key.
    - Validates the key (401 = bad key shown immediately)
    - Filters to active chat-only models
    - Returns sorted list of model ids, or [] on any failure
    """
    try:
        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=8
        )
        if resp.status_code == 401:
            st.sidebar.error("❌ Invalid API key. Check it and try again.")
            return []
        if resp.status_code == 429:
            st.sidebar.error("⏳ Groq rate limit hit. Wait a moment and try again.")
            return []
        if not resp.ok:
            st.sidebar.error(f"Could not load models (HTTP {resp.status_code}).")
            return []
        data = resp.json().get("data", [])
        return sorted([
            m["id"] for m in data
            if m.get("active", True) and is_chat_model(m["id"])
        ])
    except Exception as e:
        st.sidebar.error(f"Could not reach Groq: {e}")
        return []


# ── Control token parser ──────────────────────────────────────────────────────
# LLM appends [USED_CONTEXT] when it used retrieved repo context.
# Strips it from the answer and returns (clean_answer, used_context bool).
# used_context=True  → show sources expander
# used_context=False → hide sources expander
def parse_answer(raw_answer: str) -> tuple[str, bool]:
    lines = raw_answer.strip().splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            if lines[i].strip() == "[USED_CONTEXT]":
                clean_lines = lines[:i] + lines[i+1:]
                return "\n".join(clean_lines).strip(), True
            else:
                return raw_answer.strip(), False
    return raw_answer.strip(), False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Setup")

    # ── Groq API key input ────────────────────────────────────────────────────
    # type="password" masks input — key never shown in plain text.
    # Key is validated immediately by fetching the Groq /models endpoint.
    # ─────────────────────────────────────────────────────────────────────────
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get your free key at console.groq.com"
    )

    key_stripped = groq_api_key.strip()

    # ── Auto-validate key + fetch models ─────────────────────────────────────
    # Fires once whenever the key changes (tracked via last_fetched_key).
    # A key is considered "looks valid" if it starts with gsk_ and is >20 chars.
    # The actual validity is confirmed by the /models call (401 = bad key).
    # ─────────────────────────────────────────────────────────────────────────
    if key_stripped and key_stripped.startswith("gsk_") and len(key_stripped) > 20:
        if key_stripped != st.session_state.last_fetched_key:
            with st.spinner("Validating key & loading models..."):
                models = fetch_groq_models(key_stripped)
                if models:
                    st.session_state.available_models = models
                    st.session_state.last_fetched_key = key_stripped
                    # Default to first model if none selected yet
                    if st.session_state.active_model not in models:
                        st.session_state.active_model = models[0]

    # ── Model selectbox — only shown once models are loaded ───────────────────
    if st.session_state.available_models:
        current_idx = 0
        if st.session_state.active_model in st.session_state.available_models:
            current_idx = st.session_state.available_models.index(
                st.session_state.active_model
            )

        selected_model = st.selectbox(
            "Model",
            options=st.session_state.available_models,
            index=current_idx,
            help="Chat history is preserved when you switch models."
        )
        st.caption("💡 Switching models keeps your conversation history.")

        # Detect model switch — rebuild chain with new model, keep memory intact
        if selected_model != st.session_state.active_model:
            st.session_state.active_model = selected_model
            if st.session_state.retriever is not None:
                st.session_state.chain = build_rag_chain(
                    st.session_state.retriever,
                    key_stripped,
                    selected_model
                )
                st.success(f"Switched to `{selected_model}`. History preserved ✓")
    else:
        selected_model = st.session_state.active_model  # may be None

    github_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/owner/repo"
    )

    if st.button("Index Repository", type="primary"):
        if not key_stripped:
            st.error("Please enter your Groq API key first.")
        elif not st.session_state.available_models:
            st.error("Please enter a valid Groq API key first.")
        elif not github_url.strip():
            st.error("Please enter a GitHub repository URL.")
        else:
            with st.spinner("Fetching & indexing repo... (may take 20-30s)"):
                try:
                    retriever = ingest_github_repo(github_url)

                    # Store retriever so model switches can rebuild chain without re-indexing
                    st.session_state.retriever = retriever

                    st.session_state.chain = build_rag_chain(
                        retriever,
                        key_stripped,
                        st.session_state.active_model
                    )

                    st.session_state.messages = []
                    reset_memory(SESSION_ID)
                    st.success("Ready! Ask anything about the repo.")
                except Exception as e:
                    st.error(f"Error: {type(e).__name__}: {e}")

    st.divider()

    # ── Clear History button ──────────────────────────────────────────────────
    # Shown always — useful if context-length errors occur after model switch.
    if st.button("🗑️ Clear History"):
        st.session_state.messages = []
        reset_memory(SESSION_ID)
        st.success("Conversation history cleared.")

    # ── User-facing steps ─────────────────────────────────────────────────────
    st.markdown("""
**How to use:**
1. Get a free API key at [console.groq.com](https://console.groq.com)
2. Paste your Groq API key — models load automatically
3. Pick a model from the dropdown
4. Enter any public GitHub repository URL
5. Click **Index Repository** and wait 20-30s
6. Ask questions about the codebase in the chat
""")


# ── Display past messages ─────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📁 Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- `{src}`")

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about the repository..."):
    if st.session_state.chain is None:
        st.warning("Please index a repository first using the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.chain.invoke(
                        {"question": prompt},
                        config=MEMORY_CONFIG
                    )

                    clean_answer, used_context = parse_answer(result["answer"])
                    st.write(clean_answer)

                    if used_context and result["sources"]:
                        with st.expander("📁 Sources used"):
                            for src in result["sources"]:
                                st.markdown(f"- `{src}`")

                    msg_to_save = {"role": "assistant", "content": clean_answer}
                    if used_context and result["sources"]:
                        msg_to_save["sources"] = result["sources"]
                    st.session_state.messages.append(msg_to_save)

                except Exception as e:
                    err_str = str(e).lower()
                    # Detect context-window overflow — can happen after model switch
                    # if the new model has a smaller context limit than the history
                    if any(x in err_str for x in [
                        "context_length", "context length",
                        "maximum context", "context window",
                        "too long", "token limit", "exceeded"
                    ]):
                        st.warning(
                            "⚠️ Your conversation history is too long for this model. "
                            "Click **🗑️ Clear History** in the sidebar to continue."
                        )
                    else:
                        st.error(f"Error: {type(e).__name__}: {e}")