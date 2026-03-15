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

import streamlit as st
from ingest import ingest_github_repo
from chain import build_rag_chain, reset_memory

st.set_page_config(page_title="GitHub RAG", page_icon="💬", layout="centered")

# Title — no emoji, no subtitle caption
st.title("GitHub RAG")

# ── Session state ─────────────────────────────────────────────────────────────
if "chain" not in st.session_state:
    st.session_state.chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Memory config ─────────────────────────────────────────────────────────────
SESSION_ID = "default"
MEMORY_CONFIG = {"configurable": {"session_id": SESSION_ID}}


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
    # type="password" masks the input — key never shown in plain text
    # Users get a free key at console.groq.com
    # Key is stored in session_state for the duration of the browser session only
    # Never written to disk, never sent anywhere except Groq's API
    # ─────────────────────────────────────────────────────────────────────────
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get your free key at console.groq.com"
    )

    github_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/owner/repo"
    )

    if st.button("Index Repository", type="primary"):
        if not groq_api_key.strip():
            st.error("Please enter your Groq API key first.")
        elif not github_url.strip():
            st.error("Please enter a GitHub repository URL.")
        else:
            with st.spinner("Fetching & indexing repo... (may take 20-30s)"):
                try:
                    vectorstore = ingest_github_repo(github_url)

                    # api_key passed directly — no .env needed
                    st.session_state.chain = build_rag_chain(vectorstore, groq_api_key.strip())

                    st.session_state.messages = []
                    reset_memory(SESSION_ID)
                    st.success("Ready! Ask anything about the repo.")
                except Exception as e:
                    st.error(f"Error: {type(e).__name__}: {e}")

    st.divider()

    # ── User-facing steps ─────────────────────────────────────────────────────
    st.markdown("""
**How to use:**
1. Get a free API key at [console.groq.com](https://console.groq.com)
2. Paste your Groq API key above
3. Enter any public GitHub repository URL
4. Click **Index Repository** and wait 20-30s
5. Ask questions about the codebase in the chat
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