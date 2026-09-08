# =============================================================================
# FILE: chain.py
# PURPOSE: RAG chain with memory, source citations, control token
#
# WHAT CHANGED FROM LAST VERSION:
#   - build_rag_chain() now accepts api_key parameter
#   - ChatGroq receives api_key directly instead of reading from .env
#   - This allows users to bring their own Groq API key via the UI
#   - load_dotenv() removed — no longer needed for deployment
#   - get_retriever() removed — retriever is now built in ingest.py
#     (build_hybrid_retriever) and passed in as an EnsembleRetriever
#   - FAISS import removed — no longer referenced in this file
#
# EVERYTHING LIVES HERE:
#   - Memory storage (store dict, get_session_history, reset_memory)
#   - RAG pipeline (retriever, prompt, LLM, sources)
#   - Memory wrapping (RunnableWithMessageHistory)
#
# WHAT app.py IMPORTS FROM HERE:
#   build_rag_chain(retriever, api_key, model_id) → builds the full chain with memory
#   get_llm(model_id, api_key)                    → standalone LLM factory
#   reset_memory()                                → called when user indexes a new repo
# =============================================================================

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


# =============================================================================
# SECTION 1 — MEMORY STORAGE
# =============================================================================
#
# store: module-level dict that persists for the entire process lifetime
#   Key   = session_id string (identifies one conversation)
#   Value = ChatMessageHistory (the list of messages for that session)
#
# WHY module-level?
#   Python imports a module ONCE. After that every "from chain import ..."
#   gets the SAME module object. So store is one shared dict across the
#   entire app — survives Streamlit reruns because reruns don't restart
#   the Python process, they just re-execute the script.
#
# ChatMessageHistory:
#   A simple container for one conversation's messages.
#   Internally just a list: [HumanMessage("hi"), AIMessage("hello"), ...]
#   Methods: .add_user_message(), .add_ai_message(), .messages, .clear()
#   You never call these manually — RunnableWithMessageHistory does it.
# =============================================================================

store: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    """
    Returns the ChatMessageHistory for a given session_id.
    Creates a new empty one if this session hasn't been seen before.
    Called automatically by RunnableWithMessageHistory. Never call directly.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def reset_memory(session_id: str = "default") -> None:
    """
    Clears conversation history for a session.
    Called by app.py when user indexes a new repo.
    """
    if session_id in store:
        store[session_id].clear()


# =============================================================================
# SECTION 2 — RAG PIPELINE
# =============================================================================

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful code assistant. A user is exploring a GitHub repository and asking questions about it.

You have access to retrieved excerpts from the repository below.

How to answer:
- If the question is about this specific repository, answer using the retrieved context. Be specific and reference actual code, files, or logic from the context.
- If the question is a general programming concept (like "what is Go?" or "what is REST?"), answer it using your own knowledge and briefly connect it to the repository if relevant.
- If the question needs both, answer the general part first then explain how this repo does it specifically.
- If the answer is not in the context and it's repo-specific, say: "I don't see that in the indexed parts of this repository."
- For greetings, small talk, or questions unrelated to code and this repository, respond naturally and helpfully without using the context.
- You have access to the conversation history. Use it to understand follow-up questions and references like "it", "that", "the same function", etc.
- Never list source files in your answer — those are shown separately in the UI.
- Keep answers concise and conversational. No unnecessary headers or bullet points unless the answer genuinely needs them.

IMPORTANT — At the very end of your response, on its own line:
- If you used the retrieved repository context to answer → add exactly: [USED_CONTEXT]
- If you answered from general knowledge, small talk, or didn't need the context → do NOT add it

Retrieved context:
{context}
"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def extract_sources(docs: list) -> list[str]:
    seen = set()
    sources = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown source")
        # Skip the tree/structure overview — it's not a real file citation
        if source == "REPOSITORY_STRUCTURE":
            continue
        if source not in seen:
            seen.add(source)
            sources.append(source)
    return sources


# =============================================================================
# SECTION 3 — BUILD CHAIN WITH MEMORY
# =============================================================================

def get_llm(model_id: str, api_key: str) -> ChatGroq:
    """
    Factory — single place to instantiate the Groq LLM.
    model_id is chosen live from the sidebar dropdown (fetched from Groq's
    /models endpoint). Centralised here so LangGraph node functions can
    call get_llm() directly when the pipeline migrates to LangGraph,
    without needing to touch the graph structure.
    """
    return ChatGroq(
        model=model_id,
        temperature=0,
        api_key=api_key
    )


def build_rag_chain(retriever, api_key: str, model_id: str):
    """
    Builds and returns the full RAG chain with memory.

    WHAT CHANGED:
      - Accepts model_id from the sidebar live dropdown.
      - LLM instantiation delegated to get_llm() factory.
      - Retriever is an EnsembleRetriever (BM25 + FAISS hybrid) from ingest.py.
      - api_key passed directly from the UI — no .env needed.

    Returns: {"answer": "...", "sources": [...]}
    """
    llm = get_llm(model_id, api_key)

    extract_question = RunnableLambda(lambda x: x["question"])

    # extract_history uses .get() with default [] for safety on first message
    # when history key may not yet exist in the input dict
    extract_history  = RunnableLambda(lambda x: x.get("history", []))

    answer_chain = (
        {
            "context":  extract_question | retriever | format_docs,
            "question": extract_question,
            "history":  extract_history,    # must be explicitly passed to prompt
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    sources_chain = extract_question | retriever | extract_sources

    # ── THE FIX ───────────────────────────────────────────────────────────────
    # BEFORE (broken):
    #   RunnableWithMessageHistory(
    #       RunnableParallel(answer=..., sources=...),  ← wraps BOTH
    #       output_messages_key="answer"                ← silently drops "sources"
    #   )
    #
    # AFTER (fixed):
    #   RunnableParallel(
    #       answer = RunnableWithMessageHistory(answer_chain),  ← only answer has history
    #       sources = sources_chain                             ← passes through untouched
    #   )
    #
    # When output_messages_key is set and the inner chain produces a dict, LangChain
    # only passes through the key matching output_messages_key — "sources" was silently
    # thrown away. Wrapping only answer_chain (which returns a plain string) fixes this.
    # ─────────────────────────────────────────────────────────────────────────
    answer_with_history = RunnableWithMessageHistory(
        answer_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
        # No output_messages_key needed — answer_chain returns a plain string,
        # not a dict, so RunnableWithMessageHistory saves it directly as an AI message.
    )

    return RunnableParallel(
        answer=answer_with_history,
        sources=sources_chain
    )