# =============================================================================
# FILE: ingest.py
# PURPOSE: GitHub URL → FAISS vectorstore pipeline
#
# PIPELINE:
#   load_repo_as_document()  → fetch repo, filter junk, wrap in Document
#   split_documents()        → break into 1500-char overlapping chunks
#   build_vectorstore()      → embed all chunks → FAISS index
#
# FILES THIS IMPORTS FROM:
#   filters.py → filter_content()   strips junk files before splitting
# =============================================================================

import sys
import asyncio

# Windows fix — default SelectorEventLoop doesn't support subprocesses.
# GitIngest uses asyncio subprocesses internally to run git commands.
# ProactorEventLoop supports this. Guard with sys.platform so it only
# applies on Windows and doesn't affect Mac/Linux users.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from gitingest import ingest
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from filter import filter_content       # ← strips node_modules, lock files, binaries etc.


# =============================================================================
# STEP 1 — Fetch repo, filter junk, wrap in Document
# =============================================================================

def load_repo_as_document(github_url: str) -> list[Document]:
    """
    Fetches a GitHub repo via GitIngest, filters junk files out,
    and wraps the clean content in a LangChain Document.

    GitIngest returns three things:
      summary → repo stats (ignored)
      tree    → folder/file structure as a string
      content → all code and text files concatenated

    We combine tree + content so the LLM also knows the file structure
    when answering questions like "where is the auth logic?"
    """
    print(f"📥 Fetching repo: {github_url}")
    summary, tree, content = ingest(github_url)

    # Combine file structure + code into one text block
    full_text = f"REPOSITORY STRUCTURE:\n{tree}\n\nREPOSITORY CONTENT:\n{content}"

    # ── Filter before doing anything else ────────────────────────────────────
    # filter_content() lives in filters.py.
    # It strips out node_modules, lock files, images, binaries, build output.
    # Everything downstream (splitting, embedding, FAISS) gets cleaner input.
    # ingest.py doesn't know HOW filtering works — just calls it and moves on.
    # This is the Single Responsibility Principle — each file has one job.
    # ─────────────────────────────────────────────────────────────────────────
    filtered_text = filter_content(full_text)

    print(f"📄 Repo ready — {len(filtered_text):,} chars after filtering")

    doc = Document(
        page_content=filtered_text,
        metadata={"source": github_url}
    )

    return [doc]


# =============================================================================
# STEP 2 — Split into chunks
# =============================================================================
#
# RecursiveCharacterTextSplitter splits on natural boundaries:
#   "\n\n" → "\n" → " " → "" (in order of preference)
# chunk_size=1500    → max characters per chunk
# chunk_overlap=200  → repeated chars between chunks (prevents boundary cutoffs)
# =============================================================================

def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️  Split into {len(chunks)} chunks")
    return chunks


# =============================================================================
# STEP 3 — Embed chunks with HuggingFace
# =============================================================================
#
# HuggingFaceEmbeddings runs the model IN THIS PROCESS — no HTTP, no Ollama.
# sentence-transformers batches all chunks in one forward pass internally.
# model: all-MiniLM-L6-v2 → 22MB, 384 dimensions, fast CPU inference
# normalize_embeddings=True → unit vectors → faster FAISS dot product search
# =============================================================================

def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


# =============================================================================
# STEP 4 — Build FAISS vectorstore
# =============================================================================

def build_vectorstore(chunks: list[Document]) -> FAISS:
    embeddings = get_embeddings()
    print(f"🔢 Embedding {len(chunks)} chunks...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("✅ Vectorstore ready!")
    return vectorstore


# =============================================================================
# PUBLIC ENTRY POINT — called by app.py
# =============================================================================

def ingest_github_repo(github_url: str) -> FAISS:
    """
    Full pipeline: GitHub URL → filtered → chunked → embedded → FAISS index.
    """
    docs   = load_repo_as_document(github_url)
    chunks = split_documents(docs)
    store  = build_vectorstore(chunks)
    return store