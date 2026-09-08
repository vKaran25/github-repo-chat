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

import os
import concurrent.futures
from gitingest import ingest as _ingest_sync
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import EnsembleRetriever
except (ImportError, ModuleNotFoundError):
    try:
        from langchain.retrievers.ensemble import EnsembleRetriever
    except (ImportError, ModuleNotFoundError):
        from langchain_classic.retrievers import EnsembleRetriever
from filter import filter_content       # ← strips node_modules, lock files, binaries etc.


# =============================================================================
# LANGUAGE DETECTION — extension → Language enum for code-aware splitting
# =============================================================================
#
# RecursiveCharacterTextSplitter.from_language() uses language-specific
# separator sequences so splits respect code structure boundaries:
#   Python  → splits on "class ", "def ", "\n\n", "\n", " "
#   JS/TS   → splits on "function ", "class ", "\n\n", "\n", " "
#   Go      → splits on "func ", "type ", "\n\n", "\n", " "
#   Markdown→ splits on headings (#), "---", "\n\n", "\n", " "
#
# Files with extensions not in this map fall back to the generic splitter —
# same behaviour as before this change (YAML, JSON, Dockerfile, .env, etc.)
# Language.C is intentionally excluded — known to be buggy in some LangChain builds.
# =============================================================================

EXTENSION_TO_LANGUAGE: dict[str, Language] = {
    ".py":    Language.PYTHON,
    ".js":    Language.JS,
    ".jsx":   Language.JS,
    ".ts":    Language.TS,
    ".tsx":   Language.TS,
    ".java":  Language.JAVA,
    ".go":    Language.GO,
    ".rs":    Language.RUST,
    ".rb":    Language.RUBY,
    ".php":   Language.PHP,
    ".cpp":   Language.CPP,
    ".cc":    Language.CPP,
    ".cxx":   Language.CPP,
    ".hpp":   Language.CPP,
    ".cs":    Language.CSHARP,
    ".kt":    Language.KOTLIN,
    ".scala": Language.SCALA,
    ".swift": Language.SWIFT,
    ".md":    Language.MARKDOWN,
    ".rst":   Language.RST,
    ".html":  Language.HTML,
    ".htm":   Language.HTML,
    ".proto": Language.PROTO,
}


def get_splitter_for_file(filepath: str) -> RecursiveCharacterTextSplitter:
    """
    Returns a RecursiveCharacterTextSplitter appropriate for the file's language.
    - Known code extensions  → language-aware splitter via from_language()
    - Everything else        → generic splitter (same as the old split_documents)
    The try/except catches any broken Language.* implementations silently.
    """
    ext = os.path.splitext(filepath)[-1].lower()
    lang = EXTENSION_TO_LANGUAGE.get(ext)
    if lang is not None:
        try:
            return RecursiveCharacterTextSplitter.from_language(
                language=lang,
                chunk_size=1500,
                chunk_overlap=200,
            )
        except Exception:
            pass    # broken Language.* implementation — fall through
    # Generic fallback for YAML, JSON, plain text, Dockerfile, unknown extensions
    return RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
    )


# =============================================================================
# STEP 1 — Fetch repo, filter junk, wrap in Document
# =============================================================================

def load_repo_as_document(github_url: str) -> list[Document]:
    """
    Fetches a GitHub repo via GitIngest, filters junk files out,
    and parses the output into one Document per file.

    Returns a list of Documents — one per file — each with
    metadata={"source": filepath} so language-aware splitting
    and accurate file-path citations work downstream.

    WHY ThreadPoolExecutor?
      gitingest.ingest() calls asyncio.run() internally.
      Streamlit 1.x runs user script code inside its own asyncio event loop.
      asyncio.run() cannot be called from a running event loop — it raises
      RuntimeError. Running ingest() in a ThreadPoolExecutor worker thread
      sidesteps this: worker threads have NO event loop, so asyncio.run()
      inside gitingest works perfectly. Confirmed by diagnostic test.
    """
    print(f"📥 Fetching repo: {github_url}")

    # Run gitingest in a fresh thread — no asyncio event loop conflict
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_ingest_sync, github_url)
        summary, tree, content = future.result()

    # Combine file structure + code into one text block
    full_text = f"REPOSITORY STRUCTURE:\n{tree}\n\nREPOSITORY CONTENT:\n{content}"

    # Filter junk files out (node_modules, lock files, images, binaries, build output)
    filtered_text = filter_content(full_text)

    # Parse into one Document per file — each gets metadata={"source": filepath}
    docs = parse_into_file_documents(filtered_text)
    print(f"📄 Parsed {len(docs)} file(s) from repo")

    return docs


def parse_into_file_documents(filtered_text: str) -> list[Document]:
    """
    Parses GitIngest's filtered content string into one Document per file.

    Uses the same separator-detection logic as filter.py:
        ================================================
        File: path/to/file.ext
        ================================================
        ...file contents...

    Each resulting Document gets metadata={"source": filepath} — the actual
    file path, not the repo URL. This is what makes per-file citations work.

    Content appearing before the first "File:" header (the repo tree/structure
    section) becomes a Document with metadata={"source": "REPOSITORY_STRUCTURE"}.
    """
    lines = filtered_text.splitlines()
    docs = []
    current_source = "REPOSITORY_STRUCTURE"
    current_lines: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        is_separator = line.strip().startswith("=") and len(line.strip()) > 10

        # Detect file header: separator → ("FILE: path" or "File: path")
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if is_separator and next_line.lower().startswith("file:"):
            # Flush accumulated content as a Document for the previous file
            content = "\n".join(current_lines).strip()
            if content:
                docs.append(Document(
                    page_content=content,
                    metadata={"source": current_source}
                ))
            # Start the new file (extract path after "FILE:" or "File:")
            current_source = next_line.split(":", 1)[1].strip()
            current_lines = []
            # Skip the three header lines: separator, "FILE: ...", closing separator
            i += 3
            continue

        current_lines.append(line)
        i += 1

    # Flush the final file's content
    content = "\n".join(current_lines).strip()
    if content:
        docs.append(Document(
            page_content=content,
            metadata={"source": current_source}
        ))

    return docs


# =============================================================================
# STEP 2 — Split into chunks (language-aware, per file)
# =============================================================================
#
# Each file Document is split with a language-appropriate splitter:
#   .py  → Language.PYTHON separators (class/def boundaries)
#   .go  → Language.GO separators (func/type boundaries)
#   .js  → Language.JS separators (function/class boundaries)
#   ...anything else → generic splitter (same as before this change)
#
# Splitting per-file ensures:
#   1. Chunks never span across two different files
#   2. The file's metadata={"source": filepath} is copied to every chunk
#      automatically by LangChain's split_documents() — no extra work needed
#   3. Code blocks (functions, classes) stay intact where possible
#
# chunk_size=1500    → max characters per chunk
# chunk_overlap=200  → repeated chars between chunks (prevents boundary cutoffs)
# =============================================================================

def split_documents(file_docs: list[Document]) -> list[Document]:
    """
    Splits each file Document using a language-appropriate splitter.
    LangChain's split_documents() automatically propagates source metadata
    to every chunk — the file-path citation feature works with zero extra code.
    """
    all_chunks: list[Document] = []
    for doc in file_docs:
        filepath = doc.metadata.get("source", "")
        splitter = get_splitter_for_file(filepath)
        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)
    print(f"✂️  Split into {len(all_chunks)} chunks across {len(file_docs)} file(s)")
    return all_chunks


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
# STEP 5 — Build hybrid retriever (BM25 + FAISS EnsembleRetriever)
# =============================================================================
#
# BM25Retriever   → keyword/exact-match search (in-memory inverted index)
# FAISS retriever → dense semantic search (vector similarity)
# EnsembleRetriever → merges both lists via Reciprocal Rank Fusion (RRF)
#
# weights=[0.4, 0.6]:
#   BM25  contributes 40% of the ranking score
#   FAISS contributes 60% of the ranking score
#   (FAISS weighted higher because semantic search is more useful for code Q&A)
#
# k=5 on each retriever: each returns 5 chunks independently.
# After dedup + RRF, the ensemble returns up to 10 unique chunks ranked by fused score.
# =============================================================================

def build_hybrid_retriever(chunks: list[Document]) -> EnsembleRetriever:
    # ── Drop REPOSITORY_STRUCTURE chunks before indexing ─────────────────────
    # The directory tree listing (file structure overview) is stored with
    # source="REPOSITORY_STRUCTURE". It contains every filename and directory
    # name in the repo. If indexed in BM25, it scores highly for almost any
    # query (any keyword the user types is likely a filename in the tree),
    # polluting retrieval results and making citations always show
    # "REPOSITORY_STRUCTURE" instead of actual file paths.
    # We keep only chunks that have a real file path as their source.
    # ─────────────────────────────────────────────────────────────────────────
    indexable_chunks = [
        c for c in chunks
        if c.metadata.get("source") != "REPOSITORY_STRUCTURE"
    ]
    # Defensive safeguard: if no non-tree chunks exist, fallback to all chunks so BM25 never receives []
    if not indexable_chunks:
        indexable_chunks = chunks
    print(f"📦 Indexing {len(indexable_chunks)} chunks (tree chunks excluded)")

    # ── BM25: keyword search ──────────────────────────────────────────────────
    # from_documents() tokenises each chunk's page_content and builds an
    # in-memory BM25 index. No model download, no HTTP — pure Python.
    bm25_retriever = BM25Retriever.from_documents(indexable_chunks)
    bm25_retriever.k = 5

    # ── FAISS: dense semantic search ──────────────────────────────────────────
    # Same vectorstore build as before, then wrapped as a retriever.
    vectorstore = build_vectorstore(indexable_chunks)
    faiss_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    print("🔀 Building EnsembleRetriever (BM25 + FAISS)...")

    # ── Ensemble: fuse results via RRF ────────────────────────────────────────
    # weights order matches retrievers order: [bm25_weight, faiss_weight]
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6]
    )

    print("✅ Hybrid retriever ready!")
    return ensemble


# =============================================================================
# PUBLIC ENTRY POINT — called by app.py
# =============================================================================

def ingest_github_repo(github_url: str) -> EnsembleRetriever:
    """
    Full pipeline: GitHub URL → filtered → chunked → hybrid retriever.
    Returns an EnsembleRetriever (BM25 + FAISS) ready to be passed to chain.py.
    """
    docs   = load_repo_as_document(github_url)
    chunks = split_documents(docs)
    return build_hybrid_retriever(chunks)
