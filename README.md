# GitHub RAG — Chat With Any Repository

> Ask questions about any GitHub codebase in plain English. Get grounded answers with source file citations.

🔗 **[Live Demo](https://chatgitrepo.streamlit.app)**

## What It Does

Paste any public GitHub repository URL and start asking questions about it:

- *"How does authentication work in this repo?"*
- *"What does the main entry point do?"*
- *"What dependencies does this project use?"*
- *"What is middleware and how does this repo use it?"*

The system reads the **actual code** — not a model's memory of it. Every answer cites exactly which files it pulled from.

---

### Features

**Hybrid Retrieval-Augmented Generation (RAG)**
Combines BM25 keyword search with FAISS dense semantic embeddings using LangChain's `EnsembleRetriever` (weighted 0.4 BM25 / 0.6 FAISS). Excels at both exact identifier lookup (function names, variables, configs) and high-level conceptual questions.

**Dynamic Model Selection & Live Key Validation**
Paste your Groq API key and the app instantly validates it, fetching active chat models available to your account into a clean sidebar dropdown. Switch models seamlessly mid-session without losing chat history.

**Source Citations**
Every repo-specific answer shows which files the information came from, in a collapsible expander below the response.

**Conversation Memory**
Follow-up questions work naturally. Ask *"how does auth work?"* then *"where exactly is that implemented?"* — the system knows what *"that"* refers to. Includes a dedicated "Clear History" control.

**Smart Prompt Routing**
Handles three question types in one chain — repo-specific questions use retrieved context, general programming concepts use the LLM's own knowledge, hybrid questions combine both.

**File Filtering**
Strips `node_modules`, lock files, binaries, images, and build output before indexing. Reduces noise and speeds up embedding significantly on large repos.

**Bring Your Own Key**
Users provide their own free Groq API key — no shared credentials, no usage limits hitting a single key.

---

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Repo ingestion | GitIngest | Converts any GitHub repo to structured text |
| File filtering | Custom parser (`filter.py`) | Removes junk before embedding (state machine) |
| Chunking | `RecursiveCharacterTextSplitter` | Respects code boundaries |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | In-process inference — no HTTP, no server |
| Dense retrieval | FAISS | In-memory, sub-5ms vector similarity search |
| Sparse retrieval | `BM25Retriever` (`rank-bm25`) | In-memory exact keyword & identifier matching |
| Hybrid fusion | `EnsembleRetriever` | Merges dense + sparse ranks via Reciprocal Rank Fusion |
| LLM | Groq (`get_llm()` factory) | Fast inference, dynamic model selection from live API key |
| Memory | `RunnableWithMessageHistory` | Per-session conversation history |
| Orchestration | LangChain LCEL | Composable pipeline with `\|` pipes |
| Parallel execution | `RunnableParallel` | Answer + sources run simultaneously |
| UI | Streamlit | Chat interface with dynamic sidebar controls |

---

## Architecture

```
GitHub URL
    ↓ GitIngest
Raw repo text (tree + all files)
    ↓ filter.py — strips node_modules, lock files, binaries
Clean text
    ↓ RecursiveCharacterTextSplitter (1500 chars, 200 overlap)
Chunks
    ├──→ BM25Retriever (k=5, exact keywords)
    └──→ HuggingFaceEmbeddings → FAISS index (k=5, semantic vectors)
            ↓
    EnsembleRetriever (weights: [0.4 BM25, 0.6 FAISS], RRF fusion)

User enters Groq API Key
    ↓ GET https://api.groq.com/openai/v1/models (validates key & filters chat models)
    ↓ Populates dynamic model selectbox (rebuilds chain without re-indexing)

User asks a question
    ↓ RunnableWithMessageHistory injects conversation history
    ↓ RunnableParallel branches:
        ├── question → hybrid retriever → prompt → Groq LLM (selected model) → answer
        └── question → hybrid retriever → extract metadata → source filenames
    ↓ Control token [USED_CONTEXT] decides whether to show sources
    ↓ Streamlit renders answer + collapsible citations
```

---

## Engineering Decisions

**Why Hybrid Search (EnsembleRetriever)?**
Vector search alone often struggles with exact symbols like variable names, error codes, and config keys. BM25 catches exact keyword matches, while FAISS catches conceptual logic. Combining both with Reciprocal Rank Fusion delivers superior code search accuracy.

**Why FAISS over Chroma?**
In-memory search is sub-5ms at this scale. No server to manage, no persistence overhead. Right tool for the job.

**Why HuggingFace over Ollama embeddings?**
HuggingFace loads the model directly into the Python process — no HTTP round trips even to localhost. Embedding time dropped from 3-5 minutes to 20-30 seconds.

**Why Dynamic Model Loading via `get_llm()` Factory?**
Users get access to their latest authorized Groq models (e.g., `openai/gpt-oss-120b`, `llama-3.3-70b-versatile`) without code changes. Decoupling model creation into `get_llm()` also makes migrating to agentic frameworks like LangGraph straightforward.

**Why `RunnableParallel` for citations?**
Running answer and source extraction in parallel means one retriever call serves both. Avoids redundant vector searches.

**Why a control token `[USED_CONTEXT]`?**
Avoids a second LLM call to classify whether context was used. The LLM signals it inline — application layer reads and strips it before display.

---

## Run Locally

```bash
# 1. Clone
git clone https://github.com/YOURUSERNAME/github-rag.git
cd github-rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

Get a free Groq API key at [console.groq.com](https://console.groq.com), paste it in the sidebar (models populate automatically), enter a repo URL, and start exploring.

---

## Project Structure

```
github-rag/
├── app.py           → Streamlit UI, dynamic model picker, control token parsing
├── chain.py         → LCEL pipeline, get_llm factory, memory storage, RAG chain
├── ingest.py        → Fetch → filter → split → BM25 + FAISS EnsembleRetriever
├── filter.py        → Junk file detection (state machine parser)
└── requirements.txt
```

---

## What I Learned

Built this to get hands-on with production RAG patterns — not just "it works" but understanding why each architectural decision exists. Key takeaways:

- Hybrid search combining BM25 + dense vectors significantly outperforms pure vector search on codebases
- LangChain's abstraction layer (LCEL) makes swapping components trivial — changing from Ollama to Groq was 2 lines
- Embedding is the real bottleneck in RAG pipelines, not retrieval or generation
- Prompt engineering matters more than model choice for output quality
- Memory in LLMs is context injection — `RunnableWithMessageHistory` just automates prepending previous messages

---

## Roadmap

- [ ] Streaming UI responses
- [ ] Migration to LangGraph stateful multi-step agent
- [ ] Repo comparison mode — ask questions across two repos simultaneously  
- [ ] Evaluation script — LLM-as-a-judge scoring pipeline
- [ ] Code-aware chunking — split at function/class boundaries