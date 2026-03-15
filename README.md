# GitHub RAG — Chat With Any Repository

> Ask questions about any GitHub codebase in plain English. Get grounded answers with source file citations.

🔗 **[Live Demo](https://chatgitrepo.streamlit.app)**

---

![App Screenshot](assets/screenshot.png)

---

## What It Does

Paste any public GitHub repository URL and start asking questions about it:

- *"How does authentication work in this repo?"*
- *"What does the main entry point do?"*
- *"What dependencies does this project use?"*
- *"What is middleware and how does this repo use it?"*

The system reads the **actual code** — not a model's memory of it. Every answer cites exactly which files it pulled from.

---

## Features

**Retrieval-Augmented Generation (RAG)**
Fetches the entire repo, splits it into chunks, embeds them into a vector store, and retrieves only the most relevant chunks per question — so the LLM answers from real code, not guesswork.

**Source Citations**
Every repo-specific answer shows which files the information came from, in a collapsible expander below the response.

**Conversation Memory**
Follow-up questions work naturally. Ask *"how does auth work?"* then *"where exactly is that implemented?"* — the system knows what *"that"* refers to.

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
| File filtering | Custom parser | Removes junk before embedding (state machine) |
| Chunking | `RecursiveCharacterTextSplitter` | Respects code boundaries |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | In-process inference — no HTTP, no server |
| Vector store | FAISS | In-memory, sub-5ms similarity search |
| LLM | Groq `llama-3.3-70b-versatile` | Fastest inference available, free tier |
| Memory | `RunnableWithMessageHistory` | Per-session conversation history |
| Orchestration | LangChain LCEL | Composable pipeline with `\|` pipes |
| Parallel execution | `RunnableParallel` | Answer + sources run simultaneously |
| UI | Streamlit | Chat interface with session state |

---

## Architecture

```
GitHub URL
    ↓ GitIngest
Raw repo text (tree + all files)
    ↓ filters.py — strips node_modules, lock files, binaries
Clean text
    ↓ RecursiveCharacterTextSplitter (1500 chars, 200 overlap)
Chunks
    ↓ HuggingFaceEmbeddings — all-MiniLM-L6-v2 (in-process)
Vectors
    ↓ FAISS index (in-memory)

User asks a question
    ↓ RunnableWithMessageHistory injects conversation history
    ↓ RunnableParallel branches:
        ├── question → retriever (top-5 chunks) → prompt → Groq LLM → answer
        └── question → retriever → extract metadata → source filenames
    ↓ Control token [USED_CONTEXT] decides whether to show sources
    ↓ Streamlit renders answer + collapsible citations
```

---

## Engineering Decisions

**Why FAISS over Chroma?**
In-memory search is sub-5ms at this scale. No server to manage, no persistence overhead. Right tool for the job.

**Why HuggingFace over Ollama embeddings?**
HuggingFace loads the model directly into the Python process — no HTTP round trips even to localhost. Embedding time dropped from 3-5 minutes to 20-30 seconds.

**Why Groq over OpenAI?**
Groq's LPU hardware delivers 500-800 tokens/second — faster than any GPU-based inference. Free tier runs a 70B parameter model faster than a local 8B model.

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

Get a free Groq API key at [console.groq.com](https://console.groq.com), enter it in the sidebar, paste a repo URL, and start asking questions.

---

## Project Structure

```
github-rag/
├── app.py           → Streamlit UI, session state, control token parsing
├── chain.py         → LCEL pipeline, memory storage, RAG chain
├── ingest.py        → Fetch → filter → split → embed → FAISS
├── filters.py       → Junk file detection (state machine parser)
└── requirements.txt
```

---

## What I Learned

Built this to get hands-on with production RAG patterns — not just "it works" but understanding why each architectural decision exists. Key takeaways:

- LangChain's abstraction layer (LCEL) makes swapping components trivial — changing from Ollama to Groq was 2 lines
- Embedding is the real bottleneck in RAG pipelines, not retrieval or generation
- Prompt engineering matters more than model choice for output quality
- Memory in LLMs is context injection — `RunnableWithMessageHistory` just automates prepending previous messages

---

## Roadmap

- [ ] Streaming UI responses
- [ ] Repo comparison mode — ask questions across two repos simultaneously  
- [ ] Evaluation script — LLM-as-a-judge scoring pipeline
- [ ] Code-aware chunking — split at function/class boundaries