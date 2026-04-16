# Introduction
Smart Context Manager is an intelligent document processing engine that transforms static PDFs into searchable knowledge bases. By leveraging Retrieval-Augmented Generation (RAG), the system breaks down complex documents into semantic "chunks," indexes them in a vector database, and allows users to query their data using natural language. It acts as a bridge between raw information and actionable insights, providing precise, context-aware answers grounded directly in the user's uploaded documents.

# 🚀 Evolution of the Embedding Architecture (The "Saga")

## The Challenge: Infrastructure Constraints
Initially, the project hit a critical bottleneck during deployment to Render (Free Tier): the 500MB disk/RAM limit. Standard PyTorch installations include NVIDIA CUDA binaries exceeding 2GB, causing immediate deployment failure.

## The Pivot: External API (Google Gemini)
I attempted to offload processing to the Google Gemini API. However, persistent 404 (Batch Not Found) errors and SDK instabilities during runtime made external semantic search unreliable for this specific production environment.

## The Breakthrough: Resource Optimization & "Dependency Diet"
I returned to local processing with a lean, CPU-optimized strategy:

- **CPU-Only Implementation:** Forced torch==2.2.1+cpu via --find-links, slashing the library size from 2GB+ to ~200MB.
- **Dependency Resolution:** Manually resolved "Dependency Hell" between transformers, tokenizers, and sentence-transformers to ensure compatibility with Python 3.10.
- **The Result:** A robust, fast, and 100% free local semantic search engine that respects strict 500MB infrastructure limits.

## 🛠 Recent Updates

- Local Embeddings — Switched to local sentence-transformers for better reliability and zero API costs.
- Optimized Vector Dimensions — Normalized all embeddings to 384 dimensions (optimized for speed and storage).
- CPU-Only Build Pipeline — Custom build command for Render to stay within resource quotas.
- Pydantic v2 Migration — Backend updated to leverage the latest Pydantic performance and validation features.

## ✨ Key Features

- **Semantic Indexing** — Local embeddings with 384-bit vector representation.
- **Smart Chunking** — Configurable overlap to preserve context across document splits.
- **Vector Search** — High-speed Cosine Similarity search via pgvector.
- **Security** — X-Custom-Token middleware, prompt injection guards, and UUID-based file handling.

## 💻 Tech Stack

- **Backend:** Python 3.10+, FastAPI, SQLAlchemy, pgvector, Sentence-Transformers (PyTorch CPU).
- **Frontend:** React, Vite, Tailwind CSS.
- **Database:** PostgreSQL with pgvector extension.

## 🚀 Setup & Installation

### Backend

```bash
cd backend
python -m venv venv
# Activate venv:
# Windows: .\venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
python setup_db.py    # Initialize DB (first run only)
uvicorn app.main:app --reload
```

### Environment Variables (.env)

```env
DATABASE_URL=postgresql://user:password@host/dbname
GEMINI_API_KEY=your_key_for_llm_responses_only
EMBEDDING_DIM=384
APP_SECURITY_TOKEN=your_custom_token
```

## ⚠️ Troubleshooting

### Vector Dimension Mismatch
If you see a DB error, ensure your column is vector(384):

```sql
ALTER TABLE documents ALTER COLUMN embedding TYPE vector(384);
```

### Render Deploy
Ensure your Build Command is set to:
```
pip install --no-cache-dir -r requirements.txt
```
And PYTHON_VERSION is set to 3.10.x.