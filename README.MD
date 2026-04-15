# Smart Context Manager 🧠📂

Smart Context Manager is a professional RAG (Retrieval-Augmented Generation) system for high-precision interaction with PDF documents. It uses a modular backend (FastAPI + PostgreSQL + pgvector) and a React frontend, integrating Google Gemini for AI responses grounded in uploaded content.

## Key Features
- **Semantic Document Indexing** — 384-dim embeddings via `all-MiniLM-L6-v2`.
- **Smart Chunking Strategy** — 1,000-character chunks with 200-character overlap.
- **AI Fallback System** — automatic model fallback (flash-latest → flash-lite) on rate limits.
- **Modern UI** — React, Vite and Tailwind CSS.
- **Vectorized Search** — cosine similarity with a calibrated multilingual threshold.

## Security & Reliability
- Security token middleware requiring `X-Custom-Token`.
- Prompt-injection guards and post-generation checks.
- UUID-based filename sanitization to prevent traversal/collisions.
- Similarity threshold set to `0.15` to add one more layer of security.
- Strict CORS origin policy.

## Tech Stack
**Backend**
- Python 3.10+
- FastAPI (modular)
- Sentence-Transformers (HuggingFace)
- Google GenAI (Gemini)
- PostgreSQL + pgvector

**Frontend**
- React.js
- Tailwind CSS
- Vite

## Prerequisites
- Python 3.10+
- Node.js & npm/yarn
- PostgreSQL with `pgvector`
- Google Gemini API key

## Installation & Setup

### 1) Backend
```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
pip install -r requirements.txt
```

Create `backend/.env` with:
```env
DATABASE_URL=your_postgresql_url
GEMINI_API_KEY=your_gemini_api_key
ALLOWED_ORIGINS=http://localhost:5173
APP_SECURITY_TOKEN=your_custom_token
```

Initialize DB and run server:
```bash
python setup_db.py
uvicorn app.main:app --reload
```

### 2) Frontend
```bash
cd frontend
npm install
# create .env with VITE_API_URL and VITE_APP_SECURITY_TOKEN
npm run dev
```

## Project Structure
```text
├── backend/
│   ├── app/
│   │   ├── api/        # API routes (upload & search)
│   │   ├── core/       # security middleware & configs
│   │   ├── db/         # DB sessions & models
│   │   ├── services/   # RAG logic, embeddings & AI
│   │   └── main.py     # app entrypoint
│   ├── .env
│   ├── setup_db.py
│   └── requirements.txt
├── frontend/           # React app
└── README.MD
```

## Roadmap
- Supabase Auth for private libraries
- Content-based hashing (SHA-256) to avoid duplicate indexing
- Re-ranking (cross-encoders) and hybrid search (Vector + BM25)
- UI source citation for snippets/pages