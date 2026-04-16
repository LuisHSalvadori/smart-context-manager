# Smart Context Manager 🧠📂

Smart Context Manager is a modular RAG system for high-precision PDF interaction. It combines FastAPI + PostgreSQL(pgvector) backend with React + Vite frontend, using Google GenAI for embeddings and LLM responses.

## Recent Updates
- **Embeddings via Google GenAI** — model discovery + automatic fallback on API errors.
- **Vector normalization** — all embeddings stored at 768 dims (matches Supabase pgvector).
- **Graceful chunk skipping** — failed embeddings logged and skipped (no DB crashes).
- **Pydantic v1 pinned** — ensures FastAPI 0.135.3 compatibility.

## Key Features
- **Semantic indexing** — embeddings via Google GenAI, normalized to 768 dimensions.
- **Smart chunking** — configurable overlap to preserve context across splits.
- **AI resilience** — automatic model fallback and rate-limit retries.
- **Security** — X-Custom-Token middleware, prompt injection guards, filename UUIDing.
- **Vector search** — cosine similarity via pgvector.

## Tech Stack
**Backend**: Python 3.10+, FastAPI, SQLAlchemy, pgvector, Google GenAI  
**Frontend**: React, Vite, Tailwind CSS

## Setup

### Backend
```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
python setup_db.py    # initialize DB (first run only)
uvicorn app.main:app --reload
```

Create `backend/.env`:
```env
DATABASE_URL=postgresql://user:password@host/dbname
GEMINI_API_KEY=your_gemini_api_key
ALLOWED_ORIGINS=http://localhost:5173
APP_SECURITY_TOKEN=your_custom_token
EMBEDDING_DIM=768
```

### Frontend
```bash
cd frontend
npm install
# Create frontend/.env:
# VITE_API_URL=http://localhost:8000
# VITE_APP_SECURITY_TOKEN=your_custom_token
npm run dev
```

## Troubleshooting
- **ImportError / pydantic issues**: Ensure `pydantic<2` in requirements.txt. Recreate venv if needed:
  ```bash
  rm -r venv  # or: Remove-Item -Recurse venv (Windows)
  python -m venv venv
  venv\Scripts\activate
  pip install -r requirements.txt
  ```
- **Embedding failures**: Check logs for "Discovered embedding candidates" and "✅ Success". Verify GEMINI_API_KEY.
- **Vector dimension mismatch**: Confirm DB column is `vector(768)`. If changed, update EMBEDDING_DIM in .env and migrate DB:
  ```sql
  ALTER TABLE documents ALTER COLUMN embedding TYPE vector(768);
  ```

## Project Structure
```
├── backend/
│   ├── app/
│   │   ├── api/          # endpoints & routes
│   │   ├── core/         # config & security
│   │   ├── db/           # database & models
│   │   ├── services/     # business logic
│   │   └── main.py
│   ├── .env
│   ├── requirements.txt
│   └── setup_db.py
├── frontend/             # React app
└── README.MD
```

## Roadmap
- Supabase Auth (multi-tenant)
- Content-based deduplication (SHA-256)
- Hybrid search (BM25 + vector)
- Enhanced source citation UI