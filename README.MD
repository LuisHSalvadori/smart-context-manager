Smart Context Manager 🧠📂

Smart Context Manager is a professional RAG (Retrieval-Augmented Generation) system that allows users to chat with their PDF documents. It combines FastAPI, PostgreSQL with the pgvector extension, and Google Gemini AI to deliver precise, context-aware answers based on document content.
🚀 Key Features

    Semantic Document Indexing: Converts PDF text into 384-dimensional vector embeddings using the all-MiniLM-L6-v2 transformer model.

    Smart Chunking Strategy: Implements text fragmentation (1000 chars) with overlap (200 chars) to ensure no context is lost between paragraphs.

    AI Fallback System: High availability logic that switches between Gemini models (flash-lite and flash-latest) to ensure continuous service.

    Modern UI: A sleek, dark-themed dashboard built with React and Tailwind CSS.

    Vector Search: Real-time similarity search using Cosine Distance through the pgvector extension, optimized with a calibrated threshold for multilingual support.

🛡️ Security & Reliability Layer

This project implements advanced production-ready security practices:

    Input Guardrails: Basic pattern matching to detect and block common prompt injection attempts (e.g., "ignore all previous instructions").

    Security Token Middleware: Custom X-Custom-Token header validation required for all API requests.

    Output Guardrails: Post-generation checks to ensure the AI doesn't leak system instructions or bypass safety rules.

    Similarity Thresholding: A calibrated threshold of 0.15 to ensure the AI only answers based on highly relevant context, preventing hallucinations.

    CORS Policy: Strict origin restriction to ensure only the legitimate frontend can communicate with the backend.

🛠️ Tech Stack
Backend

    Language: Python 3.10+

    Framework: FastAPI

    AI/NLP: Sentence-Transformers (HuggingFace), Google GenAI (Gemini)

    Database: PostgreSQL + pgvector

    Security: UUID filename sanitization & Security Middleware

Frontend

    Library: React.js

    Styling: Tailwind CSS

    Environment: Vite

📦 Prerequisites

Before running the project, make sure you have:

    Docker & Docker Compose

    Python 3.10+

    Node.js & npm/yarn

    A Google Gemini API Key

🔧 Installation & Setup
1. Database & Infrastructure

The project uses Docker to provide a ready-to-use PostgreSQL instance with the vector extension.
Bash

docker-compose up -d

2. Backend Setup
Bash

cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Setup .env
# Required: GEMINI_API_KEY, DATABASE_URL, APP_SECURITY_TOKEN, ALLOWED_ORIGINS

# Initialize Database Schema
python setup_db.py

3. Frontend Setup
Bash

cd frontend
npm install

# Setup .env
# Required: VITE_API_URL, VITE_APP_SECURITY_TOKEN

npm run dev

📂 Project Structure
Plaintext

├── backend/
│   ├── main.py            # FastAPI routes, RAG logic & Security Guardrails
│   ├── database.py        # Connection and Session management (SQLAlchemy)
│   ├── embeddings.py      # AI Model for vector generation (Sentence-Transformers)
│   ├── setup_db.py        # Database schema & pgvector setup
├── frontend/
│   ├── src/App.jsx        # Modern Dashboard UI
└── docker-compose.yml     # Infrastructure (PostgreSQL + pgvector)