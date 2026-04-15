Smart Context Manager 🧠📂

Smart Context Manager is a professional RAG (Retrieval-Augmented Generation) system designed for high-precision interaction with PDF documents. Built with a modular architecture, it combines FastAPI, PostgreSQL with the pgvector extension, and Google Gemini AI to deliver context-aware answers grounded strictly in uploaded content.
🚀 Key Features

    Semantic Document Indexing: Converts PDF text into 384-dimensional vector embeddings using the all-MiniLM-L6-v2 transformer model.

    Smart Chunking Strategy: Implements optimized text fragmentation (1,000 chars) with strategic overlap (200 chars) to maintain semantic continuity.

    AI Fallback System: High-availability logic that automatically switches between Gemini models (flash-latest and flash-lite) if rate limits are hit, ensuring continuous service.

    Modern UI: A sleek, dark-themed dashboard built with React.js, Vite, and Tailwind CSS.

    Vectorized Search: Real-time similarity search using Cosine Distance, optimized with a calibrated threshold for multilingual support.

🛡️ Security & Reliability Layer

Built with production-ready standards to ensure data integrity:

    Security Token Middleware: All API requests require a custom X-Custom-Token header validation.

    Input & Output Guardrails: Integrated pattern matching to block prompt injection and post-generation checks to prevent system instruction leakage.

    Filename Sanitization: Implements UUID-based renaming for all uploads to prevent path traversal and file collisions.

    Similarity Thresholding: Calibrated at 0.15 to ensure the AI only utilizes highly relevant context, significantly reducing hallucinations.

    CORS Policy: Strict origin restrictions to ensure only authorized frontends can interact with the API.

🛠️ Tech Stack
Backend

    Language: Python 3.10+

    Framework: FastAPI (Modular Design)

    AI/NLP: Sentence-Transformers (HuggingFace), Google GenAI (Gemini)

    Database: PostgreSQL + pgvector

Frontend

    Library: React.js

    Styling: Tailwind CSS

    Build Tool: Vite

📦 Prerequisites

Before running the project, ensure you have:

    Python 3.10+

    Node.js & npm/yarn

    A PostgreSQL instance with the pgvector extension enabled

    A Google Gemini API Key

🔧 Installation & Setup
1. Backend Setup

Navigate to the backend directory and configure the environment:
Bash

cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

Create a .env file in the backend/ root with the following variables:
Snippet de código

DATABASE_URL=your_postgresql_url
GEMINI_API_KEY=your_gemini_api_key
ALLOWED_ORIGINS=http://localhost:5173
APP_SECURITY_TOKEN=your_custom_token

Initialize the database and run the server (from the backend/ folder):
Bash

# Setup Database Schema
python setup_db.py

# Run FastAPI from the modular entry point
uvicorn app.main:app --reload

2. Frontend Setup
Bash

cd frontend
npm install
# Set VITE_API_URL and VITE_APP_SECURITY_TOKEN in your .env
npm run dev

📂 Project Structure

The project follows a modular pattern for better maintainability:
Plaintext

├── backend/
│   ├── app/
│   │   ├── api/          # API Routes (Upload & Search Endpoints)
│   │   ├── core/         # Security Middleware & Configs
│   │   ├── db/           # Database Sessions & SQLAlchemy Models
│   │   ├── services/     # RAG Logic, Embeddings, & AI Service
│   │   └── main.py       # Application Entry Point
│   ├── .env              # Environment Variables
│   ├── setup_db.py       # Database Initialization Script
│   └── requirements.txt  # Python Dependencies
├── frontend/             # React Application
└── README.MD             # Project Documentation

🚀 Roadmap & Future Enhancements

    User Authentication: Integration with Supabase Auth for private document libraries.

    Intelligent File Management: Implementation of content-based hashing (SHA-256) to prevent duplicate indexing.

    Advanced RAG Techniques: Adding a Reranking Layer (Cross-Encoders) and Hybrid Search (Vector + BM25).

    Source Citation: Improving the UI to highlight specific snippets and pages used for each answer.