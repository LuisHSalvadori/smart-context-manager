Smart Context Manager 🧠📂

Smart Context Manager is a professional RAG (Retrieval-Augmented Generation) system that allows users to chat with their PDF documents. It combines FastAPI, PostgreSQL with the pgvector extension, and Google Gemini AI to deliver precise, context-aware answers based on document content.
🚀 Key Features

    Semantic Document Indexing: Converts PDF text into 384-dimensional vector embeddings using the all-MiniLM-L6-v2 transformer model.

    Smart Chunking Strategy: Implements text fragmentation with overlap to ensure no context is lost between paragraphs.

    AI Fallback System: High availability logic that switches between Gemini models (flash-lite and flash-latest) to ensure continuous service.

    Modern UI: A sleek, dark-themed dashboard built with React and Tailwind CSS.

    Vector Search: Real-time similarity search using Cosine Distance through the pgvector extension.

🛠️ Tech Stack
Backend

    Language: Python 3.10+

    Framework: FastAPI

    ORM: SQLAlchemy

    AI/NLP: Sentence-Transformers (HuggingFace), Google GenAI (Gemini)

    Database: PostgreSQL + pgvector

Frontend

    Library: React.js

    Styling: Tailwind CSS

    State Management: React Hooks (useState)

📦 Prerequisites

Before running the project, make sure you have:

    Docker & Docker Compose

    Python 3.10+

    Node.js & npm/yarn (for the frontend)

    A Google Gemini API Key

🔧 Installation & Setup
1. Database & Infrastructure

The project uses Docker to provide a ready-to-use PostgreSQL instance with the vector extension.
Bash

docker-compose up -d

2. Backend Setup

Navigate to the /backend folder:
Bash

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure Environment Variables
# Copy .env.example to .env and fill in your GEMINI_API_KEY
copy .env.example .env

# Initialize Database Schema
python setup_db.py

3. Frontend Setup

Navigate to the /frontend folder:
Bash

npm install
npm run dev

📂 Project Structure
Plaintext

├── backend/
│   ├── main.py            # FastAPI routes & RAG logic
│   ├── database.py        # Connection and Session management
│   ├── embeddings.py      # AI Model for vector generation
│   ├── setup_db.py        # Database schema & pgvector setup
│   └── index_document.py  # Utility for manual indexing
├── frontend/
│   ├── src/App.jsx        # Modern Dashboard UI
└── docker-compose.yml     # Infrastructure (PostgreSQL + pgvector)