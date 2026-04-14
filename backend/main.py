import os
import io
import psycopg2
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai

# Local modules
from embeddings import generate_embedding

load_dotenv()

app = FastAPI(title="Smart Context Manager API")

# Configuration
API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not API_KEY or not DATABASE_URL:
    raise ValueError("Missing critical environment variables (GEMINI_API_KEY or DATABASE_URL)")

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI Client Initialization
client = genai.Client(api_key=API_KEY)

class DocumentRequest(BaseModel):
    content: str

@app.get("/")
def health_check():
    return {"status": "online", "service": "Smart Context Manager API"}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Extracts text from PDF, splits it into chunks with overlap, 
    and stores them as vector embeddings in PostgreSQL.
    """
    try:
        pdf_content = await file.read()
        pdf_reader = PdfReader(io.BytesIO(pdf_content))
        
        full_text = ""
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                full_text += f"\n[Page {i+1}] " + page_text

        # Chunking Logic (1000 chars with 200 char overlap)
        chunk_size = 1000
        overlap = 200
        chunks = []
        
        for i in range(0, len(full_text), chunk_size - overlap):
            chunk = full_text[i : i + chunk_size]
            chunks.append(chunk)

        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        for chunk in chunks:
            vector = generate_embedding(chunk)
            content_to_store = f"Source: {file.filename} | {chunk}"
            cur.execute(
                "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
                (content_to_store, str(vector))
            )
        
        conn.commit()
        cur.close()
        conn.close()
        
        return {"filename": file.filename, "chunks_processed": len(chunks)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Processing Error: {str(e)}")

@app.get("/search")
async def search_documents(query: str, limit: int = 3):
    """
    Performs semantic search and generates an answer using LLM 
    with an automatic fallback strategy for model availability.
    """
    try:
        # 1. Vector Search
        query_vector = generate_embedding(query)
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT content, 1 - (embedding <=> %s) AS similarity
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT %s;
        """, (str(query_vector), str(query_vector), limit))
        
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return {"query": query, "answer": "No relevant information found in the database."}

        context_text = "\n---\n".join([row[0] for row in rows])

        # 2. LLM Prompt Construction
        prompt = f"""
        You are a document analysis assistant.
        Answer the user's question based ONLY on the context provided.
        If the information is not present, state that it was not found.

        CONTEXT:
        {context_text}

        USER QUESTION:
        {query}
        """

        # 3. Model Fallback Strategy
        models_to_try = ["gemini-2.0-flash-lite", "gemini-flash-latest"]
        response = None
        
        for model_name in models_to_try:
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                break 
            except Exception:
                continue # Try next model if quota is exhausted

        if not response:
            raise HTTPException(status_code=503, detail="AI Service temporarily unavailable")

        return {
            "query": query, 
            "answer": response.text, 
            "source_count": len(rows)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))