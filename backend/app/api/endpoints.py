from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from app.core.config import settings
from app.api.deps import verify_token
from app.services.pdf_service import index_pdf_content
from app.services.vector_service import generate_embedding
from app.services.db_service import search_documents as db_search
from app.services.ai_service import generate_safe_answer
import logging

# Standard logging setup for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(verify_token)])

# Security: List of patterns to prevent prompt injection attacks
BANNED_PATTERNS = [
    "ignore all previous instructions",
    "ignore previous instructions",
    "system prompt",
    "reveal your instructions",
    "forget everything",
    "as an administrator",
    "become a hacker",
    "prompt injection",
]

def validate_query_safety(query: str):
    """
    Checks the user query for potential malicious patterns or excessive length.
    """
    clean_query = query.lower().strip()
    for pattern in BANNED_PATTERNS:
        if pattern in clean_query:
            return False, f"Potential prompt injection detected. The phrase '{pattern}' is not allowed."
    if len(query) > 500:
        return False, "Query is too long. Please limit your question to 500 characters."
    return True, None

@router.get("/health", tags=["Health"], summary="Health Check")
def health_check():
    return {"status": "online", "service": "Smart Context Manager API"}

@router.post(
    "/upload-pdf", 
    tags=["Processing"],
    summary="Upload and index PDF"
)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Receives a PDF, splits it into chunks, generates embeddings, 
    and stores them in the vector database.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    pdf_bytes = await file.read()
    file_size_bytes = len(pdf_bytes)
    
    logger.info(f"--- UPLOAD START: {file.filename} ({file_size_bytes} bytes) ---")

    if file_size_bytes > settings.max_file_size:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 2MB.")

    # Calculate readable file size for the frontend
    if file_size_bytes < 1024 * 1024:
        readable_size = f"{round(file_size_bytes / 1024, 2)} KB"
    else:
        readable_size = f"{round(file_size_bytes / (1024 * 1024), 2)} MB"

    # Process the PDF and get metadata
    safe_filename, chunks, pages, chars = index_pdf_content(file.filename, pdf_bytes)
    
    logger.info(f"INDEXING COMPLETE: {chunks} chunks stored for {safe_filename}")

    if chunks == 0:
        raise HTTPException(status_code=400, detail="Empty PDF content or unreadable text.")
    
    return {
        "status": "success",
        "metadata": {
            "filename": safe_filename,
            "original_name": file.filename,
            "file_size": readable_size,
            "page_count": pages,
            "total_characters": chars,
            "chunks_processed": chunks
        }
    }

@router.get(
    "/search", 
    tags=["AI Search"],
    summary="Search and Generate Answer"
)
async def handle_search(query: str, limit: int = 10): # Defaulted to 10 to match your new frontend limit
    """
    Main RAG Endpoint:
    1. Validates query safety.
    2. Converts query to a vector.
    3. Retrieves 'limit' number of chunks from the DB.
    4. Sends context to the AI for the final answer.
    """
    logger.info(f"--- SEARCH START: '{query}' with limit {limit} ---")

    is_safe, error_message = validate_query_safety(query)
    if not is_safe:
        logger.warning(f"BLOCKED QUERY: {error_message}")
        raise HTTPException(status_code=400, detail=error_message)

    # Convert the search string into a mathematical vector
    query_vector = generate_embedding(query)
    
    # FIX: We now pass the 'limit' directly to the DB search
    # We fetch slightly more chunks (limit + 2) to account for similarity filtering
    results = db_search(query_vector, limit=limit + 2)
    
    # Debug: See which chunks are being picked up
    for i, row in enumerate(results):
        logger.info(f"Raw Result {i+1}: Score {row[1]:.4f} | Text snippet: {row[0][:50]}...")

    # Filter results by similarity score (0.12) and slice by the requested limit
    filtered = [row for row in results if row[1] > 0.12][:limit]
    
    if not filtered:
        logger.info("SEARCH END: No context found above threshold.")
        return {
            "query": query, 
            "answer": "No relevant information found in the documents.",
            "results": []
        }

    # Combine all retrieved text chunks into one context block for the AI
    context_text = "\n---\n".join([row[0] for row in filtered])
    
    logger.info(f"AI CALL: Sending prompt with {len(filtered)} context chunks.")

    # Building the final prompt for Gemini
    prompt = f"""
    INSTRUCTION: You are a professional document analysis assistant.
    Answer the user's question accurately using ONLY the context provided below.
    If the answer is not in the context, say that you don't have that information.

    CONTEXT:
    {context_text}

    USER QUESTION:
    {query}
    """
    
    try:
        answer = generate_safe_answer(prompt)
        logger.info("--- SEARCH SUCCESS: AI Answer Generated ---")
    except Exception as exc:
        logger.error(f"--- SEARCH FAILURE: AI Service Error: {exc} ---")
        raise HTTPException(
            status_code=503, 
            detail=f"AI Service unavailable. Internal error: {str(exc)}"
        )

    return {
        "query": query, 
        "answer": answer, 
        "source_count": len(filtered),
        # Return chunks and scores to the frontend for transparency
        "results": [{"content": row[0], "similarity": row[1]} for row in filtered]
    }