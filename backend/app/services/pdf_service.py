import io
import uuid
from pypdf import PdfReader
from app.services.vector_service import generate_embedding
from app.services.db_service import insert_document

def extract_text_chunks(pdf_bytes: bytes, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Reads a PDF from bytes, extracts text page by page, and splits it into 
    smaller, overlapping chunks to improve RAG retrieval accuracy.
    """
    pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
    full_text = ""
    
    for i, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        if page_text:
            full_text += f"\n[Page {i+1}] " + page_text

    if not full_text.strip():
        return []

    chunks = []
    for i in range(0, len(full_text), chunk_size - overlap):
        chunks.append(full_text[i : i + chunk_size])
    return chunks

def index_pdf_content(filename: str, pdf_bytes: bytes) -> tuple[str, int, int, int]:
    """
    Orchestrates the indexing process with safety checks for embeddings.
    """
    pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
    page_count = len(pdf_reader.pages)
    
    safe_filename = f"{uuid.uuid4()}_{filename.replace(' ', '_')}"
    
    chunks = extract_text_chunks(pdf_bytes)
    total_chars = sum(len(chunk) for chunk in chunks)
    
    # Process each chunk with a safety gate
    for chunk in chunks:
        embedding = generate_embedding(chunk)
        
        # CRITICAL FIX: Only attempt to insert into DB if embedding generation succeeded.
        # This prevents 'psycopg2.errors.InvalidTextRepresentation' (None value in vector column).
        if embedding is not None:
            insert_document(f"Source: {safe_filename} | {chunk}", embedding)
        else:
            # We log the failure but continue processing other chunks to avoid crashing the whole upload.
            print(f"⚠️ Warning: Skipping chunk in {filename} due to embedding failure.")
        
    return safe_filename, len(chunks), page_count, total_chars