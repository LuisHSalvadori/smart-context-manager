import io
import uuid
from pypdf import PdfReader
from app.services.vector_service import generate_embedding
from app.services.db_service import insert_document

# OPTIMIZED SETTINGS:
# We reduce chunk_size to 500 characters. 
# This creates more 'data points' in the database, ensuring specific sections 
# like "Languages" or "Tech Stack" on Page 2 are not drowned out by Page 1 headers.
def extract_text_chunks(pdf_bytes: bytes, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Reads a PDF from bytes, extracts text page by page, and splits it into 
    smaller, overlapping chunks to improve RAG retrieval accuracy.
    """
    pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
    full_text = ""
    
    # Iterate through pages and append text with a page marker 
    for i, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        if page_text:
            # Adding page numbers helps the AI identify where information was found
            full_text += f"\n[Page {i+1}] " + page_text

    # Return an empty list if no text could be extracted
    if not full_text.strip():
        return []

    # Sliding window chunking logic
    chunks = []
    # (chunk_size - overlap) ensures we don't lose context between the split points
    for i in range(0, len(full_text), chunk_size - overlap):
        chunks.append(full_text[i : i + chunk_size])
    return chunks

def index_pdf_content(filename: str, pdf_bytes: bytes) -> tuple[str, int, int, int]:
    """
    Orchestrates the indexing process:
    1. Generates a unique file ID.
    2. Splits PDF into optimized chunks.
    3. Generates vector embeddings for each chunk.
    4. Saves content to the vector database.
    """
    pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
    page_count = len(pdf_reader.pages)
    
    # Generate a unique identifier to avoid filename collisions
    safe_filename = f"{uuid.uuid4()}_{filename.replace(' ', '_')}"
    
    # Extract chunks using our new, smaller default sizes
    chunks = extract_text_chunks(pdf_bytes)
    total_chars = sum(len(chunk) for chunk in chunks)
    
    # Batch process each chunk
    for chunk in chunks:
        # We generate a vector embedding (a mathematical map) of the text
        embedding = generate_embedding(chunk)
        
        # We store the metadata alongside the text so the AI knows the source
        insert_document(f"Source: {safe_filename} | {chunk}", embedding)
        
    return safe_filename, len(chunks), page_count, total_chars