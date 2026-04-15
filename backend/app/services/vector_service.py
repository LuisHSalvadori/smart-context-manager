from sentence_transformers import SentenceTransformer

# Light and efficient model for generating 384-dimensional vector embeddings
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def generate_embedding(text: str) -> list[float]:
    """
    Converts a string of text into a numerical vector (embedding).
    Returns an empty list if the input text is empty or contains only whitespace.
    """
    if not text or not text.strip():
        return []
    
    # Generate the embedding using the SentenceTransformer model
    embedding = model.encode(text)
    
    # Convert the numpy array result to a standard Python list for database compatibility
    return embedding.tolist()