from sentence_transformers import SentenceTransformer

# Initialize the model globally to avoid reloading on every request
# 'all-MiniLM-L6-v2' is a lightweight and efficient model for semantic search
MODEL_NAME = 'all-MiniLM-L6-v2'
model = SentenceTransformer(MODEL_NAME)

def generate_embedding(text: str) -> list[float]:
    """
    Converts a string of text into a numerical vector (embedding) 
    using the sentence-transformers model.
    """
    if not text.strip():
        return []
        
    embedding = model.encode(text)
    return embedding.tolist()

if __name__ == "__main__":
    # Local module testing
    sample_text = "How to manage documents using artificial intelligence?"
    vector = generate_embedding(sample_text)
    
    print(f"Model: {MODEL_NAME}")
    print(f"Text: {sample_text}")
    print(f"Dimensions: {len(vector)}")
    print(f"Preview (first 5): {vector[:5]}")