import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from app.core.config import settings

# Logger configuration for tracking service operations
logger = logging.getLogger(__name__)

# Pre-loading the model at the module level to ensure it's warmed up during app startup.
# 'all-MiniLM-L6-v2' is a lightweight, high-performance model optimized for document retrieval.
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("SentenceTransformer model 'all-MiniLM-L6-v2' loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load SentenceTransformer model: {str(e)}")
    raise

def generate_embedding(text: str) -> Optional[List[float]]:
    """
    Generates a high-dimensional vector (embedding) for the given text using 
    a local SentenceTransformer model.

    Args:
        text (str): The input string to be vectorized.

    Returns:
        Optional[List[float]]: A list of floats representing the embedding, 
                               or None if the input is invalid or an error occurs.
    """
    if not text or not text.strip():
        logger.warning("Received empty or whitespace-only text for embedding.")
        return None

    try:
        # Generate the embedding using CPU-based inference
        # The result is a numpy array by default
        embedding = model.encode(text)
        
        # Convert numpy array to a standard Python list for database compatibility (pgvector)
        vector_list = embedding.tolist()
        
        logger.debug(f"Embedding generated successfully. Dimensions: {len(vector_list)}")
        return vector_list
            
    except Exception as e:
        logger.error(f"Inference failure during local embedding generation: {str(e)}")
        return None