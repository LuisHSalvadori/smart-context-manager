import time
import logging
from google import genai
from app.core.config import settings

# Logger setup
logger = logging.getLogger(__name__)

# Initialize the new Gemini Client
client = genai.Client(api_key=settings.GEMINI_API_KEY)

def generate_embedding(text: str) -> list[float]:
    """
    Generates 768-dimension embeddings using the NEW Google GenAI SDK.
    This bypasses the deprecated google-generativeai 404 errors.
    """
    if not text or not text.strip():
        return None

    # Modern model names
    fallbacks = ["text-embedding-004", "embedding-001"]

    for model_name in fallbacks:
        try:
            # The new SDK call format
            result = client.models.embed_content(
                model=model_name,
                contents=text,
                config={
                    "task_type": "RETRIEVAL_DOCUMENT",
                    "output_dimensionality": 768
                }
            )
            
            if result and result.embeddings:
                # The new SDK returns a list of embeddings
                logger.info(f"✅ Success: Embedding generated with {model_name}")
                return result.embeddings[0].values
        
        except Exception as e:
            logger.error(f"❌ Failure on model {model_name}: {str(e)[:100]}")
            continue

    logger.critical("🚨 All embedding models failed with the new SDK.")
    return None