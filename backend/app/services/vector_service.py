import logging
from google import genai
from google.genai import types 
from app.core.config import settings

logger = logging.getLogger(__name__)

# Client Configuration: Forcing 'v1' to bypass the 404 errors seen in logs.
client = genai.Client(
    api_key=settings.GEMINI_API_KEY,
    http_options={'api_version': 'v1'}
)

def generate_embedding(text: str) -> list[float]:
    """
    Generates embeddings using the most compatible model and handling response structure.
    """
    if not text or not text.strip():
        return None

    # Adding 'models/' prefix as suggested by the API error message.
    model_name = "models/embedding-001"

    try:
        # Note: model 001 does NOT support output_dimensionality. 
        # It is fixed at 768.
        result = client.models.embed_content(
            model=model_name,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            )
        )
        
        # Checking both 'embedding' and 'embeddings' to be safe across SDK versions.
        if result:
            if hasattr(result, 'embedding') and result.embedding:
                logger.info(f"✅ Success: Embedding generated via .embedding")
                return result.embedding.values
            elif hasattr(result, 'embeddings') and result.embeddings:
                logger.info(f"✅ Success: Embedding generated via .embeddings[0]")
                return result.embeddings[0].values
            
    except Exception as e:
        logger.error(f"❌ Failure on model {model_name}: {str(e)}")

    return None