import logging
from google import genai
from google.genai import types 
from app.core.config import settings

logger = logging.getLogger(__name__)

# Client Configuration: Explicitly forcing 'v1' to prevent 404 errors 
# that occur when the SDK defaults to the 'v1beta' endpoint.
client = genai.Client(
    api_key=settings.GEMINI_API_KEY,
    http_options={'api_version': 'v1'}
)

def generate_embedding(text: str) -> list[float]:
    """
    Generates 768-dimension embeddings using the modern SDK and official types.
    """
    if not text or not text.strip():
        return None

    # Stable production model recommended for general retrieval tasks.
    model_name = "text-embedding-004"

    try:
        # Utilizing the exact structure defined in the official Google GenAI documentation.
        result = client.models.embed_content(
            model=model_name,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768
            )
        )
        
        # The new SDK returns a list of embedding objects.
        # Accessing the first result and extracting its numerical values.
        if result and result.embeddings:
            logger.info(f"✅ Success: Embedding generated with {model_name}")
            return result.embeddings[0].values
            
    except Exception as e:
        logger.error(f"❌ Failure on model {model_name}: {str(e)}")

    return None