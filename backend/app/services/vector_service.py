import time
from google import genai
from app.core.config import settings

# Initialize the Google GenAI client
client = genai.Client(api_key=settings.GEMINI_API_KEY)

def generate_embedding(text: str) -> list[float]:
    """
    Generates 768-dimension embeddings with multiple fallbacks and semantic retry.
    Models included: text-embedding-004, text-multilingual-embedding-002,
    text-embedding-001, and textembedding-gecko@003.
    """
    if not text or not text.strip():
        return None

    # List of models based on technical research. 
    # Names vary between Vertex AI and AI Studio, so we include variations.
    fallbacks = [
        "models/text-embedding-004",
        "models/text-multilingual-embedding-002",
        "models/embedding-001",
        "models/text-embedding-gecko@003"
    ]

    for model_name in fallbacks:
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Call with dimensionality parameter (MRL) to ensure 768 dimensions
                result = client.models.embed_content(
                    model=model_name,
                    contents=text,
                    config={
                        "output_dimensionality": 768
                    }
                )

                if result and result.embeddings:
                    print(f"✅ Success: Embedding generated with model: {model_name}")
                    return result.embeddings[0].values

            except Exception as e:
                # For quota errors (429), we wait longer (Exponential Backoff)
                wait_time = (attempt + 1) * 2 
                
                print(f"❌ Failure on model {model_name}: {str(e)[:100]}... Retrying in {wait_time}s")
                
                # If model is not found (404), skip retries and move to next fallback
                if "404" in str(e) or "not found" in str(e).lower():
                    break
                
                time.sleep(wait_time)
        
        print(f"⚠️ Switching to next fallback model after failures on: {model_name}")

    print("🚨 CRITICAL: All embedding models failed.")
    return None