import time
import google.generativeai as genai
from app.core.config import settings

# Configure the SDK with your API Key
genai.configure(api_key=settings.GEMINI_API_KEY)

def generate_embedding(text: str) -> list[float]:
    """
    Generates 768-dimension embeddings using the stable Google Generative AI SDK.
    Includes fallbacks for high availability and MRL for consistent dimensionality.
    """
    if not text or not text.strip():
        return None

    # List of stable model identifiers for Google AI Studio
    fallbacks = [
        "models/text-embedding-004",
        "models/embedding-001"
    ]

    for model_name in fallbacks:
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Using the stable embed_content method
                # output_dimensionality=768 ensures compatibility with Supabase
                result = genai.embed_content(
                    model=model_name,
                    content=text,
                    task_type="retrieval_document",
                    output_dimensionality=768
                )

                if result and 'embedding' in result:
                    print(f"✅ Success: Embedding generated with model: {model_name}")
                    return result['embedding']

            except Exception as e:
                # Exponential backoff for rate limits or temporary hiccups
                wait_time = (attempt + 1) * 2
                error_msg = str(e).lower()
                
                print(f"❌ Failure on model {model_name}: {str(e)[:100]}... Retrying in {wait_time}s")
                
                # If the model is explicitly not found, don't bother retrying
                if "404" in error_msg or "not found" in error_msg:
                    break
                
                time.sleep(wait_time)
        
        print(f"⚠️ Switching to next fallback model after failures on: {model_name}")

    print("🚨 CRITICAL: All embedding models failed.")
    return None