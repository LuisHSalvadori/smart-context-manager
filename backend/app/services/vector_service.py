import time
import google.generativeai as genai
from app.core.config import settings

# Force REST transport to avoid gRPC routing issues on platforms like Render
# This also helps ensure we hit the v1 stable endpoints
genai.configure(api_key=settings.GEMINI_API_KEY, transport="rest")

def generate_embedding(text: str) -> list[float]:
    """
    Generates 768-dimension embeddings using the stable Google Generative AI SDK.
    Forces usage of v1 API via REST to avoid 404/v1beta errors.
    """
    if not text or not text.strip():
        return None

    # Official model identifiers for the stable v1 API
    fallbacks = [
        "models/text-embedding-004",
        "models/embedding-001"
    ]

    for model_name in fallbacks:
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Execution call using the stable SDK method
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
                error_msg = str(e).lower()
                wait_time = (attempt + 1) * 2
                
                print(f"❌ Failure on model {model_name}: {str(e)[:100]}")
                
                # If it's a rate limit error (429), we wait and retry.
                # If it's a 404 or other structural error, we break and switch models.
                if "429" in error_msg:
                    time.sleep(wait_time)
                else:
                    break 
        
        print(f"⚠️ Switching to next fallback model after failures on: {model_name}")

    print("🚨 CRITICAL: All embedding models failed.")
    return None