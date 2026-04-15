import logging
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_message
from google import genai
from app.core.config import settings

# Initialize logger for tracking AI service performance and errors
logger = logging.getLogger(__name__)

# Initialize the Gemini client using the API key from centralized settings
client = genai.Client(api_key=settings.GEMINI_API_KEY)

# List of models to try in order of preference. 
# We start with the stable latest and move to the 2.5 lite version as a backup.
FALLBACK_MODELS = ["gemini-flash-latest", "gemini-2.5-flash-lite"]

@retry(
    # Exponential backoff: waits 1s, 2s... up to 4s between retries
    wait=wait_exponential(multiplier=1, min=1, max=4), 
    # Stops attempting after 2 failed tries for a specific model
    stop=stop_after_attempt(2),
    # Only triggers a retry if the error is a 429 (Rate Limit Exceeded)
    retry=retry_if_exception_message(match=".*429.*"),
    # Logs a warning before the next retry attempt within the same model
    before_sleep=lambda retry_state: logger.warning(
        f"Rate limit hit. Retrying AI call... Attempt {retry_state.attempt_number}"
    )
)
def fetch_ai_response(model_name: str, prompt: str) -> str:
    """
    Executes the remote call to the Google GenAI API for a specific model.
    This function is wrapped with retry logic specifically for handling rate limits.
    """
    response = client.models.generate_content(model=model_name, contents=prompt)
    
    # Check if the response exists and contains valid text content
    if response and getattr(response, "text", None):
        return response.text.strip()
    
    # Raise error if response is empty to trigger the retry or the next fallback
    raise ValueError(f"Model {model_name} returned an empty response.")

def generate_safe_answer(prompt: str) -> str:
    """
    Orchestrates the AI response generation by iterating through available models.
    Provides a safety net by falling back to alternative models if the primary fails.
    """
    for model_name in FALLBACK_MODELS:
        try:
            logger.info(f"Attempting to generate answer using model: {model_name}")
            return fetch_ai_response(model_name, prompt)
            
        except Exception as e:
            error_msg = str(e)
            
            # If we hit a 429 even after retries, log it and move to the next model
            if "429" in error_msg:
                logger.error(f"Quota exceeded for model {model_name} after retries.")
            else:
                logger.error(f"Unexpected failure for model {model_name}: {e}")
            
            # Continue to the next model in FALLBACK_MODELS
            continue
            
    # Final exception if all models in the list fail or are throttled
    raise RuntimeError("All AI models failed or are currently unavailable. Please try again later.")