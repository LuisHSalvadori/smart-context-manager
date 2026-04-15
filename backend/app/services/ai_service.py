import logging
from google import genai
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception
from app.core.config import settings

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize the modern Gemini Client
# The new SDK uses a single client instance
client = genai.Client(api_key=settings.GEMINI_API_KEY)

# Modern model names for the new SDK
FALLBACK_MODELS = ["gemini-1.5-flash", "gemini-1.5-flash-8b"]

def is_rate_limit_error(exception):
    """Checks if the error is a 429 Rate Limit to trigger retry logic."""
    return "429" in str(exception)

@retry(
    wait=wait_exponential(multiplier=1, min=1, max=4), 
    stop=stop_after_attempt(2),
    retry=retry_if_exception(is_rate_limit_error),
    before_sleep=lambda retry_state: logger.warning(
        f"Rate limit hit. Retrying AI call... Attempt {retry_state.attempt_number}"
    )
)
def fetch_ai_response(model_name: str, prompt: str) -> str:
    """
    Executes the generation call using the NEW google-genai SDK.
    """
    # In the new SDK, we call models.generate_content directly from the client
    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )
    
    if response and response.text:
        return response.text.strip()
    
    raise ValueError(f"Model {model_name} returned an empty response.")

def generate_safe_answer(prompt: str) -> str:
    """
    Orchestrates AI response generation with fallbacks using the modern SDK.
    """
    for model_name in FALLBACK_MODELS:
        try:
            logger.info(f"Attempting to generate answer using model: {model_name}")
            return fetch_ai_response(model_name, prompt)
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                logger.error(f"Quota exceeded for model {model_name} after retries.")
            else:
                logger.error(f"Unexpected failure for model {model_name}: {error_msg[:100]}")
            continue
            
    raise RuntimeError("All AI models failed or are currently unavailable.")