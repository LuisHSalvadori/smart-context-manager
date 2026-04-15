import logging
from google import genai
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception
from app.core.config import settings

logger = logging.getLogger(__name__)

# Cliente unificado forçando a versão estável da API
client = genai.Client(
    api_key=settings.GEMINI_API_KEY,
    http_options={'api_version': 'v1'}
)

def is_rate_limit_error(exception):
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
    # Chamada direta via client.models seguindo a nova doc
    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )
    
    if response and response.text:
        return response.text.strip()
    
    raise ValueError(f"Model {model_name} returned an empty response.")

def generate_safe_answer(prompt: str) -> str:
    for model_name in ["gemini-1.5-flash", "gemini-1.5-flash-8b"]:
        try:
            logger.info(f"Attempting to generate answer using model: {model_name}")
            return fetch_ai_response(model_name, prompt)
        except Exception as e:
            logger.error(f"Unexpected failure for model {model_name}: {str(e)[:100]}")
            continue
            
    raise RuntimeError("All AI models failed.")