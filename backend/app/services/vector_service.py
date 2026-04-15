from google import genai
from app.core.config import settings

# Inicializa o cliente do Google GenAI usando sua API Key das configurações
client = genai.Client(api_key=settings.GEMINI_API_KEY)

def generate_embedding(text: str) -> list[float]:
    """
    Converte uma string de texto em um vetor numérico (embedding) usando a API do Google.
    O modelo 'text-embedding-004' gera vetores de 768 dimensões.
    """
    if not text or not text.strip():
        return []

    try:
        # Gera o embedding usando o modelo profissional do Google
        result = client.models.embed_content(
            model="text-embedding-004",
            contents=text
        )
        
        # O resultado vem como uma lista de floats dentro do objeto de resposta
        return result.embeddings[0].values
    
    except Exception as e:
        print(f"Erro ao gerar embedding no Google: {e}")
        # Em caso de erro na API, retorna lista vazia para não quebrar o fluxo
        return []