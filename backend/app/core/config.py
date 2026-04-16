import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    # API Keys and Security
    GEMINI_API_KEY: str
    DATABASE_URL: str
    APP_SECURITY_TOKEN: str
    
    # Renamed to uppercase to correctly map ALLOWED_ORIGINS from environment
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"
    max_file_size: int = 2 * 1024 * 1024
    EMBEDDING_DIM: int = 384

    @property
    def origins(self) -> list[str]:
        """Converts the ALLOWED_ORIGINS string into a list for CORS configuration."""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin]

settings = Settings()