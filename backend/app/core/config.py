import os
from dotenv import load_dotenv
# If you encounter an import error, run: pip install pydantic-settings
from pydantic_settings import BaseSettings 

load_dotenv()

class Settings(BaseSettings):
    # Defining in uppercase to match standard .env naming conventions
    GEMINI_API_KEY: str
    DATABASE_URL: str
    APP_SECURITY_TOKEN: str
    
    # Default values to simplify local development
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    max_file_size: int = 2 * 1024 * 1024

    @property
    def origins(self) -> list[str]:
        """Converts the allowed_origins string into a list for CORS configuration."""
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Ignores additional variables present in the .env file not defined here
        extra = "ignore" 

settings = Settings()