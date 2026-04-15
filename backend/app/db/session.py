import psycopg2
from app.core.config import settings

def get_db_connection():
    """
    Create new connection with supabase.
    """
    try:
        conn = psycopg2.connect(settings.DATABASE_URL)
        return conn
    except Exception as e:
        print(f"Erro ao conectar ao banco de dados: {e}")
        raise e
    
    