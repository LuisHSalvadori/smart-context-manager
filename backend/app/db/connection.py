import psycopg2
from app.core.config import settings

def get_connection():
    """
    Creates and returns a new connection to the PostgreSQL database 
    using the credentials defined in the environment settings.
    """
    return psycopg2.connect(settings.DATABASE_URL)