import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def setup_database():
    """
    Initializes the database by enabling the pgvector extension 
    and creating the necessary schema for document storage.
    """
    conn = None
    try:
        db_url = os.getenv("DATABASE_URL")
        
        if not db_url:
            print("Error: DATABASE_URL not found in environment variables.")
            return

        print("Connecting to PostgreSQL...")
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        # Enable pgvector extension to support vector data types
        print("Enabling pgvector extension...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Create documents table
        # vector(384) corresponds to the dimensions of all-MiniLM-L6-v2 model
        print("Initializing schema...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT,
                embedding vector(768),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        conn.commit()
        cur.close()
        print("Database initialized successfully.")
        
    except Exception as e:
        print(f"Database setup failed: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    setup_database()