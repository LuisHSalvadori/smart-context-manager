import os
import psycopg2
from dotenv import load_dotenv
from embeddings import generate_embedding

load_dotenv()

def save_to_database(content: str):
    """
    Generates an embedding and persists the content and its 
    vector representation in the PostgreSQL database.
    """
    conn = None
    try:
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL is not defined in environment variables")

        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        
        print(f"Processing content: '{content[:50]}...'")
        vector = generate_embedding(content)
        
        # Insert both raw content and the generated vector
        cur.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
            (content, str(vector))
        )
        
        conn.commit()
        print("Successfully indexed document to database.")
        
        cur.close()
    except Exception as e:
        print(f"Failed to index document: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # Sample data for indexing test
    test_content = "The Smart Context Manager uses FastAPI and PostgreSQL."
    save_to_database(test_content)