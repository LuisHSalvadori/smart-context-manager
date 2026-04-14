import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_connection():
    """
    Simple sanity check to verify if the application can 
    successfully connect to the PostgreSQL/Docker instance.
    """
    conn = None
    try:
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            print("Error: DATABASE_URL variable not found.")
            return

        print(f"Attempting to connect to database...")
        conn = psycopg2.connect(db_url)
        
        print("Connection successful: Database is reachable.")
        
    except Exception as e:
        print(f"Connection failed: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    test_connection()