import os
import psycopg2
from dotenv import load_dotenv
from embeddings import generate_embedding

load_dotenv()

def semantic_search(query: str, limit: int = 3):
    """
    Executes a vector similarity search in PostgreSQL using the pgvector <=> operator.
    Returns the most contextually relevant documents based on cosine distance.
    """
    conn = None
    try:
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL is not set")

        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        
        # Convert natural language query to vector embedding
        query_embedding = generate_embedding(query)
        
        # The <=> operator calculates cosine distance (1 - distance = similarity)
        cur.execute("""
            SELECT content, 1 - (embedding <=> %s) AS similarity
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT %s;
        """, (str(query_embedding), str(query_embedding), limit))
        
        results = cur.fetchall()
        
        if not results:
            print("No matching documents found.")
            return

        print(f"\n--- Search results for: '{query}' ---")
        for i, (content, similarity) in enumerate(results, 1):
            # Display similarity score and a snippet of the content
            snippet = content.replace('\n', ' ')[:120]
            print(f"{i}. [Score: {similarity:.4f}] {snippet}...")
            
        cur.close()
    except Exception as e:
        print(f"Search operation failed: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # Test query for terminal validation
    sample_query = "What technologies are used in this project?"
    semantic_search(sample_query)