from typing import List, Tuple
from app.db.connection import get_connection

def insert_document(content: str, embedding: List[float]):
    """
    Inserts a text chunk and its corresponding vector embedding into the database.
    Ensures the connection is closed even if the operation fails.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
            (content, str(embedding))
        )
        conn.commit()
        cur.close()
    finally:
        # Always close the connection to prevent memory leaks or hung processes
        conn.close()

def search_documents(query_vector: List[float], limit: int = 5) -> List[Tuple[str, float]]:
    """
    Performs a vector similarity search (Nearest Neighbor) using the cosine distance operator (<=>).
    Returns a list of tuples containing the content and the calculated similarity score.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        # 1 - (distance) is used to convert the distance into a similarity score (0 to 1)
        cur.execute("""
            SELECT content, 1 - (embedding <=> %s) AS similarity
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT %s;
        """, (str(query_vector), str(query_vector), limit))
        results = cur.fetchall()
        cur.close()
        return results
    finally:
        conn.close()