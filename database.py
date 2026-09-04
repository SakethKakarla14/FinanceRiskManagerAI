import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "unified_user.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create the user stats table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            ltv REAL DEFAULT 0.0,
            prior_returns INTEGER DEFAULT 0,
            prior_fraud INTEGER DEFAULT 0,
            last_login TEXT
        )
    """)
    conn.commit()
    
    # Insert some mock users for the sandbox UI to simulate against
    users = [
        ("user_safe", 500.0, 0, 0, "2026-09-01"),
        ("user_wardrober", 300.0, 3, 0, "2026-09-01"),
        ("user_scammer", 100.0, 5, 1, "2026-09-01"),
        ("user_chronic", 450.0, 4, 0, "2026-09-01"),
        ("LIVE_USER", 250.0, 1, 0, "2026-09-01")   # Default fallback user
    ]

    for user in users:
        cursor.execute("""
            INSERT OR IGNORE INTO users (user_id, ltv, prior_returns, prior_fraud, last_login)
            VALUES (?, ?, ?, ?, ?)
        """, user)

    conn.commit()
    conn.close()

def get_user_profile(user_id: str) -> dict:
    """Fetch user metrics from the unified database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None

def update_user_profile(user_id: str, ltv_increment: float, returned: bool, fraud: bool):
    """Update user metrics after a transaction."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE users 
        SET ltv = ltv + ?, 
            prior_returns = prior_returns + ?, 
            prior_fraud = prior_fraud + ?
        WHERE user_id = ?
    """, (ltv_increment, 1 if returned else 0, 1 if fraud else 0, user_id))
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized.")
