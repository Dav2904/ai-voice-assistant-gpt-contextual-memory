import os
import sqlite3
import time
import threading
from typing import List, Tuple, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "chat_history.db")


def _ensure():
    os.makedirs(DATA_DIR, exist_ok=True)


class ChatStore:
    """
    Persists *all* chat messages across refresh/restart.
    Schema:
      sessions(user_id)
      messages(user_id, role, text, ts)
    Thread-safe for Streamlit.
    """
    def __init__(self):
        _ensure()
        self.lock = threading.Lock()
        self.con = sqlite3.connect(DB_PATH, check_same_thread=False)

        with self.lock:
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    user_id TEXT PRIMARY KEY,
                    created_at REAL
                )
            """)
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,      -- 'user' or 'assistant'
                    text TEXT NOT NULL,
                    ts REAL NOT NULL
                )
            """)
            self.con.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_ts ON messages(user_id, ts)")
            self.con.commit()

    def ensure_session(self, user_id: str):
        with self.lock:
            self.con.execute(
                "INSERT OR IGNORE INTO sessions(user_id, created_at) VALUES (?, ?)",
                (user_id, time.time())
            )
            self.con.commit()

    def add_message(self, user_id: str, role: str, text: str):
        text = (text or "").strip()
        if not text:
            return
        if role not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")

        self.ensure_session(user_id)

        with self.lock:
            self.con.execute(
                "INSERT INTO messages(user_id, role, text, ts) VALUES (?, ?, ?, ?)",
                (user_id, role, text, time.time())
            )
            self.con.commit()

    def load_history(self, user_id: str, limit: Optional[int] = None) -> List[Tuple[str, str]]:
        self.ensure_session(user_id)
        with self.lock:
            if limit:
                cur = self.con.execute(
                    "SELECT role, text FROM messages WHERE user_id=? ORDER BY ts ASC LIMIT ?",
                    (user_id, limit)
                )
            else:
                cur = self.con.execute(
                    "SELECT role, text FROM messages WHERE user_id=? ORDER BY ts ASC",
                    (user_id,)
                )
            return [(r, t) for (r, t) in cur.fetchall()]

    def clear_history(self, user_id: str):
        with self.lock:
            self.con.execute("DELETE FROM messages WHERE user_id=?", (user_id,))
            self.con.commit()

    def close(self):
        with self.lock:
            self.con.close()