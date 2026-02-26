import os
import sqlite3
import time
import threading
from typing import List, Tuple

import numpy as np
import faiss

from llm import ollama_embed

# Absolute project directory (folder where memory.py exists)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "memory.db")
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


class MemoryStore:
    """
    Long-term memory:
    - SQLite stores text
    - FAISS stores embeddings
    - Thread-safe for Streamlit
    """

    def __init__(self):
        _ensure_data_dir()

        self.lock = threading.Lock()
        self.con = sqlite3.connect(DB_PATH, check_same_thread=False)

        with self.lock:
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL,
                    text TEXT NOT NULL
                )
            """)
            self.con.commit()

        self.index = None
        self.dim = None

        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            self.dim = self.index.d

    def _save_index(self):
        if self.index is not None:
            faiss.write_index(self.index, INDEX_PATH)

    def _rows_in_insert_order(self) -> List[Tuple[int, str]]:
        with self.lock:
            cur = self.con.cursor()
            cur.execute("SELECT id, text FROM memories ORDER BY id ASC")
            return cur.fetchall()

    def add(self, text: str):
        text = text.strip()
        if not text:
            return

        emb = np.array(ollama_embed(text), dtype=np.float32)

        if self.dim is None:
            self.dim = emb.shape[0]
            self.index = faiss.IndexFlatIP(self.dim)

        emb = _normalize(emb).reshape(1, -1)

        with self.lock:
            cur = self.con.cursor()
            cur.execute("INSERT INTO memories(created_at, text) VALUES (?, ?)", (time.time(), text))
            self.con.commit()

        self.index.add(emb)
        self._save_index()

    def search(self, query: str, k: int = 5) -> List[str]:
        if self.index is None or self.index.ntotal == 0:
            return []

        q = np.array(ollama_embed(query), dtype=np.float32)
        q = _normalize(q).reshape(1, -1)

        _, idxs = self.index.search(q, k)
        idxs = idxs[0].tolist()

        rows = self._rows_in_insert_order()
        results = []
        for pos in idxs:
            if 0 <= pos < len(rows):
                results.append(rows[pos][1])
        return results

    def close(self):
        with self.lock:
            self._save_index()
            self.con.close()