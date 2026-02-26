import requests
from typing import List, Dict, Any

OLLAMA_URL = "http://127.0.0.1:11434"
CHAT_MODEL = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text:latest"


def ollama_generate(messages: List[Dict[str, str]], temperature: float = 0.4) -> str:
    """
    Calls Ollama chat API safely.
    - Returns readable error text instead of crashing Streamlit
    - Limits output tokens to reduce load
    """
    payload: Dict[str, Any] = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_predict": 256,
        },
    }

    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=180)
    except requests.RequestException as e:
        return f"[Ollama request failed] {e}"

    if r.status_code >= 400:
        return f"[Ollama {r.status_code}] {r.text[:900]}"

    try:
        return r.json()["message"]["content"]
    except Exception:
        return f"[Ollama parse error] {r.text[:900]}"


def ollama_embed(text: str) -> List[float]:
    """
    Calls Ollama embeddings API and returns a vector.
    """
    text = (text or "").strip()
    if not text:
        # return a small dummy vector; memory.py will ignore empty text anyway
        return [0.0]

    payload = {"model": EMBED_MODEL, "prompt": text}

    r = requests.post(f"{OLLAMA_URL}/api/embeddings", json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()

    # Ollama embeddings response typically: {"embedding":[...]}
    if "embedding" in data:
        return data["embedding"]

    # fallback (some versions might nest)
    if "data" in data and data["data"] and "embedding" in data["data"][0]:
        return data["data"][0]["embedding"]

    raise ValueError(f"Unexpected embeddings response: {str(data)[:300]}")