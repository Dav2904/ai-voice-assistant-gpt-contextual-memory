import uuid
import re
import os
import tempfile

import streamlit as st
import sounddevice as sd
import numpy as np
import pyttsx3

from stt import SpeechToText
from llm import ollama_generate
from memory import MemoryStore
from chat_store import ChatStore


SYSTEM = """You are a voice assistant with contextual memory.
You DO have access to the chat history and stored memory provided to you below.
Do NOT claim you are a new conversation if history/memory is present.
Keep responses short unless asked for detail.
"""


def get_or_create_user_id() -> str:
    qp = st.query_params
    if "uid" in qp and qp["uid"]:
        return qp["uid"]
    uid = str(uuid.uuid4())
    st.query_params["uid"] = uid
    return uid


def record_audio(seconds=5, sample_rate=16000) -> np.ndarray:
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return audio.squeeze()


def tts_wav_bytes(text: str, rate: int = 175) -> bytes:
    text = (text or "").strip()
    if not text:
        return b""

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = tmp.name
    tmp.close()

    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.save_to_file(text, tmp_path)
    engine.runAndWait()

    with open(tmp_path, "rb") as f:
        audio_bytes = f.read()

    try:
        os.remove(tmp_path)
    except Exception:
        pass

    return audio_bytes


st.set_page_config(page_title="Das – The Personal Voice Assistant", layout="wide")
st.title("🎙️ Das – The Personal Voice Assistant")

user_id = get_or_create_user_id()

# Init stores once per Streamlit session
if "chatdb" not in st.session_state:
    st.session_state.chatdb = ChatStore()
if "memdb" not in st.session_state:
    st.session_state.memdb = MemoryStore()
if "history" not in st.session_state:
    st.session_state.history = st.session_state.chatdb.load_history(user_id)

chatdb: ChatStore = st.session_state.chatdb
memdb: MemoryStore = st.session_state.memdb
history = st.session_state.history

# Whisper model (load once)
if "stt" not in st.session_state:
    st.session_state.stt = SpeechToText(model_size="base", device="cpu", compute_type="int8")
stt = st.session_state.stt

# Sidebar controls
with st.sidebar:
    st.markdown("### Controls")
    st.write("User ID:", user_id[:8] + "…")

    if st.button("🧹 Clear chat history"):
        chatdb.clear_history(user_id)
        st.session_state.history = []
        st.session_state.last_tts = b""
        st.rerun()

    st.markdown("### Audio")
    st.session_state.tts_enabled = st.toggle("🔊 Speak replies", value=True)

    st.markdown("### Debug")
    try:
        st.write("Stored memories:", memdb.memory_count())
        st.write("FAISS vectors:", memdb.faiss_count())
    except Exception:
        st.write("Stored memories: n/a")
        st.write("FAISS vectors: n/a")


# Main UI
col1, col2 = st.columns([1, 2])
user_text = None

with col1:
    st.markdown("### Input")
    mode = st.radio("Mode", ["🎤 Voice (5 sec)", "⌨️ Type"], horizontal=True)

    if mode.startswith("🎤"):
        if st.button("🎙 Record"):
            st.info("Recording...")
            audio = record_audio(seconds=5)
            user_text = stt.transcribe(audio)
            st.success(f"You said: {user_text}")

    else:
        # ✅ FIXED: deterministic submit using a form
        with st.form("text_form", clear_on_submit=True):
            typed = st.text_input("Type here")
            submitted = st.form_submit_button("Send")

        if submitted and typed.strip():
            user_text = typed.strip()
        else:
            user_text = None

with col2:
    st.markdown("### 💬 Conversation (Persistent)")
    for role, msg in history:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Assistant:** {msg}")

    # Play last reply audio in browser
    if st.session_state.get("last_tts"):
        st.audio(st.session_state["last_tts"], format="audio/wav", autoplay=True)


# Process input
if user_text:
    user_text = user_text.strip()
    if user_text:
        # Save user message (disk + session)
        chatdb.add_message(user_id, "user", user_text)
        history.append(("user", user_text))

        # Explicit memory command (optional)
        if user_text.lower().startswith("remember"):
            content = user_text.replace("remember", "", 1).strip()
            m = re.search(r"\bmy name is\s+(.+)$", content, re.IGNORECASE)
            if m:
                name = m.group(1).strip()
                memdb.add(f"USER_NAME: {name}")
                reply = f"Ok. I’ll remember your name is {name}."
            else:
                memdb.add(content)
                reply = f"Ok. I’ll remember: {content}"

            chatdb.add_message(user_id, "assistant", reply)
            history.append(("assistant", reply))

            if st.session_state.get("tts_enabled", True):
                st.session_state.last_tts = tts_wav_bytes(reply)
            else:
                st.session_state.last_tts = b""

            st.rerun()

        # Retrieve long-term memory (semantic)
        retrieved = memdb.search(user_text, k=5)

        # Build messages (memory + bounded chat window)
        messages = [{"role": "system", "content": SYSTEM}]

        if retrieved:
            mem_block = "\n".join(f"- {m}" for m in retrieved)
            messages.append({"role": "system", "content": f"Persistent user memory:\n{mem_block}"})

        # ✅ IMPORTANT: keep prompt small to avoid Ollama 500
        for role, msg in history[-12:]:
            messages.append({"role": role, "content": msg[:1200]})

        reply = ollama_generate(messages, temperature=0.4).strip()

        # Save assistant message (disk + session)
        chatdb.add_message(user_id, "assistant", reply)
        history.append(("assistant", reply))

        # TTS for browser
        if st.session_state.get("tts_enabled", True):
            st.session_state.last_tts = tts_wav_bytes(reply)
        else:
            st.session_state.last_tts = b""

        st.rerun()
#source .venv/Scripts/activate
#python -m streamlit run streamlit_app.py