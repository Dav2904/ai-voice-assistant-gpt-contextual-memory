from typing import List, Tuple, Dict
from stt import SpeechToText
from tts import TextToSpeech
from memory import MemoryStore
from llm import ollama_generate

SYSTEM = """You are a voice assistant with contextual memory.
- Keep answers short and direct unless asked for detail.
- Use retrieved memory only if it helps the current request.
- If the user says 'remember ...', store it and confirm.
"""

def is_remember_cmd(text: str) -> str | None:
    t = text.strip()
    low = t.lower()
    if low.startswith("remember that "):
        return t[len("remember that "):].strip()
    if low.startswith("remember "):
        return t[len("remember "):].strip()
    return None

def build_messages(recent: List[Tuple[str,str]], memories: List[str], user_text: str) -> List[Dict[str,str]]:
    msgs = [{"role":"system","content":SYSTEM}]
    if memories:
        mem_block = "\n".join(f"- {m}" for m in memories[:5])
        msgs.append({"role":"developer","content":f"Relevant memories:\n{mem_block}"})
    for role, text in recent[-20:]:
        msgs.append({"role":role,"content":text})
    msgs.append({"role":"user","content":user_text})
    return msgs

def main():
    stt = SpeechToText(model_size="base", device="cpu", compute_type="int8")
    tts = TextToSpeech(rate=175)
    mem = MemoryStore()
    recent: List[Tuple[str,str]] = []

    print("=== FREE Offline Voice Assistant (Ollama + Whisper + Memory) ===")
    print("Enter = record 6s | /text = type | /quit = exit")
    print("Say/type: 'remember ...' to store long-term memory.\n")

    while True:
        cmd = input(">> ").strip()
        if cmd == "/quit":
            break

        if cmd == "/text":
            user_text = input("You: ").strip()
        else:
            audio = stt.record(seconds=6.0, sample_rate=16000)
            user_text = stt.transcribe(audio)

        if not user_text:
            print("[STT] Didn’t catch that. Try again.")
            continue

        print("You:", user_text)

        remember = is_remember_cmd(user_text)
        if remember:
            mem.add(remember)
            reply = f"Ok. I’ll remember: {remember}"
            print("Assistant:", reply)
            tts.speak(reply)
            recent += [("user", user_text), ("assistant", reply)]
            continue

        memories = mem.search(user_text, k=5)
        msgs = build_messages(recent, memories, user_text)
        reply = ollama_generate(msgs, temperature=0.4).strip()

        print("Assistant:", reply)
        tts.speak(reply)
        recent += [("user", user_text), ("assistant", reply)]

    mem.close()

if __name__ == "__main__":
    main()