import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

class SpeechToText:
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def record(self, seconds=6.0, sample_rate=16000) -> np.ndarray:
        print(f"[STT] Recording {seconds:.1f}s... speak now.")
        audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
        return audio.squeeze()

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(audio, language="en", vad_filter=True)
        return "".join(seg.text for seg in segments).strip()