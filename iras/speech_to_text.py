import os
import tempfile
import time

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
from faster_whisper import WhisperModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class SpeechToText:
    """
    Handles microphone recording and converts speech to text
    using Groq Whisper as primary and Faster-Whisper as fallback.
    """
    LOCAL_FALLBACK_DELAY_SECONDS = 0.7

    def __init__(self, model_size: str = "small.en", samplerate: int = 16000):
        self.samplerate = samplerate
        self.local_model_size = model_size
        self.local_model = None
        self.groq_client = None
        self.groq_model = "whisper-large-v3-turbo"
        self.groq_response_format = "json"
        self.local_compute_type = "int8"
        self.min_audio_peak = 0.015
        self.groq_prompt = (
            "This is a job interview response in English. "
            "Transcribe every spoken word exactly as said. "
            "Do not summarize. Do not paraphrase. "
            "Keep filler words if they are spoken. "
            "Return only the transcript text."
        )

        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            try:
                print("🔄 Initializing Groq STT client...")
                self.groq_client = Groq(api_key=groq_api_key)
                print("✅ Groq STT client initialized successfully.")
            except Exception as e:
                print(f"⚠️ Failed to initialize Groq STT client: {e}")
                self.groq_client = None
        else:
            print("⚠️ GROQ_API_KEY not found. Groq STT disabled.")

    def _load_local_model(self):
        if self.local_model is None:
            print(f"🔄 Loading local Whisper STT model ({self.local_model_size})...")
            self.local_model = WhisperModel(self.local_model_size, compute_type=self.local_compute_type)
            print("✅ Local Whisper model loaded.")

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        audio = audio.astype(np.float32).flatten()
        if audio.size == 0:
            return audio

        audio = audio - np.mean(audio)
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(np.square(audio))) if audio.size else 0.0

        if peak > 0:
            target_peak = 0.95
            gain = min(target_peak / peak, 8.0)
            if rms < 0.03:
                gain = min(gain * 1.8, 12.0)
            audio = audio * gain

        audio = np.clip(audio, -1.0, 1.0)
        return audio

    def record_audio(self, duration: int = 10, samplerate: int = None):
        """
        Record audio from the microphone.

        Args:
            duration (int): recording time in seconds
            samplerate (int): audio sampling rate

        Returns:
            str: path to recorded audio file
        """

        samplerate = samplerate or self.samplerate
        print(f"🎤 Recording for {duration} seconds...")
        print("🎙️ Speak clearly and keep the mic close to your mouth.")

        recording = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype="float32"
        )
        sd.wait()

        audio = self._normalize_audio(recording)
        audio = np.asarray(audio, dtype=np.float32).flatten()
        audio = np.clip(audio, -1.0, 1.0)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        print(f"🔎 Audio levels -> peak: {peak:.4f}, rms: {rms:.4f}")
        if peak < self.min_audio_peak:
            print("⚠️ Your voice was recorded very softly. Move closer to the mic or increase input volume.")

        audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)

        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav.write(temp_file.name, samplerate, audio_int16)

        print("✅ Recording complete")
        return temp_file.name

    def _transcribe_with_groq(self, audio_path: str):
        if not self.groq_client:
            return ""

        try:
            print("🧠 Transcribing audio with Groq Whisper...")
            print(f"📂 Transcribing file: {audio_path}")
            with open(audio_path, "rb") as audio_file:

                transcription = self.groq_client.audio.transcriptions.create(
                    file=audio_file,
                    model=self.groq_model,
                    prompt=self.groq_prompt,
                    response_format=self.groq_response_format,
                    language="en",
                    temperature=0.0,
                )

            text_output = ""
            if hasattr(transcription, "text") and transcription.text:
                text_output = transcription.text.strip()
            elif isinstance(transcription, dict):
                text_output = (transcription.get("text") or "").strip()
            elif hasattr(transcription, "get"):
                text_output = (transcription.get("text") or "").strip()

            if text_output:
                print("✅ Groq Whisper transcription successful.")
            return text_output

        except Exception as e:
            print(f"⚠️ Groq Whisper failed, switching to local fallback: {e}")
            return ""

    def _transcribe_with_local_whisper(self, audio_path: str):
        self._load_local_model()

        print("🧠 Transcribing audio with local Whisper fallback...")
        segments, info = self.local_model.transcribe(
            audio_path,
            beam_size=10,
            best_of=10,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 250,
                "speech_pad_ms": 500,
            },
            language="en",
            condition_on_previous_text=False,
            temperature=0.0,
            compression_ratio_threshold=2.2,
            no_speech_threshold=0.45,
            log_prob_threshold=-1.0,
            initial_prompt=(
                "This is an English job interview answer. "
                "Transcribe every spoken word exactly, including short words and filler words."
            ),
        )

        collected_segments = []
        for segment in segments:
            cleaned = (segment.text or "").strip()
            if cleaned:
                collected_segments.append(cleaned)

        text_output = " ".join(collected_segments).strip()

        if info.language_probability is not None:
            print(f"🌐 Detected language: {info.language} ({info.language_probability:.2f})")

        return text_output

    def transcribe(self, audio_path: str):
        """
        Convert recorded audio into text.
        """

        groq_text = self._transcribe_with_groq(audio_path)
        if groq_text and len(groq_text.split()) >= 2:
            return groq_text

        if groq_text:
            print("⚠️ Groq transcript looks too short. Trying local Whisper for verification...")
        else:
            print(f"⏳ Waiting {self.LOCAL_FALLBACK_DELAY_SECONDS:.1f}s before local fallback...")
            time.sleep(self.LOCAL_FALLBACK_DELAY_SECONDS)

        local_text = self._transcribe_with_local_whisper(audio_path)
        if local_text and (
            not groq_text or
            len(local_text.split()) > len(groq_text.split())
        ):
            print("✅ Using local Whisper transcript because it captured more speech.")
            return local_text

        return groq_text or local_text


# --- Convenience wrapper for modules that expect simple functions ---
# Create a single global SpeechToText instance so clients are initialized once.
_stt_instance = SpeechToText()


def record_audio(duration: int = 10, samplerate: int = 16000):
    """
    Wrapper function to record audio using the global STT instance.
    """
    return _stt_instance.record_audio(duration, samplerate)



def transcribe(audio_path: str):
    """
    Wrapper function to transcribe audio using the global STT instance.
    """
    return _stt_instance.transcribe(audio_path)


if __name__ == "__main__":
    stt = SpeechToText()
    audio_file = stt.record_audio(duration=5)
    text = stt.transcribe(audio_file)

    print("\nTranscription:")
    print(text)
