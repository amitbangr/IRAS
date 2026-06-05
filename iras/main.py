# main.py
from iras.controller import run_interview, state
from iras.interview_engine import generate_questions, evaluate
from iras.speech_to_text import SpeechToText
from iras.text_to_speech import TextToSpeech
import time
import numpy as np
import sounddevice as sd
import tempfile
import soundfile as sf

# Initialize once
stt = SpeechToText()
tts = TextToSpeech()

SILENCE_THRESHOLD = 0.015
SILENCE_DURATION = 0.3
COOLDOWN_TIME = 0.3


def record_until_silence(max_duration=30):
    sample_rate = 16000
    chunk_duration = 0.1
    chunk_size = int(sample_rate * chunk_duration)

    THRESHOLD = 0.013
    SILENCE_LIMIT = 1.5

    audio_buffer = []
    silence_time = 0
    started = False
    start_time = time.time()

    print("🎧 Waiting for speech...")

    while True:
        audio = sd.rec(chunk_size, samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()

        rms = np.sqrt(np.mean(audio**2))

        if not started:
            if rms > THRESHOLD:
                started = True
                print("🗣️ Speech detected. Recording...")
                audio_buffer.append(audio)
            continue

        audio_buffer.append(audio)

        if rms < THRESHOLD:
            silence_time += chunk_duration
        else:
            silence_time = 0

        if silence_time >= SILENCE_LIMIT:
            print("🛑 Silence detected. Stopping recording.")
            break

        if time.time() - start_time > max_duration:
            print("⏱️ Max duration reached.")
            break

    if not audio_buffer:
        return None

    audio_data = np.concatenate(audio_buffer, axis=0)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_file.name, audio_data, sample_rate)

    return temp_file.name


def listen():
    while state["cooldown"] or state["is_speaking"]:
        time.sleep(0.1)

    time.sleep(0.1)

    audio_path = record_until_silence()

    if not audio_path:
        return ""

    text = stt.transcribe(audio_path)

    if not text or len(text.strip().split()) < 3:
        return ""

    return text


def clean_answer(answer: str, question: str):
    if not answer:
        return ""

    answer_lower = answer.lower()
    question_lower = question.lower()

    if question_lower in answer_lower:
        answer = answer_lower.replace(question_lower, "").strip()

    if len(answer.split()) < 3:
        return ""

    return answer


def speak(text: str):
    state["is_speaking"] = True
    try:
        tts.speak(text)
    finally:
        state["is_speaking"] = False


if __name__ == "__main__":
    report = run_interview(
        generate_questions=generate_questions,
        speak=speak,
        listen=listen,
        evaluate=evaluate,
    )

    print("\n--- FINAL REPORT ---\n")
    for item in report:
        print(item)
        print()
