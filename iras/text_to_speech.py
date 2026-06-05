import pyttsx3


class TextToSpeech:
    """
    Handles converting text to spoken audio so the AI interviewer
    can ask questions aloud during the IRAS interview.
    """

    def __init__(self):
        print("🔄 Initializing Text-to-Speech engine...")

        self.engine = pyttsx3.init()

        # Optional voice configuration
        self.engine.setProperty("rate", 170)   # speech speed
        self.engine.setProperty("volume", 1.0) # max volume

        print("✅ Text-to-Speech engine ready.")

    def speak(self, text: str):
        """
        Convert text into spoken audio.

        Args:
            text (str): Text the AI interviewer should speak
        """

        print(f"🔊 AI Interviewer: {text}")

        self.engine.say(text)
        self.engine.runAndWait()


if __name__ == "__main__":

    # Simple standalone test
    tts = TextToSpeech()

    question = "Hello, welcome to the IRAS AI interview. Can you explain object oriented programming?"

    tts.speak(question)
