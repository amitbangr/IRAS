import os
from dotenv import load_dotenv

from groq import Groq
load_dotenv()

GROQ_MODEL = "llama-3.1-8b-instant"
MIN_OUTPUT_LENGTH = 20


class GroqLLM:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")

        print("🔄 Initializing Groq API client...")
        self.client = Groq(api_key=api_key)
        print("✅ Groq API client initialized successfully.")

    def generate(self, prompt, max_tokens=500, temperature=0.2):
        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            text = ""
            if response and response.choices and response.choices[0].message:
                text = (response.choices[0].message.content or "").strip()

            if text and len(text) >= MIN_OUTPUT_LENGTH:
                return text
            return text

        except Exception as e:
            print(f"❌ Groq API Error: {e}")
            return ""


class FallbackLLM:
    def generate(self, prompt, max_tokens=500, temperature=0.2):
        print("⚠️ No working LLM provider available. Returning empty response.")
        return ""


_llm_instance = None


def get_llm():
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    groq_key = os.getenv("GROQ_API_KEY")

    if groq_key:
        try:
            _llm_instance = GroqLLM()
            test_response = _llm_instance.generate("Reply with only: ready", max_tokens=10, temperature=0)
            if test_response:
                print("✅ Using Groq as primary LLM.")
                return _llm_instance
            print("⚠️ Groq is configured but did not return a valid test response.")
        except Exception as e:
            print(f"⚠️ Groq initialization failed: {e}")

    _llm_instance = FallbackLLM()
    return _llm_instance


if __name__ == "__main__":
    llm = get_llm()
    print(llm.generate("Reply with only: LLM connected"))