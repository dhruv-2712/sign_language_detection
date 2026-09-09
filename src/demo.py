"""Phase 5 demo: recognized glosses -> Claude (fluent sentence) -> offline TTS.

Requires ANTHROPIC_API_KEY in the environment for the Claude call.
pyttsx3 is offline and needs no key.
"""
import os
import joblib
import pyttsx3
from anthropic import Anthropic

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLF_PATH = os.path.join(_REPO_ROOT, "models", "baseline_classifier.joblib")

client = Anthropic()
engine = pyttsx3.init()


def to_sentence(words):
    prompt = (
        "Convert these sign-language glosses into one fluent English sentence, "
        f"reply with only the sentence: {' '.join(words)}"
    )
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=50,
        messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text.strip()


if __name__ == "__main__":
    # Hardcoded gloss list for the end-to-end wiring check.
    words = ["hello", "name", "help"]
    sentence = to_sentence(words)
    print(sentence)
    engine.say(sentence)
    engine.runAndWait()
