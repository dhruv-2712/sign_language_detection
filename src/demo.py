"""Phase 5 demo: recognized glosses -> Claude (fluent sentence) -> offline TTS.

Requires ANTHROPIC_API_KEY in the environment for the Claude call.
pyttsx3 is offline and needs no key.
"""
import os
import pyttsx3
from anthropic import Anthropic

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_client = None
_engine = None


def _get_client():
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


def speak(text):
    """Speak text with offline TTS (engine created lazily)."""
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
    _engine.say(text)
    _engine.runAndWait()


def to_sentence(words):
    prompt = (
        "Convert these sign-language glosses into one fluent English sentence, "
        f"reply with only the sentence: {' '.join(words)}"
    )
    msg = _get_client().messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=50,
        messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text.strip()


if __name__ == "__main__":
    # Hardcoded gloss list for the end-to-end wiring check.
    words = ["hello", "name", "help"]
    sentence = to_sentence(words)
    print(sentence)
    speak(sentence)
