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


PRETTY = {"howareyou": "how are you", "goodmorning": "good morning",
          "goodafternoon": "good afternoon", "goodevening": "good evening",
          "goodnight": "good night", "youplural": "you (plural)", "biglarge": "big/large",
          "smalllittle": "small/little", "storeorshop": "store/shop",
          "streetorroad": "street/road", "secondnumber": "second", "raceethnicity": "race",
          "exmonsoon": "monsoon", "trainstation": "train station", "trainticket": "train ticket",
          "cellphone": "cell phone", "tshirt": "t-shirt"}


def pretty(gloss):
    return PRETTY.get(gloss, gloss)


def to_sentence(words, alternatives=None):
    """words: recognised glosses in order. alternatives: optional parallel list of
    [(gloss, prob), ...] runner-up guesses so the LLM can fix likely misreads from context."""
    if alternatives:
        lines = "\n".join(
            f"{i + 1}. " + ", ".join(f"{pretty(g)} ({p:.0%})" for g, p in alts)
            for i, alts in enumerate(alternatives))
        prompt = (
            "A sign-language recognizer (Indian Sign Language, one word per sign, no grammar) "
            "produced these ranked guesses for each sign in order:\n" + lines +
            "\nThe top guess is usually right, but pick a lower one if the sentence only makes "
            "sense that way. Convert the signs into one fluent English sentence. "
            "Reply with only the sentence.")
    else:
        prompt = (
            "Convert these sign-language glosses into one fluent English sentence, "
            f"reply with only the sentence: {' '.join(pretty(w) for w in words)}")
    msg = _get_client().messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=80,
        messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text.strip()


if __name__ == "__main__":
    # Hardcoded gloss list for the end-to-end wiring check.
    words = ["hello", "name", "help"]
    sentence = to_sentence(words)
    print(sentence)
    speak(sentence)
