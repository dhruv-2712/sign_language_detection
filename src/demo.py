"""Phase 5 demo: recognized glosses -> Groq LLM (fluent sentence) -> offline TTS.

Requires GROQ_API_KEY in the environment for the LLM call (GROQ_MODEL optionally overrides the model).
pyttsx3 is offline and needs no key.
"""
import os
import queue
import subprocess
import threading

from groq import Groq

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# gpt-oss-20b is available on the project's Groq account (llama-3.3-70b-versatile is not).
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

_client = None


def _load_key():
    """Make GROQ_API_KEY available even in a terminal opened before it was set (Windows)."""
    if os.environ.get("GROQ_API_KEY") or os.name != "nt":
        return
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            os.environ["GROQ_API_KEY"] = winreg.QueryValueEx(k, "GROQ_API_KEY")[0]
    except OSError:
        pass


def _get_client():
    global _client
    if _client is None:
        _load_key()
        _client = Groq()
    return _client


# ---- text to speech: a worker thread speaks one utterance at a time, so the video never freezes
_SAPI = ("Add-Type -AssemblyName System.Speech; "
         "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak($env:SIGN_TTS_TEXT)")
_speech_q = queue.Queue()
_speech_thread = None
_speech_errors = []
_POWERSHELL = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32",
                           "WindowsPowerShell", "v1.0", "powershell.exe")


def _say(text):
    """Speak with Windows' built-in voice; fall back to pyttsx3 elsewhere / on failure."""
    if os.name == "nt":
        try:
            r = subprocess.run([_POWERSHELL, "-NoProfile", "-Command", _SAPI], capture_output=True,
                               env={**os.environ, "SIGN_TTS_TEXT": text})
            if r.returncode == 0:
                return
            print("Windows speech failed:", r.stderr.decode(errors="ignore")[:200])
        except OSError as e:
            print("Windows speech unavailable:", e)
    import pyttsx3
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def _speech_worker():
    while True:
        text = _speech_q.get()
        try:
            _say(text)
        except Exception as e:
            _speech_errors.append(str(e))
            print("TTS failed:", e)
        finally:
            _speech_q.task_done()


def speak(text):
    """Queue text to be spoken; returns immediately."""
    global _speech_thread
    if _speech_thread is None:
        _speech_thread = threading.Thread(target=_speech_worker, daemon=True)
        _speech_thread.start()
    _speech_q.put(text)


def wait_speech():
    """Block until everything queued has been spoken."""
    _speech_q.join()


def speak_blocking(text):
    """Speak and wait; raises if no speech engine worked."""
    n = len(_speech_errors)
    speak(text)
    wait_speech()
    if len(_speech_errors) > n:
        raise RuntimeError("text-to-speech failed: " + _speech_errors[-1])


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
    extra = {"reasoning_effort": "low"} if "gpt-oss" in MODEL else {}
    # Reasoning models spend completion tokens on thinking, so leave generous headroom.
    msg = _get_client().chat.completions.create(
        model=MODEL, max_completion_tokens=400, temperature=0.2,
        messages=[{"role": "user", "content": prompt}], **extra)
    text = (msg.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("LLM returned an empty reply")
    return text


if __name__ == "__main__":
    # Hardcoded gloss list for the end-to-end wiring check.
    words = ["hello", "name", "help"]
    sentence = to_sentence(words)
    print(sentence)
    speak_blocking(sentence)
