"""to_sentence builds a sensible prompt and returns the model text (fake client, no network)."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import demo  # noqa: E402


class _FakeClient:
    def __init__(self):
        self.prompt = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, max_completion_tokens, temperature, messages, **extra):
        self.prompt = messages[0]["content"]
        msg = SimpleNamespace(content="  Hello, how are you?  ")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_prompt_includes_alternatives_and_pretty_names():
    fake = _FakeClient()
    demo._client = fake
    out = demo.to_sentence(["hello", "howareyou"],
                           [[("hello", 0.67), ("warm", 0.26)], [("howareyou", 0.97), ("wet", 0.01)]])
    assert out == "Hello, how are you?"
    assert "how are you (97%)" in fake.prompt and "warm (26%)" in fake.prompt
    assert fake.prompt.index("1. hello") < fake.prompt.index("2. how are you")


def test_plain_gloss_prompt():
    fake = _FakeClient()
    demo._client = fake
    demo.to_sentence(["goodmorning", "friend"])
    assert "good morning friend" in fake.prompt
