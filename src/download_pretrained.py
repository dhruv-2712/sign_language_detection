"""Download the AI4Bharat INCLUDE pretrained Transformer (263 ISL words, ~205 MB)
and the heavier MediaPipe pose model into models/. Safe to re-run."""
import os
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = {
    "pretrained/include_large.pth":
        "https://api.wandb.ai/files/abdur-ai4bharat/include-no-cnn/1nywb73r/augs_transformer.pth",
}

for rel, url in FILES.items():
    path = os.path.join(REPO_ROOT, "models", rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"Downloading {rel}...")
        urllib.request.urlretrieve(url, path)
