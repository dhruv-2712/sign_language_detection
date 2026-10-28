"""Download ONLY the held-out INCLUDE *test-split* videos for chosen categories.

Uses HTTP range requests against the Zenodo zips (remotezip), so we fetch just the
needed members instead of the multi-GB archives. These clips are in the official
INCLUDE test split, so the pretrained model never saw them during training.

Usage:
    python src/fetch_include_test.py --categories Greetings Pronouns Days_and_Time
"""
import argparse
import os

from remotezip import RemoteZip

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZENODO = "https://zenodo.org/records/4010759/files/{}?download=1"
PARTS = {"Greetings": 2, "Pronouns": 2, "Days_and_Time": 3, "Seasons": 1, "People": 5,
         "Places": 4, "Jobs": 2, "Home": 4, "Society": 3, "Adjectives": 8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=["Greetings", "Pronouns", "Days_and_Time"])
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "data", "include_test"))
    args = ap.parse_args()

    listing = os.path.join(REPO_ROOT, "INCLUDE", "train_test_paths", f"include_{args.split}.txt")
    wanted = {l.strip() for l in open(listing) if l.strip()}
    for cat in args.categories:
        want_cat = {w for w in wanted if w.split("/")[0] == cat}
        for part in range(1, PARTS[cat] + 1):
            url = ZENODO.format(f"{cat}_{part}of{PARTS[cat]}.zip")
            with RemoteZip(url) as z:
                names = [n for n in z.namelist() if n in want_cat]
                print(f"{cat} part {part}: {len(names)} {args.split} videos")
                for n in names:
                    dest = os.path.join(args.out, n)
                    if not os.path.exists(dest):
                        z.extract(n, args.out)


if __name__ == "__main__":
    main()
