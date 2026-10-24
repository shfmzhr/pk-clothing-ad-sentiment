import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
LABELED = ROOT / "data" / "labeled"

# how many comments to hand-label per brand
TARGETS = {"Khaadi": 250, "Gul Ahmed": 250, "Sana Safinaz": 100}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_RE = re.compile(r"[A-Za-z]")
SPAM_PAT = r"subscribe|whatsapp|telegram|giveaway|click (?:the )?link|follow me|check my channel|earn money"

TURKISH_CHARS = set("ğşıİĞŞ")
TURKISH_WORDS = {"cok", "guzel", "icin", "harika", "tesekkurler",
                 "muhtesem", "seni", "hayranim", "bence"}
ROMAN_URDU_WORDS = {"hai", "hain", "kya", "nahi", "nahin", "bohat", "bahut",
                    "boht", "acha", "achi", "achha", "accha", "pyara", "pyari",
                    "lekin", "magar", "mein", "aur", "yeh", "woh", "bhi",
                    "tha", "thi", "kitna", "kitni", "kitne", "zabardast",
                    "mashallah", "kapra", "kapray", "bekar", "mehngi", "mehnga"}


def guess_lang(text):
    """Rough heuristic. Verify during manual labeling."""
    n_ar = len(ARABIC_RE.findall(text))
    n_la = len(LATIN_RE.findall(text))
    if n_ar == 0 and n_la == 0:
        return "emoji_only" if any(ord(c) > 127 for c in text) else "no_text"
    if n_ar > n_la:
        return "urdu_script"
    words = set(re.findall(r"[a-z]+", text.lower()))
    if any(c in TURKISH_CHARS for c in text) or words & TURKISH_WORDS:
        return "turkish_likely"
    return "roman_urdu" if words & ROMAN_URDU_WORDS else "english"


def main():
    files = sorted(RAW.glob("comments_*.csv"))
    df = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files],
                   ignore_index=True)
    df = df[df["brand"].isin(TARGETS)]
    steps = [("raw comments", len(df))]

    df["comment_clean"] = (
        df["comment_text"].astype(str)
        .str.replace(HTML_RE, " ", regex=True)
        .str.replace(URL_RE, " ", regex=True)
        .str.replace(SPACE_RE, " ", regex=True)
        .str.strip()
    )
    df = df[df["comment_clean"].str.len() >= 1]

    df = df[~df["comment_clean"].str.contains(SPAM_PAT, case=False, regex=True)]
    steps.append(("after removing empty and spam", len(df)))

    df["_key"] = df["comment_clean"].str.lower()
    df = df.drop_duplicates(subset=["brand", "_key"]).drop(columns="_key")
    steps.append(("after removing duplicates", len(df)))

    df["lang_guess"] = df["comment_clean"].apply(guess_lang)
    df = df[df["lang_guess"] != "no_text"].reset_index(drop=True)
    df.insert(0, "comment_id", range(1, len(df) + 1))
    steps.append(("after removing digit/punctuation-only", len(df)))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    LABELED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED / "comments_clean.csv", index=False, encoding="utf-8-sig")

    usable = df[df["lang_guess"].isin(
        ["english", "roman_urdu", "urdu_script", "emoji_only"])]
    parts = []
    for brand, n in TARGETS.items():
        g = usable[usable["brand"] == brand]
        rare = g[g["lang_guess"].isin(["roman_urdu", "urdu_script"])]
        rest = g[~g.index.isin(rare.index)]
        k = max(0, n - len(rare))
        parts.append(pd.concat([rare, rest.sample(n=min(k, len(rest)), random_state=42)]))
        sample = pd.concat(parts)[
        ["comment_id", "brand", "video_title", "comment_clean", "lang_guess"]
    ].copy()
    sample["sentiment"] = ""
    sample["aspect"] = ""
    sample.to_csv(LABELED / "to_label.csv", index=False, encoding="utf-8-sig")

    print("--- cleaning steps ---")
    for name, n in steps:
        print(f"{n:>6}  {name}")
    print("\n--- comments per brand ---")
    print(df["brand"].value_counts().to_string())
    print("\n--- language guess (all brands) ---")
    print(df["lang_guess"].value_counts().to_string())
    print(f"\nLabeling sample: {len(sample)} comments -> data/labeled/to_label.csv")


if __name__ == "__main__":
    main()