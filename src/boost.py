import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent

clean = pd.read_csv(ROOT / "data/processed/comments_clean.csv", encoding="utf-8-sig")
used = pd.read_csv(ROOT / "data/labeled/to_label.csv", encoding="utf-8-sig")["comment_id"]

NEG = (r"expensive|overpriced|costly|mehng|mahng|mehanga|bekar|bakwas|bakwaas|worst|waste|"
       r"disappoint|poor|cheap|ghatiya|bura|buri|boring|cringe|hate|ugly|fake|scam|refund|"
       r"delivery|\blate\b|overrated|zyada|not good|not worth|pasand nahi|nahi pasand|"
       r"برا|مہنگ|بکواس")

pool = clean[~clean["comment_id"].isin(used)]
pool = pool[pool["lang_guess"].isin(["english", "roman_urdu", "urdu_script"])]
hits = pool[pool["comment_clean"].str.contains(NEG, case=False, regex=True)]

print("unlabeled comments:", len(pool))
print("keyword matches:", len(hits))
print(hits.groupby("brand").size().to_string())

out = hits.sample(n=min(150, len(hits)), random_state=7)[
    ["comment_id", "brand", "video_title", "comment_clean", "lang_guess"]
].copy()
out["source"] = "boost"
out["sentiment"] = ""
out["aspect"] = ""
out.to_csv(ROOT / "data/labeled/boost_candidates.csv", index=False, encoding="utf-8-sig")
print("saved", len(out), "-> data/labeled/boost_candidates.csv")