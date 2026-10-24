import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["negative", "neutral", "positive"]
lines = []


def out(s=""):
    print(s)
    lines.append(str(s))


def build_model():
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                 min_df=2, sublinear_tf=True)),
    ])
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=2.0)
    return Pipeline([("tfidf", features), ("clf", clf)])


def main():
    d = pd.read_csv(ROOT / "data/labeled/labeled.csv", encoding="utf-8-sig")
    d = d[d["sentiment"].isin(LABELS)].reset_index(drop=True)
    X, y = d["comment_clean"].astype(str), d["sentiment"]

    out(f"Labeled comments used: {len(d)}")
    out(y.value_counts().to_string())

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    base = cross_val_predict(DummyClassifier(strategy="most_frequent"),
                             np.zeros((len(y), 1)), y, cv=cv)
    pred = cross_val_predict(build_model(), X, y, cv=cv)

    out("\n=== Baseline: always predict the majority class ===")
    out(f"macro-F1: {f1_score(y, base, average='macro', zero_division=0):.3f}")

    out("\n=== TF-IDF + Logistic Regression (5-fold cross-validation) ===")
    out(classification_report(y, pred, labels=LABELS, digits=3, zero_division=0))

    out("=== By language guess ===")
    for lang, g in d.assign(pred=pred).groupby("lang_guess"):
        present = sorted(set(g["sentiment"]))
        f1 = f1_score(g["sentiment"], g["pred"], labels=present,
                      average="macro", zero_division=0)
        acc = (g["sentiment"] == g["pred"]).mean()
        out(f"{lang:<12} n={len(g):<4} accuracy={acc:.3f}  macro-F1={f1:.3f}")

    (ROOT / "reports").mkdir(exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(y, pred, labels=LABELS, cmap="Blues")
    plt.title("Confusion matrix (5-fold cross-validated)")
    plt.tight_layout()
    plt.savefig(ROOT / "reports/confusion_matrix.png", dpi=150)
    plt.close()

    wrong = d.assign(predicted=pred)
    wrong = wrong[wrong["sentiment"] != wrong["predicted"]]
    wrong[["comment_id", "brand", "lang_guess", "comment_clean",
           "sentiment", "predicted"]].to_csv(
        ROOT / "reports/errors.csv", index=False, encoding="utf-8-sig")
    out(f"\nMisclassified: {len(wrong)} -> reports/errors.csv")

    model = build_model().fit(X, y)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models/sentiment_model.joblib")

    clean = pd.read_csv(ROOT / "data/processed/comments_clean.csv", encoding="utf-8-sig")
    clean = clean[clean["brand"].isin(["Khaadi", "Gul Ahmed", "Sana Safinaz"])].copy()
    proba = model.predict_proba(clean["comment_clean"].astype(str))
    clean["sentiment_pred"] = model.classes_[proba.argmax(axis=1)]
    clean["confidence"] = proba.max(axis=1).round(3)
    clean["hand_labeled"] = clean["comment_id"].isin(d["comment_id"])
    clean.to_csv(ROOT / "data/processed/predictions.csv", index=False, encoding="utf-8-sig")
    out(f"Predictions for {len(clean)} comments -> data/processed/predictions.csv")

    (ROOT / "reports/metrics.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()