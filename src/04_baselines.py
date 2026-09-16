"""
04_baselines.py — the two required baselines, scored on the FIXED test split
(golden/test.jsonl) so they're directly comparable to the LLM classifier, which
is scored on the same 81 examples.

  trivial : always predict the majority intent (from train)
  simple  : TF-IDF + logistic regression (fit on train)

Primary metric: macro-F1 (weights rare intents equally). We also print 5-fold
cross-validated macro-F1 over all 202 as a robustness check, because an 81-item
test set has wide confidence intervals — a point for the "misleading headline"
section of the report.

Run:  python src/04_baselines.py
"""
import argparse
import json

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from common import read_jsonl, clean_text, INTENTS


def xy(rows):
    return [clean_text(r["customer_msg"]) for r in rows], [r["intent"] for r in rows]


def tfidf_lr():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def evaluate(name, est, Xtr, ytr, Xte, yte, save_preds=None):
    est.fit(Xtr, ytr)
    pred = est.predict(Xte)
    mf1 = f1_score(yte, pred, average="macro", zero_division=0)
    acc = accuracy_score(yte, pred)
    print(f"\n=== {name}  (fixed test split, n={len(yte)}) ===")
    print(f"  macro-F1: {mf1:.3f}   accuracy: {acc:.3f}")
    rep = classification_report(yte, pred, labels=INTENTS, output_dict=True, zero_division=0)
    if save_preds:
        with open(save_preds, "w") as f:
            for m, p, t in zip(Xte, pred, yte):
                f.write(json.dumps({"text": m, "pred": p, "true": t}) + "\n")
    return {"name": name, "macro_f1": float(mf1), "accuracy": float(acc),
            "per_class_f1": {i: round(rep[i]["f1-score"], 3) for i in INTENTS if i in rep}}


def cv_macro_f1(build, X, y, k=5):
    X, y = np.array(X, dtype=object), np.array(y)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    s = []
    for tr, te in skf.split(X, y):
        est = build(); est.fit(X[tr].tolist(), y[tr])
        s.append(f1_score(y[te], est.predict(X[te].tolist()), average="macro", zero_division=0))
    return float(np.mean(s)), float(np.std(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval_out/baselines.json")
    args = ap.parse_args()

    train = read_jsonl("golden/train.jsonl")
    test = read_jsonl("golden/test.jsonl")
    Xtr, ytr = xy(train)
    Xte, yte = xy(test)

    results = []
    results.append(evaluate("trivial (majority class)",
                            DummyClassifier(strategy="most_frequent"),
                            Xtr, ytr, Xte, yte))
    results.append(evaluate("simple (TF-IDF + LogReg)",
                            tfidf_lr(), Xtr, ytr, Xte, yte,
                            save_preds="eval_out/tfidf_preds.jsonl"))

    print("\nPer-class F1 (TF-IDF + LogReg, test split):")
    for i, f in results[1]["per_class_f1"].items():
        print(f"  {i:26} {f:.3f}")

    allrows = train + test
    Xa, ya = xy(allrows)
    m, s = cv_macro_f1(tfidf_lr, Xa, ya)
    print(f"\n[robustness] TF-IDF 5-fold CV macro-F1 over all {len(ya)}: {m:.3f} (+/- {s:.3f})")
    results.append({"name": "simple_cv_all", "macro_f1_mean": m, "macro_f1_std": s})

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
