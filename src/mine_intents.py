"""
mine_intents.py — surface candidate intents from data (not from guessing).

We take the FIRST customer message of every thread (that's the "incoming
message" the agent must classify), clean it, TF-IDF vectorise, and KMeans
cluster it. For each cluster we print the top terms and a few short example
snippets. This is exploratory: it informs a small, human-authored taxonomy —
it is NOT the final label set.
"""
import argparse
import json
import re

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

MENTION = re.compile(r"@\w+")
URL = re.compile(r"https?://\S+")
WS = re.compile(r"\s+")


def clean(t):
    t = MENTION.sub(" ", t)
    t = URL.sub(" ", t)
    t = t.replace("&amp;", "&")
    return WS.sub(" ", t).strip()


def first_customer_msgs(path):
    msgs = []
    for line in open(path):
        th = json.loads(line)
        for turn in th["turns"]:
            if turn["role"] == "customer":
                c = clean(turn["text"])
                if len(c) >= 8:
                    msgs.append(c)
                break
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", required=True)
    ap.add_argument("--k", type=int, default=12)
    args = ap.parse_args()

    msgs = first_customer_msgs(args.threads)
    print(f"{len(msgs):,} first-customer messages\n")

    vec = TfidfVectorizer(max_features=3000, stop_words="english",
                          ngram_range=(1, 2), min_df=20)
    X = vec.fit_transform(msgs)
    km = KMeans(n_clusters=args.k, random_state=0, n_init=5)
    labels = km.fit_predict(X)
    terms = vec.get_feature_names_out()

    import numpy as np
    order = km.cluster_centers_.argsort()[:, ::-1]
    for c in range(args.k):
        size = int((labels == c).sum())
        top = [terms[i] for i in order[c, :10]]
        print(f"### cluster {c}  (n={size}, {100*size/len(msgs):.1f}%)")
        print("   top terms:", ", ".join(top))
        # two shortest examples for readability
        ex = sorted([msgs[i] for i in range(len(msgs)) if labels[i] == c], key=len)
        for e in ex[3:5]:
            print("   e.g.:", e[:90])
        print()


if __name__ == "__main__":
    main()
