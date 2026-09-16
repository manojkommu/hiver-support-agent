"""
03_split_golden.py — deterministic stratified split of the golden set.

All methods (trivial, TF-IDF, LLM) are evaluated on the SAME test set so the
comparison is apples-to-apples. The train split is used to fit the TF-IDF model
and to draw the LLM's few-shot demonstrations.

Run:  python src/03_split_golden.py --golden golden/golden_set.jsonl
Writes: golden/train.jsonl, golden/test.jsonl
"""
import argparse
from collections import Counter

from sklearn.model_selection import train_test_split

from common import read_jsonl, write_jsonl, INTENTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="golden/golden_set.jsonl")
    ap.add_argument("--test_size", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = [r for r in read_jsonl(args.golden) if r.get("intent")]
    y = [r["intent"] for r in rows]
    train, test = train_test_split(
        rows, test_size=args.test_size, random_state=args.seed, stratify=y)

    write_jsonl("golden/train.jsonl", train)
    write_jsonl("golden/test.jsonl", test)

    print(f"train: {len(train)}   test: {len(test)}")
    dtr, dte = Counter(r["intent"] for r in train), Counter(r["intent"] for r in test)
    print(f"\n{'intent':26} {'train':>5} {'test':>5}")
    for i in INTENTS:
        print(f"{i:26} {dtr.get(i,0):>5} {dte.get(i,0):>5}")


if __name__ == "__main__":
    main()
