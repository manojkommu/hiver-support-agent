"""
06_build_index.py — build the grounding corpus + retrieval index.

Corpus = the brand's RESOLVED threads (agent helped in-thread rather than just
pushing to DM). Each item is {thread_id, customer_msg, resolution}, where the
resolution is what the agent actually said. New messages retrieve against these
so replies are grounded in real past resolutions.

Run:  python src/06_build_index.py            (uses data/spotify_threads.jsonl)
      python src/06_build_index.py --backend tfidf --n 5000
Writes: data/index.pkl
"""
import argparse
import re

from common import read_jsonl, clean_text
from retrieval import Retriever

DM = re.compile(r"\b(DM|dm|direct message|private message|sent you a|slide into)", re.I)


def build_corpus(threads, max_n):
    corpus = []
    for t in threads:
        turns = t["turns"]
        first_cust = next((x for x in turns if x["role"] == "customer"), None)
        agent_turns = [x for x in turns if x["role"] == "agent"]
        if not first_cust or not agent_turns:
            continue
        # resolved = the first agent reply is NOT just "take it to DM"
        if DM.search(agent_turns[0]["text"]):
            continue
        cust = clean_text(first_cust["text"])
        resolution = " ".join(clean_text(a["text"]) for a in agent_turns)
        if len(cust) < 8 or len(resolution) < 8:
            continue
        corpus.append({
            "thread_id": t["thread_id"],
            "customer_msg": cust,
            "resolution": resolution[:600],
        })
        if len(corpus) >= max_n:
            break
    return corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="data/spotify_threads.jsonl")
    ap.add_argument("--backend", default="auto", choices=["auto", "embed", "tfidf"])
    ap.add_argument("--n", type=int, default=6000, help="max resolved threads to index")
    ap.add_argument("--out", default="data/index.pkl")
    args = ap.parse_args()

    threads = read_jsonl(args.threads)
    corpus = build_corpus(threads, args.n)
    print(f"corpus: {len(corpus)} resolved threads")

    r = Retriever(backend=args.backend)
    r.build(corpus)
    r.save(args.out)

    # sanity: show retrieval for a couple of probe queries
    for probe in ["I was charged twice for premium",
                  "the app keeps crashing when I press play",
                  "how do I get my playlist on a featured list"]:
        hits = r.query(probe, k=2)
        print(f"\nQ: {probe}")
        for h in hits:
            print(f"   [{h['score']:.2f}] cust: {h['customer_msg'][:70]}")
            print(f"          res : {h['resolution'][:70]}")


if __name__ == "__main__":
    main()
