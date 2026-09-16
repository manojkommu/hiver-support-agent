"""
sample_golden.py — build the pool of examples for hand-labeling.

Strategy (hybrid, documented for the report's sampling note):

  1. STRATIFIED pool (~coverage): for each intent we have a set of seed
     keyword patterns. We pull candidates matching those patterns so every
     intent — including rare ones like playlist_help — has enough examples
     to compute a real per-class F1. Keyword matching is ONLY used to ensure
     coverage; it is NOT the label. The human relabels every item from
     scratch, and is expected to disagree with the keyword hint often.

  2. RANDOM pool (~true distribution): a straight random sample of first
     customer messages, so the golden set also reflects the real class
     balance. Metrics on this subset tell us honest real-world performance;
     metrics on the stratified subset tell us per-class capability.

We keep each item's full thread (customer opener + the agent's real reply)
so the labeler has context AND so the same threads can be reused later as
the retrieval corpus for grounded replies.

Output: golden/to_label.jsonl  (NO intent labels — the human adds those.)
"""
import argparse
import json
import random
import re

MENTION = re.compile(r"@\w+")
URL = re.compile(r"https?://\S+")
WS = re.compile(r"\s+")

# Seed patterns per intent — for COVERAGE sampling only, never a final label.
SEEDS = {
    "account_access": r"\b(log ?in|login|log ?ged|password|reset|hacked|can'?t access|locked out|change.*email|email address)\b",
    "billing_payment": r"\b(charge|charged|refund|payment|paid|money back|billed|double charg|student discount|£|\$\d)\b",
    "subscription_management": r"\b(cancel|unsubscrib|upgrade|downgrade|family plan|family premium|join premium|renew|subscription)\b",
    "content_availability": r"\b(why isn'?t|not on spotify|when will.*available|add .* (album|artist|song)|release|new album|available on spotify)\b",
    "playlist_help": r"\b(playlist|collaborative|on a playlist|make me a playlist|my playlist)\b",
    "playback_app_issue": r"\b(crash|won'?t play|not working|keeps? (stopping|skipping)|shuffle|offline|download(ing|ed)?|buffer|freez|glitch|stops? playing)\b",
    "praise_or_feedback": r"\b(love|thank|best app|amazing|awesome|great job|you guys are)\b",
}


def clean(t):
    t = MENTION.sub("", t)
    t = URL.sub("", t)
    t = t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return WS.sub(" ", t).strip()


def load_openers(path):
    items = []
    for line in open(path):
        th = json.loads(line)
        opener, replies = None, []
        for turn in th["turns"]:
            if turn["role"] == "customer" and opener is None:
                opener = clean(turn["text"])
            elif turn["role"] == "agent":
                replies.append(clean(turn["text"]))
        if opener and len(opener) >= 8:
            items.append({
                "thread_id": th["thread_id"],
                "customer_msg": opener,
                "agent_reply": replies[0] if replies else "",
            })
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_intent", type=int, default=16, help="stratified target per seeded intent")
    ap.add_argument("--random_n", type=int, default=80, help="pure random additions")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    random.seed(args.seed)
    items = load_openers(args.threads)
    # dedupe by lowercased text
    seen, uniq = set(), []
    for it in items:
        key = it["customer_msg"].lower()
        if key not in seen:
            seen.add(key)
            uniq.append(it)
    random.shuffle(uniq)

    chosen, used_tid = [], set()

    # 1) stratified coverage
    for intent, pat in SEEDS.items():
        rx = re.compile(pat, re.I)
        pool = [it for it in uniq
                if it["thread_id"] not in used_tid and rx.search(it["customer_msg"])]
        for it in pool[:args.per_intent]:
            used_tid.add(it["thread_id"])
            chosen.append({**it, "hint": intent})

    # 2) random additions (true distribution)
    rand_pool = [it for it in uniq if it["thread_id"] not in used_tid]
    for it in rand_pool[:args.random_n]:
        used_tid.add(it["thread_id"])
        chosen.append({**it, "hint": "random"})

    random.shuffle(chosen)
    for i, it in enumerate(chosen):
        it["id"] = i
        it["intent"] = None  # <-- human fills this

    with open(args.out, "w") as f:
        for it in chosen:
            f.write(json.dumps(it) + "\n")

    from collections import Counter
    print(f"wrote {len(chosen)} items -> {args.out}")
    print("coverage hint breakdown:", dict(Counter(c["hint"] for c in chosen)))


if __name__ == "__main__":
    main()
