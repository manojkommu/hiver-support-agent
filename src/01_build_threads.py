"""
threading.py — reconstruct conversation threads for a single brand from the
raw Customer Support on Twitter dataset (twcs.csv).

Each row in twcs.csv is one tweet with:
  tweet_id                 unique id
  author_id                brand handle (e.g. 'SpotifyCares') or anonymised customer id (numeric string)
  inbound                  True  = message FROM a customer TO the brand
                           False = message FROM the brand
  created_at               timestamp string
  text                     tweet text
  response_tweet_id        comma-separated ids of tweets that REPLY to this one (children)
  in_response_to_tweet_id  id of the tweet this one replies to (parent)

We rebuild threads by treating the reply links as an undirected graph and
collecting every connected component that contains at least one tweet from the
target brand. We do a BFS out from the brand's tweets so we only touch the
brand's neighbourhood, not all 2.8M tweets.

Output: one JSON object per thread, e.g.
  {
    "thread_id": <root tweet_id>,
    "brand": "SpotifyCares",
    "turns": [
      {"tweet_id":.., "role":"customer"|"agent", "text":..., "created_at":..},
      ...
    ]
  }
"""
import argparse
import json
import sys
from collections import defaultdict, deque

import pandas as pd


def load_links(csv_path):
    """Load only the columns needed to build the reply graph (memory-light)."""
    df = pd.read_csv(
        csv_path,
        usecols=["tweet_id", "author_id", "inbound",
                 "response_tweet_id", "in_response_to_tweet_id"],
        dtype={"tweet_id": "int64", "author_id": "str",
               "response_tweet_id": "str"},
    )
    return df


def build_adjacency(df):
    """Undirected adjacency list over tweet_ids from both link columns."""
    adj = defaultdict(set)
    # parent edges: in_response_to_tweet_id
    parent = df[["tweet_id", "in_response_to_tweet_id"]].dropna()
    for tid, pid in zip(parent["tweet_id"].tolist(),
                        parent["in_response_to_tweet_id"].astype("int64").tolist()):
        adj[tid].add(pid)
        adj[pid].add(tid)
    # child edges: response_tweet_id (comma-separated)
    child = df[["tweet_id", "response_tweet_id"]].dropna()
    for tid, kids in zip(child["tweet_id"].tolist(),
                         child["response_tweet_id"].tolist()):
        for k in str(kids).split(","):
            k = k.strip()
            if k.isdigit():
                k = int(k)
                adj[tid].add(k)
                adj[k].add(tid)
    return adj


def brand_threads(df, adj, brand):
    """BFS out from brand tweets, return list of thread member-id sets."""
    brand_ids = set(df.loc[df["author_id"] == brand, "tweet_id"].tolist())
    seen = set()
    threads = []
    for seed in brand_ids:
        if seed in seen:
            continue
        comp = set()
        q = deque([seed])
        seen.add(seed)
        while q:
            node = q.popleft()
            comp.add(node)
            for nb in adj.get(node, ()):
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        threads.append(comp)
    return threads


def hydrate(csv_path, member_ids, brand):
    """Second pass: pull text/role/time for the member tweets only."""
    keep = set(member_ids)
    rows = {}
    for chunk in pd.read_csv(
        csv_path,
        usecols=["tweet_id", "author_id", "inbound", "created_at", "text"],
        dtype={"tweet_id": "int64", "author_id": "str", "text": "str"},
        chunksize=400_000,
    ):
        hit = chunk[chunk["tweet_id"].isin(keep)]
        for r in hit.itertuples(index=False):
            role = "agent" if (r.author_id == brand or not r.inbound) else "customer"
            rows[r.tweet_id] = {
                "tweet_id": int(r.tweet_id),
                "role": role,
                "text": "" if pd.isna(r.text) else str(r.text),
                "created_at": str(r.created_at),
            }
    return rows


def assemble(threads, rows):
    """Order each thread's turns by time and drop degenerate threads."""
    out = []
    for comp in threads:
        turns = [rows[t] for t in comp if t in rows]
        if not turns:
            continue
        turns.sort(key=lambda x: pd.to_datetime(x["created_at"], errors="coerce")
                   if x["created_at"] else pd.Timestamp.min)
        has_cust = any(t["role"] == "customer" for t in turns)
        has_agent = any(t["role"] == "agent" for t in turns)
        if not (has_cust and has_agent):
            continue  # need at least one of each to be a real support exchange
        out.append({
            "thread_id": int(min(t["tweet_id"] for t in turns)),
            "brand": None,  # filled by caller
            "turns": turns,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to twcs.csv")
    ap.add_argument("--brand", default="SpotifyCares")
    ap.add_argument("--out", required=True, help="output .jsonl path")
    args = ap.parse_args()

    print(f"[1/4] loading link columns from {args.csv} ...", file=sys.stderr)
    df = load_links(args.csv)
    print(f"      {len(df):,} rows", file=sys.stderr)

    print("[2/4] building reply graph ...", file=sys.stderr)
    adj = build_adjacency(df)

    print(f"[3/4] collecting threads for {args.brand} ...", file=sys.stderr)
    comps = brand_threads(df, adj, args.brand)
    member_ids = set().union(*comps) if comps else set()
    print(f"      {len(comps):,} raw threads, {len(member_ids):,} tweets", file=sys.stderr)

    print("[4/4] hydrating text ...", file=sys.stderr)
    rows = hydrate(args.csv, member_ids, args.brand)
    threads = assemble(comps, rows)
    for t in threads:
        t["brand"] = args.brand

    with open(args.out, "w") as f:
        for t in threads:
            f.write(json.dumps(t) + "\n")
    print(f"      wrote {len(threads):,} usable threads -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
 