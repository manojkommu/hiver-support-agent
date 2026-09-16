"""
07_agent.py — the support agent. For each incoming customer message it:

  1. classifies intent (LLM few-shot, same taxonomy as 05),
  2. retrieves the most similar RESOLVED past threads (retrieval.py),
  3. decides auto-handle vs escalate — with a stated reason, and
  4. drafts a reply grounded in the retrieved resolutions (only if auto-handling).

Escalation policy (conservative, per design decision):
  - If no retrieved resolution is similar enough (best score < FLOOR), ESCALATE —
    we won't guess a reply we can't ground. This is the primary rule.
  - Sensitive intents (billing_payment, account_access) typically need a private
    account lookup, so they escalate unless the grounding match is strong.
  - Otherwise AUTO-HANDLE with a grounded draft.

Run:  python src/07_agent.py                          (demo on a few messages)
      python src/07_agent.py --infile golden/test.jsonl --out eval_out/agent_out.jsonl
"""
import argparse
import json
from collections import defaultdict

from common import read_jsonl, clean_text, llm, parse_json, INTENTS
from retrieval import Retriever

FLOOR = 0.30            # min retrieval similarity to attempt an auto reply
STRONG = 0.45           # similarity needed to auto-handle a sensitive intent
SENSITIVE = {"billing_payment", "account_access"}

DEFINITIONS = {
    "playback_app_issue": "app crashing, songs won't play, skipping, shuffle, downloads/offline not working",
    "account_access": "can't log in, password reset, email change, hacked/compromised account",
    "billing_payment": "wrong charge, double charge, refund, student-discount billing, payment failed",
    "subscription_management": "upgrade/downgrade/cancel Premium, join or manage a Family plan",
    "content_availability": "why isn't an artist/album/song on Spotify, add content, new-release timing",
    "playlist_help": "create or edit playlists, get a playlist featured, collaborative playlists",
    "praise_or_feedback": "compliments or feedback, no support action needed",
    "other_unclear": "spam, off-topic, insults, or too ambiguous to action",
}
CLS_SYSTEM = (
    "You are an intent classifier for Spotify support tweets. Classify into exactly ONE of:\n"
    + "\n".join(f"- {k}: {v}" for k, v in DEFINITIONS.items())
    + '\nRespond ONLY with JSON: {"intent": "<label>"}'
)
DRAFT_SYSTEM = (
    "You are a Spotify customer-support agent replying on Twitter. Write ONE short, "
    "friendly reply (<= 280 chars). Ground your reply ONLY in the example resolutions "
    "provided — do not invent steps or promises not supported by them. If the examples "
    "don't actually address the customer's problem, reply with exactly: NEEDS_ESCALATION"
)


def classify(msg, demos, model=None):
    prompt = (f"Examples:\n{demos}\n\n" if demos else "") + \
             f'Classify:\nMessage: "{clean_text(msg)}"\nIntent:'
    try:
        out = parse_json(llm(prompt, system=CLS_SYSTEM, model=model)).get("intent", "")
        return out if out in INTENTS else "other_unclear"
    except Exception:
        return "other_unclear"


def draft_reply(msg, hits, model=None):
    ctx = "\n".join(
        f'{i+1}. Customer: "{h["customer_msg"][:150]}"\n   Resolved with: "{h["resolution"][:200]}"'
        for i, h in enumerate(hits)
    )
    prompt = (f"Similar past resolutions:\n{ctx}\n\n"
              f'New customer message: "{clean_text(msg)}"\n\nYour reply:')
    return llm(prompt, system=DRAFT_SYSTEM, model=model).strip()


def decide(intent, hits):
    best = hits[0]["score"] if hits else 0.0
    if best < FLOOR:
        return "escalate", f"no similar past resolution to ground on (best {best:.2f} < {FLOOR})"
    if intent in SENSITIVE and best < STRONG:
        return "escalate", f"{intent} usually needs a private account lookup and match is only {best:.2f}"
    return "auto_handle", f"grounded on {len(hits)} similar resolutions (best {best:.2f})"


def build_demos(train_path, per_class=2):
    by = defaultdict(list)
    for r in read_jsonl(train_path):
        if r.get("intent"):
            by[r["intent"]].append(clean_text(r["customer_msg"]))
    lines = []
    for intent in INTENTS:
        for m in by[intent][:per_class]:
            lines.append(f'Message: "{m}"\nIntent: {{"intent": "{intent}"}}')
    return "\n\n".join(lines)


def handle(msg, retriever, demos, model=None):
    intent = classify(msg, demos, model)
    hits = retriever.query(clean_text(msg), k=3)
    action, reason = decide(intent, hits)
    result = {"message": msg, "intent": intent, "action": action, "reason": reason,
              "evidence": [{"customer_msg": h["customer_msg"], "score": round(h["score"], 3)} for h in hits]}
    if action == "auto_handle":
        try:
            reply = draft_reply(msg, hits, model)
        except Exception as e:
            result["action"] = "escalate"
            result["reason"] = f"drafting unavailable ({str(e)[:40]}) — escalated"
            return result
        if reply.strip() == "NEEDS_ESCALATION":
            result["action"] = "escalate"
            result["reason"] = "drafting model judged retrieved resolutions insufficient"
        else:
            result["reply"] = reply
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index.pkl")
    ap.add_argument("--train", default="golden/train.jsonl")
    ap.add_argument("--infile", default=None, help="jsonl with customer_msg; default = a few demo messages")
    ap.add_argument("--out", default="eval_out/agent_out.jsonl")
    ap.add_argument("--limit", type=int, default=15)
    args = ap.parse_args()

    retriever = Retriever.load(args.index)
    demos = build_demos(args.train)

    if args.infile:
        rows = read_jsonl(args.infile)[:args.limit]
        msgs = [r["customer_msg"] for r in rows]
    else:
        msgs = [
            "I was charged twice for premium this month, please refund",
            "the app keeps crashing every time I hit play on my android",
            "how do I get my playlist added to a featured list?",
            "why isn't the new Taylor Swift album on spotify yet",
            "you guys are the best, love the new wrapped feature!",
        ]

    out = []
    with open(args.out, "w") as fout:
        for m in msgs:
            try:
                r = handle(m, retriever, demos)
            except Exception as e:
                r = {"message": m, "intent": "other_unclear", "action": "escalate",
                     "reason": f"pipeline error ({str(e)[:40]}) — escalated", "evidence": []}
            out.append(r)
            fout.write(json.dumps(r) + "\n"); fout.flush()  # save as we go
            print(f"\n[{r['action'].upper()}] intent={r['intent']}  ({r['reason']})")
            print(f"  msg: {m[:80]}")
            if r.get("reply"):
                print(f"  reply: {r['reply'][:160]}")
    print(f"\nSaved -> {args.out}  ({len(out)} messages)")


if __name__ == "__main__":
    main()
