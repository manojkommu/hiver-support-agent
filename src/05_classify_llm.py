"""
05_classify_llm.py — LLM few-shot intent classifier (Gemini).

Scored on the SAME fixed test split as the baselines (golden/test.jsonl), so the
macro-F1 numbers are directly comparable. Few-shot demonstrations are drawn from
golden/train.jsonl only — never from the test set — so there's no leakage.

Every call is cached to disk (see common.llm), so re-running is free and
deterministic. On the free tier this processes ~81 messages in ~8-9 minutes the
first time (6.5s/call), then instantly on re-runs.

Run:  python src/05_classify_llm.py
"""
import argparse
import json
from collections import defaultdict

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from common import read_jsonl, clean_text, llm, parse_json, INTENTS

DEFINITIONS = {
    "playback_app_issue": "app crashing, songs won't play, skipping, shuffle problems, downloads/offline not working",
    "account_access": "can't log in, password reset, email change, hacked or compromised account",
    "billing_payment": "wrong charge, double charge, refund request, student-discount billing, payment failed",
    "subscription_management": "upgrade / downgrade / cancel Premium, join or manage a Family plan",
    "content_availability": "why isn't an artist/album/song on Spotify, requests to add content, new-release timing",
    "playlist_help": "create or edit playlists, get a playlist featured, collaborative playlists",
    "praise_or_feedback": "compliments, opinions, or general feedback with no support action needed",
    "other_unclear": "spam, off-topic, insults, or too ambiguous to action",
}

SYSTEM = (
    "You are an intent classifier for Spotify customer-support tweets. "
    "Classify the customer's message into exactly ONE of these intents:\n"
    + "\n".join(f"- {k}: {v}" for k, v in DEFINITIONS.items())
    + "\n\nRules:\n"
    "- Choose the customer's PRIMARY ask if several are present.\n"
    "- A billing dispute (money already charged) is billing_payment, even if it mentions Premium.\n"
    "- Managing a plan (cancel/upgrade/family) with no charge dispute is subscription_management.\n"
    "- Pure compliments or opinions with no problem are praise_or_feedback.\n"
    'Respond with ONLY a JSON object: {"intent": "<one_of_the_labels>"}'
)


def build_demos(train, per_class=2):
    by = defaultdict(list)
    for r in train:
        by[r["intent"]].append(clean_text(r["customer_msg"]))
    lines = []
    for intent in INTENTS:
        for msg in by[intent][:per_class]:
            lines.append(f'Message: "{msg}"\nIntent: {{"intent": "{intent}"}}')
    return "\n\n".join(lines)


_reported = [False]

# Free tier caps each model at 20 requests/day, but the cap is PER MODEL — so we
# rotate through comparable Flash models, each contributing its own daily quota.
# All are same-family Gemini Flash class; disclosed in the decision log.
MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    
    "gemini-flash-lite-latest",
    
]
_mi = [0]  # current model index (advances only forward, persists across messages)

def _is_daily_quota(e):
    m = str(e).lower()
    return "perday" in m or "generaterequestsperday" in m or "requests_per_day" in m

def _should_switch_model(e):
    # daily quota, or a model that stayed overloaded/unreachable through all retries
    m = str(e).lower()
    return _is_daily_quota(e) or any(s in m for s in (
        "503", "unavailable", "overloaded", "high load", "timed out", "read operation",
    ))

def classify(msg, demos):
    prompt = (
        (f"Examples:\n{demos}\n\n" if demos else "")
        + f'Now classify this message.\nMessage: "{clean_text(msg)}"\nIntent:'
    )
    raw = None
    while _mi[0] < len(MODELS):
        model = MODELS[_mi[0]]
        try:
            raw = llm(prompt, system=SYSTEM, model=model)
            break
        except Exception as e:
            if _should_switch_model(e) and _mi[0] + 1 < len(MODELS):
                reason = "daily cap" if _is_daily_quota(e) else "overloaded"
                print(f"    [switch] {model} {reason} -> {MODELS[_mi[0]+1]}")
                _mi[0] += 1
                continue
            if not _reported[0]:
                print(f"\n  !! LLM call failed — run `python src/diagnose.py`. Error: {e}\n")
                _reported[0] = True
            return "other_unclear"
    if raw is None:
        return "other_unclear"  # all models exhausted for today
    # 1) try strict JSON
    try:
        intent = parse_json(raw).get("intent", "")
        if intent in INTENTS:
            return intent
    except Exception:
        pass
    # 2) fallback: find any label name mentioned in the raw text
    for i in INTENTS:
        if i in raw:
            return i
    return "other_unclear"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_class", type=int, default=2, help="few-shot demos per intent (0 = zero-shot)")
    ap.add_argument("--out", default="eval_out/llm_classifier.json")
    args = ap.parse_args()

    train = read_jsonl("golden/train.jsonl")
    test = read_jsonl("golden/test.jsonl")
    demos = build_demos(train, args.per_class) if args.per_class else ""

    preds, trues, recs = [], [], []
    for i, r in enumerate(test):
        p = classify(r["customer_msg"], demos)
        preds.append(p); trues.append(r["intent"])
        recs.append({"text": clean_text(r["customer_msg"]), "pred": p, "true": r["intent"]})
        print(f"  [{i+1}/{len(test)}] pred={p:26} true={r['intent']}")

    mf1 = f1_score(trues, preds, average="macro", zero_division=0)
    acc = accuracy_score(trues, preds)
    print(f"\n=== LLM few-shot ({args.per_class}/class)  test n={len(test)} ===")
    print(f"  macro-F1: {mf1:.3f}   accuracy: {acc:.3f}")

    rep = classification_report(trues, preds, labels=INTENTS, output_dict=True, zero_division=0)
    print("\nPer-class F1:")
    for i in INTENTS:
        print(f"  {i:26} {rep[i]['f1-score']:.3f}")

    print("\nConfusion matrix (rows=true, cols=pred), order:")
    print("  " + " ".join(f"{i[:6]:>6}" for i in INTENTS))
    cm = confusion_matrix(trues, preds, labels=INTENTS)
    for i, row in zip(INTENTS, cm):
        print(f"  {i[:22]:22} " + " ".join(f"{v:>6}" for v in row))

    from common import write_jsonl
    write_jsonl("eval_out/llm_preds.jsonl", recs)
    with open(args.out, "w") as f:
        json.dump({"macro_f1": float(mf1), "accuracy": float(acc),
                   "per_class_f1": {i: round(rep[i]["f1-score"], 3) for i in INTENTS}}, f, indent=2)
    print(f"\nSaved -> {args.out} and eval_out/llm_preds.jsonl")


if __name__ == "__main__":
    main()
