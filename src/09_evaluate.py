"""
09_evaluate.py — one command that consolidates every metric into a report-ready
summary and saves eval_out/summary.json.

Pulls together whatever has been produced:
  - classification : baselines.json + llm_classifier.json  (macro-F1, accuracy)
  - escalation     : compares the agent's auto/escalate decision on the test set
                     against a weak ground truth (did Spotify actually push to DM?)
                     -> precision/recall/F1 for the ESCALATE decision.
                     Needs: python src/07_agent.py --infile golden/test.jsonl \
                            --out eval_out/agent_test.jsonl
  - reply quality  : judge_scores.jsonl (mean) + judge_agreement.json (kappa)

Missing pieces are skipped with a note, so you can run it at any stage.

Run:  python src/09_evaluate.py
"""
import json
import os
import re

from common import read_jsonl

DM = re.compile(r"\b(DM|dm|direct message|private message|sent you a|slide into)", re.I)


def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def section(title):
    print("\n" + "=" * 58 + f"\n{title}\n" + "=" * 58)


def classification(summary):
    section("1) INTENT CLASSIFICATION (test split, macro-F1)")
    base = load_json("eval_out/baselines.json")
    llm = load_json("eval_out/llm_classifier.json")
    rows = []
    if base:
        for b in base:
            if "macro_f1" in b:
                rows.append((b["name"], b["macro_f1"], b.get("accuracy")))
    if llm:
        rows.append(("LLM few-shot", llm["macro_f1"], llm["accuracy"]))
    if not rows:
        print("  (no classification results yet)")
        return
    print(f"  {'method':28} {'macroF1':>8} {'acc':>6}")
    for name, f1, acc in rows:
        print(f"  {name:28} {f1:>8.3f} {('' if acc is None else f'{acc:>6.3f}')}")
    summary["classification"] = [{"method": n, "macro_f1": f, "accuracy": a} for n, f, a in rows]


def escalation(summary):
    section("2) ESCALATION DECISION (agent vs. did-Spotify-DM weak label)")
    test_path = "golden/test.jsonl"
    agent_path = "eval_out/agent_test.jsonl"
    if not os.path.exists(agent_path):
        print("  (run: python src/07_agent.py --infile golden/test.jsonl --out eval_out/agent_test.jsonl)")
        return
    test = read_jsonl(test_path)
    agent = read_jsonl(agent_path)
    truth = {r["customer_msg"]: bool(DM.search(r.get("agent_reply", ""))) for r in test}
    tp = fp = tn = fn = 0
    for a in agent:
        t = truth.get(a["message"])
        if t is None:
            continue
        pred_esc = a["action"] == "escalate"
        if pred_esc and t: tp += 1
        elif pred_esc and not t: fp += 1
        elif not pred_esc and not t: tn += 1
        else: fn += 1
    n = tp + fp + tn + fn
    if not n:
        print("  (no overlap between agent output and test set)")
        return
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    acc = (tp + tn) / n
    print(f"  n={n}   escalate precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}  acc={acc:.3f}")
    print(f"  (tp={tp} fp={fp} tn={tn} fn={fn}; 'positive' = agent escalated)")
    print("  NOTE: ground truth here is the DM heuristic — itself imperfect (see report).")
    summary["escalation"] = {"n": n, "precision": prec, "recall": rec, "f1": f1,
                             "accuracy": acc, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def reply_quality(summary):
    section("3) REPLY QUALITY (LLM-as-judge + human agreement)")
    scores = read_jsonl("eval_out/judge_scores.jsonl") if os.path.exists("eval_out/judge_scores.jsonl") else []
    agree = load_json("eval_out/judge_agreement.json")
    if scores:
        vals = [s["judge_score"] for s in scores if s.get("judge_score")]
        mean = sum(vals) / len(vals) if vals else 0
        dist = {k: sum(1 for v in vals if v == k) for k in range(1, 6)}
        print(f"  replies judged: {len(vals)}   mean score: {mean:.2f} / 5")
        print(f"  score distribution 1..5: {[dist[k] for k in range(1,6)]}")
        summary["reply_quality"] = {"n": len(vals), "mean": mean, "dist": dist}
    else:
        print("  (no judge scores yet: python src/08_judge.py score)")
    if agree:
        print(f"  judge-vs-human: exact={agree['exact']:.0%}  within1={agree['within1']:.0%}  kappa={agree['kappa']:.3f}")
        summary.setdefault("reply_quality", {})["agreement"] = agree
    else:
        print("  (no judge-human agreement yet: hand-score, then 08_judge.py agree)")


def main():
    summary = {}
    classification(summary)
    escalation(summary)
    reply_quality(summary)
    with open("eval_out/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    section("SAVED")
    print("  eval_out/summary.json")


if __name__ == "__main__":
    main()
