"""
08_judge.py — LLM-as-judge for REPLY QUALITY, plus evidence that the judge
agrees with a human.

Two parts:
  (A) score_replies : the judge scores each auto-handled reply on a 1-5 rubric
      (grounded? correct? helpful? on-brand tone?) and returns JSON.
  (B) agreement     : compares the judge's scores against YOUR hand-scores on a
      small calibration set and reports agreement — exact %, within-1 %, and
      Cohen's kappa (quadratic-weighted, since scores are ordinal).

Why this matters: a judge you haven't validated is just a second opinion of
unknown quality. Measuring judge-vs-human agreement is what lets you trust (or
distrust) every other number the judge produces.

Workflow:
  1. python src/08_judge.py score   --infile eval_out/agent_out.jsonl
       -> writes eval_out/judge_scores.jsonl
  2. Hand-score ~25-30 of those replies yourself into golden/human_scores.jsonl
       (id, human_score 1-5). A tiny helper builds the template for you:
       python src/08_judge.py template --infile eval_out/judge_scores.jsonl
  3. python src/08_judge.py agree   --judge eval_out/judge_scores.jsonl \
                                    --human golden/human_scores.jsonl
"""
import argparse
import json

from common import read_jsonl, write_jsonl, llm, parse_json

RUBRIC = """You are evaluating a Spotify support reply. Score it 1-5 on overall quality:
5 = grounded in the evidence, correct, directly helpful, on-brand friendly tone
4 = good, minor issues
3 = acceptable but generic or partially helpful
2 = weak: vague, slightly off, or not clearly grounded
1 = wrong, ungrounded, or unhelpful
Consider especially: is the reply GROUNDED in the provided past resolutions, or invented?
Respond ONLY as JSON: {"score": <1-5>, "reason": "<one short sentence>"}"""


def score_replies(infile, out):
    rows = read_jsonl(infile)
    scored = []
    for i, r in enumerate(rows):
        if r.get("action") != "auto_handle" or not r.get("reply"):
            continue
        ev = "\n".join(f'- {e["customer_msg"][:150]}' for e in r.get("evidence", [])[:3])
        prompt = (f'Customer message: "{r["message"]}"\n\n'
                  f'Past resolutions the reply was grounded on:\n{ev}\n\n'
                  f'Agent reply to evaluate: "{r["reply"]}"')
        try:
            out_j = parse_json(llm(prompt, system=RUBRIC))
            score = int(out_j.get("score", 0))
            reason = out_j.get("reason", "")
        except Exception as e:
            score, reason = 0, f"judge_error: {str(e)[:50]}"
        rec = {"id": i, "message": r["message"], "reply": r["reply"],
               "judge_score": score, "judge_reason": reason}
        scored.append(rec)
        print(f"  [{len(scored)}] score={score}  {r['message'][:55]}")
    write_jsonl(out, scored)
    print(f"\nSaved -> {out}  ({len(scored)} replies scored)")


def make_template(infile, out):
    rows = read_jsonl(infile)
    tmpl = [{"id": r["id"], "message": r["message"], "reply": r["reply"],
             "human_score": None} for r in rows]
    write_jsonl(out, tmpl)
    print(f"Wrote blank scoring template -> {out}")
    print("Fill in human_score (1-5) for each, then run the 'agree' step.")


def agreement(judge_path, human_path):
    import numpy as np
    judge = {r["id"]: r["judge_score"] for r in read_jsonl(judge_path)}
    human = {r["id"]: r["human_score"] for r in read_jsonl(human_path)
             if r.get("human_score") is not None}
    ids = sorted(set(judge) & set(human))
    if not ids:
        print("No overlapping ids with human scores yet — fill in human_scores.jsonl.")
        return
    j = np.array([judge[i] for i in ids])
    h = np.array([human[i] for i in ids])
    exact = float((j == h).mean())
    within1 = float((np.abs(j - h) <= 1).mean())
    kappa = quadratic_weighted_kappa(h, j)
    print(f"n = {len(ids)} replies with both scores")
    print(f"  exact agreement : {exact:.1%}")
    print(f"  within-1 agreement: {within1:.1%}")
    print(f"  quadratic-weighted Cohen's kappa: {kappa:.3f}")
    print("  (kappa > 0.6 = substantial agreement; > 0.8 = near-perfect)")
    out = {"n": len(ids), "exact": exact, "within1": within1, "kappa": kappa}
    with open("eval_out/judge_agreement.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved -> eval_out/judge_agreement.json")


def quadratic_weighted_kappa(a, b, lo=1, hi=5):
    import numpy as np
    n = hi - lo + 1
    O = np.zeros((n, n))
    for x, y in zip(a, b):
        if lo <= x <= hi and lo <= y <= hi:
            O[int(x) - lo, int(y) - lo] += 1
    if O.sum() == 0:
        return float("nan")
    w = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            w[i, k] = (i - k) ** 2 / (n - 1) ** 2
    act_a = O.sum(1); act_b = O.sum(0)
    E = np.outer(act_a, act_b) / O.sum()
    return float(1 - (w * O).sum() / (w * E).sum())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score"); s.add_argument("--infile", default="eval_out/agent_out.jsonl"); s.add_argument("--out", default="eval_out/judge_scores.jsonl")
    t = sub.add_parser("template"); t.add_argument("--infile", default="eval_out/judge_scores.jsonl"); t.add_argument("--out", default="golden/human_scores.jsonl")
    a = sub.add_parser("agree"); a.add_argument("--judge", default="eval_out/judge_scores.jsonl"); a.add_argument("--human", default="golden/human_scores.jsonl")
    args = ap.parse_args()

    if args.cmd == "score":
        score_replies(args.infile, args.out)
    elif args.cmd == "template":
        make_template(args.infile, args.out)
    elif args.cmd == "agree":
        agreement(args.judge, args.human)


if __name__ == "__main__":
    main()
