
import json
from pathlib import Path

p = Path("golden/human_scores.jsonl")
rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

print("Score each reply 1-5:")
print("  5=grounded/correct/on-brand  4=good  3=generic/partial  2=weak/off  1=wrong/ungrounded")
print("  (Enter to skip, q to save & quit)\n")

for r in rows:
    if r.get("human_score") is not None:
        continue
    print("=" * 70)
    print("MESSAGE:", r["message"][:200])
    print("REPLY  :", r["reply"][:200])
    while True:
        ans = input("your score [1-5 / Enter=skip / q=quit]: ").strip().lower()
        if ans == "q":
            p.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
            print("saved. bye."); raise SystemExit
        if ans == "":
            break
        if ans in {"1", "2", "3", "4", "5"}:
            r["human_score"] = int(ans)
            break
        print("  please type 1-5, Enter, or q")
    # save after every entry so nothing is lost
    p.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")

done = sum(1 for r in rows if r.get("human_score") is not None)
print(f"\nDone. {done}/{len(rows)} scored -> {p}")
print("Next: python src/08_judge.py agree")