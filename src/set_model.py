"""
set_model.py — change GEMINI_MODEL in .env without touching your key.

Run:  python src/set_model.py                    (defaults to gemini-3.6-flash)
  or: python src/set_model.py gemini-3.6-flash
"""
import sys
from pathlib import Path

model = sys.argv[1] if len(sys.argv) > 1 else "gemini-3.6-flash"
env = Path(__file__).resolve().parent.parent / ".env"

if not env.exists():
    print("No .env found — run `python src/set_key.py` first.")
    sys.exit(1)

lines = env.read_text(encoding="utf-8").splitlines()
out, found = [], False
for l in lines:
    if l.strip().startswith("GEMINI_MODEL="):
        out.append(f"GEMINI_MODEL={model}"); found = True
    else:
        out.append(l)
if not found:
    out.append(f"GEMINI_MODEL={model}")

env.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"Set GEMINI_MODEL={model}")
print("Next: python src/diagnose.py")