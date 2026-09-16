"""
set_key.py — write the .env file correctly (Windows-safe).

Run:  python src/set_key.py
Paste your Gemini key when asked (the input is hidden on purpose — you won't
see characters as you type/paste; just paste and press Enter).
"""
import sys
from getpass import getpass
from pathlib import Path

root = Path(__file__).resolve().parent.parent
env_path = root / ".env"

print("This writes:", env_path)
try:
    key = getpass("Paste your Gemini API key (hidden), then press Enter: ").strip()
except Exception:
    # some terminals don't support hidden input; fall back to visible
    key = input("Paste your Gemini API key, then press Enter: ").strip()

if not key or key == "your_key_here":
    print("No key entered — nothing written.")
    sys.exit(1)

env_path.write_text(
    f"GEMINI_API_KEY={key}\n"
    f"GEMINI_MODEL=gemini-2.5-flash\n"
    f"LLM_MIN_INTERVAL=6.5\n",
    encoding="utf-8",
)
print(f"\nDone. Wrote {env_path}")
print(f"Key length: {len(key)}  (starts with {key[:4]!r})")
print("Next: python src/diagnose.py")
