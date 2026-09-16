"""
diagnose.py — figure out why LLM calls are failing.

Prints: SDK version, whether the key is loaded, the models your key can call,
and the full traceback / raw response from a single test call.

Run:  python src/diagnose.py
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(__file__))
from common import MODEL, MIN_INTERVAL  # noqa

print("=" * 60)
print("1) SDK + key")
print("=" * 60)
try:
    import google.genai as g
    print("google-genai version:", getattr(g, "__version__", "unknown"))
except Exception as e:
    print("could not import google.genai:", e)

key = os.environ.get("GEMINI_API_KEY", "")
print("GEMINI_API_KEY loaded:", bool(key) and key != "your_key_here",
      f"(len={len(key)}, starts {key[:4]!r})" if key else "(missing)")
print("Configured GEMINI_MODEL:", MODEL)

print("\n" + "=" * 60)
print("2) Models your key can use for generateContent")
print("=" * 60)
try:
    from google import genai
    client = genai.Client(api_key=key)
    usable = []
    for m in client.models.list():
        name = getattr(m, "name", "?")
        actions = getattr(m, "supported_actions", None) \
            or getattr(m, "supported_generation_methods", None) or []
        if not actions or "generateContent" in actions:
            usable.append(name)
        print(f"  {name:45} {actions}")
    print("\n>>> Candidates for GEMINI_MODEL (strip the 'models/' prefix):")
    for n in usable:
        if "flash" in n.lower():
            print("   ", n)
except Exception:
    print("listing models failed:")
    traceback.print_exc()

print("\n" + "=" * 60)
print("3) Single test call with configured model")
print("=" * 60)
try:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=key)
    resp = client.models.generate_content(
        model=MODEL,
        contents='Reply with only this JSON: {"intent": "billing_payment"}',
        config=types.GenerateContentConfig(temperature=0.0),
    )
    print("SUCCESS. resp.text =", repr(resp.text))
except Exception:
    print("CALL FAILED — this is the real error:")
    traceback.print_exc()