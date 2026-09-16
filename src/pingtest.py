"""
pingtest.py — how often does the connection actually succeed?
Tries 5 real generate calls and reports the hit rate, so we know whether the
retry-enabled classifier will push through or whether we must switch networks.

Run:  python src/pingtest.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import _get_client, MODEL
from google.genai import types

print(f"model: {MODEL}\ntrying 5 calls...\n")
ok = 0
for i in range(5):
    try:
        client = _get_client(fresh=True)  # new connection each time
        r = client.models.generate_content(
            model=MODEL,
            contents='Reply with only this JSON: {"intent": "billing_payment"}',
            config=types.GenerateContentConfig(temperature=0.0),
        )
        txt = (r.text or "").strip()
        print(f"  attempt {i+1}: OK   -> {txt[:60]}")
        ok += 1
    except Exception as e:
        print(f"  attempt {i+1}: FAIL -> {str(e)[:90]}")
    time.sleep(2)

print(f"\n{ok}/5 succeeded")
if ok == 0:
    print("Network is blocking every call -> switch to phone hotspot / drop VPN.")
elif ok < 5:
    print("Intermittent resets -> the classifier's 8x retry should still complete. Run it.")
else:
    print("All good -> run: python src/05_classify_llm.py")
