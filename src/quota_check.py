"""
quota_check.py — make one call and print the FULL error (untruncated), so we can
read exactly which quota was hit (per-minute vs per-day) and any retry hint.

Run:  python src/quota_check.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import _get_client, MODEL
from google.genai import types

client = _get_client(fresh=True)
try:
    r = client.models.generate_content(
        model=MODEL,
        contents='Reply with only this JSON: {"intent": "billing_payment"}',
        config=types.GenerateContentConfig(temperature=0.0),
    )
    print("OK ->", (r.text or "").strip())
except Exception as e:
    print("FULL ERROR BELOW")
    print("=" * 70)
    print(repr(e))
    print("=" * 70)
    # try to surface structured details if present
    for attr in ("message", "details", "response_json", "code", "status"):
        if hasattr(e, attr):
            print(f"{attr}: {getattr(e, attr)}")
