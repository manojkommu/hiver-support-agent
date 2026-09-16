"""
common.py — shared helpers used by the numbered scripts.

The important bit is llm(): a thin wrapper over the Gemini API that
  (1) CACHES every response to disk keyed by (model, system, prompt), so
      re-running the eval costs zero additional quota and is reproducible, and
  (2) RATE-LIMITS calls to respect the free tier (~10 req/min) with retry+backoff.

The Gemini SDK is imported lazily inside llm(), so scripts that don't call the
API (e.g. the baselines) run even if google-genai isn't installed.
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path

# ---- config / env -----------------------------------------------------------

def _load_dotenv():
    p = Path(__file__).resolve().parent.parent / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_dotenv()

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
MIN_INTERVAL = float(os.environ.get("LLM_MIN_INTERVAL", "6.5"))
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# ---- text cleaning ----------------------------------------------------------

_MENTION = re.compile(r"@\w+")
_URL = re.compile(r"https?://\S+")
_WS = re.compile(r"\s+")

def clean_text(t):
    if t is None:
        return ""
    t = _MENTION.sub("", str(t))
    t = _URL.sub("", t)
    t = t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return _WS.sub(" ", t).strip()

# ---- jsonl io ---------------------------------------------------------------

def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

# ---- cached, rate-limited LLM call ------------------------------------------

_last_call = [0.0]
_client = [None]

def _get_client(fresh=False):
    if fresh or _client[0] is None:
        from google import genai  # lazy import
        from google.genai import types as _t
        key = os.environ.get("GEMINI_API_KEY")
        if not key or key == "your_key_here":
            raise RuntimeError("Set GEMINI_API_KEY in your .env (see .env.example).")
        # generous timeout; a fresh client avoids reusing a pooled socket that was reset
        http_opts = _t.HttpOptions(timeout=120_000)  # ms
        _client[0] = genai.Client(api_key=key, http_options=http_opts)
    return _client[0]

def _cache_key(model, system, prompt):
    h = hashlib.sha256(f"{model}\x00{system}\x00{prompt}".encode()).hexdigest()
    return CACHE_DIR / f"{h}.json"

# Free tier caps each model at 20 requests/day, but the cap is PER MODEL. When a
# call isn't pinned to a specific model, llm() rotates through these on a daily
# cap so any script (classifier, agent, judge) keeps working. Same Flash family.
MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-flash-lite-latest",
]
_model_idx = [0]

def _is_daily_quota(msg):
    return "perday" in msg or "generaterequestsperday" in msg or "requests_per_day" in msg

def llm(prompt, system=None, model=None, temperature=0.0, use_cache=True, max_retries=8):
    """Return the model's text response. Cached, rate-limited, and — when no model
    is pinned — auto-rotating across MODELS when a model's daily quota is hit."""
    pinned = model is not None
    if not pinned:
        if _model_idx[0] >= len(MODELS):
            model = MODELS[-1]
        else:
            model = MODELS[_model_idx[0]]

    # cache: check every model's cache entry (a hit under any model is still valid)
    if use_cache:
        for m in ([model] if pinned else MODELS):
            ck = _cache_key(m, system or "", prompt)
            if ck.exists():
                return json.loads(ck.read_text())["response"]

    from google.genai import types  # lazy
    client = _get_client()
    cfg = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system if system else None,
    )

    attempt = 0
    while attempt < max_retries:
        wait = MIN_INTERVAL - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
            _last_call[0] = time.time()
            try:
                text = (resp.text or "").strip()
            except Exception:
                text = ""
            if use_cache:
                _cache_key(model, system or "", prompt).write_text(json.dumps({"response": text}))
            return text
        except Exception as e:
            _last_call[0] = time.time()
            msg = str(e).lower()
            # per-DAY quota: rotate to the next model (if not pinned), else give up
            if _is_daily_quota(msg):
                if not pinned and _model_idx[0] + 1 < len(MODELS):
                    _model_idx[0] += 1
                    model = MODELS[_model_idx[0]]
                    print(f"    [llm] daily cap -> switching to {model}")
                    continue  # free switch, doesn't count against retries
                raise
            is_conn = any(s in msg for s in (
                "10054", "connection", "reset", "forcibly closed",
                "connecterror", "remotedisconnected", "eof occurred",
                "timeout", "timed out", "readtimeout", "read operation",
            ))
            transient = is_conn or any(s in msg for s in
                                       ("429", "rate", "quota", "resource_exhausted",
                                        "503", "unavailable"))
            if attempt == max_retries - 1 or not transient:
                raise
            if is_conn:
                client = _get_client(fresh=True)  # drop the poisoned socket, reconnect
            backoff = min(45, MIN_INTERVAL * (1.6 ** attempt))
            print(f"    [llm] transient error, retry {attempt+1}/{max_retries} in {backoff:.0f}s: {str(e)[:80]}")
            time.sleep(backoff)
            attempt += 1

def parse_json(text):
    """Robustly pull a JSON object out of an LLM response (strips code fences)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return json.loads(text)

# ---- the intent taxonomy (single source of truth) ---------------------------

INTENTS = [
    "playback_app_issue",
    "account_access",
    "billing_payment",
    "subscription_management",
    "content_availability",
    "playlist_help",
    "praise_or_feedback",
    "other_unclear",
]
