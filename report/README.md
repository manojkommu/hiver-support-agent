# Spotify support agent (Hiver SDE take-home)

An AI support agent for Spotify built from the *Customer Support on Twitter*
dataset. It (1) classifies each customer message into 8 data-derived intents,
(2) drafts a reply grounded in Spotify's past resolutions, and (3) decides
auto-handle vs escalate with a stated reason — evaluated against two baselines,
an LLM-as-judge validated against human scores, and a documented failure
analysis.

The emphasis is the **proof**, not the system: a simple agent, measured honestly.

## Reproduce headline results in <15 minutes

The golden set is already labelled (`golden/golden_set.jsonl`) and split, so:

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # then put your Gemini key in it (or: python src/set_key.py)

python src/04_baselines.py     # trivial + TF-IDF baselines (no API key needed)
python src/05_classify_llm.py  # LLM classifier on the same test split
python src/09_evaluate.py      # consolidated summary
```

`04` needs no key and prints the baselines immediately (macro-F1 ≈ 0.39 test /
0.52 CV). `05` needs a free Gemini key.

## Full pipeline (rebuild from scratch)

```bash
# 1. reconstruct threads for one brand from the raw dataset (data/twcs.csv)
python src/01_build_threads.py --csv data/twcs.csv --brand SpotifyCares --out data/spotify_threads.jsonl
# 2. sample messages to hand-label (hybrid stratified + random)
python src/02_sample_golden.py --threads data/spotify_threads.jsonl --out golden/to_label.jsonl
python src/build_label_tool.py                 # -> golden/label_tool.html (label in browser)
# 3. fixed stratified train/test split
python src/03_split_golden.py
# 4. baselines / 5. LLM classifier
python src/04_baselines.py
python src/05_classify_llm.py
# 6. build retrieval index over resolved threads
python src/06_build_index.py
# 7. run the agent (classify -> retrieve -> escalate/auto -> grounded reply)
python src/07_agent.py --infile golden/test.jsonl --out eval_out/agent_test.jsonl
# 8. LLM-as-judge + human agreement
python src/08_judge.py score
python src/08_judge.py template                # hand-score ~25-30 -> golden/human_scores.jsonl
python src/08_judge.py agree
# 9. consolidate everything
python src/09_evaluate.py
```

## Notes on the free tier (important)

- **Get a key** at https://aistudio.google.com/apikey (starts with `AIza` or
  `AQ.`; no credit card). Save it with `python src/set_key.py`.
- **Daily cap is 20 requests per model.** The classifier automatically **rotates
  across Gemini Flash models** when one is exhausted; responses are **cached to
  `.cache/`** so re-runs cost no quota. If you run out, quotas reset ~midnight US
  Pacific.
- **Resilience:** calls retry on connection resets / 503s and switch models on
  persistent failures, so a flaky network won't kill a run.
- Retrieval uses local embeddings (`sentence-transformers`) with an automatic
  **TF-IDF fallback** — no API, no download required to still work.

## Layout

```
src/   01..09 pipeline scripts (run in order), common.py, retrieval.py, helpers
golden/  golden_set.jsonl (labels), train/test splits, label_tool.html
data/    spotify_threads.jsonl, index.pkl  (twcs.csv is downloaded, not committed)
eval_out/  metrics json + prediction dumps
report/  report.md, decision_log.md
```

Dataset: Customer Support on Twitter (Kaggle,
`thoughtvector/customer-support-on-twitter`). A subsample is used throughout, as
the brief encourages.
