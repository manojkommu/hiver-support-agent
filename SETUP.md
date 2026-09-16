# Setup & quickstart (runnable core)

This is the classification half of the pipeline — reproducible today. The
retrieval + reply + escalation agent and the eval/judge harness are added next.

## 1. Environment

```bash
cd hiver-support-agent
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. API key

```bash
cp .env.example .env
# open .env and paste your Gemini key into GEMINI_API_KEY
```

Get a free key at https://aistudio.google.com → "Get API key". No credit card.

## 3. Run it (in order)

The golden set is already labelled and split, so you can go straight to results.

```bash
# (a) Baselines — NO API key needed. Confirms your env works.
python src/04_baselines.py

# (b) LLM classifier — uses your key. ~8–9 min the first time (free-tier pacing),
#     instant on re-runs (responses are cached to .cache/).
python src/05_classify_llm.py
```

`04` prints the trivial + TF-IDF macro-F1 on the fixed 81-item test split.
`05` prints the LLM's macro-F1 on the *same* test split, plus per-class F1 and a
confusion matrix, and writes `eval_out/llm_preds.jsonl` for failure analysis.

## Rebuilding from scratch (optional, for the reproducibility section)

```bash
# needs data/twcs.csv from Kaggle (thoughtvector/customer-support-on-twitter)
python src/01_build_threads.py --csv data/twcs.csv --brand SpotifyCares --out data/spotify_threads.jsonl
python src/02_sample_golden.py --threads data/spotify_threads.jsonl --out golden/to_label.jsonl
python src/build_label_tool.py           # regenerates golden/label_tool.html
# ... hand-label via label_tool.html -> golden/golden_set.jsonl ...
python src/03_split_golden.py
```

## What each file is

| File | Purpose |
|------|---------|
| `src/common.py` | shared helpers; the cached, rate-limited Gemini client |
| `src/01_build_threads.py` | rebuild conversation threads for one brand from twcs.csv |
| `src/02_sample_golden.py` | hybrid stratified+random sampling of messages to label |
| `src/mine_intents.py` | (exploratory) cluster messages to derive the taxonomy |
| `src/build_label_tool.py` | generate the self-contained HTML labeling app |
| `src/03_split_golden.py` | fixed stratified train/test split |
| `src/04_baselines.py` | trivial + TF-IDF baselines on the test split |
| `src/05_classify_llm.py` | LLM few-shot intent classifier on the same split |
