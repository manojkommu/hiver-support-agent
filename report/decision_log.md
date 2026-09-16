# Decision log

Non-obvious decisions and why we made them. (Bullet form, as allowed.)

1. **Optimised for evaluation over system.** The brief says "the proof is worth
   more than the system," so effort went into the golden set, baselines, judge
   validation and failure analysis rather than a fancy agent. A defensible,
   simple agent that is measured honestly beats an elaborate one that isn't.

2. **Chose Spotify from data, not vibe.** Compared five high-volume brands on two
   axes: is the product *bounded* (→ clean intents) and is the escalate/auto
   decision *real* (both classes populated). Spotify: bounded product, ~31%
   "send to DM" rate (both classes well-populated), reviewer-legible. Amazon
   escalated <1% (degenerate escalation task); Apple/Amazon sprawl across every
   product (fuzzy intents).

3. **Mined the taxonomy, didn't guess it.** Clustered the first customer message
   of every thread (TF-IDF + KMeans) to surface natural groups, then authored a
   small 8-intent set from what actually appeared.

4. **Split `billing_payment` from `subscription_management`.** They look similar
   but behave oppositely for triage: a billing dispute usually needs a private
   account lookup (→ escalate), while "how do I cancel Premium" can be answered
   with instructions (→ auto-handle). Keeping them separate lets the agent act
   on that difference.

5. **Included two non-support intents** (`praise_or_feedback`, `other_unclear`).
   A large share of brand @-mentions aren't problems; recognising "no action
   needed" is itself good triage, and it stops the classifier from being
   flattered by forcing everything into a problem class.

6. **Hybrid golden-set sampling.** A *stratified* portion guarantees every intent
   has enough examples for a real per-class F1; a *random* portion preserves the
   true class distribution. The two answer different questions (per-class
   capability vs honest real-world performance) and they differ — which is
   itself reported.

7. **Hand-labelled all 202 examples** (human ground truth). Tie-break rule when a
   message spanned two intents: label the customer's *primary grievance* (e.g.
   "charged and now it won't play" → billing_payment).

8. **One fixed stratified train/test split (121/81)**, used by every method, so
   comparisons are apples-to-apples. LLM few-shot demos are drawn only from
   train — never from test — so there is no leakage.

9. **Macro-F1 as the headline metric**, not accuracy — it weights rare intents
   equally, so a model can't win by only nailing the majority class.

10. **Two baselines as the bar, not throwaways.** Trivial (majority class) sets
    the floor; TF-IDF + logistic regression is the "is an LLM even worth it?"
    test. The LLM must clear TF-IDF to justify itself.

11. **LLM classifier is few-shot (2 demos/intent)**, JSON-constrained output,
    temperature 0 for reproducibility.

12. **Local embeddings for retrieval, with automatic TF-IDF fallback.** Keeps
    retrieval off the API (saves quota, no rate limits) and still runs if the
    embedding model can't download — robustness that mattered given a hostile
    network.

13. **Grounding corpus = resolved threads only.** A thread counts as "resolved"
    if the first agent reply is substantive rather than "DM us." Those same
    "DM us" threads become weak *escalation* labels — one signal, two uses.

14. **Conservative escalation policy with stated reasons.** Escalate when no
    retrieved resolution is similar enough to ground a reply (won't guess);
    sensitive intents (billing, account) escalate unless the match is strong;
    otherwise auto-handle. Every decision emits a reason string, per the brief.

15. **Engineered around free-tier limits.** Disk-cached every LLM response (reruns
    cost no quota, and are reproducible), rate-limited calls, and — on hitting
    the 20-requests/day-*per-model* free-tier cap — rotated across same-family
    Gemini Flash models. Consequence to disclose: final classifications come
    from a mix of Flash models, not a single one.

16. **Validated the judge before trusting it.** The LLM-as-judge is checked
    against ~25-30 human-scored replies via quadratic-weighted Cohen's kappa. An
    unvalidated judge is an opinion of unknown quality; the kappa is what lets us
    trust (or discount) its scores.
