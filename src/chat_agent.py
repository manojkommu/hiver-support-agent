"""
chat_agent.py — a multi-turn conversation layer over the one-shot agent.

Reuses the existing pipeline (07_agent.py) unchanged:
  - The FIRST customer message runs full triage: classify intent, retrieve
    grounding, decide auto-handle vs escalate (07_agent.classify/decide).
  - Once a conversation is AUTO-HANDLED, follow-up turns continue naturally,
    grounded in the same retrieved resolutions and the running history — so the
    agent remembers context ("app crashing" -> "I'm on Android 14") without
    re-triaging every message.
  - At any turn the agent can escalate (it decides it needs a human, or the
    drafting call fails), which ends the auto-conversation.

The app keeps one `state` dict per conversation and calls send() each turn.
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import clean_text, llm  # noqa: E402

# load the numbered agent module (can't `import 07_agent`)
_spec = importlib.util.spec_from_file_location(
    "agent", os.path.join(os.path.dirname(__file__), "07_agent.py"))
agent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agent)

CHAT_SYSTEM = (
    "You are a friendly Spotify customer-support agent chatting with a customer. "
    "Use the past resolutions provided as your source of truth — stay grounded in "
    "them and do NOT invent steps, links, or promises they don't support. Keep "
    "each reply short and natural, like a real support rep. Ask a brief clarifying "
    "question when you need more detail (device, OS, app version, etc.). When the "
    "customer's issue looks resolved, close warmly. If you cannot help without "
    "private account access, or the issue is clearly unresolved after your best "
    "effort, reply with exactly: NEEDS_ESCALATION"
)


def new_state():
    return {"intent": None, "reason": None, "hits": [], "history": [], "status": "new"}


def _evidence(hits):
    return [{"customer_msg": h["customer_msg"], "score": round(h["score"], 3)} for h in hits]


def _ground_block(hits):
    return "\n".join(
        f'- A customer asked: "{h["customer_msg"][:120]}"\n'
        f'  and it was resolved with: "{h["resolution"][:220]}"'
        for h in hits
    )


def _chat_reply(intent, hits, history, model=None):
    convo = "\n".join(
        ("Customer" if t["role"] == "customer" else "Agent") + ": " + t["text"]
        for t in history
    )
    prompt = (
        f"Detected issue type: {intent}\n\n"
        f"Relevant past resolutions (your source of truth):\n{_ground_block(hits)}\n\n"
        f"Conversation so far:\n{convo}\n\n"
        f"Write the agent's next reply (or NEEDS_ESCALATION):"
    )
    return llm(prompt, system=CHAT_SYSTEM, model=model).strip()


def send(state, msg, retriever, demos, model=None):
    """Advance the conversation by one customer turn.
    Returns (state, event) where event is a dict the UI renders:
      {"type": "reply"|"escalate", "text"?, "intent", "reason", "evidence"}
    """
    state["history"].append({"role": "customer", "text": msg})

    # first turn: full triage
    if state["status"] == "new":
        intent = agent.classify(msg, demos, model)
        hits = retriever.query(clean_text(msg), k=3)
        action, reason = agent.decide(intent, hits)
        state["intent"], state["hits"], state["reason"] = intent, hits, reason
        if action == "escalate":
            state["status"] = "escalated"
            return state, {"type": "escalate", "intent": intent, "reason": reason,
                           "evidence": _evidence(hits)}
        state["status"] = "active"

    # active conversation: draft the next grounded turn
    try:
        reply = _chat_reply(state["intent"], state["hits"], state["history"], model)
    except Exception as e:
        state["status"] = "escalated"
        return state, {"type": "escalate", "intent": state["intent"],
                       "reason": f"drafting unavailable ({str(e)[:40]})",
                       "evidence": _evidence(state["hits"])}

    if reply.strip() == "NEEDS_ESCALATION":
        state["status"] = "escalated"
        return state, {"type": "escalate", "intent": state["intent"],
                       "reason": "agent judged this needs a human",
                       "evidence": _evidence(state["hits"])}

    state["history"].append({"role": "agent", "text": reply})
    return state, {"type": "reply", "text": reply, "intent": state["intent"],
                   "reason": state["reason"], "evidence": _evidence(state["hits"])}
