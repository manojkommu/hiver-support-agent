"""
app.py — a local multi-turn web chat over the Spotify support agent.

The first message runs full triage (classify -> retrieve -> decide). Once a
conversation is auto-handled, follow-up messages continue naturally, grounded in
the same retrieved resolutions and the running history, until the issue is
resolved or the agent escalates to a human.

Run:  streamlit run app.py
"""
import importlib.util
import os
import sys

import streamlit as st

sys.path.insert(0, "src")
from retrieval import Retriever  # noqa: E402

_spec = importlib.util.spec_from_file_location("chat_agent", os.path.join("src", "chat_agent.py"))
ca = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ca)

INTENT_COLORS = {
    "playback_app_issue": "#1db954", "account_access": "#e67e22",
    "billing_payment": "#e74c3c", "subscription_management": "#9b59b6",
    "content_availability": "#3498db", "playlist_help": "#16a085",
    "praise_or_feedback": "#f1c40f", "other_unclear": "#95a5a6",
}

st.set_page_config(page_title="Spotify Support Agent", page_icon="🎧", layout="centered")


@st.cache_resource(show_spinner="Loading agent (index + demos)…")
def load_agent():
    retriever = Retriever.load("data/index.pkl")
    demos = ca.agent.build_demos("golden/train.jsonl")
    return retriever, demos


def badge(text, color):
    return (f"<span style='background:{color};color:#fff;padding:2px 10px;"
            f"border-radius:12px;font-size:0.8rem;font-weight:600'>{text}</span>")


def reset_conversation():
    st.session_state["state"] = ca.new_state()
    st.session_state["messages"] = []


if "state" not in st.session_state:
    reset_conversation()
retriever, demos = load_agent()

with st.sidebar:
    st.header("🎧 Support Agent")
    st.caption("A multi-turn AI support agent for Spotify, built from real "
               "Twitter support conversations. It triages the first message, "
               "then holds a grounded back-and-forth — and escalates when it "
               "shouldn't guess.")
    st.button("🔄 New conversation", use_container_width=True, on_click=reset_conversation)
    st.divider()
    st.markdown("**Intents it recognizes:**")
    for i, c in INTENT_COLORS.items():
        st.markdown(badge(i, c), unsafe_allow_html=True)
    st.divider()
    st.markdown("**Try starting with:**")
    for ex in ["the app keeps crashing when I press play",
               "I was charged twice for premium this month",
               "why isn't the new Taylor Swift album on spotify yet",
               "how do I get my playlist featured?"]:
        if st.button(ex, use_container_width=True):
            st.session_state["pending"] = ex

st.title("Spotify Support Agent")
st.caption("Chat like a customer — the agent remembers the conversation.")

for m in st.session_state["messages"]:
    with st.chat_message(m["role"], avatar="🎧" if m["role"] == "assistant" else None):
        if m["role"] == "assistant" and m.get("meta"):
            meta = m["meta"]
            color = INTENT_COLORS.get(meta["intent"], "#95a5a6")
            line = badge(meta["intent"], color)
            if meta["type"] == "escalate":
                line += " &nbsp; " + badge("ESCALATED", "#e67e22")
            st.markdown(line, unsafe_allow_html=True)
        if m["role"] == "assistant" and m.get("escalated"):
            st.info(f"**Escalated to a human agent.** — {m.get('reason','')}")
        else:
            st.write(m["content"])
        if m["role"] == "assistant" and m.get("meta") and m["meta"].get("evidence"):
            with st.expander("🔍 grounding & decision"):
                st.markdown(f"**Decision reason:** {m['meta'].get('reason','')}")
                st.markdown("**Retrieved similar past cases:**")
                for e in m["meta"]["evidence"]:
                    st.markdown(f"- _{e['customer_msg'][:110]}_ — sim **{e['score']}**")

prompt = st.chat_input("Type a message as the customer…")
if "pending" in st.session_state and not prompt:
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant", avatar="🎧"):
        with st.spinner("Thinking…"):
            state, event = ca.send(st.session_state["state"], prompt, retriever, demos)
            st.session_state["state"] = state
        color = INTENT_COLORS.get(event["intent"], "#95a5a6")
        line = badge(event["intent"], color)
        if event["type"] == "escalate":
            line += " &nbsp; " + badge("ESCALATED", "#e67e22")
        st.markdown(line, unsafe_allow_html=True)

        msg = {"role": "assistant", "meta": event}
        if event["type"] == "escalate":
            st.info(f"**Escalated to a human agent.** — {event['reason']}")
            msg["escalated"] = True
            msg["reason"] = event["reason"]
            msg["content"] = ""
        else:
            st.write(event["text"])
            msg["content"] = event["text"]
        with st.expander("🔍 grounding & decision"):
            st.markdown(f"**Decision reason:** {event.get('reason','')}")
            if event.get("evidence"):
                st.markdown("**Retrieved similar past cases:**")
                for e in event["evidence"]:
                    st.markdown(f"- _{e['customer_msg'][:110]}_ — sim **{e['score']}**")
        st.session_state["messages"].append(msg)

    if st.session_state["state"]["status"] == "escalated":
        st.caption("This conversation was escalated. Click **New conversation** in "
                   "the sidebar to start over.")
