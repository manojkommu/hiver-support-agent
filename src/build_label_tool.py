"""build_label_tool.py — generate a self-contained HTML labeling app with the
sample data embedded, so the user opens one file and labels with keys 1-8.
Exports golden/golden_set.jsonl. Labels are the human's; nothing is pre-filled.
"""
import json

items = [json.loads(l) for l in open("golden/to_label.jsonl")]

INTENTS = [
    ("playback_app_issue", "app crash · won't play · skip · shuffle · offline/download"),
    ("account_access", "login · password reset · email change · hacked account"),
    ("billing_payment", "wrong/double charge · refund · student-discount billing · payment failed"),
    ("subscription_management", "upgrade/downgrade/cancel Premium · Family plan"),
    ("content_availability", "\u201cwhy isn't X on Spotify\u201d · add artist/album · release timing"),
    ("playlist_help", "create/edit playlists · get featured · collaborative"),
    ("praise_or_feedback", "compliments · opinions · no support action needed"),
    ("other_unclear", "spam · off-topic · insult · too ambiguous to action"),
]

DATA = json.dumps(items)
INTENT_JSON = json.dumps([i[0] for i in INTENTS])
rows = "\n".join(
    f'<button class="lab" data-i="{n}"><span class="k">{n+1}</span>'
    f'<span class="nm">{name}</span><span class="df">{desc}</span></button>'
    for n, (name, desc) in enumerate(INTENTS)
)

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Golden set labeling \u00b7 Spotify support agent</title>
<style>
  :root {{
    --bg:#12141a; --panel:#1a1d26; --line:#2a2f3b; --ink:#e8ebf2;
    --dim:#98a0b3; --accent:#1db954; --accent-ink:#04160b; --warn:#f0a020;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    line-height:1.5; }}
  .wrap {{ max-width:720px; margin:0 auto; padding:24px 20px 80px; }}
  header {{ display:flex; align-items:center; justify-content:space-between; gap:16px;
    padding:14px 0; position:sticky; top:0; background:var(--bg); z-index:5; }}
  h1 {{ font-size:15px; font-weight:600; margin:0; letter-spacing:.2px; }}
  .count {{ font-variant-numeric:tabular-nums; color:var(--dim); font-size:14px; }}
  .bar {{ height:4px; background:var(--line); border-radius:99px; overflow:hidden; margin:2px 0 22px; }}
  .bar > i {{ display:block; height:100%; width:0; background:var(--accent); transition:width .18s; }}
  .msg {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:22px 22px; }}
  .who {{ font-size:12px; color:var(--dim); margin-bottom:6px; }}
  .cust {{ font-size:22px; line-height:1.4; }}
  .reply {{ margin-top:16px; padding-top:16px; border-top:1px dashed var(--line);
    font-size:14px; color:var(--dim); }}
  .reply b {{ color:#c3cad9; font-weight:600; }}
  .hint {{ margin-top:10px; font-size:12px; color:var(--dim); }}
  .hint em {{ color:var(--warn); font-style:normal; }}
  .labs {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:20px; }}
  .lab {{ display:grid; grid-template-columns:auto 1fr; grid-template-rows:auto auto;
    column-gap:10px; text-align:left; background:var(--panel); color:var(--ink);
    border:1px solid var(--line); border-radius:12px; padding:12px 14px; cursor:pointer;
    font:inherit; transition:border-color .12s, background .12s; }}
  .lab:hover {{ border-color:#3a4152; background:#20242f; }}
  .lab .k {{ grid-row:1/3; align-self:center; width:26px; height:26px; border-radius:7px;
    background:var(--line); color:var(--ink); display:grid; place-items:center;
    font-size:13px; font-weight:700; }}
  .lab .nm {{ font-size:14px; font-weight:600; }}
  .lab .df {{ font-size:11.5px; color:var(--dim); }}
  .lab.chosen {{ border-color:var(--accent); background:#12281b; }}
  .lab.chosen .k {{ background:var(--accent); color:var(--accent-ink); }}
  .nav {{ display:flex; align-items:center; justify-content:space-between; margin-top:20px; }}
  .nav button {{ background:none; border:1px solid var(--line); color:var(--ink);
    border-radius:9px; padding:8px 14px; font:inherit; cursor:pointer; }}
  .nav button:disabled {{ opacity:.4; cursor:default; }}
  .foot {{ position:fixed; left:0; right:0; bottom:0; background:var(--panel);
    border-top:1px solid var(--line); padding:12px 20px; display:flex; gap:12px;
    align-items:center; justify-content:space-between; }}
  .foot .k {{ font-size:12.5px; color:var(--dim); }}
  .dl {{ background:var(--accent); color:var(--accent-ink); border:none; border-radius:9px;
    padding:10px 16px; font:inherit; font-weight:700; cursor:pointer; }}
  kbd {{ background:var(--line); border-radius:5px; padding:1px 6px; font-size:12px;
    font-family:ui-monospace,monospace; }}
</style></head>
<body><div class="wrap">
  <header>
    <h1>Golden set \u00b7 Spotify support</h1>
    <span class="count" id="count"></span>
  </header>
  <div class="bar"><i id="fill"></i></div>

  <div class="msg">
    <div class="who">Incoming customer message</div>
    <div class="cust" id="cust"></div>
    <div class="reply" id="reply"></div>
    <div class="hint" id="hint"></div>
  </div>

  <div class="labs" id="labs">{rows}</div>

  <div class="nav">
    <button id="prev">\u2190 Prev</button>
    <button id="skip">Skip (unlabel)</button>
    <button id="next">Next \u2192</button>
  </div>
</div>

<div class="foot">
  <span class="k">Keys <kbd>1</kbd>\u2013<kbd>8</kbd> label &amp; advance \u00b7 <kbd>\u2190</kbd><kbd>\u2192</kbd> move \u00b7 <kbd>u</kbd> unlabel</span>
  <button class="dl" id="dl">Download golden_set.jsonl</button>
</div>

<script>
const DATA = {DATA};
const INTENTS = {INTENT_JSON};
const KEY = "spotify_golden_v1";
let labels = {{}};
try {{ labels = JSON.parse(localStorage.getItem(KEY)) || {{}}; }} catch(e) {{ labels = {{}}; }}
let idx = 0;

const $ = id => document.getElementById(id);
function save() {{ try {{ localStorage.setItem(KEY, JSON.stringify(labels)); }} catch(e) {{}} }}

function render() {{
  const it = DATA[idx];
  $("cust").textContent = it.customer_msg;
  $("reply").innerHTML = it.agent_reply ? "<b>Spotify replied:</b> " + escapeHtml(it.agent_reply) : "<b>Spotify replied:</b> \u2014";
  $("hint").innerHTML = it.hint === "random"
    ? "sampled at random (reflects true distribution)"
    : "sampled for coverage of <em>" + it.hint + "</em> \u2014 label what you actually see, ignore this if wrong";
  const done = Object.keys(labels).length;
  $("count").textContent = done + " / " + DATA.length + " labelled  \u00b7  item " + (idx+1);
  $("fill").style.width = (100*done/DATA.length) + "%";
  document.querySelectorAll(".lab").forEach(b => {{
    b.classList.toggle("chosen", labels[it.id] === INTENTS[+b.dataset.i]);
  }});
  $("prev").disabled = idx===0;
  $("next").disabled = idx===DATA.length-1;
}}
function escapeHtml(s) {{ const d=document.createElement("div"); d.textContent=s; return d.innerHTML; }}

function choose(i) {{ labels[DATA[idx].id] = INTENTS[i]; save(); if (idx<DATA.length-1) idx++; render(); }}
function go(d) {{ idx = Math.max(0, Math.min(DATA.length-1, idx+d)); render(); }}

document.querySelectorAll(".lab").forEach(b =>
  b.addEventListener("click", () => choose(+b.dataset.i)));
$("prev").onclick = () => go(-1);
$("next").onclick = () => go(1);
$("skip").onclick = () => {{ delete labels[DATA[idx].id]; save(); if (idx<DATA.length-1) idx++; render(); }};

document.addEventListener("keydown", e => {{
  if (e.key >= "1" && e.key <= "8") choose(+e.key - 1);
  else if (e.key === "ArrowLeft") go(-1);
  else if (e.key === "ArrowRight") go(1);
  else if (e.key.toLowerCase() === "u") {{ delete labels[DATA[idx].id]; save(); render(); }}
}});

$("dl").onclick = () => {{
  const lines = DATA
    .filter(it => labels[it.id])
    .map(it => JSON.stringify({{
      id: it.id, thread_id: it.thread_id,
      customer_msg: it.customer_msg, agent_reply: it.agent_reply,
      intent: labels[it.id]
    }}));
  const blob = new Blob([lines.join("\\n")], {{type:"application/x-ndjson"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "golden_set.jsonl";
  a.click();
}};

render();
</script>
</body></html>"""

open("golden/label_tool.html", "w").write(html)
print("wrote golden/label_tool.html", len(html), "bytes")
