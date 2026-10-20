"""Executive explanation in three parts: WHAT? WHY? WHAT SHOULD I DO?

Built from the decision object only (no other facts). Azure OpenAI GPT-4o rewrites it when
AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY are set; it must keep the same three labelled
lines and may only use the supplied numbers. Without credentials (or on any error) the
deterministic template is used, so the demo never breaks.
"""
from __future__ import annotations

import json
import re
import urllib.request

from vantage import config


def _m(v: float) -> str:
    return f"EUR {v / 1e6:,.1f}M"


def _who(label: str) -> str:
    return "all stores" if label == "All stores" else label


def _a(p: float) -> str:
    n = round(p * 100)
    return "an" if n in (8, 11, 18) or 80 <= n <= 89 else "a"


def three_part(dec: dict) -> dict:
    """Deterministic WHAT / WHY / WHAT SHOULD I DO from a decision object."""
    what = (f"{dec['month_label']} sales for {_who(dec['segment_label'])} are forecast at {_m(dec['forecast'])}, "
            f"{dec['change_vs_last_pct']:+.1f}% vs last month (80% range {_m(dec['lo80'])} to {_m(dec['hi80'])})")
    if dec["p_hit"] is not None:
        what += f", with {_a(dec['p_hit'])} {dec['p_hit'] * 100:.0f}% chance of reaching the {_m(dec['target'])} target."
    else:
        what += "."
    drivers = ", ".join(f"{x['name'].lower()} {x['pct']:+.1f}%" for x in dec["top_drivers"][:2])
    lead = dec["leading_indicator"]
    why = f"{dec['top_reason']}."
    why += f" Biggest modeled drivers vs last month: {drivers}."
    rec = dec["recommended"]
    if rec["promo_delta"]:
        do = (f"Among the tested interventions, {rec['label']} gives the strongest modeled improvement: "
              f"{rec['impact_text']}. This is a model-based scenario estimate, not causal proof.")
    elif dec["action_required"]:
        do = "None of the tested promo interventions improves the outlook enough; review other levers."
    else:
        do = "Keep the current plan and monitor weekly footfall."
    return dict(what=what, why=why, do=do)


def as_text(parts: dict) -> str:
    return f"WHAT: {parts['what']}\nWHY: {parts['why']}\nWHAT SHOULD I DO: {parts['do']}"


def narrate(dec: dict) -> dict:
    parts = three_part(dec)
    base = dict(parts, text=as_text(parts), source="template")
    if not (config.AZURE_OPENAI_ENDPOINT and config.AZURE_OPENAI_KEY):
        return base
    url = (f"{config.AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/"
           f"{config.AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={config.AZURE_OPENAI_API_VERSION}")
    facts = {k: dec[k] for k in ("segment_label", "month_label", "forecast", "lo80", "hi80", "target", "p_hit",
                                 "change_vs_last_pct", "top_reason", "top_drivers", "recommended", "risk_level")}
    body = {"messages": [
        {"role": "system", "content": (
            "You write for a retail executive. Output exactly three lines starting with 'WHAT:', 'WHY:' and "
            "'WHAT SHOULD I DO:'. Use only the numbers and facts provided; do not add any new numbers, causes or "
            "actions. Say that recommendations are model-based scenario estimates, not causal proof. Max 35 words per line.")},
        {"role": "user", "content": "Facts (JSON): " + json.dumps(facts, default=str) + "\nDraft: " + base["text"]}],
        "temperature": 0.1, "max_tokens": 220}
    try:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
            "Content-Type": "application/json", "api-key": config.AZURE_OPENAI_KEY})
        with urllib.request.urlopen(req, timeout=20) as r:
            text = json.loads(r.read())["choices"][0]["message"]["content"].strip()
        m = re.search(r"WHAT:\s*(.+?)\s*WHY:\s*(.+?)\s*WHAT SHOULD I DO:\s*(.+)", text, re.S)
        if not m:
            return base  # keep the guaranteed structure
        parts = dict(what=m.group(1).strip(), why=m.group(2).strip(), do=m.group(3).strip())
        return dict(parts, text=as_text(parts), source=f"Azure OpenAI {config.AZURE_OPENAI_DEPLOYMENT}")
    except Exception as e:  # never break the demo
        return dict(base, source=f"template (Azure OpenAI error: {type(e).__name__})")
