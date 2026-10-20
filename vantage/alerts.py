"""Early-warning alerts and the Teams Adaptive Card.

Two kinds of alert:
1. Leading-indicator anomaly: weekly footfall (customers) vs what the model expected,
   scored as a z-score against the segment's own weekly error history.
2. Target risk: probability of hitting next month's sales target below 40% (critical below 25%).
"""
from __future__ import annotations

import json
import urllib.request
import numpy as np
import pandas as pd

from vantage.config import SEGMENT_LABELS, TEAMS_WEBHOOK_URL, DASHBOARD_URL

Z_FLAG = -2.5


def leading_indicators(res: dict) -> dict:
    """Last month's footfall vs model expectation for every segment (the leading indicator).

    The expectation comes from the LightGBM model trained up to the day before that month,
    so it is a genuine one-month-ahead expectation, not a fit to the month itself.
    """
    wk = res["weekly"]
    wk = wk[(wk.metric == "customers") & (wk.open >= 20)].copy()
    wk["resid"] = wk.actual / wk.pred - 1
    last_month = wk.month.max()
    out = {}
    for seg, g in wk.groupby("segment"):
        hist, recent = g[g.month < last_month], g[g.month == last_month].sort_values("week")
        sd = max(hist.resid.std(), 0.01)
        z = recent.resid / sd
        out[seg] = dict(month=pd.Timestamp(last_month).strftime("%Y-%m"),
                        gap_pct=float((recent.actual.sum() / recent.pred.sum() - 1) * 100),
                        weeks=int(len(recent)), weeks_flagged=int((z <= Z_FLAG).sum()),
                        z_min=float(z.min()), weekly_sd_pct=float(sd * 100))
    return out


def detect(res: dict) -> list[dict]:
    wk = res["weekly"].copy()
    wk = wk[wk.open >= 20]
    wk["resid"] = wk.actual / wk.pred - 1
    last_month = wk.month.max()
    alerts = []
    for (seg, metric), g in wk.groupby(["segment", "metric"]):
        hist, recent = g[g.month < last_month], g[g.month == last_month].sort_values("week")
        sd = max(hist.resid.std(), 0.01)
        recent = recent.assign(z=recent.resid / sd)
        flagged = recent[recent.z <= Z_FLAG]
        if metric != "customers" or len(flagged) < 2:
            continue
        gap = float(recent.actual.sum() / recent.pred.sum() - 1)
        severity = "critical" if len(flagged) >= 3 or recent.z.min() <= -4 else "warning"
        stores = res["store_last"]
        worst = (stores[(stores.store_type == seg) | (seg == "All")]
                 .assign(gap=lambda s: s.actual / s.pred - 1).nsmallest(5, "gap"))
        alerts.append(dict(
            kind="leading_indicator", severity=severity, segment=seg,
            segment_label=SEGMENT_LABELS[seg], metric="Footfall (customers)",
            title=f"{SEGMENT_LABELS[seg]}: footfall {gap * 100:+.1f}% vs expected",
            detail=(f"{len(flagged)} of the last {len(recent)} weeks were more than 2.5 standard "
                    f"deviations below the model. Footfall leads sales, so next month is at risk."),
            weeks=[dict(week=r.week.strftime("%d %b"), resid_pct=float(r.resid * 100), z=float(r.z))
                   for r in recent.itertuples()],
            worst_stores=[dict(store=int(r.store_id), gap_pct=float(r.gap * 100)) for r in worst.itertuples()],
        ))
    # target risk on next month
    nxt = res["future_months"][0]
    sf = res["sales_fc"]
    for r in sf[(sf.month == nxt) & sf.p_hit.notna()].itertuples():
        if r.p_hit < 0.4:
            alerts.append(dict(
                kind="target_risk", severity="critical" if r.p_hit < 0.25 else "warning",
                segment=r.segment, segment_label=SEGMENT_LABELS[r.segment], metric="Sales",
                title=f"{SEGMENT_LABELS[r.segment]}: {r.p_hit * 100:.0f}% chance to hit {pd.Timestamp(nxt):%B} target",
                detail=(f"Forecast EUR {r.p50 / 1e6:,.1f}M vs target EUR {r.target / 1e6:,.1f}M "
                        f"(80% range EUR {r.lo80 / 1e6:,.1f}M to {r.hi80 / 1e6:,.1f}M)."),
                p_hit=float(r.p_hit)))
    order = {"critical": 0, "warning": 1}
    return sorted(alerts, key=lambda a: (order[a["severity"]], a["segment"] != "All"))


def teams_card(decision: dict | None, alerts: list[dict], narrative: dict, as_of: str) -> dict:
    """Adaptive Card 1.4: 'Vantage Horizon - Action required' built from the decision object.

    Falls back to a plain alert digest when there is no decision. Buttons open the dashboard
    pre-filtered to the segment (Action.OpenUrl works in Teams without any bot backend)."""
    body = [{"type": "TextBlock", "text": "VANTAGE HORIZON — ACTION REQUIRED" if decision and decision["action_required"]
             else "VANTAGE HORIZON — FORECAST UPDATE", "weight": "Bolder", "size": "Medium",
             "color": "Attention" if decision and decision["action_required"] else "Default"},
            {"type": "TextBlock", "text": f"Data as of {as_of} · model-based estimates, not guarantees",
             "isSubtle": True, "spacing": "None", "size": "Small", "wrap": True}]
    actions = []
    if decision:
        d = decision
        rec = d["recommended"]
        facts = [
            {"title": "Affected segment", "value": d["segment_label"]},
            {"title": "Risk level", "value": d["risk_level"]},
            {"title": "Forecast", "value": f"EUR {d['forecast'] / 1e6:,.1f}M ({d['month_label']})"},
            {"title": "Target", "value": f"EUR {d['target'] / 1e6:,.1f}M"},
            {"title": "Probability of target", "value": f"{d['p_hit'] * 100:.0f}%"},
            {"title": "Leading indicator", "value": d["leading_indicator"]["text"]},
            {"title": "Top reason", "value": d["top_reason"]},
            {"title": "Recommended scenario", "value": rec["label"]},
            {"title": "Modeled impact", "value": rec["impact_text"]},
        ]
        body.append({"type": "FactSet", "facts": facts})
        body.append({"type": "TextBlock", "wrap": True, "size": "Small", "isSubtle": True,
                     "text": "Scenario estimate from the forecast model; it is not causal proof."})
        q = f"?segment={d['segment']}&month={d['month']}"
        actions = [{"type": "Action.OpenUrl", "title": "Open dashboard", "url": f"{DASHBOARD_URL}/{q}"},
                   {"type": "Action.OpenUrl", "title": "View forecast", "url": f"{DASHBOARD_URL}/{q}#why"},
                   {"type": "Action.OpenUrl", "title": "Open scenario", "url": f"{DASHBOARD_URL}/{q}#scenarios"}]
    if narrative:
        body.append({"type": "TextBlock", "wrap": True, "text": narrative.get("text", "")})
    for a in alerts[:3]:
        body.append({"type": "Container", "style": "attention" if a["severity"] == "critical" else "warning",
                     "items": [{"type": "TextBlock", "text": f"{a['severity'].upper()} · {a['title']}",
                                "weight": "Bolder", "wrap": True},
                               {"type": "TextBlock", "text": a["detail"], "wrap": True, "size": "Small"}]})
    card = {"$schema": "http://adaptivecards.io/schemas/adaptive-card.json", "type": "AdaptiveCard",
            "version": "1.4", "body": body}
    if actions:
        card["actions"] = actions
    return card


def post_to_teams(card: dict) -> str:
    if not TEAMS_WEBHOOK_URL:
        return "skipped (set TEAMS_WEBHOOK_URL to post)"
    payload = {"type": "message", "attachments": [
        {"contentType": "application/vnd.microsoft.card.adaptive", "content": card}]}
    req = urllib.request.Request(TEAMS_WEBHOOK_URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return f"posted ({r.status})"
