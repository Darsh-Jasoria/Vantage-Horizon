"""Rebuild decisions, narratives and the Teams card from an existing dashboard_data.json.

    python -m vantage.refresh

Useful after changing decision settings in config.py without re-running the forecast.
"""
from __future__ import annotations

import json

from vantage import alerts as alerts_mod, decision as decision_mod, narrate as narrate_mod
from vantage.config import OUT_DIR


def refresh(data: dict) -> dict:
    data["decisions"] = decision_mod.build_all(data)
    nxt = data["future_months"][0]
    data["narratives"] = {}
    for m in data["future_months"]:
        data["narratives"][m] = {}
        for s in data["segments"]:
            dec = data["decisions"][m][s["id"]]
            if m == nxt and s["id"] == "All":
                data["narratives"][m][s["id"]] = narrate_mod.narrate(dec)
            else:
                parts = narrate_mod.three_part(dec)
                data["narratives"][m][s["id"]] = dict(parts, text=narrate_mod.as_text(parts), source="template")
    top = decision_mod.top_decision(data["decisions"], nxt)
    card = alerts_mod.teams_card(top, data["alerts"], data["narratives"][nxt][top["segment"]], data["meta"]["as_of"])
    (OUT_DIR / "teams_alert.json").write_text(json.dumps(card, indent=2))
    data["meta"]["top_decision"] = dict(month=nxt, segment=top["segment"])
    return data


if __name__ == "__main__":
    path = OUT_DIR / "dashboard_data.json"
    d = refresh(json.loads(path.read_text()))
    path.write_text(json.dumps(d))
    t = d["meta"]["top_decision"]
    print("refreshed decisions; top:", t, "->", d["decisions"][t["month"]][t["segment"]]["recommended"]["label"])
