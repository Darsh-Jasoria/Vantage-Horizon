"""Decision engine: simulate -> score -> recommend.

Works on the exported forecast data (the same dict the API serves), so the dashboard, the API,
the Teams card and the tests all use one implementation.

What it does for one segment and month
--------------------------------------
1. Reads the forecast (median + 80/95% range) and target.
2. Tests interventions on the only lever the model actually represents: chain-wide promo days.
   The promo response comes from re-scoring the LightGBM store-day model with a changed promo
   calendar (computed in forecast.promo_curves). It is a MODELED association learned from history,
   not a causal estimate.
3. Scores each tested scenario with a transparent formula (weights in config.DECISION_WEIGHTS):

       score = prob_pp            * gain in probability of hitting target, counted only up to
                                    config.CONFIDENCE_CAP (percentage points)
             + sales_pct          * modeled sales change (%)
             - complexity_per_day * promo days added            (intervention complexity proxy;
                                                                   the data has no cost information)
             - extrapolation_pp   * promo share above the highest share seen in history (pp)
                                                                  (penalises scenarios outside the
                                                                   range the model has learned from)

4. Recommends the highest-scoring scenario only if the segment is at risk
   (probability below config.ACTION_THRESHOLD) and the score is positive; otherwise "keep current plan".

Scenario intervals keep the forecast's relative width; probabilities use the same log-normal
error model as the forecast: P(hit) = 1 - Phi((ln target - ln median) / sigma).
"""
from __future__ import annotations

import math
from statistics import NormalDist

from vantage import config

Z80 = NormalDist().inv_cdf(0.9)
_PHI = NormalDist().cdf


FRIENDLY_DRIVER = {"SARIMAX cross-check": "Monthly trend model", "Last month's deviation": "Last month's momentum"}


def _drivers(d: dict, month: str, seg: str) -> list[dict]:
    return [dict(x, name=FRIENDLY_DRIVER.get(x["name"], x["name"])) for x in d["drivers"][month][seg]["drivers"]]


def risk_level(p: float) -> str:
    return "Critical" if p < 0.25 else "High" if p < 0.40 else "Medium" if p < config.ACTION_THRESHOLD else "Low"


def _fc(d: dict, seg: str, month: str) -> dict:
    return next(f for f in d["series"][seg]["sales"]["forecast"] if f["month"] == month)


def sigma_of(f: dict) -> float:
    """Recover the log-scale sigma from the forecast's 80% range."""
    return math.log(f["hi80"] / f["p50"]) / Z80


def p_hit(median: float, target: float | None, sigma: float) -> float | None:
    if not target:
        return None
    return 1 - _PHI((math.log(target) - math.log(median)) / sigma)


def promo_ratio(d: dict, seg: str, month: str, delta: int) -> float:
    curve = d["whatif"][month]["curve"][seg]
    delta = max(min(delta, max(curve["delta"])), min(curve["delta"]))
    return curve["sales"][curve["delta"].index(delta)]


def scenario(d: dict, seg: str, month: str, promo: int = 0, foot_pct: float = 0.0, ticket_pct: float = 0.0,
             name: str = "Custom", assumptions: list[str] | None = None) -> dict:
    """Sales outcome for a set of assumptions. Promo days re-use the model's response curve;
    footfall/ticket changes are manual multipliers (not modeled)."""
    f, t = _fc(d, seg, month), d["targets"][seg].get(month)
    sd = sigma_of(f)
    ratio = promo_ratio(d, seg, month, promo) * (1 + foot_pct / 100) * (1 + ticket_pct / 100)
    p50 = f["p50"] * ratio
    base_p = p_hit(f["p50"], t, sd)
    new_p = p_hit(p50, t, sd)
    base_days = d["whatif"][month]["base_promo_days"]
    return dict(name=name, promo_delta=promo, foot_pct=foot_pct, ticket_pct=ticket_pct,
                promo_days=base_days + promo, ratio=ratio, p50=p50, lo80=f["lo80"] * ratio, hi80=f["hi80"] * ratio,
                lo95=f["lo95"] * ratio, hi95=f["hi95"] * ratio, target=t, p_hit=new_p,
                delta_sales=p50 - f["p50"], delta_sales_pct=(ratio - 1) * 100,
                delta_p_pp=None if new_p is None else (new_p - base_p) * 100,
                assumptions=assumptions or [])


def _hist_max_promo_share(d: dict, seg: str) -> float:
    actual = set(d["actual_months"])
    shares = [v["promo_share"] for m, v in d["calendar"][seg].items() if m in actual]
    return max(shares) if shares else 1.0


def capped_gain_pp(sc: dict, cap: float | None = None) -> float:
    """Gain in target probability, counting only the part below the confidence cap."""
    cap = config.CONFIDENCE_CAP if cap is None else cap
    if sc["p_hit"] is None or sc["delta_p_pp"] is None:
        return 0.0
    base = sc["p_hit"] - sc["delta_p_pp"] / 100
    return (min(sc["p_hit"], cap) - min(base, cap)) * 100


def score(sc: dict, extrapolation_pp: float, w: dict | None = None) -> dict:
    """Transparent decision score with its components (see module docstring)."""
    w = w or config.DECISION_WEIGHTS
    parts = {
        "probability_gain": w["prob_pp"] * capped_gain_pp(sc),
        "sales_gain": w["sales_pct"] * sc["delta_sales_pct"],
        "complexity_penalty": -w["complexity_per_day"] * max(sc["promo_delta"], 0),
        "extrapolation_penalty": -w["extrapolation_pp"] * extrapolation_pp,
    }
    return dict(score=sum(parts.values()), parts=parts)


def tested_interventions(d: dict, seg: str, month: str, options=None) -> list[dict]:
    options = options if options is not None else config.DECISION_PROMO_OPTIONS
    cal = d["calendar"][seg][month]
    base_days = d["whatif"][month]["base_promo_days"]
    hist_max = _hist_max_promo_share(d, seg)
    out = []
    for k in options:
        label = "Current plan" if k == 0 else f"+{k} promo day{'s' if k > 1 else ''}"
        sc = scenario(d, seg, month, promo=k, name=label,
                      assumptions=[f"{base_days + k} promo days in the month (plan: {base_days})"])
        share = cal["promo_share"] * (base_days + k) / base_days if base_days else cal["promo_share"]
        extra_pp = max(0.0, share - hist_max) * 100
        s = score(sc, extra_pp)
        sc.update(label=label, promo_share=share, extrapolation_pp=extra_pp, **s)
        out.append(sc)
    return out


def _impact_text(base: dict, rec: dict) -> str:
    if rec["promo_delta"] == 0:
        return "No change to the current plan"
    txt = f"+EUR {rec['delta_sales'] / 1e6:,.1f}M modeled sales"
    if rec["p_hit"] is not None:
        txt += f"; target probability {base['p_hit'] * 100:.0f}% → {rec['p_hit'] * 100:.0f}%"
    return txt


def build_decision(d: dict, seg: str, month: str) -> dict:
    labels = {s["id"]: s["label"] for s in d["segments"]}
    f, t = _fc(d, seg, month), d["targets"][seg].get(month)
    sd = sigma_of(f)
    base_p = p_hit(f["p50"], t, sd)
    lead = d.get("leading", {}).get(seg)
    tests = tested_interventions(d, seg, month)
    base = tests[0]
    candidates = [x for x in tests if x["promo_delta"] > 0]
    best = max(candidates, key=lambda x: x["score"]) if candidates else base
    action_required = base_p is not None and base_p < config.ACTION_THRESHOLD
    rec = best if (action_required and best["score"] > 0) else base
    for x in tests:
        x["recommended"] = x is rec

    # Downside stress test: footfall falls a further max(3%, latest shortfall) %, no extra promos
    shortfall = -lead["gap_pct"] if lead and lead["gap_pct"] < 0 else 0.0
    shock = max(config.DOWNSIDE_MIN_FOOTFALL_PCT, round(shortfall, 1))
    downside = scenario(d, seg, month, foot_pct=-shock, name="Downside",
                        assumptions=[f"Footfall a further −{shock:.1f}% vs forecast",
                                     "Latest shortfall repeats" if shortfall >= config.DOWNSIDE_MIN_FOOTFALL_PCT
                                     else f"Generic {config.DOWNSIDE_MIN_FOOTFALL_PCT:.0f}% stress test", "No extra promo days"])
    recovery = dict(best, name="Recovery", assumptions=best["assumptions"] + ["Best-scoring tested intervention"])
    base_s = dict(base, name="Base", assumptions=["Current promo and holiday calendar", "Latest momentum carried forward"])

    # Leading indicator and top reason
    if lead:
        flagged = lead["weeks_flagged"] >= 2
        lead_txt = f"Footfall {lead['gap_pct']:+.1f}% vs expected" + (
            f" ({lead['weeks_flagged']} of {lead['weeks']} weeks flagged)" if flagged else "")
    else:
        flagged, lead_txt = False, "No leading-indicator data"
    drv = _drivers(d, month, seg)
    neg = [x for x in drv if x["pct"] < 0]
    if flagged and lead["gap_pct"] < -2:
        top_reason = f"Sustained footfall shortfall: customers {abs(lead['gap_pct']):.1f}% below expected last month"
    elif t and f["p50"] < t:
        top_reason = f"Forecast is {((t / f['p50']) - 1) * 100:.1f}% short of target" + (
            f"; main drag: {neg[0]['name'].lower()} ({neg[0]['pct']:+.1f}%)" if neg else "")
    else:
        top_reason = f"Forecast is on or above target; main driver: {drv[0]['name'].lower()} ({drv[0]['pct']:+.1f}%)"

    tr = d.get("trust", {})
    cov = tr.get("coverage", {}).get("sales")
    ens = next((a for a in tr.get("accuracy", []) if a["model"] == "Vantage ensemble"), None)
    evidence = ["Model-based scenario estimate; promo response is learned from history, not causal proof",
                "Controlled synthetic dataset (Rossmann format)"]
    if ens:
        evidence.insert(0, f"Backtest error {ens['mape_total']:.1f}% (one month ahead, out-of-sample)")
    if cov:
        evidence.insert(1, f"{cov['inside80']}/{cov['n']} backtest outcomes inside the 80% range")
    if rec["promo_delta"] > 0 and rec["extrapolation_pp"] > 0:
        evidence.append("Recommended promo share is above anything seen in history (extrapolation)")
    level = "Moderate" if not (rec["promo_delta"] > 0 and rec["extrapolation_pp"] > 0) else "Low to moderate"

    last_sales = d["series"][seg]["sales"]["actual"][-1]
    return dict(
        segment=seg, segment_label=labels[seg], month=month, horizon=f["horizon"],
        month_label=_month_label(month), risk_level=risk_level(base_p) if base_p is not None else "Unknown",
        action_required=action_required,
        forecast=f["p50"], lo80=f["lo80"], hi80=f["hi80"], lo95=f["lo95"], hi95=f["hi95"],
        change_vs_last_pct=(f["p50"] / last_sales - 1) * 100, target=t, p_hit=base_p,
        leading_indicator=dict(text=lead_txt, flagged=flagged, **(lead or {})),
        top_reason=top_reason, top_drivers=drv[:3],
        levers=[dict(name="Promo days", modeled=True, note="Re-scored with the LightGBM store-day model"),
                dict(name="Footfall %", modeled=False, note="Manual assumption (multiplier)"),
                dict(name="Average ticket %", modeled=False, note="Manual assumption (multiplier)")],
        tested=tests,
        recommended=dict(label=rec["label"] if rec["promo_delta"] else "Keep current plan",
                         promo_delta=rec["promo_delta"], p50=rec["p50"], p_hit=rec["p_hit"],
                         delta_sales=rec["delta_sales"], delta_p_pp=rec["delta_p_pp"], score=rec["score"],
                         impact_text=_impact_text(base, rec),
                         why=("Highest decision score among tested interventions" if rec["promo_delta"]
                              else "Target probability already acceptable" if not action_required
                              else "No tested intervention scored above the current plan")),
        scenarios=dict(base=base_s, recovery=recovery, downside=downside),
        impact=dict(target_gap=(t - f["p50"]) if t else None, downside_exposure=f["p50"] - f["lo80"],
                    modeled_upside=rec["delta_sales"], probability_gain_pp=rec["delta_p_pp"] or 0.0),
        evidence=dict(level=level, points=evidence),
        score_weights=dict(config.DECISION_WEIGHTS, confidence_cap=config.CONFIDENCE_CAP,
                           action_threshold=config.ACTION_THRESHOLD),
    )


def _month_label(ym: str) -> str:
    import calendar
    y, m = ym.split("-")
    return f"{calendar.month_name[int(m)]} {y}"


def build_all(d: dict) -> dict:
    return {m: {s["id"]: build_decision(d, s["id"], m) for s in d["segments"]} for m in d["future_months"]}


def top_decision(decisions: dict, month: str) -> dict:
    """The most urgent decision for a month: lowest target probability among store types,
    falling back to 'All stores'."""
    segs = [v for k, v in decisions[month].items() if k != "All" and v["p_hit"] is not None]
    worst = min(segs, key=lambda v: v["p_hit"]) if segs else decisions[month]["All"]
    return worst if worst["action_required"] else decisions[month]["All"]
