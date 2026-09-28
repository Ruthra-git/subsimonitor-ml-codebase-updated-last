"""
insight.py — early-warning intelligence for the SubsiMonitor twin
=================================================================
  forecast(eng)      trend-based time-to-CRITICAL estimate with a 95 % range
  explain(eng)       plain-language "why is this alert firing" for supervisors
  render_card(eng)   HTML card combining both
  forecast_fig(eng)  history + projection cone toward the CRITICAL line
  shift_report(eng)  markdown shift report (events, peaks, status)

Everything here is derived from the engine's own history; it adds no new data source.
Times are in the twin's simulated seconds. The forecast is a statistical trend estimate,
not a guarantee: it widens when the signal is noisy and says "stable" when there is no trend.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import plotly.graph_objects as go

from twin_engine import EVAC_DWELL_S

WARN_LEVEL, CRIT_LEVEL = 33.0, 66.0
WARN_TILT, WARN_VIB, WARN_TEMP, BASE_TEMP = 6.0, 0.22, 36.0, 27.0
COL = {"normal": "#4fa88a", "warning": "#d9a441", "critical": "#e0543a"}


def _series(eng, n: int = 40):
    h = list(eng.history)[-n:]
    if len(h) < 8:
        return None, None
    t = np.array([r["t"] for r in h], float)
    y = np.array([np.mean([r[f"{k}_ai"] for k in "ABC"]) for r in h], float)
    return t, y


def forecast(eng, target: float = CRIT_LEVEL, n: int = 40) -> dict:
    t, y = _series(eng, n)
    if t is None:
        return {"state": "warming"}
    cur = float(y[-1])
    if cur >= target:
        return {"state": "critical", "cur": cur}
    x = t - t[-1]
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, pred), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ np.array([slope, pred])
    sxx = float(((t - t.mean()) ** 2).sum())
    se = math.sqrt(float(resid @ resid) / max(1, len(t) - 2) / sxx) if sxx > 0 else 0.0
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss_tot if ss_tot > 1e-9 else 0.0
    lo_s, hi_s = slope - 1.96 * se, slope + 1.96 * se
    eta = lambda s: (target - pred) / s if s > 1e-6 else math.inf
    if slope <= 0.05 or hi_s <= 0.05:
        return {"state": "stable", "cur": cur, "slope": float(slope), "pred": float(pred), "r2": r2}
    return {"state": "rising", "cur": cur, "slope": float(slope), "pred": float(pred), "r2": r2,
            "lo_s": float(lo_s), "hi_s": float(hi_s),
            "eta": eta(slope), "eta_fast": eta(hi_s), "eta_slow": eta(lo_s)}


def _fmt_s(v: float) -> str:
    if not math.isfinite(v):
        return "no bound"
    return f"{v:.0f} s" if v < 120 else f"{v / 60:.1f} min" if v < 3600 else "> 1 h"


def explain(eng) -> dict:
    rows = []
    for k, n in eng.nodes.items():
        ratios = {"tilt": abs(n["tilt"]) / WARN_TILT, "vibration": n["vib"] / WARN_VIB,
                  "temperature": max(0.0, (n["temp"] - BASE_TEMP) / (WARN_TEMP - BASE_TEMP))}
        drv = max(ratios, key=ratios.get)
        val = {"tilt": f"{n['tilt']:.1f}°", "vibration": f"{n['vib']:.2f} g", "temperature": f"{n['temp']:.1f} °C"}[drv]
        rows.append(dict(node=k, ai=n["ml_pct"], rule=n["rule"], fused=n["fused"], drift=n.get("drift", False),
                         driver=drv, ratio=ratios[drv], value=val))
    rows.sort(key=lambda r: -r["ai"])
    top = rows[0]
    elevated = [r for r in rows if r["fused"] != "normal"]
    lines = []
    if eng.sys == "normal" and top["ai"] < WARN_LEVEL:
        head = f"All nodes in normal range. Highest AI risk: Node {top['node']} at {top['ai']:.0f}/100."
    else:
        head = (f"Node {top['node']} is the main driver: {top['driver']} {top['value']} "
                f"({top['ratio'] * 100:.0f}% of its WARNING threshold). AI {top['ai']:.0f}/100, rules {top['rule'].upper()}.")
    if elevated:
        lines.append(f"{len(elevated)}/3 nodes elevated — " +
                     ("corroborated, alert is active." if eng.confirmed else
                      "not yet corroborated; a lone bad node is treated as possible sensor drift."))
    for r in rows:
        if r["drift"]:
            lines.append(f"Node {r['node']} flagged DRIFT? and suppressed from the alert.")
    if eng.sys == "critical" and not eng.evacuated:
        lines.append(f"Evacuation fires in {max(0.0, EVAC_DWELL_S - eng.dwell):.1f} s if CRITICAL persists.")
    if eng.evacuated:
        lines.append("Evacuation signal active: siren on, barrier down, safe route lit.")
    return {"head": head, "lines": lines, "rows": rows}


def render_card(eng) -> str:
    f, e = forecast(eng), explain(eng)
    if f["state"] == "warming":
        big, sub, col = "…", "collecting trend data", "#9096a0"
    elif f["state"] == "critical":
        big, sub, col = "NOW", "risk is already at CRITICAL level", COL["critical"]
    elif f["state"] == "stable":
        big, sub, col = "STABLE", "no upward trend detected", COL["normal"]
    else:
        big = "≈ " + _fmt_s(f["eta"])
        sub = f"95% range {_fmt_s(f['eta_fast'])} – {_fmt_s(f['eta_slow'])} · trend fit R² {f['r2']:.2f}"
        col = COL["critical"] if f["eta"] < 30 else COL["warning"]
    bullets = "".join(f"<li>{s}</li>" for s in e["lines"])
    return (f"<div class='sm-card'><div class='sm-h'>Early-warning intelligence</div>"
            f"<div style='font-size:12px;color:#9096a0'>Estimated time to CRITICAL (simulated seconds)</div>"
            f"<div style='font-size:30px;font-weight:700;color:{col};line-height:1.1'>{big}</div>"
            f"<div style='font-size:12px;color:#9096a0;margin-bottom:8px'>{sub}</div>"
            f"<div style='font-size:13px;margin-bottom:4px'><b>Why:</b> {e['head']}</div>"
            f"<ul style='margin:4px 0 0 16px;padding:0;font-size:12px;color:#c9ccd2'>{bullets}</ul></div>")


def forecast_fig(eng) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="plotly_dark", height=290, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      title=dict(text="Mean AI risk and projection to CRITICAL", font=dict(size=13)),
                      yaxis=dict(range=[0, 105], title="risk %"), xaxis=dict(title="sim time (s)"),
                      legend=dict(orientation="h", y=-0.25))
    t, y = _series(eng, 60)
    if t is None:
        fig.add_annotation(text="collecting data…", showarrow=False)
        return fig
    fig.add_hline(y=WARN_LEVEL, line=dict(color="rgba(217,164,65,0.6)", dash="dot"))
    fig.add_hline(y=CRIT_LEVEL, line=dict(color="rgba(224,84,58,0.7)", dash="dot"))
    fig.add_scatter(x=t, y=y, mode="lines", name="mean AI risk", line=dict(color="#c97f4a", width=2))
    f = forecast(eng)
    if f["state"] == "rising":
        H = 120.0 if not math.isfinite(f["eta_slow"]) else min(120.0, max(10.0, f["eta_slow"] * 1.1))
        xs = [t[-1], t[-1] + H]
        up = [f["pred"], min(105, f["pred"] + f["hi_s"] * H)]
        lo = [f["pred"], max(0, f["pred"] + f["lo_s"] * H)]
        mid = [f["pred"], min(105, f["pred"] + f["slope"] * H)]
        fig.add_scatter(x=xs, y=up, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=xs, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                        fillcolor="rgba(201,127,74,0.20)", name="95% range", hoverinfo="skip")
        fig.add_scatter(x=xs, y=mid, mode="lines", name="projection", line=dict(color="#e8d9b0", dash="dash"))
    return fig


def shift_report(eng) -> str:
    h = list(eng.history)
    peaks = {k: (max((r[f"{k}_ai"] for r in h), default=0.0)) for k in "ABC"}
    f = forecast(eng)
    out = [f"# SubsiMonitor shift report",
           f"Generated {datetime.now():%Y-%m-%d %H:%M} · simulated Zone A, Gallery 3 · twin time T+{eng.t:.0f}s", "",
           "## Status",
           f"- System stage: **{eng.stage()}**",
           f"- Evacuation active: **{'YES' if eng.evacuated else 'no'}**",
           f"- Trend: **{f['state']}**" + (f" (≈ {_fmt_s(f['eta'])} to CRITICAL)" if f["state"] == "rising" else ""), "",
           "## Peak AI risk per node (0-100)"] + [f"- Node {k}: {v:.0f}" for k, v in peaks.items()] + [
           "", "## Why (current)", explain(eng)["head"], "", "## Recent events (newest first)"]
    out += [f"- T+{ev['t']}s [{ev['sev']}] {ev['msg']}" for ev in list(eng.events)[:40]]
    out += ["", "_Simulated data. AI scoring uses real held-out coal-mine shifts through a documented sensor-to-shift adapter._"]
    return "\n".join(out)
