"""
viz.py — rendering helpers for the Streamlit dashboard.
Everything here is a pure function of the TwinEngine state, so the same
engine object can be drawn in several places without side effects.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from twin_engine import WC_COLS, WC_ROWS

C = {"normal": "#4fa88a", "warning": "#d9a441", "critical": "#e0543a", "drift": "#8b7fc7",
     "copper": "#c97f4a", "dim": "#9096a0", "panel": "#1b1e23", "bg": "#14161a", "text": "#e9e6df"}


def temp_color(t: float) -> str:
    return "#2e6fd9" if t < 27 else "#4fa88a" if t < 33 else "#d9a441" if t < 39 else "#e0543a"


def pill(text: str, lvl: str) -> str:
    c = C.get(lvl, C["dim"])
    return (f'<span style="font-family:monospace;font-size:11px;font-weight:600;padding:2px 9px;'
            f'border-radius:10px;color:{c};background:{c}22;border:1px solid {c}55;white-space:nowrap">{text}</span>')


# ------------------------------------------------------------------ mine scene
def mine_svg(eng) -> str:
    n = eng.nodes
    crit_conf = eng.confirmed and eng.sys == "critical"
    P = []
    a = P.append
    a('<svg viewBox="0 0 960 520" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;'
      'background:#1b1e23;border-radius:8px;border:1px solid #2c3138">')
    a('<defs><pattern id="smh" width="9" height="9" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">'
      '<line x1="0" y1="0" x2="0" y2="9" stroke="#000" stroke-opacity=".18"/></pattern>'
      '<radialGradient id="smv" cx="50%" cy="35%" r="75%"><stop offset="0%" stop-color="#18181a"/>'
      '<stop offset="100%" stop-color="#050506"/></radialGradient>'
      '<linearGradient id="smsky" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#2a3446"/>'
      '<stop offset="100%" stop-color="#1b2230"/></linearGradient>'
      '<linearGradient id="smscale" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#2e6fd9"/>'
      '<stop offset="35%" stop-color="#4fa88a"/><stop offset="65%" stop-color="#d9a441"/>'
      '<stop offset="100%" stop-color="#e0543a"/></linearGradient>'
      '<filter id="smb" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="22"/></filter>'
      '</defs>')
    # surface + strata
    a('<rect width="960" height="46" fill="url(#smsky)"/>'
      '<path d="M60 46 L60 8 L92 8 L92 46 M60 8 L92 22 M92 8 L60 22" stroke="#4a4038" stroke-width="3" fill="none" opacity=".55"/>'
      '<text x="76" y="20" text-anchor="middle" font-family="monospace" font-size="7.5" fill="#7d8492">SHAFT</text>')
    for y, h, col in [(46, 110, "#241f1a"), (156, 90, "#2e281f"), (246, 150, "#1c1a18"), (396, 80, "#2a251d"), (476, 44, "#231f19")]:
        a(f'<rect x="0" y="{y}" width="960" height="{h}" fill="{col}"/>')
    a('<rect x="0" y="46" width="960" height="474" fill="url(#smh)"/>')
    a('<path d="M 40 281 Q 480 251 920 281 L 920 376 Q 480 398 40 376 Z" fill="url(#smv)" stroke="#000" stroke-width="2"/>')
    a('<g stroke="#4a4038" stroke-width="4" opacity=".55"><path d="M170 376 L170 278"/><path d="M370 381 L370 264"/>'
      '<path d="M590 381 L590 264"/><path d="M790 376 L790 278"/></g>')
    # temperature heat blobs
    pos = {"A": (260, 246), "B": (650, 246), "C": (460, 304)}
    a('<g filter="url(#smb)" style="mix-blend-mode:screen" opacity=".6">')
    for k, (x, y) in pos.items():
        r = 65 + min(45, max(0, n[k]["temp"] - 27) * 3.2)
        a(f'<circle cx="{x}" cy="{y}" r="{r:.0f}" fill="{temp_color(n[k]["temp"])}"/>')
    a('</g>')
    # cracks
    if crit_conf:
        for d in ("M 250 256 L 262 272 L 245 284 L 267 298 L 253 312", "M 650 256 L 662 273 L 640 286 L 664 299 L 648 313"):
            a(f'<path d="{d}" fill="none" stroke="{C["critical"]}" stroke-width="4" opacity=".5"/>'
              f'<path d="{d}" fill="none" stroke="#0a0a0a" stroke-width="2"/>')
    # cross-check triangle when a single node deviates
    if eng.calibrating:
        for (x1, y1, x2, y2) in [(266, 242, 454, 290), (466, 290, 644, 242), (266, 238, 644, 238)]:
            a(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{C["drift"]}" stroke-width="1.6" stroke-dasharray="5,4" opacity=".85"/>')
    # LoRa links + gateway
    for k, (x, y) in pos.items():
        a(f'<path d="M {x} {y-4} Q {(x+460)/2} 110 460 100" fill="none" stroke="#3a4049" stroke-width="2" stroke-dasharray="4,5"/>')
    a('<path d="M 460 100 C 560 65, 720 58, 850 85" fill="none" stroke="#3a4049" stroke-width="2" stroke-dasharray="4,5"/>'
      '<line x1="460" y1="98" x2="460" y2="126" stroke="#c97f4a" stroke-width="3"/>'
      '<circle cx="460" cy="92" r="6" fill="none" stroke="#c97f4a" stroke-width="2"/>'
      '<text x="460" y="142" text-anchor="middle" font-family="monospace" font-size="9.5" fill="#9096a0">GATEWAY</text>'
      '<ellipse cx="858" cy="80" rx="26" ry="14" fill="#20242a" stroke="#3a4049"/>'
      '<text x="858" y="114" text-anchor="middle" font-family="monospace" font-size="9" fill="#9096a0">AI RISK ENGINE</text>')
    # nodes
    labels = {"A": "NODE A · ROOF TILT", "B": "NODE B · ROOF TILT", "C": "NODE C · FLOOR VIB"}
    for k, (x, y) in pos.items():
        s = n[k]
        lvl = "drift" if s["drift"] else s["fused"]
        col = C[lvl]
        a(f'<g transform="translate({x},{y})">')
        a(f'<circle r="16" fill="none" stroke="{col}" stroke-width="2.4" opacity=".9"/>')
        a('<rect x="-10" y="-8" width="20" height="16" rx="3" fill="#20242a" stroke="#3a4049"/>')
        if k == "C":
            amp = 3 + min(9, s["vib"] * 18)
            a(f'<path d="M -8 0 L -4 0 L -2 {-amp:.1f} L 1 {amp*0.6:.1f} L 3 {-amp*0.4:.1f} L 8 0" fill="none" stroke="{col}" stroke-width="1.6"/>')
        else:
            ang = max(-32, min(32, s["tilt"]))
            a(f'<g transform="rotate({ang:.1f})"><line x1="0" y1="0" x2="0" y2="15" stroke="{col}" stroke-width="2.4" stroke-linecap="round"/></g>')
        a(f'<text y="-30" text-anchor="middle" font-family="monospace" font-weight="600" font-size="9.5" fill="{col}">'
          f'{"DRIFT?" if s["drift"] else s["fused"].upper()}</text>')
        a(f'<text y="38" text-anchor="middle" font-family="monospace" font-size="9.5" fill="#9096a0">{labels[k]}</text>')
        a(f'<text y="51" text-anchor="middle" font-family="monospace" font-size="9" fill="#c9c2b0">'
          f'AI {s["ml_pct"]:.0f}% · {s["temp"]:.1f}°C</text>')
        a('</g>')
    # evacuation overlay
    if eng.evacuated:
        a('<path d="M 460 340 C 300 380, 130 300, 76 46" fill="none" stroke="#d9a441" stroke-width="2.6" stroke-dasharray="7,6"/>'
          '<polygon points="76,46 70,60 82,60" fill="#d9a441"/>'
          '<g transform="translate(60,38) rotate(78)"><line x1="0" y1="0" x2="34" y2="0" stroke="#d9a441" stroke-width="4" stroke-linecap="round"/>'
          '<line x1="6" y1="0" x2="10" y2="0" stroke="#1c1a18" stroke-width="4"/><line x1="18" y1="0" x2="22" y2="0" stroke="#1c1a18" stroke-width="4"/></g>')
        for (cx, cy) in [(250, 350), (262, 356), (240, 358), (650, 350), (664, 356), (640, 358)]:
            a(f'<polygon points="{cx},{cy-6} {cx+7},{cy+4} {cx-6},{cy+6}" fill="#2b241d" stroke="#151210"/>')
        for (cx, cy) in [(250, 335), (650, 335)]:
            a(f'<ellipse cx="{cx}" cy="{cy}" rx="34" ry="12" fill="#8a8578" opacity=".25"/>')
        a('<rect x="8" y="54" width="944" height="458" fill="none" stroke="#e0543a" stroke-width="6" opacity=".55"/>'
          f'<rect x="300" y="56" width="360" height="26" rx="5" fill="#e0543a" opacity=".92"/>'
          f'<text x="480" y="74" text-anchor="middle" font-family="monospace" font-size="13" font-weight="700" fill="#fff">'
          f'⚠ EVACUATE — T+{eng.evac_elapsed:.0f}s</text>')
    # legend + caption
    a('<g transform="translate(700,20)"><rect width="150" height="8" rx="2" fill="url(#smscale)"/>'
      '<text y="20" font-family="monospace" font-size="8" fill="#7d848f">18°C</text>'
      '<text x="140" y="20" text-anchor="end" font-family="monospace" font-size="8" fill="#7d848f">45°C</text>'
      '<text x="75" y="-4" text-anchor="middle" font-family="monospace" font-size="8" fill="#9096a0">ROCK-FACE TEMP</text></g>')
    a('<text x="480" y="512" text-anchor="middle" font-family="monospace" font-size="10" fill="#5d636d">'
      'Zone A — Gallery 3 (simulated cross-section · 3 virtual nodes · scored live by the trained model)</text></svg>')
    return "".join(P)


# ------------------------------------------------------------------ charts
def gauge_fig(value: float, sys_lvl: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number={"suffix": "%", "font": {"size": 34}},
        gauge={"axis": {"range": [0, 100], "tickcolor": "#5d636d"}, "bar": {"color": C[sys_lvl], "thickness": .35},
               "bgcolor": "#20242a", "borderwidth": 0,
               "steps": [{"range": [0, 33], "color": "rgba(79,168,138,0.20)"}, {"range": [33, 66], "color": "rgba(217,164,65,0.20)"},
                         {"range": [66, 100], "color": "rgba(224,84,58,0.20)"}]}))
    fig.update_layout(height=190, margin=dict(l=20, r=20, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      font={"color": C["text"], "family": "monospace"})
    return fig


def history_fig(eng) -> go.Figure:
    h = list(eng.history)
    fig = go.Figure()
    if h:
        t = [r["t"] for r in h]
        for k, col in zip("ABC", ["#c97f4a", "#4fa88a", "#8b7fc7"]):
            fig.add_trace(go.Scatter(x=t, y=[r[f"{k}_ai"] for r in h], name=f"Node {k} AI risk", mode="lines",
                                     line=dict(color=col, width=2)))
    fig.add_hrect(y0=66, y1=100, fillcolor="#e0543a", opacity=.10, line_width=0)
    fig.add_hrect(y0=33, y1=66, fillcolor="#d9a441", opacity=.08, line_width=0)
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), title="Trained-model risk per node (0-100)",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1b1e23", font={"color": C["text"]},
                      yaxis=dict(range=[0, 100], gridcolor="#2c3138"), xaxis=dict(title="sim seconds", gridcolor="#2c3138"),
                      legend=dict(orientation="h", y=-0.3))
    return fig


def worstcase_fig(grid: np.ndarray) -> go.Figure:
    z = grid
    text = [[f"{WC_ROWS[i]}{WC_COLS[j]}<br><b>{z[i][j]:.0f}</b>" + ("<br>GALLERY 3" if (i, j) == (1, 2) else "")
             for j in range(len(WC_COLS))] for i in range(len(WC_ROWS))]
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(c) for c in WC_COLS], y=WC_ROWS, zmin=0, zmax=100, text=text, texttemplate="%{text}",
        colorscale=[[0, "#4fa88a"], [.33, "#4fa88a"], [.331, "#d9a441"], [.66, "#d9a441"], [.661, "#e0543a"], [1, "#e0543a"]],
        xgap=6, ygap=6, showscale=False))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font={"color": "#fff", "family": "monospace"})
    return fig


# ------------------------------------------------------------------ html tables
def node_table(eng) -> str:
    rows = ""
    for k, s in eng.nodes.items():
        status = pill("DRIFT?", "drift") if s["drift"] else pill(s["fused"].upper(), s["fused"])
        ai = pill("%.0f%% %s" % (s["ml_pct"], s["ml"].upper()), s["ml"])
        rl = pill(s["rule"].upper(), s["rule"])
        rows += ("<tr><td><b>%s</b></td><td>%.1f°</td><td>%.2f g</td><td>%.1f°C</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                 % (k, s["tilt"], s["vib"], s["temp"], rl, ai, status))
    return _table(["Node", "Tilt", "Vib", "Temp", "Rules say", "AI says", "Fused"], rows)


def rules_table(eng) -> str:
    rows = "".join(f"<tr><td><b>{r['id']}</b></td><td>{r['rule']}</td><td style='color:#7d848f'>{r['layer']}</td>"
                   f"<td>{pill(r['status'], r['lvl'])}</td></tr>" for r in eng.rules())
    return _table(["Rule", "Condition", "Layer", "Live status"], rows)


def log_table(eng, n=8) -> str:
    rows = "".join(f"<tr><td style='color:#7d848f'>T+{e['t']:.0f}s</td><td>{pill(e['sev'].upper(), e['sev'])}</td><td>{e['msg']}</td></tr>"
                   for e in list(eng.events)[:n])
    return _table(["Time", "Level", "Event"], rows)


def stage_stepper(eng) -> str:
    stages = ["NORMAL", "WATCH", "WARNING", "CRITICAL", "EVACUATION"]
    cur = eng.stage()
    out = ""
    for s in stages:
        active = s == cur
        col = {"NORMAL": C["normal"], "WATCH": C["drift"], "WARNING": C["warning"], "CRITICAL": C["critical"], "EVACUATION": C["critical"]}[s]
        style = (f"background:{col};color:#fff;" if active else f"background:#20242a;color:#7d848f;")
        out += (f"<span style='{style}padding:5px 11px;border-radius:14px;font-family:monospace;font-size:11px;"
                f"font-weight:600;margin-right:4px;border:1px solid {col}66'>{s}</span>")
    if eng.sys == "critical" and not eng.evacuated:
        out += f"<span style='color:#d9a441;font-family:monospace;font-size:11px;margin-left:6px'>dwell {eng.dwell:.1f}s / 5s</span>"
    return out


def _table(head, rows) -> str:
    th = "".join(f"<th style='text-align:left;font-family:monospace;font-size:10px;color:#5d636d;text-transform:uppercase;"
                 f"padding:5px 8px;border-bottom:1px solid #2c3138'>{h}</th>" for h in head)
    return (f"<table style='width:100%;border-collapse:collapse;font-size:12.5px'><thead><tr>{th}</tr></thead>"
            f"<tbody>{rows}</tbody></table>").replace("<tr><td", "<tr style='border-bottom:1px solid #2c3138'><td")
