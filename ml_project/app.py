"""
app.py — SubsiMonitor Streamlit dashboard
=========================================
Run:   streamlit run app.py

Everything on the "Live Twin" tab is driven by src/twin_engine.py, which calls
the TRAINED RandomForest (models/risk_engine_rf.joblib) on every simulated
second for every node — so the AI Risk Engine box is a live component here,
not an illustration.

Tabs
  1  Live Twin        mine cross-section, gauge, per-node rules-vs-AI table,
                      escalation stepper, R1-R7 rule matrix, SQLite event log
  2  Worst-Case Map   mine-wide critical propagation projection (Gallery 3 epicentre)
  3  ML Model         metrics, plots, what-if probe, edge tree C code, retrain
  4  Real-Data Replay real held-out coal-mine shifts streamed through the model
  5  Hardware View    the standalone HTML twin (ESP32 pipeline + camera feed)
"""

import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import insight
import viz
from data_loader import load_dataset
from twin_engine import EVAC_DWELL_S, EventDB, ModelBridge, TwinEngine, band

def _stretch(fn) -> dict:
    """width='stretch' on new Streamlit, use_container_width=True on older ones."""
    return {"width": "stretch"} if "width" in inspect.signature(fn).parameters else {"use_container_width": True}


st.set_page_config(page_title="SubsiMonitor — Digital Twin", page_icon="⛏️", layout="wide")
st.markdown("""
<style>
  .block-container{padding-top:1.4rem;max-width:1400px}
  h1,h2,h3{letter-spacing:.2px}
  div[data-testid="stMetric"]{background:#1b1e23;border:1px solid #2c3138;border-radius:8px;padding:10px 14px}
  .sm-card{background:#1b1e23;border:1px solid #2c3138;border-radius:8px;padding:12px 14px;margin-bottom:10px}
  .sm-h{font-family:monospace;font-size:11px;letter-spacing:.7px;text-transform:uppercase;color:#9096a0;margin:0 0 8px}
  .sm-banner{padding:10px 16px;border-radius:8px;font-family:monospace;font-weight:600;margin-bottom:8px}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ resources
@st.cache_resource(show_spinner="Loading trained risk engine…")
def get_bridge() -> ModelBridge:
    return ModelBridge()


@st.cache_resource
def get_db() -> EventDB:
    return EventDB()


@st.cache_data
def get_dataset():
    X_train, y_train, X_test, y_test, feats = load_dataset()
    return X_test.reset_index(drop=True), y_test


bridge = get_bridge()
if "engine" not in st.session_state:
    st.session_state.engine = TwinEngine(bridge, get_db())
    st.session_state.live = True
    st.session_state.speed = 1
    st.session_state.replay = {"i": 0, "playing": False, "rows": [], "crit_run": 0, "evac": 0}
eng: TwinEngine = st.session_state.engine

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## ⛏️ SubsiMonitor")
    st.caption("SIH26025 · HexaNovaTech · EdgeAI mine-subsidence digital twin")
    st.toggle("Live simulation", key="live", help="Auto-advances the twin once per second.")
    st.select_slider("Sim speed", options=[1, 2, 4], key="speed", format_func=lambda x: f"{x}×")
    st.markdown("**Scenarios**")
    st.button("🌋 Real subsidence event", on_click=eng.trigger, args=("subsidence",), **_stretch(st.button),
              help="All 3 nodes ramp together over ~18 s → corroborated → evacuation.")
    st.button("⚠️ Isolated sensor fault (Node A)", on_click=eng.trigger, args=("fault",), **_stretch(st.button),
              help="Only Node A goes bad → 2-of-3 gate suppresses it as DRIFT.")
    st.button("⚡ Force critical hit now", on_click=eng.trigger, args=("criticalhit",), **_stretch(st.button),
              help="Skips the ramp: all nodes critical immediately → 5 s dwell → evacuation.")
    st.button("↺ Reset simulation", on_click=eng.reset, **_stretch(st.button))
    if st.button("⏭ Step +1 s", **_stretch(st.button), disabled=st.session_state.live):
        eng.step(1.0)
    st.divider()
    st.caption(f"**AI risk engine:** RandomForest ({len(bridge.model.estimators_)} trees) · "
               f"calibrated p_idle={bridge.p_idle:.3f}, p_critical={bridge.p_crit:.3f}")
    st.caption("Scoring uses real held-out coal-mine shifts picked by a sensor→shift adapter "
               "(see ML Model tab for what that means and its limits).")

# ------------------------------------------------------------------ view selector
# Only the selected view is rendered on each run. (Rendering every st.tab on every run while several
# auto-refreshing fragments were ticking let a refresh interrupt the full render, leaving later tabs blank.)
# Only the Live Twin view auto-steps the engine; the other views are static or refresh themselves.
V_LIVE, V_WC, V_ML, V_REPLAY, V_HW, V_DEMO = ("🛰️ Live Twin", "🔥 Worst-Case Heatmap", "🧠 ML Model",
                                             "📼 Real-Data Replay", "🔧 Hardware View", "🎬 Live Mine Twin")
st.markdown("# SubsiMonitor — Digital Twin")
view = st.radio("View", [V_LIVE, V_DEMO, V_WC, V_ML, V_REPLAY, V_HW], horizontal=True, key="view",
                label_visibility="collapsed")
run_every = 1.0 if (st.session_state.live and view == V_LIVE) else None
dt = float(st.session_state.speed)
with st.sidebar:
    st.markdown("---")
    if st.button("📝 Generate shift report", key="mkreport"):
        st.session_state["report_md"] = insight.shift_report(eng)
    if st.session_state.get("report_md"):
        st.download_button("⬇ Download shift report (.md)", st.session_state["report_md"],
                           file_name="subsimonitor_shift_report.md", mime="text/markdown", key="dlreport")
if view != V_LIVE:
    st.caption(f"Twin status: **{eng.stage()}** — the engine simulation pauses while another view is open; "
               f"switch back to 🛰️ Live Twin to continue it.")


# ------------------------------------------------------------------ 1. live twin
def _live_view():
    if st.session_state.live:
        eng.step(dt)
    lvl = "critical" if eng.evacuated else eng.sys
    st.markdown(f"<div style='margin:-6px 0 8px'>{viz.pill('● SYSTEM ' + eng.stage(), lvl)}</div>", unsafe_allow_html=True)
    if eng.evacuated:
        st.markdown(f"<div class='sm-banner' style='background:#e0543a22;border:1px solid #e0543a88;color:#ff9d8a'>"
                    f"🚨 EVACUATION SIGNAL ACTIVE — Zone A · Gallery 3 · T+{eng.evac_elapsed:.0f}s — "
                    f"siren on, barrier down, safe route to shaft lit.</div>", unsafe_allow_html=True)
    st.markdown(viz.stage_stepper(eng), unsafe_allow_html=True)
    left, right = st.columns([2.1, 1])
    with left:
        st.markdown(viz.mine_svg(eng), unsafe_allow_html=True)
    with right:
        st.markdown("<div class='sm-h'>System risk</div>", unsafe_allow_html=True)
        st.plotly_chart(viz.gauge_fig(eng.risk_value, eng.sys), **_stretch(st.plotly_chart), key="gauge")
        m1, m2 = st.columns(2)
        m1.metric("Corroboration", "confirmed 2/3" if eng.confirmed else ("verifying…" if eng.calibrating else "idle"))
        m2.metric("CRITICAL dwell", f"{eng.dwell:.1f}s / {EVAC_DWELL_S:.0f}s")
    st.markdown("<div class='sm-card'><div class='sm-h'>Per-node fusion — deterministic rules vs trained AI</div>"
                + viz.node_table(eng) + "</div>", unsafe_allow_html=True)
    fa, fb = st.columns([1, 1.3])
    with fa:
        st.markdown(insight.render_card(eng), unsafe_allow_html=True)
    with fb:
        st.plotly_chart(insight.forecast_fig(eng), **_stretch(st.plotly_chart), key="fcast")
    a, b = st.columns(2)
    with a:
        st.markdown("<div class='sm-card'><div class='sm-h'>Deterministic response matrix</div>"
                    + viz.rules_table(eng) + "</div>", unsafe_allow_html=True)
    with b:
        st.plotly_chart(viz.history_fig(eng), **_stretch(st.plotly_chart), key="hist")
    st.markdown("<div class='sm-card'><div class='sm-h'>Event log</div>" + viz.log_table(eng, 8) + "</div>",
                unsafe_allow_html=True)


if view == V_LIVE:
    st.fragment(run_every=run_every)(_live_view)()
    with st.expander("📚 Event history (SQLite — data/events.db)"):
        st.dataframe(get_db().recent(60), **_stretch(st.dataframe), hide_index=True)


# ------------------------------------------------------------------ 2. worst case
def _wc_view():
    running = eng.wc["running"]
    now = time.time()
    grid, radius = eng.wc_grid(now)
    st.markdown("A standalone **what-if projection**, separate from the live twin: if Gallery 3 (B3) goes fully "
                "CRITICAL and stays there, how far does the hazard spread across the wider mine layout?")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(viz.worstcase_fig(grid), **_stretch(st.plotly_chart), key="wcfig")
    with c2:
        crit, warn = int((grid >= 66).sum()), int(((grid >= 33) & (grid < 66)).sum())
        k1, k2 = st.columns(2)
        k1.metric("Critical galleries", f"{crit} / 20")
        k2.metric("Warning buffer", f"{warn} / 20")
        k3, k4 = st.columns(2)
        k3.metric("Propagation radius", f"{radius:.1f}")
        k4.metric("Elapsed", f"{(now - eng.wc['t0']) if running else 0:.1f}s")
        if st.button("▶ Simulate worst-case collapse", **_stretch(st.button), key="wcrun"):
            eng.wc_start(time.time())
            st.rerun()
        if st.button("↺ Reset heatmap", **_stretch(st.button), key="wcreset"):
            eng.wc_reset()
            st.rerun()
        st.caption("Severity decays with distance from the epicentre and ramps in as the wavefront passes. "
                   "Colours use the same 0-33 / 33-66 / 66-100 bands as the gauge.")
    if running and not eng.wc.get("settled") and (now - eng.wc["t0"]) > 12:
        eng.wc["settled"] = True      # fully propagated: stop the refresh timer
        st.rerun()


if view == V_WC:
    _wc_active = eng.wc["running"] and not eng.wc.get("settled")
    st.fragment(run_every=0.6 if _wc_active else None)(_wc_view)()


# ------------------------------------------------------------------ 3. ML model
if view == V_ML:
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    st.markdown("### Trained on real coal-mine hazard data")
    st.markdown("UCI **seismic-bumps** (Sikora & Wrobel, 2010): 1,809 shift-level seismic/seismoacoustic records from a "
                "Polish coal mine, labelled for whether a dangerous high-energy bump followed. Only ~6% of shifts are "
                "hazardous — the same rare-event imbalance our CRITICAL class has.")
    k = st.columns(4)
    k[0].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    k[1].metric("PR-AUC", f"{metrics['pr_auc']:.3f}", help="No-skill baseline ≈ 0.06")
    k[2].metric("Recall (hazardous)", f"{metrics['recall_hazardous']:.2f}")
    k[3].metric("Precision (hazardous)", f"{metrics['precision_hazardous']:.2f}")
    st.caption("Honest numbers: 6%-positive hazard forecasting is hard; published results on this dataset are similar. "
               "That is why the twin never lets the AI fire alone — rules R1-R3, the 2-of-3 gate (R4) and the dwell "
               "timer (R6) sit on top.")
    p1, p2, p3 = st.columns(3)
    p1.image(str(ROOT / "reports" / "confusion_matrix.png"), caption="Confusion matrix (held-out)")
    p2.image(str(ROOT / "reports" / "pr_roc_curves.png"), caption="ROC & Precision-Recall")
    imp = pd.Series(bridge.model.feature_importances_, index=bridge.feature_names).nlargest(12)[::-1]
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color="#4fa88a"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), title="Top feature importances (live from model)",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1b1e23", font={"color": "#e9e6df"})
    p3.plotly_chart(fig, **_stretch(st.plotly_chart), key="fi")

    st.markdown("### 🔬 What-if probe — score a sensor reading with the trained model")
    st.caption("Sensor→shift adapter: your tilt / vibration / temperature are fused into a severity (0-1); the adapter "
               "then draws real held-out mine shifts of that severity and the RandomForest scores them. The scoring is "
               "real and out-of-sample; the sensor→shift mapping is a proxy until SubsiMonitor has field data.")
    s1, s2, s3 = st.columns(3)
    tilt = s1.slider("Tilt (°)", 0.0, 30.0, 4.0, 0.5)
    vib = s2.slider("Vibration (g)", 0.0, 0.8, 0.08, 0.01)
    temp = s3.slider("Rock-face temp (°C)", 20.0, 50.0, 27.0, 0.5)
    pr = bridge.probe(tilt, vib, temp)
    from twin_engine import rule_class
    q = st.columns(4)
    q[0].metric("Fused severity", f"{pr['severity']:.2f}")
    q[1].metric("AI risk", f"{pr['risk_pct']:.0f}%", band(pr["risk_pct"]).upper())
    q[2].metric("Rules say", rule_class(tilt, vib).upper())
    q[3].metric("Edge-tree trip rate", f"{100 * pr['edge_trip_rate']:.0f}%")

    st.markdown("### 📟 On-device failsafe (exported to the ESP32)")
    hdr = ROOT / "models" / "edge_rules.h"
    st.code(hdr.read_text(encoding="utf-8") if hdr.exists() else "run src/export_edge_rules.py", language="c")

    st.markdown("### ♻️ Retrain")
    st.caption("Re-runs src/train.py and src/export_edge_rules.py, reloads the model, and restarts the twin.")
    if st.button("Retrain risk engine now"):
        with st.spinner("Training RandomForest + edge tree…"):
            for script in ("train.py", "export_edge_rules.py"):
                res = subprocess.run([sys.executable, str(ROOT / "src" / script)], capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT / "src")
                if res.returncode != 0:
                    st.error(res.stderr[-800:])
                    break
            else:
                st.cache_resource.clear()
                st.session_state.pop("engine", None)
                st.success("Retrained. Reloading…")
                st.rerun()


# ------------------------------------------------------------------ 4. replay
def _replay_view():
    X_test, y_test = get_dataset()
    R = st.session_state.replay
    c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
    play = c1.toggle("▶ Play", value=R["playing"], key="rp_play")
    if play != R["playing"]:
        R["playing"] = play
        st.rerun()   # full rerun so the fragment's auto-refresh timer is re-armed
    if c2.button("⏭ Step", key="rp_step"):
        R["i"] += 1
        _advance(R, X_test, y_test)
    if c3.button("↺ Restart", key="rp_reset"):
        R.update({"i": 0, "rows": [], "crit_run": 0, "evac": 0})
    if R["playing"]:
        R["i"] += 1
        _advance(R, X_test, y_test)
    df = pd.DataFrame(R["rows"])
    st.caption("Held-out real shifts streamed through the trained model, one per tick — the same NORMAL/WARNING/CRITICAL "
               "bands and 5-shift dwell rule as the live twin. ✖ marks shifts that were truly hazardous.")
    if df.empty:
        st.info("Press ▶ Play or ⏭ Step to start streaming shifts.")
        return
    tp = int(((df.band == "critical") & (df.truth == 1)).sum())
    fp = int(((df.band == "critical") & (df.truth == 0)).sum())
    fn = int(((df.band != "critical") & (df.truth == 1)).sum())
    m = st.columns(5)
    m[0].metric("Shifts streamed", len(df))
    m[1].metric("Hazards caught (CRITICAL)", tp)
    m[2].metric("False alarms", fp)
    m[3].metric("Missed hazards", fn)
    m[4].metric("Evacuations fired", R["evac"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["shift"], y=df["risk"], mode="lines", line=dict(color="#c97f4a", width=2), name="AI risk"))
    hz = df[df.truth == 1]
    fig.add_trace(go.Scatter(x=hz["shift"], y=hz["risk"], mode="markers", name="truly hazardous",
                             marker=dict(symbol="x", size=11, color="#e0543a")))
    fig.add_hrect(y0=66, y1=100, fillcolor="#e0543a", opacity=.10, line_width=0)
    fig.add_hrect(y0=33, y1=66, fillcolor="#d9a441", opacity=.08, line_width=0)
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="#1b1e23", font={"color": "#e9e6df"}, yaxis=dict(range=[0, 100], gridcolor="#2c3138"),
                      xaxis=dict(title="shift #", gridcolor="#2c3138"), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, **_stretch(st.plotly_chart), key="rpfig")
    last = df.tail(8).iloc[::-1]
    rows = "".join(f"<tr><td>#{r['shift']}</td><td>{r['risk']:.0f}%</td><td>{viz.pill(r['band'].upper(), r['band'])}</td>"
                   f"<td>{'⚠ HAZARD' if r['truth'] else 'safe'}</td><td>{r['note']}</td></tr>" for _, r in last.iterrows())
    st.markdown("<div class='sm-card'>" + viz._table(["Shift", "AI risk", "Band", "Ground truth", "Action"], rows) + "</div>",
                unsafe_allow_html=True)


def _advance(R, X_test, y_test):
    idx = (R["i"] - 1) % len(X_test)
    p = float(bridge.model.predict_proba(X_test.iloc[[idx]][bridge.feature_names])[0, 1])
    risk = bridge.to_risk_pct(p)
    bd = band(risk)
    R["crit_run"] = R["crit_run"] + 1 if bd == "critical" else 0
    note = ""
    if R["crit_run"] >= 5:
        R["evac"] += 1
        R["crit_run"] = 0
        note = "🚨 EVACUATION (CRITICAL held 5 shifts)"
    R["rows"].append({"shift": R["i"], "risk": risk, "band": bd, "truth": int(y_test[idx]), "note": note})
    R["rows"] = R["rows"][-300:]


if view == V_REPLAY:
    st.fragment(run_every=0.7 if st.session_state.replay["playing"] else None)(_replay_view)()


# ------------------------------------------------------------------ 5. hardware view
if view == V_HW:
    st.caption("The standalone visual twin: ESP32 board, LoRa → gateway → MQTT → risk-engine pipeline, temperature "
               "heatmap and camera feed. It runs its own in-browser simulation; the Live Twin tab is the one wired to the model.")
    html_path = ROOT / "web" / "subsimonitor_dashboard.html"
    if html_path.exists():
        if hasattr(st, "iframe"):
            st.iframe(html_path, height=2600)
        else:
            components.html(html_path.read_text(encoding="utf-8"), height=2600, scrolling=True)
    else:
        st.warning("Copy subsimonitor_dashboard.html into the web/ folder to show it here.")


# ------------------------------------------------------------------ 6. live mine twin (panel demo)
if view == V_DEMO:
    st.caption("Self-animating mine cross-section: ground sag, cracks, dust, five sensor nodes, composite risk, "
               "evacuation guidance, and a worst-case heatmap. It starts from the live twin's current risk, then runs its "
               "own in-browser simulation (same 0-33 / 33-66 / 66-100 bands). The Live Twin view is the one wired to the model.")
    demo_path = ROOT / "web" / "subsimonitor_live_twin.html"
    if demo_path.exists():
        seed = json.dumps({"sev": float(eng.risk_value)})
        components.html(demo_path.read_text(encoding="utf-8").replace("/*SEED*/null", seed), height=1150, scrolling=True)
    else:
        st.warning("Copy subsimonitor_live_twin.html into the web/ folder to show it here.")
