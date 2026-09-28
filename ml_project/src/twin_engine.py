"""
twin_engine.py
--------------
The Python port of the SubsiMonitor digital twin, with the trained AI risk
engine (models/risk_engine_rf.joblib) wired in as a LIVE component.

Per simulated second, for each virtual node (A, B, C):

  1. simulate MEMS-style tilt / vibration / temperature (idle jitter + the
     scenario profile you trigger: subsidence, isolated fault, critical hit)
  2. deterministic layer  -> threshold class (rules R1-R3), like the ESP32 would
  3. AI layer             -> the RandomForest scores a real mine-shift record
                             chosen by the SensorToShiftAdapter (see below)
  4. fuse                 -> node class = worse of (rules, AI)
  5. corroborate          -> 2-of-3 nodes must agree, else the outlier is DRIFT
  6. escalate             -> CRITICAL held >= 5 s => EVACUATION (rule R6),
                             with hysteresis on the way back down

HONEST NOTE ON THE ADAPTER
The model was trained on geophone shift summaries (UCI seismic-bumps), not on
tilt/vibration/temperature. Until SubsiMonitor has its own field data, the
SensorToShiftAdapter bridges the gap: it turns a node's fused sensor severity
(0..1) into a real, held-out mine-shift record of matching severity (safe
shifts for low severity, hazardous shifts increasingly for high severity) and
lets the model score that record. The scoring is real and out-of-sample; the
sensor -> record mapping is a documented proxy that should be replaced by a
model retrained on real tilt/vibration/temperature data.
"""

from __future__ import annotations

import sqlite3
import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data_loader import load_dataset

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DB_PATH = ROOT / "data" / "events.db"

LEVELS = ["normal", "warning", "critical"]
EVAC_DWELL_S = 5.0        # CRITICAL must hold this long before evacuation (R6)
CRIT_GRACE_S = 1.5        # a dip shorter than this doesn't reset the dwell timer
STAND_DOWN_S = 4.0        # readings must stay below CRITICAL this long to stand down
WC_ROWS, WC_COLS = ["A", "B", "C", "D"], [1, 2, 3, 4, 5]
WC_EPI = ("B", 3)


def worse(a: str, b: str) -> str:
    return LEVELS[max(LEVELS.index(a), LEVELS.index(b))]


def rule_class(tilt: float, vib: float) -> str:
    """Deterministic on-device thresholds (same numbers as the HTML twin)."""
    dev = abs(tilt)
    if dev > 14 or vib > 0.5:
        return "critical"
    if dev > 6 or vib > 0.22:
        return "warning"
    return "normal"


def band(risk_pct: float) -> str:
    return "normal" if risk_pct < 33 else "warning" if risk_pct < 66 else "critical"


# --------------------------------------------------------------------------
# Model bridge: trained RandomForest + edge tree + sensor->shift adapter
# --------------------------------------------------------------------------
class ModelBridge:
    def __init__(self, seed: int = 7):
        bundle = joblib.load(MODELS / "risk_engine_rf.joblib")
        self.model = bundle["model"]
        self.feature_names = bundle["feature_names"]

        edge = MODELS / "edge_tree.joblib"
        self.edge = joblib.load(edge) if edge.exists() else None

        # Out-of-sample pools only: the held-out test split, never training rows.
        _, _, X_test, y_test, _ = load_dataset()
        X_test = X_test.reset_index(drop=True)
        act = (X_test["genergy"].rank(pct=True) + X_test["gpuls"].rank(pct=True)
               + X_test["energy"].rank(pct=True)) / 3
        self.pools = {}
        for k in (0, 1):
            sel = (y_test == k)
            P = X_test[sel].assign(_a=act[sel]).sort_values("_a").drop(columns="_a")
            self.pools[k] = P.reset_index(drop=True)
        self.rng = np.random.default_rng(seed)

        # Calibrate the 0-100 risk scale so idle ~ 0 and a fully critical node ~ 100.
        p_idle = self._probs([0.02] * 300).mean()
        p_crit = self._probs([1.0] * 300).mean()
        self.p_idle, self.p_crit = float(p_idle), float(max(p_crit, p_idle + 1e-3))

    # -- adapter ---------------------------------------------------------
    def draw_row(self, sev: float) -> pd.DataFrame:
        h = float(np.clip((sev - 0.35) / 0.6, 0, 1))      # chance of a hazardous shift
        k = 1 if self.rng.random() < h else 0
        P = self.pools[k]
        q = float(np.clip(sev + self.rng.normal(0, 0.15), 0, 0.999))
        return P.iloc[[int(q * len(P))]][self.feature_names]

    def _probs(self, sevs) -> np.ndarray:
        X = pd.concat([self.draw_row(s) for s in sevs])
        return self.model.predict_proba(X)[:, 1]

    def score(self, sevs: list[float]):
        """Returns (probabilities, edge_trip flags, rows) for a batch of severities."""
        rows = pd.concat([self.draw_row(s) for s in sevs])
        p = self.model.predict_proba(rows)[:, 1]
        if self.edge:
            trips = self.edge["tree"].predict(rows[self.edge["features"]].values).astype(bool)
        else:
            trips = np.zeros(len(sevs), dtype=bool)
        return p, trips, rows

    def to_risk_pct(self, p: float) -> float:
        return float(np.clip(100 * (p - self.p_idle) / (self.p_crit - self.p_idle), 0, 100))

    def probe(self, tilt: float, vib: float, temp: float, n: int = 40) -> dict:
        """One-off 'what if' query used by the ML tab's sliders."""
        sev = node_severity(tilt, vib, temp)
        p, trips, _ = self.score([sev] * n)
        pm = float(p.mean())
        return {"severity": sev, "p": pm, "risk_pct": self.to_risk_pct(pm),
                "edge_trip_rate": float(trips.mean())}


def node_severity(tilt: float, vib: float, temp: float) -> float:
    t = min(abs(tilt) / 20.0, 1.0)
    v = min(vib / 0.6, 1.0)
    h = float(np.clip((temp - 27.0) / 14.0, 0, 1))
    return float(np.clip(0.45 * t + 0.35 * v + 0.20 * h, 0, 1))


# --------------------------------------------------------------------------
# Event log (SQLite, matches the stack on the feasibility slide)
# --------------------------------------------------------------------------
class EventDB:
    def __init__(self, path: Path = DB_PATH):
        path.parent.mkdir(exist_ok=True)
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.execute("""CREATE TABLE IF NOT EXISTS events(
            id INTEGER PRIMARY KEY AUTOINCREMENT, wall TEXT, sim_t REAL, sev TEXT, msg TEXT)""")
        self.con.commit()

    def add(self, sim_t, sev, msg):
        self.con.execute("INSERT INTO events(wall,sim_t,sev,msg) VALUES(?,?,?,?)",
                         (time.strftime("%Y-%m-%d %H:%M:%S"), sim_t, sev, msg))
        self.con.commit()

    def recent(self, n=50) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT wall AS time, sim_t AS sim_seconds, sev AS severity, msg AS event "
            "FROM events ORDER BY id DESC LIMIT ?", self.con, params=(n,))


# --------------------------------------------------------------------------
# The twin
# --------------------------------------------------------------------------
class TwinEngine:
    def __init__(self, bridge: ModelBridge, db: EventDB | None = None, seed: int = 11):
        self.bridge = bridge
        self.db = db
        self.rng = np.random.default_rng(seed)
        self.reset(log=False)

    # -- lifecycle -------------------------------------------------------
    def reset(self, log=True):
        self.t = 0.0
        self.event = None                     # {"type","t0"}
        base = {"A": 27.0, "B": 26.4, "C": 27.8}
        self.nodes = {n: dict(tilt=0.0, vib=0.06, temp=base[n], sev=0.0, p=self.bridge.p_idle,
                              ml_pct=0.0, rule="normal", ml="normal", fused="normal",
                              drift=False, edge_trip=False) for n in "ABC"}
        self.sys = "normal"
        self.confirmed = False
        self.calibrating = False
        self.risk_value = 6.0
        self.crit_since = None
        self.nonc_since = None
        self.evacuated = False
        self.evac_start = None
        self.events: deque = deque(maxlen=40)
        self.history: deque = deque(maxlen=240)
        self.wc = {"running": False, "t0": None}
        if log:
            self.log("sys", "Simulation reset.")
        else:
            self.log("sys", "Digital twin online — 3 nodes streaming, AI risk engine loaded.")

    def log(self, sev: str, msg: str):
        self.events.appendleft({"t": round(self.t, 1), "sev": sev, "msg": msg})
        if self.db:
            self.db.add(self.t, sev, msg)

    def trigger(self, kind: str):
        self.event = {"type": kind, "t0": self.t}
        msg = {"subsidence": "Demo: real subsidence event — all 3 nodes will move together.",
               "fault": "Demo: isolated sensor fault on Node A — watch the cross-check.",
               "criticalhit": "Demo: critical hit forced on all nodes — watch the 5 s dwell timer."}[kind]
        self.log({"subsidence": "warning", "fault": "drift", "criticalhit": "critical"}[kind], msg)

    # -- one simulated step ---------------------------------------------
    def step(self, dt: float = 1.0):
        self.t += dt
        ev = self.event
        a_tilt, a_vib, a_temp = (1 - (1 - x) ** (dt / 0.12) for x in (0.12, 0.15, 0.08))
        rnd = self.rng.random

        for name, n in self.nodes.items():
            tt = (rnd() - 0.5) * 1.2
            tv = 0.05 + rnd() * 0.05
            tT = 27 + {"A": 0.0, "B": -0.6, "C": 0.8}[name] + (rnd() - 0.5) * 0.6
            if ev:
                el = self.t - ev["t0"]
                if ev["type"] == "subsidence":
                    pr = min(el / 18, 1)
                    tt = pr * 20 + (rnd() - 0.5) * 2
                    tv = 0.06 + pr * 0.5 + max(0, rnd() - 0.7) * 0.3
                    tT = 27 + pr * 14 + (rnd() - 0.5) * 1.2
                elif ev["type"] == "fault" and name == "A":
                    if el < 12:
                        tt = 18 + (rnd() - 0.5) * 2
                        tv = 0.1 + rnd() * 0.05
                        tT = 27 + min(el * 1.4, 13) + (rnd() - 0.5) * 1.5
                elif ev["type"] == "criticalhit":
                    tt = 22 + (rnd() - 0.5) * 2
                    tv = 0.6 + rnd() * 0.15
                    tT = 39 + (rnd() - 0.5) * 2
            n["tilt"] += (tt - n["tilt"]) * a_tilt
            n["vib"] += (tv - n["vib"]) * a_vib
            n["temp"] += (tT - n["temp"]) * a_temp
            n["sev"] = node_severity(n["tilt"], n["vib"], n["temp"])
            n["rule"] = rule_class(n["tilt"], n["vib"])

        # --- AI layer: the trained RandomForest scores each node (batched) ---
        names = list(self.nodes)
        p, trips, _ = self.bridge.score([self.nodes[k]["sev"] for k in names])
        for k, pk, tk in zip(names, p, trips):
            n = self.nodes[k]
            n["p"] = 0.5 * n["p"] + 0.5 * float(pk)           # light smoothing
            n["ml_pct"] = self.bridge.to_risk_pct(n["p"])
            n["ml"] = band(n["ml_pct"])
            n["edge_trip"] = bool(tk)
            n["fused"] = worse(n["rule"], n["ml"])

        # --- 2-of-3 corroboration -----------------------------------------
        elevated = [k for k, n in self.nodes.items() if n["fused"] != "normal"]
        self.confirmed = len(elevated) >= 2
        for k, n in self.nodes.items():
            n["drift"] = (n["fused"] != "normal") and not self.confirmed
        self.calibrating = len(elevated) == 1

        prev = self.sys
        self.sys = "normal"
        if self.confirmed:
            self.sys = "critical" if any(self.nodes[k]["fused"] == "critical" for k in elevated) else "warning"
        if self.sys != prev:
            self.log(self.sys if self.sys != "normal" else "normal",
                     {"critical": "Corroborated CRITICAL — 2-of-3 nodes confirm ground movement.",
                      "warning": "Corroborated WARNING — 2-of-3 nodes trending together.",
                      "normal": "System back to normal range."}[self.sys])

        # --- dwell timer / evacuation with hysteresis ---------------------
        if self.sys == "critical":
            self.nonc_since = None
            if self.crit_since is None:
                self.crit_since = self.t
        else:
            if self.nonc_since is None:
                self.nonc_since = self.t
            if self.t - self.nonc_since >= CRIT_GRACE_S:
                self.crit_since = None
        if self.sys == "critical" and not self.evacuated and self.crit_since is not None \
                and self.t - self.crit_since >= EVAC_DWELL_S:
            self._evacuate()
        if self.evacuated and self.sys != "critical" and self.nonc_since is not None \
                and self.t - self.nonc_since >= STAND_DOWN_S:
            self._stand_down()

        # --- gauge ---------------------------------------------------------
        avg = np.mean([n["ml_pct"] for n in self.nodes.values()])
        target = 92 if self.sys == "critical" else 55 if self.sys == "warning" else max(4, avg * 0.5)
        self.risk_value += (target - self.risk_value) * (1 - 0.92 ** (dt / 0.12))

        self.history.append({"t": round(self.t, 1), "sys": LEVELS.index(self.sys),
                             **{f"{k}_ai": n["ml_pct"] for k, n in self.nodes.items()},
                             **{f"{k}_tilt": n["tilt"] for k, n in self.nodes.items()},
                             **{f"{k}_vib": n["vib"] for k, n in self.nodes.items()},
                             **{f"{k}_temp": n["temp"] for k, n in self.nodes.items()}})

    def _evacuate(self):
        self.evacuated = True
        self.evac_start = self.t
        for m in ("siren + evacuation signal dispatched to all collars in Zone A",
                  "access barrier lowered at the Gallery 3 shaft entrance",
                  "safe route to shaft highlighted on all worker displays",
                  "power isolated and ventilation reversed in Zone A"):
            self.log("critical", "ACTION — " + m + ".")

    def _stand_down(self):
        self.evacuated = False
        self.log("normal", "ACTION — all-clear confirmed, barrier raised, siren stood down.")

    # -- derived views -----------------------------------------------------
    @property
    def dwell(self) -> float:
        return 0.0 if self.crit_since is None else self.t - self.crit_since

    @property
    def evac_elapsed(self) -> float:
        return 0.0 if not self.evacuated else self.t - self.evac_start

    def rules(self) -> list[dict]:
        n = self.nodes
        hot = [k for k in n if n[k]["temp"] > 36]
        tilt = [k for k in n if abs(n[k]["tilt"]) > 6]
        vib = [k for k in n if n[k]["vib"] > 0.22]
        edge = [k for k in n if n[k]["edge_trip"]]
        crit_conf = self.confirmed and any(n[k]["fused"] == "critical" for k in n)
        def lab(lst): return ("FIRED · " + ",".join(lst)) if lst else "CLEAR"
        r6 = ("FIRED · EVACUATION", "critical") if self.evacuated else (
            (f"ARMED · {self.dwell:.1f}s / {EVAC_DWELL_S:.0f}s", "warning") if self.sys == "critical" else ("CLEAR", "normal"))
        return [
            {"id": "R1", "rule": "Any node tilt > 6°", "layer": "deterministic", "status": lab(tilt), "lvl": "warning" if tilt else "normal"},
            {"id": "R2", "rule": "Any node vibration > 0.22 g", "layer": "deterministic", "status": lab(vib), "lvl": "warning" if vib else "normal"},
            {"id": "R3", "rule": "Rock-face temperature > 36 °C", "layer": "deterministic", "status": lab(hot), "lvl": "warning" if hot else "normal"},
            {"id": "R4", "rule": "2-of-3 nodes elevated (corroboration)", "layer": "gate", "status": "ARMED" if self.confirmed else "CLEAR", "lvl": "warning" if self.confirmed else "normal"},
            {"id": "R5", "rule": "Corroborated AND any node critical", "layer": "gate", "status": "FIRED" if crit_conf else "CLEAR", "lvl": "critical" if crit_conf else "normal"},
            {"id": "R6", "rule": f"CRITICAL held ≥ {EVAC_DWELL_S:.0f}s (dwell timer)", "layer": "escalation", "status": r6[0], "lvl": r6[1]},
            {"id": "R7", "rule": "Learned edge tree trips (distilled from data)", "layer": "learned failsafe", "status": lab(edge), "lvl": "warning" if edge else "normal"},
        ]

    def stage(self) -> str:
        if self.evacuated:
            return "EVACUATION"
        return {"normal": "WATCH" if self.calibrating else "NORMAL",
                "warning": "WARNING", "critical": "CRITICAL"}[self.sys]

    # -- worst-case projection (independent 'what if' grid) ------------------
    def wc_start(self, now: float):
        self.wc = {"running": True, "t0": now}
        self.log("critical", "Worst-case projection started — propagating outward from Gallery 3 (B3).")

    def wc_reset(self):
        self.wc = {"running": False, "t0": None}

    def wc_grid(self, now: float):
        speed, ramp = 0.4, 1.2
        radius = 0.0 if not self.wc["running"] else (now - self.wc["t0"]) * speed
        grid = np.full((len(WC_ROWS), len(WC_COLS)), 6.0)
        for i, r in enumerate(WC_ROWS):
            for j, c in enumerate(WC_COLS):
                d = float(np.hypot(i - WC_ROWS.index(WC_EPI[0]), c - WC_EPI[1]))
                plateau = float(np.clip(96 - d * 24, 8, 96))
                if self.wc["running"]:
                    k = float(np.clip(((radius - d) / speed) / ramp, 0, 1))
                    grid[i, j] = 6 + (plateau - 6) * k
        return grid, radius
