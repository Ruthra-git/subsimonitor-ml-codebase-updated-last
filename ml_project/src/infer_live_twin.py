"""
infer_live_twin.py
-------------------
Streams the held-out test rows one at a time through the trained AI risk
engine, printing output in the same NORMAL / WARNING / CRITICAL vocabulary
and per-shift cadence as the SubsiMonitor dashboard — i.e. this script is
the offline stand-in for the MQTT -> AI risk engine -> dashboard leg of the
pipeline, driven by real historical mine data instead of the browser's
synthetic tilt/vibration jitter.

Each "shift" here (one row of the dataset) plays the same role as one tick
of node telemetry in the live twin: a probability is produced, mapped onto
the same 0-33 / 33-66 / 66-100 gauge bands used in the HTML risk gauge, and
a log line is printed exactly like the dashboard's event log.

Run:
    python src/infer_live_twin.py [--speed 0.4] [--limit 40]
"""

import argparse
import time
from pathlib import Path

import joblib

from data_loader import load_dataset

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "risk_engine_rf.joblib"


def band(prob_pct: float) -> str:
    # same thresholds as the gauge/rule logic in the digital twin HTML
    if prob_pct < 33:
        return "NORMAL"
    if prob_pct < 66:
        return "WARNING"
    return "CRITICAL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=0.4, help="seconds between simulated shifts")
    ap.add_argument("--limit", type=int, default=60, help="number of test rows to stream")
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit("No trained model found — run `python src/train.py` first.")

    bundle = joblib.load(MODEL_PATH)
    model, feature_names = bundle["model"], bundle["feature_names"]

    _, _, X_test, y_test, _ = load_dataset()
    X_test = X_test.reset_index(drop=True)

    n = min(args.limit, len(X_test))
    print(f"Streaming {n} held-out shifts through the AI risk engine "
          f"(ground-truth hazardous rate in this slice: "
          f"{100*y_test[:n].mean():.1f}%)\n")

    criticalSince = None
    for i in range(n):
        row = X_test.iloc[[i]]
        prob = model.predict_proba(row)[0, 1] * 100
        state = band(prob)
        truth = "ACTUAL HAZARD" if y_test[i] == 1 else "actual: safe"
        tag = "***" if (state == "CRITICAL" and y_test[i] == 1) else ("!!!" if state == "CRITICAL" and y_test[i] == 0 else "")

        if state == "CRITICAL":
            criticalSince = criticalSince or i
            dwell = i - criticalSince
            if dwell >= 5:
                print(f"  shift {i:03d} | risk {prob:5.1f}% | {state:8s} | {truth:14s} {tag}  "
                      f"-> EVACUATION (CRITICAL held {dwell} shifts)")
                criticalSince = i  # reset dwell after firing, same as the HTML twin
                time.sleep(args.speed)
                continue
        else:
            criticalSince = None

        print(f"  shift {i:03d} | risk {prob:5.1f}% | {state:8s} | {truth:14s} {tag}")
        time.sleep(args.speed)


if __name__ == "__main__":
    main()
