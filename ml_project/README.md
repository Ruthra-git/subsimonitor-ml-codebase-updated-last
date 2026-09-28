# SubsiMonitor — AI Risk Engine training codebase

Training pipeline for the AI risk engine in the SubsiMonitor digital twin
(SIH26025, Team HexaNovaTech), trained on a real, open, mine-hazard dataset
rather than synthetic numbers.

## Quick start — the Streamlit dashboard (live model connected)

```bash
pip install -r requirements.txt
python src/train.py && python src/export_edge_rules.py   # only needed if models/ is empty
streamlit run app.py
```

`app.py` is the front end for the whole codebase. On the **Live Twin** tab, `src/twin_engine.py`
simulates three virtual nodes (tilt / vibration / temperature) once per simulated second and, for every node,
calls the **trained RandomForest** (`models/risk_engine_rf.joblib`). Each node is then classified twice:

| layer | what it is | where it lives |
|---|---|---|
| Rules | hard thresholds (tilt > 6°, vib > 0.22 g, temp > 36 °C) | on-device, deterministic |
| AI | trained model, calibrated to a 0-100 risk (bands 33 / 66) | PC / cloud risk engine |
| Fused | worse of the two | per node |

The fused classes go through the **2-of-3 corroboration gate** (a lone bad node is shown as `DRIFT?` and suppressed),
then the **5-second CRITICAL dwell timer** (rule R6) before EVACUATION fires, with hysteresis on the way back down
(1.5 s grace before the dwell resets, 4 s below CRITICAL before stand-down). R7 is the distilled edge decision tree
(`models/edge_tree.joblib`, exported to `models/edge_rules.h` for the ESP32). Events are stored in SQLite (`data/events.db`).

Views (one radio selector, only the selected view renders per run so nothing goes blank): **Live Twin**, **Live Mine Twin** (self-animating mine cross-section in `web/subsimonitor_live_twin.html`, seeded from the engine's current risk), **Worst-Case Heatmap**, **ML Model** (metrics, plots, what-if probe, C code, retrain button),
**Real-Data Replay** (real held-out shifts through the model), **Hardware View** (the standalone HTML twin in `web/`).

### The sensor-to-shift adapter (read this before presenting)

The model was trained on geophone shift summaries, not tilt/vibration/temperature. Until SubsiMonitor has field
data, `ModelBridge` in `src/twin_engine.py` fuses a node's tilt/vibration/temperature into a severity (0-1) and draws a
**real, held-out mine shift** of that severity (safe shifts at low severity, increasingly hazardous ones at high
severity) for the model to score. Scoring is real and out-of-sample; the sensor-to-shift mapping is a documented proxy.
The 0-100 scale is calibrated so an idle node is ~0 and a fully critical node is ~100. Replace `ModelBridge.draw_row`
with a model trained on real SubsiMonitor sensor data when you have it.

## Dataset: UCI "seismic-bumps"

**Why this dataset:** there is no public sensor-level dataset for tilt /
vibration mine-subsidence monitoring specifically, but the UCI **seismic-bumps**
set is the closest real match available: shift-by-shift seismic /
seismoacoustic summaries from **two longwalls in a Polish coal mine**, each
labelled with whether a high-energy (>10⁴ J) seismic bump — a rockburst —
occurred in the *next* shift. That is structurally the same problem
SubsiMonitor solves (fuse recent sensor activity -> forecast a near-term
ground-hazard event), just measured with geophones instead of MEMS
tilt/vibration/temperature sensors.

- Source: Sikora M., Wrobel L. (2010), *UCI Machine Learning Repository*,
  https://doi.org/10.24432/C5W902 — CC BY 4.0
- Mirror used here (public GitHub, ARFF format):
  `https://raw.githubusercontent.com/Thomas-K-John/Seismic-Bumps-Forecasting/master/train.arff`
- 1,809 labelled shifts, 18 features (seismic/seismoacoustic hazard grades,
  geophone energy & pulse counts, bump counts by energy band) + 1 binary
  label. Only **6.2% of shifts are "hazardous"** — a realistic, heavily
  imbalanced safety-event rate, same character as our CRITICAL class.
- `data/test.arff` from the same mirror ships **without** a label column (it
  was a held-out competition file), so it isn't usable for evaluation. We
  instead take the 1,809 labelled rows and carve our own stratified 80/20
  split (`src/data_loader.py`) — same ~6% positive rate in both halves.

## What's in this repo

```
data/                    raw ARFF files (downloaded from the mirror above)
src/data_loader.py       ARFF parsing, cleaning, one-hot encoding, stratified split
src/train.py             trains + evaluates the cloud-side AI risk engine
src/export_edge_rules.py distills a shallow tree into C code for the ESP32
src/infer_live_twin.py   streams test rows through the model in the terminal
src/twin_engine.py       Python digital twin: sensors, rules, AI scoring, corroboration, escalation, SQLite log
src/viz.py               mine SVG, gauge, charts, tables for the dashboard
app.py                   Streamlit dashboard (streamlit run app.py)
web/                     standalone HTML twin (ESP32 pipeline, camera feed)
models/                  risk_engine_rf.joblib, edge_rules.h (generated)
reports/                 metrics.json, confusion_matrix.png, etc. (generated)
requirements.txt
```

## Model 1 — cloud/PC-side AI risk engine (`train.py`)

A `RandomForestClassifier` (400 trees, depth 8, `class_weight="balanced"`)
is trained on all 24 engineered features (numeric geophone readings +
one-hot encoded hazard-grade categoricals). This is the model that sits in
the **AI RISK ENGINE** box of the HTML pipeline diagram.

A RandomForest — not a deep net — was chosen deliberately: the dataset is
small (1,447 training rows) and tabular, exactly the regime where
tree ensembles reliably beat neural nets, and it matches the
Scikit-learn/TensorFlow stack already named on the feasibility slide.

**Held-out test results** (362 shifts, 22 truly hazardous):

| metric | value | reading |
|---|---|---|
| ROC-AUC | 0.798 | good rank-ordering of risky vs. safe shifts |
| PR-AUC (avg. precision) | 0.279 | far above the 6% no-skill baseline |
| Recall (hazardous) | 0.273 | catches ~1 in 4 true hazards at the default 0.5 cut |
| Precision (hazardous) | 0.286 | ~1 in 3 CRITICAL flags is a real hazard |

These numbers are **consistent with published results on this exact
dataset** — several papers report F1 in the 0.2–0.35 range for the minority
class, because 6%-positive, low-feature-signal seismic hazard forecasting
is a genuinely hard problem. We report it honestly rather than tuning the
threshold to look better: this is exactly why the digital twin never
relies on the AI score alone — it's paired with the deterministic
threshold layer (rules R1–R6) as a second, independent safety net, and
with the 2-of-3 node corroboration gate to suppress single-sensor noise.
Lowering the decision threshold trades some precision for materially
better recall, which is the right trade for a safety alert — see
`reports/pr_roc_curves.png` to pick an operating point.

Artifacts: `models/risk_engine_rf.joblib`, `reports/metrics.json`,
`reports/confusion_matrix.png`, `reports/pr_roc_curves.png`,
`reports/feature_importance.png`.

## Model 2 — on-device edge failsafe (`export_edge_rules.py`)

The 400-tree forest is too heavy for the ESP32, so a second, **depth-4
DecisionTreeClassifier** is fit on just 4 edge-measurable features
(`genergy`, `gpuls`, `gdenergy`, `gdpuls` — the geophone-world analogue of
our tilt/vibration amplitude and drift-from-baseline). It's exported as
plain nested if/else **C code** (`models/edge_rules.h`) that can be pasted
directly into ESP32 firmware with zero ML runtime — this is literally the
deterministic response matrix (rules R1–R6) shown in the dashboard, learned
from data instead of hand-set thresholds. Test accuracy: ~0.86 on the same
split (a much easier metric than the forest's recall/precision on the rare
class, because this tree is intentionally a coarse local trip-wire, not the
primary classifier).

## Model 3 — live-twin inference (`infer_live_twin.py`)

Streams the held-out test rows one at a time through the saved forest and
prints output in the **same NORMAL / WARNING / CRITICAL vocabulary and
gauge bands (0–33 / 33–66 / 66–100)** used in the SubsiMonitor HTML, and
applies the same 5-shift CRITICAL dwell timer before declaring an
EVACUATION line — this is the offline, real-data stand-in for the
MQTT → AI risk engine → dashboard leg of the pipeline.

## Usage

```bash
pip install -r requirements.txt

python src/train.py                 # trains + evaluates the risk engine
python src/export_edge_rules.py     # generates models/edge_rules.h
python src/infer_live_twin.py --speed 0.3 --limit 60   # dashboard-style replay
```

## Honest limitations (for the judging panel)

- This is a **proxy dataset**, not SubsiMonitor's own sensor data — no such
  public dataset exists yet. It validates the *modelling approach*
  (imbalanced binary hazard forecasting from recent sensor activity) on
  real mining data, not the exact tilt/vibration thresholds used in the demo.
- Recall on the rare hazardous class is modest (0.27) at the default
  threshold, which is why the system design never lets the AI model be the
  sole trigger — see the deterministic rule layer and 2-of-3 corroboration
  gate in the main dashboard.
- Once real SubsiMonitor field data exists, the same `data_loader.py` /
  `train.py` structure — swap the loader for one reading tilt/vibration/
  temperature CSVs — retrains the same two-model (cloud + edge) pipeline.


## Deploying for a demo / panel

- **Local (safest):** `pip install -r requirements.txt && streamlit run app.py`
- **Streamlit Community Cloud:** push this folder to GitHub, create a new app, set the main file to `app.py`.
  `models/` and `data/` must be committed. `data/events.db` is created at runtime and is not needed in the repo.
- **Backup with no Python:** open `web/subsimonitor_live_twin.html` (or `web/subsimonitor_dashboard.html`) directly in a browser.
  Both are self-contained and work offline.


## Early-warning intelligence (src/insight.py)

Shown on the Live Twin view under the node table:

- **Time to CRITICAL**: a linear trend fit on the recent mean AI risk gives an estimated time to the CRITICAL line with a 95% range and an R² fit quality. It says STABLE when there is no upward trend and NOW when already critical. Times are simulated seconds; it is a statistical trend estimate, not a guarantee.
- **Why is this alert firing**: names the driving node and feature, its share of the WARNING threshold, whether the 2-of-3 corroboration gate is satisfied, DRIFT? suppression, and the evacuation countdown.
- **Projection chart**: history plus the projected cone toward the CRITICAL line.
- **Shift report**: sidebar button generates a downloadable markdown report (status, peak risk per node, why, recent events).

## Deployment note

Pin `scikit-learn` in `requirements.txt` to the version that trained `models/*.joblib` (`pip show scikit-learn`),
or use the retrain button after deploying, so the models load without version warnings.
