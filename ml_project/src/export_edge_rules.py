"""
export_edge_rules.py
---------------------
The full RandomForest (400 trees) trained in train.py is a "cloud / PC side"
model — it belongs in the MQTT / risk-engine box of the pipeline, not on the
ESP32 itself. For the edge node we instead fit a *shallow* single decision
tree (depth-limited, few features) and export it as plain C if/else code,
so it can be pasted straight into ESP32 firmware and evaluated in
microseconds with no ML runtime on-device at all.

This mirrors exactly the two-layer safety design in the digital twin HTML:
  - a rich AI risk engine (this repo's RandomForest) running in the cloud
  - a small deterministic rule layer running on the ESP32 itself, so a
    local hazard is still caught even if LoRa / the cloud link is down.

We deliberately restrict the tree to the 4 features that are the closest
analogue to what the ESP32 can actually measure on-device (energy / pulse
counts are the geophone equivalent of the tilt+vibration amplitude our
MEMS sensors report), so the exported logic stays genuinely embeddable.

Run:
    python src/export_edge_rules.py
Outputs:
    models/edge_rules.h   - C header, ready to #include in ESP32 firmware
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree

from data_loader import load_dataset

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# The 4 edge-measurable proxies: total recorded energy, pulse count, and the
# two "deviation from recent average" signals (the geophone-world equivalent
# of our tilt/vibration drift-from-baseline check).
EDGE_FEATURES = ["genergy", "gpuls", "gdenergy", "gdpuls"]


def tree_to_c(tree, feature_names, func_name="edge_risk_classify"):
    """Walks an sklearn DecisionTreeClassifier and emits an equivalent
    nested if/else C function returning 0 (NORMAL) or 1 (CRITICAL)."""
    t = tree.tree_
    lines = [
        "// Auto-generated from a depth-limited DecisionTreeClassifier — see export_edge_rules.py",
        "// Returns 1 if the edge node should raise a local CRITICAL flag, 0 otherwise.",
        f"int {func_name}(float genergy, float gpuls, float gdenergy, float gdpuls) {{",
    ]

    def recurse(node, depth):
        indent = "    " * (depth + 1)
        if t.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[t.feature[node]]
            threshold = t.threshold[node]
            lines.append(f"{indent}if ({name} <= {threshold:.4f}f) {{")
            recurse(t.children_left[node], depth + 1)
            lines.append(f"{indent}}} else {{")
            recurse(t.children_right[node], depth + 1)
            lines.append(f"{indent}}}")
        else:
            # majority class at this leaf
            counts = t.value[node][0]
            pred = int(np.argmax(counts))
            lines.append(f"{indent}return {pred};  // leaf: {counts.tolist()}")

    recurse(0, 0)
    lines.append("}")
    return "\n".join(lines)


def main():
    X_train, y_train, X_test, y_test, feature_names = load_dataset()
    idx = [feature_names.index(f) for f in EDGE_FEATURES]
    Xtr = X_train.values[:, idx]
    Xte = X_test.values[:, idx]

    edge_tree = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=8, class_weight="balanced", random_state=42
    )
    edge_tree.fit(Xtr, y_train)

    train_acc = edge_tree.score(Xtr, y_train)
    test_acc = edge_tree.score(Xte, y_test)
    print(f"Edge tree (depth<=4, 4 features) — train acc {train_acc:.3f}, test acc {test_acc:.3f}")
    print("This is intentionally a small, interpretable failsafe layer, not the primary")
    print("classifier — see reports/metrics.json for the full RandomForest's numbers.")

    joblib.dump({"tree": edge_tree, "features": EDGE_FEATURES}, MODELS_DIR / "edge_tree.joblib")
    c_code = tree_to_c(edge_tree, EDGE_FEATURES)
    header = (
        "#ifndef EDGE_RULES_H\n#define EDGE_RULES_H\n\n"
        "/* SubsiMonitor — on-device deterministic failsafe classifier\n"
        " * Distilled from the cloud RandomForest via a depth-4 decision tree\n"
        " * so it can run directly on the ESP32 with no ML runtime.\n"
        f" * train acc={train_acc:.3f}  test acc={test_acc:.3f}\n"
        " */\n\n"
        f"{c_code}\n\n#endif // EDGE_RULES_H\n"
    )
    out_path = MODELS_DIR / "edge_rules.h"
    out_path.write_text(header, encoding="utf-8")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
