"""
train.py
--------
Trains the SubsiMonitor "AI risk engine" classifier on the seismic-bumps
coal-mine hazard dataset and evaluates it the way this problem actually
needs to be evaluated: the positive class (a dangerous bump next shift) is
only ~6% of the data, so plain accuracy is meaningless (predicting "safe"
every time already scores ~94%). We optimise and report on recall /
precision / F1 / ROC-AUC / PR-AUC for the hazardous class instead, and use
class_weight="balanced" so the model isn't just told to ignore the rare
class.

Outputs (written to ../models and ../reports):
  models/risk_engine_rf.joblib   - full-precision RandomForest (cloud / PC side)
  reports/metrics.json           - held-out test metrics
  reports/confusion_matrix.png
  reports/feature_importance.png
  reports/pr_roc_curves.png

Run:
    python src/train.py
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, f1_score,
)

from data_loader import load_dataset

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


def train_model(X_train, y_train):
    """
    A RandomForest is used (rather than a deep net) because: the dataset is
    small (1,447 training rows), tabular, and mostly count/energy features —
    exactly the regime where gradient-boosted / random forests reliably beat
    neural nets. class_weight='balanced' re-weights the loss so the rare
    hazardous class isn't drowned out by the 94% safe class.
    """
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, feature_names):
    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    report = classification_report(y_test, preds, target_names=["safe", "hazardous"], output_dict=True)
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    f1 = f1_score(y_test, preds)

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1_hazardous": f1,
        "precision_hazardous": report["hazardous"]["precision"],
        "recall_hazardous": report["hazardous"]["recall"],
        "support_hazardous": report["hazardous"]["support"],
        "n_test": len(y_test),
    }
    print("\n=== Held-out test performance (hazardous / CRITICAL class) ===")
    for k, v in metrics.items():
        print(f"  {k:22s}: {v:.4f}" if isinstance(v, float) else f"  {k:22s}: {v}")

    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # confusion matrix
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Oranges")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["safe", "hazardous"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["safe", "hazardous"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix — held-out test set")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=140)
    plt.close(fig)

    # ROC + PR curves
    fpr, tpr, _ = roc_curve(y_test, proba)
    prec, rec, _ = precision_recall_curve(y_test, proba)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(fpr, tpr, color="#c97f4a"); axes[0].plot([0, 1], [0, 1], "--", color="#5d636d")
    axes[0].set_title(f"ROC curve (AUC={roc_auc:.3f})"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[1].plot(rec, prec, color="#e0543a")
    axes[1].set_title(f"Precision-Recall (AP={pr_auc:.3f})"); axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "pr_roc_curves.png", dpi=140)
    plt.close(fig)

    # feature importance
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:12]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([feature_names[i] for i in order][::-1], importances[order][::-1], color="#4fa88a")
    ax.set_title("Top 12 feature importances — AI risk engine")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "feature_importance.png", dpi=140)
    plt.close(fig)

    return metrics


def main():
    X_train, y_train, X_test, y_test, feature_names = load_dataset()
    print(f"Training rows: {len(X_train)}  (positives: {y_train.sum()}, {100*y_train.mean():.1f}%)")
    print(f"Test rows:     {len(X_test)}  (positives: {y_test.sum()}, {100*y_test.mean():.1f}%)")

    model = train_model(X_train, y_train)
    evaluate(model, X_test, y_test, feature_names)

    joblib.dump({"model": model, "feature_names": feature_names}, MODELS_DIR / "risk_engine_rf.joblib")
    print(f"\nSaved model -> {MODELS_DIR / 'risk_engine_rf.joblib'}")
    print(f"Saved reports -> {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
