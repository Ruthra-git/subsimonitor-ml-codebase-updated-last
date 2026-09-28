"""
data_loader.py
--------------
Loads and cleans the UCI "seismic-bumps" coal-mine hazard dataset
(Sikora & Wrobel, 2010 — https://doi.org/10.24432/C5W902).

We are using this dataset as the training proxy for SubsiMonitor's AI risk
engine because it is the closest openly-available real-world dataset to our
problem: shift-level seismic / seismoacoustic sensor summaries from a Polish
coal mine, labelled with whether a dangerous high-energy seismic bump
("rockburst") occurred in the following shift. That target (hazardous vs.
non-hazardous, next-period) is structurally the same prediction task as our
digital twin's tilt/vibration/temperature -> NORMAL/WARNING/CRITICAL
classifier, just measured with different sensors (geophone energy and pulse
counts instead of MEMS tilt/vibration/temperature).

Source file (data/train.arff) was pulled from a public mirror of the UCI set:
https://raw.githubusercontent.com/Thomas-K-John/Seismic-Bumps-Forecasting/master/train.arff
Original dataset: https://archive.ics.uci.edu/dataset/266/seismic+bumps

Note: that mirror's companion test.arff ships WITHOUT a class column (it was
a held-out competition file), so it has no ground truth to evaluate against.
We instead use the 1,809 labelled rows in train.arff and carve our own
stratified train/test split, which keeps the same ~6% positive class rate
in both halves.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.io import arff
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Attr1 in the source file is just a row index, not a feature -> dropped.
COLUMN_MAP = {
    "Attr2": "seismic",        # nominal a-d: shift hazard assessment (seismic method)
    "Attr3": "seismoacoustic",  # nominal a-c: shift hazard assessment (seismoacoustic method)
    "Attr4": "shift",          # nominal N/W: shift type (preparation / coal-getting)
    "Attr5": "genergy",        # numeric: seismic energy recorded by the most active geophone
    "Attr6": "gpuls",          # numeric: pulse count recorded by the most active geophone
    "Attr7": "gdenergy",       # numeric: % deviation of energy from the previous 8-shift average
    "Attr8": "gdpuls",         # numeric: % deviation of pulse count from the previous 8-shift average
    "Attr9": "ghazard",        # nominal a-c: hazard assessment from the most active geophone alone
    "Attr10": "nbumps",        # numeric: number of seismic bumps in the shift
    "Attr11": "nbumps2",       # numeric: number of bumps with 10^2-10^3 J energy
    "Attr12": "nbumps3",       # numeric: number of bumps with 10^3-10^4 J energy
    "Attr13": "nbumps4",       # numeric: number of bumps with 10^4-10^5 J energy
    "Attr14": "nbumps5",       # numeric: number of bumps with 10^5-10^6 J energy
    "Attr15": "nbumps6",       # numeric: number of bumps with 10^6-10^7 J energy
    "Attr16": "nbumps7",       # numeric: number of bumps with 10^7-10^8 J energy
    "Attr17": "nbumps89",      # numeric: number of bumps with >=10^8 J energy
    "Attr18": "energy",        # numeric: total energy of bumps recorded in the shift
    "Attr19": "maxenergy",     # numeric: maximum energy of a single bump in the shift
    "class": "label",          # target: 1 = hazardous state (bump >10^4 J next shift), 0 = safe
}

NOMINAL_COLS = ["seismic", "seismoacoustic", "shift", "ghazard"]
NUMERIC_COLS = [c for c in COLUMN_MAP.values() if c not in NOMINAL_COLS + ["label"]]


def _load_labelled() -> pd.DataFrame:
    data, _meta = arff.loadarff(DATA_DIR / "train.arff")
    df = pd.DataFrame(data)
    df = df.drop(columns=["Attr1"])  # row index, not a feature
    df = df.rename(columns=COLUMN_MAP)

    # scipy decodes ARFF nominal values as bytes (b'a') -> plain strings
    for col in NOMINAL_COLS:
        df[col] = df[col].apply(lambda v: v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v)

    df["label"] = df["label"].astype(int)
    return df


def load_raw():
    """Loads the labelled rows and splits them ourselves (stratified)."""
    df = _load_labelled()
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label"]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def clean(df: pd.DataFrame, medians: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Imputes the handful of missing numeric readings (ARFF '?' -> NaN) with the
    supplied medians, or computes fresh medians from this dataframe if none
    are given (fit on train, re-used on test to avoid leakage).
    """
    df = df.copy()
    if medians is None:
        medians = {c: df[c].median() for c in NUMERIC_COLS}
    for c in NUMERIC_COLS:
        df[c] = df[c].fillna(medians[c])
    return df, medians


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encodes the nominal hazard-assessment columns and returns a
    fully-numeric feature frame (label excluded).
    """
    X = pd.get_dummies(df.drop(columns=["label"]), columns=NOMINAL_COLS)
    return X


def load_dataset():
    """
    Convenience entry point used by train.py / infer scripts.
    Returns X_train, y_train, X_test, y_test, feature_names
    """
    train_df, test_df = load_raw()
    train_df, medians = clean(train_df)
    test_df, _ = clean(test_df, medians=medians)

    X_train = build_features(train_df)
    X_test = build_features(test_df)
    # align columns in case a rare category only appears in one split
    X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=0)

    y_train = train_df["label"].values
    y_test = test_df["label"].values
    return X_train, y_train, X_test, y_test, list(X_train.columns)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, feats = load_dataset()
    print(f"Train: {X_train.shape}, positives: {y_train.sum()} / {len(y_train)} "
          f"({100*y_train.mean():.1f}%)")
    print(f"Test:  {X_test.shape}, positives: {y_test.sum()} / {len(y_test)} "
          f"({100*y_test.mean():.1f}%)")
    print(f"Features ({len(feats)}): {feats}")
