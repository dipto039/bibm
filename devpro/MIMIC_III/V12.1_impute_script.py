#!/usr/bin/env python3
"""
V12.1_impute_script.py
-------------------
Leakage-safe imputation for the wide hourly MIMIC-III LOS dataset.

Fix:
- Patient-level train split using SUBJECT_ID.
- Train-only column means for remaining NaNs after LOCF.
"""

import os
import re
import logging
import argparse
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

from los_split_utils import ensure_subject_id, patient_level_split_indices

# ---------------- Logging ----------------
logger = logging.getLogger("v12.1_impute")
logger.setLevel(logging.INFO)
logger.handlers = []
_sh = logging.StreamHandler()
_sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_sh)

# ---------------- Config / CLI ----------------
# Same original_data/processed_data locations as clean_LOS and
# V12.1_build_script.py -- see that script's CONFIG section for why this
# walks up 3 levels (this script's own directory is 3 levels below the
# project root that clean_LOS sits directly under).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))

DEF_BASE = os.path.join(_PROJECT_ROOT, "processed_data", "structured_LOS_dynamic")
DEF_DATA = os.path.join(_PROJECT_ROOT, "original_data")
DEF_IN   = f"{DEF_BASE}/mimic_los_timewindows_wide_v12.1.csv"
DEF_OUT  = f"{DEF_BASE}/mimic_los_timewindows_wide_v12.1_imputed.csv"
DEF_ADM  = f"{DEF_DATA}/ADMISSIONS.csv"

STATIC_COLS_CANON = ["HADM_ID", "SUBJECT_ID", "AGE", "GENDER", "ADMISSION_TYPE"]
LABEL_COLS_CANON  = ["LOS_label", "LOS_label_3class", "LOS_DAYS"]

HOUR_SUFFIX_RE = re.compile(r"_(\d+)_(\d+)h$")

# ---------------- Helpers ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEF_IN)
    ap.add_argument("--output", default=DEF_OUT)
    ap.add_argument("--admissions", default=DEF_ADM)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--imputer", choices=["knn", "mean", "median"], default="knn",
        help="How to fill whatever's left after LOCF: KNN (default, borrows "
             "values from the most similar TRAIN admissions), or a plain "
             "train-only column mean/median fallback. median is more robust "
             "to the skewed distributions common in labs/vitals than mean.",
    )
    ap.add_argument(
        "--n_neighbors", type=int, default=5,
        help="K for KNNImputer (only used when --imputer=knn).",
    )
    return ap.parse_args()

def is_hourly_col(col: str) -> bool:
    return bool(HOUR_SUFFIX_RE.search(col))

def feature_base(col: str) -> str:
    m = HOUR_SUFFIX_RE.search(col)
    return col[:m.start()] if m else col

def hour_start(col: str) -> int:
    m = HOUR_SUFFIX_RE.search(col)
    return int(m.group(1)) if m else -1

# ensure_subject_id / patient_level_split_indices now live in
# los_split_utils.py (shared with V12.1_preprocess_script.py) so the two
# scripts can never silently disagree on which SUBJECT_IDs are "train".

# ---------------- Main ----------------
def main():
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    logger.info(f"Loading: {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded shape: rows={df.shape[0]:,} cols={df.shape[1]:,}")

    df = ensure_subject_id(df, args.admissions)

    static_cols = [c for c in STATIC_COLS_CANON if c in df.columns]
    label_cols  = [c for c in LABEL_COLS_CANON if c in df.columns]

    feature_cols = [
        c for c in df.columns
        if c not in static_cols + label_cols
        and is_hourly_col(c)
        and not c.endswith("_0_24h")
    ]

    if not feature_cols:
        logger.warning("No hourly feature columns detected. Nothing to impute.")
        df.to_csv(args.output, index=False)
        return

    groups = {}
    for c in feature_cols:
        groups.setdefault(feature_base(c), []).append(c)

    for base, cols in groups.items():
        cols.sort(key=hour_start)

    logger.info("Applying row-wise LOCF across hours...")
    for base, cols in groups.items():
        df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
        df[cols] = df[cols].ffill(axis=1)

    train_idx, valid_idx, test_idx = patient_level_split_indices(df, args.seed)
    logger.info(
        f"Patient-level split rows: train={len(train_idx):,}, "
        f"valid={len(valid_idx):,}, test={len(test_idx):,}"
    )

    binary_like = []
    for c in feature_cols:
        vals = df.loc[train_idx, c].dropna().unique()
        if len(vals) > 0:
            vals = set(pd.Series(vals).astype(float).round(10).tolist())
            if vals.issubset({0.0, 1.0}):
                binary_like.append(c)

    # Columns with zero observed values in TRAIN carry no information for
    # either a mean or a KNN fill -- pin these to 0.0 up front (same
    # behavior as before), then impute everything else.
    col_means = df.loc[train_idx, feature_cols].mean(axis=0, skipna=True)
    entirely_nan = col_means[col_means.isna()].index.tolist()
    if entirely_nan:
        logger.warning(f"{len(entirely_nan)} columns entirely NaN in train; filling with 0.0")
        df[entirely_nan] = df[entirely_nan].fillna(0.0)

    impute_cols = [c for c in feature_cols if c not in entirely_nan]

    if args.imputer == "knn" and impute_cols:
        logger.info(
            f"Filling remaining NaNs with a TRAIN-only-fitted KNNImputer "
            f"(n_neighbors={args.n_neighbors}) over {len(impute_cols)} columns..."
        )
        # KNN distance is scale-sensitive (HR ~60-150 would swamp SpO2
        # ~90-100 or lactate ~0-10 otherwise), so z-score with TRAIN-only
        # mean/std before imputing, then convert back to clinical units so
        # the imputed CSV's contract (raw units) is unchanged for everything
        # downstream. Fit once on train; transform() re-uses those same
        # fitted train neighbors for valid/test, so nothing leaks.
        train_mean = df.loc[train_idx, impute_cols].mean(axis=0, skipna=True)
        train_std = df.loc[train_idx, impute_cols].std(axis=0, skipna=True).replace(0.0, 1.0).fillna(1.0)

        knn = KNNImputer(n_neighbors=args.n_neighbors)
        knn.fit(((df.loc[train_idx, impute_cols] - train_mean) / train_std).values)

        for split_name, idx in (("train", train_idx), ("valid", valid_idx), ("test", test_idx)):
            scaled = ((df.loc[idx, impute_cols] - train_mean) / train_std).values
            imputed_scaled = knn.transform(scaled)
            df.loc[idx, impute_cols] = imputed_scaled * train_std.values + train_mean.values
            logger.info(f"  KNN-imputed {split_name}: {len(idx):,} rows")
    elif impute_cols:
        stat_name = "medians" if args.imputer == "median" else "means"
        logger.info(f"Filling remaining NaNs with TRAIN-ONLY column {stat_name}...")
        if args.imputer == "median":
            fill_values = df.loc[train_idx, impute_cols].median(axis=0, skipna=True)
        else:
            fill_values = df.loc[train_idx, impute_cols].mean(axis=0, skipna=True)
        df[impute_cols] = df[impute_cols].fillna(fill_values)

    if binary_like:
        df[binary_like] = (df[binary_like] > 0.5).astype(int)

    remaining = int(df[feature_cols].isna().sum().sum())
    if remaining:
        raise RuntimeError(f"Imputation failed: {remaining} NaNs remain.")
    logger.info(
        "Validation passed: no NaNs remain in per-hour feature columns "
        "(this check intentionally excludes *_0_24h aggregate columns, "
        "which the preprocess script imputes separately as static features)."
    )

    feature_cols_ordered = []
    for base in sorted(groups.keys()):
        feature_cols_ordered.extend(groups[base])

    other = [c for c in df.columns if c not in static_cols + feature_cols_ordered + label_cols]
    ordered_cols = static_cols + feature_cols_ordered + other + label_cols
    df = df[ordered_cols]

    df.to_csv(args.output, index=False)
    logger.info(f"Saved imputed dataset: {args.output} rows={len(df):,} cols={df.shape[1]:,}")

if __name__ == "__main__":
    main()