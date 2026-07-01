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
from sklearn.model_selection import train_test_split

# ---------------- Logging ----------------
logger = logging.getLogger("v12.1_impute")
logger.setLevel(logging.INFO)
logger.handlers = []
_sh = logging.StreamHandler()
_sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_sh)

# ---------------- Config / CLI ----------------
DEF_BASE = "/lustre/home/rahas2/mimic_projects/outputs"
DEF_DATA = "/lustre/home/rahas2/mimic_projects/data"
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
    return ap.parse_args()

def is_hourly_col(col: str) -> bool:
    return bool(HOUR_SUFFIX_RE.search(col))

def feature_base(col: str) -> str:
    m = HOUR_SUFFIX_RE.search(col)
    return col[:m.start()] if m else col

def hour_start(col: str) -> int:
    m = HOUR_SUFFIX_RE.search(col)
    return int(m.group(1)) if m else -1

def ensure_subject_id(df: pd.DataFrame, admissions_path: str) -> pd.DataFrame:
    if "SUBJECT_ID" in df.columns:
        return df

    if "HADM_ID" not in df.columns:
        raise ValueError("Need HADM_ID to recover SUBJECT_ID.")

    if not os.path.exists(admissions_path):
        raise FileNotFoundError(
            f"SUBJECT_ID missing and ADMISSIONS.csv not found: {admissions_path}"
        )

    logger.info("SUBJECT_ID missing; merging from ADMISSIONS.csv...")
    adm = pd.read_csv(admissions_path, usecols=["HADM_ID", "SUBJECT_ID"])
    adm = adm.drop_duplicates("HADM_ID")

    df = df.merge(adm, on="HADM_ID", how="left")

    if df["SUBJECT_ID"].isna().any():
        n = int(df["SUBJECT_ID"].isna().sum())
        raise ValueError(f"Failed to recover SUBJECT_ID for {n} rows.")

    return df

def patient_level_split_indices(df: pd.DataFrame, seed: int):
    y = pd.to_numeric(df["LOS_label"], errors="coerce").fillna(-1).astype(int)
    valid_mask = y != -1

    work = df.loc[valid_mask, ["SUBJECT_ID"]].copy()
    work["LOS_label"] = y.loc[valid_mask].values

    subj_labels = (
        work.groupby("SUBJECT_ID")["LOS_label"]
        .max()
        .reset_index()
    )

    train_subj, rest_subj = train_test_split(
        subj_labels["SUBJECT_ID"],
        test_size=0.30,
        stratify=subj_labels["LOS_label"],
        random_state=seed,
    )

    rest_labels = subj_labels[subj_labels["SUBJECT_ID"].isin(rest_subj)]

    valid_subj, test_subj = train_test_split(
        rest_labels["SUBJECT_ID"],
        test_size=0.50,
        stratify=rest_labels["LOS_label"],
        random_state=seed,
    )

    train_idx = df.index[df["SUBJECT_ID"].isin(set(train_subj))].to_numpy()
    valid_idx = df.index[df["SUBJECT_ID"].isin(set(valid_subj))].to_numpy()
    test_idx  = df.index[df["SUBJECT_ID"].isin(set(test_subj))].to_numpy()

    return train_idx, valid_idx, test_idx

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

    logger.info("Filling remaining NaNs with TRAIN-ONLY column means...")
    col_means = df.loc[train_idx, feature_cols].mean(axis=0, skipna=True)

    entirely_nan = col_means[col_means.isna()].index.tolist()
    if entirely_nan:
        logger.warning(f"{len(entirely_nan)} columns entirely NaN in train; filling with 0.0")
        col_means.loc[entirely_nan] = 0.0

    df[feature_cols] = df[feature_cols].fillna(col_means)

    if binary_like:
        df[binary_like] = (df[binary_like] > 0.5).astype(int)

    remaining = int(df[feature_cols].isna().sum().sum())
    if remaining:
        raise RuntimeError(f"Imputation failed: {remaining} NaNs remain.")
    logger.info("Validation passed: no NaNs remain in hourly features.")

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