#!/usr/bin/env python3
"""
V1_preprocess_eicu_LSTM.py
--------------------------

Preprocesses the imputed eICU LOS wide dataset for LSTM / hybrid models.

Input:
    /lustre/home/rahas2/mimic_projects/outputs/eicu_los_timewindows_wide_v1_imputed.csv

Outputs:
    X_seq_train.npy / valid / test
    X_dyn_train.npy / valid / test
    X_static_train.npy / valid / test
    y_bin_train.npy / valid / test
    y_3c_train.npy / valid / test
    preprocess_meta_eicu_v1.json

This mirrors the MIMIC-III V12.1 preprocessing architecture:
- dynamic hourly features -> X_dyn, shape (N, 24, F_dynamic)
- static features -> X_static, shape (N, F_static)
- combined repeated static+dynamic sequence -> X_seq
- patient-level train/valid/test split with stratification
- train-only normalization to avoid leakage
"""

import os
import re
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = "/lustre/home/rahas2/mimic_projects/outputs"
INPUT_CSV = f"{BASE_DIR}/eicu_los_timewindows_wide_v1_imputed.csv"
OUTPUT_DIR = f"{BASE_DIR}/eicu_lstm_data_v1"
PATIENT_CSV = "/lustre/home/rahas2/mimic_projects/eicu_data/patient.csv.gz"

DEFAULT_SEED = 42

REQUIRED_STATIC = {"HADM_ID", "AGE", "GENDER", "ADMISSION_TYPE"}
REQUIRED_LABELS = {"LOS_label", "LOS_label_3class"}

HOUR_SUFFIX_RE = re.compile(r"_(\d+)_(\d+)h$")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=INPUT_CSV)
    ap.add_argument("--outdir", default=OUTPUT_DIR)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return ap.parse_args()


def is_dynamic_col(col: str) -> bool:
    return bool(HOUR_SUFFIX_RE.search(col))


def extract_hour_start(col: str) -> int:
    m = HOUR_SUFFIX_RE.search(col)
    return int(m.group(1)) if m else -1


def base_name(col: str) -> str:
    m = HOUR_SUFFIX_RE.search(col)
    return col[:m.start()] if m else col


def one_hot(series: pd.Series, vocab, prefix: str) -> pd.DataFrame:
    s = series.fillna("UNKNOWN").astype(str).str.upper()
    s = s.where(s.isin(vocab), "OTHER")

    d = pd.get_dummies(s, prefix=prefix)

    col_order = [f"{prefix}_{v}" for v in vocab + ["OTHER"]]

    for c in col_order:
        if c not in d.columns:
            d[c] = 0

    return d[col_order].astype(np.float32)


def zscore(df: pd.DataFrame, mean: pd.Series, std: pd.Series, eps=1e-8) -> pd.DataFrame:
    return (df - mean) / (std.replace(0.0, eps) + eps)


def assert_and_clean(arr: np.ndarray, name: str):
    if not np.isfinite(arr).all():
        print(f"WARNING: {name} contained NaN/Inf; replacing with 0.0")
        arr[~np.isfinite(arr)] = 0.0
    else:
        print(f"{name} verified: no NaN/Inf")
    return arr


def ensure_uniquepid(df):
    if "uniquepid" in df.columns:
        return df

    if not os.path.exists(PATIENT_CSV):
        raise FileNotFoundError(f"uniquepid missing and patient file not found: {PATIENT_CSV}")

    print("[eICU preprocess] uniquepid missing; merging from patient.csv.gz")
    patient = pd.read_csv(
        PATIENT_CSV,
        compression="gzip",
        usecols=["patientunitstayid", "uniquepid"],
        low_memory=False,
    ).drop_duplicates("patientunitstayid")

    patient = patient.rename(columns={"patientunitstayid": "HADM_ID"})
    df = df.merge(patient, on="HADM_ID", how="left")

    if df["uniquepid"].isna().any():
        n = int(df["uniquepid"].isna().sum())
        raise ValueError(f"Could not recover uniquepid for {n} rows.")

    return df


def patient_level_split_indices(df, y_bin_all, seed):
    work = df[["uniquepid"]].copy()
    work["LOS_label"] = y_bin_all.values

    patient_labels = (
        work.groupby("uniquepid")["LOS_label"]
        .max()
        .reset_index()
    )

    train_pid, rest_pid = train_test_split(
        patient_labels["uniquepid"],
        test_size=0.30,
        stratify=patient_labels["LOS_label"],
        random_state=seed,
    )

    rest_labels = patient_labels[patient_labels["uniquepid"].isin(rest_pid)]

    valid_pid, test_pid = train_test_split(
        rest_labels["uniquepid"],
        test_size=0.50,
        stratify=rest_labels["LOS_label"],
        random_state=seed,
    )

    train_idx = df.index[df["uniquepid"].isin(set(train_pid))].to_numpy()
    valid_idx = df.index[df["uniquepid"].isin(set(valid_pid))].to_numpy()
    test_idx = df.index[df["uniquepid"].isin(set(test_pid))].to_numpy()

    return train_idx, valid_idx, test_idx


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[eICU preprocess] loading: {args.input}")
    df = pd.read_csv(args.input, low_memory=False)

    print(f"Loaded shape: {df.shape}")

    df = ensure_uniquepid(df)

    required = REQUIRED_STATIC | REQUIRED_LABELS | {"uniquepid"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --------------------------------------------------
    # Detect dynamic hourly columns
    # --------------------------------------------------

    dyn_cols = [c for c in df.columns if is_dynamic_col(c)]

    if not dyn_cols:
        raise ValueError("No dynamic hourly columns detected.")

    groups = {}

    for c in dyn_cols:
        b = base_name(c)
        groups.setdefault(b, []).append(c)

    for b in groups:
        groups[b].sort(key=extract_hour_start)

    base_features = sorted(groups.keys())

    print(f"Dynamic columns: {len(dyn_cols):,}")
    print(f"Dynamic base features: {len(base_features):,}")

    # --------------------------------------------------
    # Drop invalid labels
    # --------------------------------------------------

    y_bin_all = pd.to_numeric(df["LOS_label"], errors="coerce").fillna(-1).astype(int)
    valid_mask = y_bin_all.isin([0, 1])

    df = df.loc[valid_mask].reset_index(drop=True)
    y_bin_all = y_bin_all.loc[valid_mask].reset_index(drop=True)

    y_3c_all = pd.to_numeric(df["LOS_label_3class"], errors="coerce").fillna(-1).astype(int)

    valid_3c = y_3c_all.isin([0, 1, 2])

    df = df.loc[valid_3c].reset_index(drop=True)
    y_bin_all = y_bin_all.loc[valid_3c].reset_index(drop=True)
    y_3c_all = y_3c_all.loc[valid_3c].reset_index(drop=True)

    if len(df) < 10:
        raise ValueError("Too few valid rows after label filtering.")

    print("Binary label counts:")
    print(y_bin_all.value_counts().sort_index())

    print("3-class label counts:")
    print(y_3c_all.value_counts().sort_index())

    # --------------------------------------------------
    # Train / valid / test split
    # --------------------------------------------------

    train_idx, valid_idx, test_idx = patient_level_split_indices(
        df=df,
        y_bin_all=y_bin_all,
        seed=args.seed,
    )

    print(
        f"Patient-level row split: train={len(train_idx):,}, "
        f"valid={len(valid_idx):,}, test={len(test_idx):,}"
    )

    print(
        f"Unique patients: train={df.loc[train_idx, 'uniquepid'].nunique():,}, "
        f"valid={df.loc[valid_idx, 'uniquepid'].nunique():,}, "
        f"test={df.loc[test_idx, 'uniquepid'].nunique():,}"
    )

    # --------------------------------------------------
    # Numeric cleanup
    # --------------------------------------------------

    df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce").clip(0, 120)

    df[dyn_cols] = df[dyn_cols].replace([np.inf, -np.inf], np.nan)
    df[dyn_cols] = df[dyn_cols].apply(pd.to_numeric, errors="coerce")

    dyn_train_medians = df.loc[train_idx, dyn_cols].median(skipna=True).fillna(0.0)

    for split_idx in [train_idx, valid_idx, test_idx]:
        df.loc[split_idx, dyn_cols] = df.loc[split_idx, dyn_cols].fillna(dyn_train_medians)

    # --------------------------------------------------
    # Static categorical features
    # --------------------------------------------------

    df["GENDER"] = (
        df["GENDER"]
        .astype(str)
        .str.upper()
        .replace({"MALE": "M", "FEMALE": "F"})
    )

    Xg = one_hot(
        df["GENDER"],
        ["M", "F"],
        prefix="GENDER",
    )

    Xa = one_hot(
        df["ADMISSION_TYPE"],
        ["EMERGENCY", "URGENT", "ELECTIVE", "OTHER HOSPITAL", "TRANSFER"],
        prefix="ADM",
    )

    Xcu = one_hot(
        df.get("FIRST_CAREUNIT", pd.Series(["UNKNOWN"] * len(df))),
        [
            "MICU",
            "SICU",
            "CCU",
            "CSRU",
            "CTICU",
            "CCU-CTICU",
            "MED-SURG ICU",
            "NEURO ICU",
        ],
        prefix="CU",
    )

    Xeth = one_hot(
        df.get("ETHNICITY", pd.Series(["UNKNOWN"] * len(df))),
        [
            "CAUCASIAN",
            "AFRICAN AMERICAN",
            "HISPANIC",
            "ASIAN",
            "NATIVE AMERICAN",
        ],
        prefix="ETH",
    )

    # --------------------------------------------------
    # Static numeric features
    # --------------------------------------------------

    num_static_candidates = [
        "apacheaps",
        "apachescore",
    ]

    num_statics = []

    for c in num_static_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            num_statics.append(c)

    if num_statics:
        mu_ns = df.loc[train_idx, num_statics].mean().fillna(0.0)
        std_ns = df.loc[train_idx, num_statics].std().replace(0.0, 1.0).fillna(1.0)
        df[num_statics] = zscore(df[num_statics], mu_ns, std_ns).fillna(0.0)

    # --------------------------------------------------
    # Diagnosis flags
    # --------------------------------------------------

    dx_cols = [c for c in df.columns if c.startswith("admitdx_")]

    if dx_cols:
        df[dx_cols] = df[dx_cols].fillna(0).astype(np.float32)

    # --------------------------------------------------
    # Scale dynamic features and AGE using train only
    # --------------------------------------------------

    scale_cols = dyn_cols + ["AGE"]

    mu = df.loc[train_idx, scale_cols].mean().fillna(0.0)
    std = df.loc[train_idx, scale_cols].std().replace(0.0, 1.0).fillna(1.0)

    df[scale_cols] = zscore(df[scale_cols], mu, std).fillna(0.0)

    # --------------------------------------------------
    # Build static block
    # --------------------------------------------------

    static_pieces = [
        df["AGE"].astype(np.float32),
        Xg,
        Xa,
        Xcu,
        Xeth,
    ]

    if num_statics:
        static_pieces.append(df[num_statics].astype(np.float32))

    if dx_cols:
        static_pieces.append(df[dx_cols].astype(np.float32))

    static_block = pd.concat(static_pieces, axis=1).astype(np.float32)

    static_cols = list(static_block.columns)
    F_static = static_block.shape[1]

    print(f"Static features: {F_static}")

    # --------------------------------------------------
    # Build X_seq and X_dyn
    # --------------------------------------------------

    N = len(df)
    hours = list(range(24))
    F_dynamic = len(base_features)

    X_dyn = np.zeros((N, 24, F_dynamic), dtype=np.float32)

    for h in hours:
        cols_h = []

        for b in base_features:
            col_h = f"{b}_{h}_{h+1}h"

            if col_h in df.columns:
                vals = df[col_h].values
            else:
                vals = np.zeros(N, dtype=np.float32)

            cols_h.append(vals)

        X_dyn[:, h, :] = np.vstack(cols_h).T.astype(np.float32)

    X_static = static_block.values.astype(np.float32)

    X_seq = np.zeros((N, 24, F_dynamic + F_static), dtype=np.float32)

    for h in hours:
        X_seq[:, h, :] = np.concatenate(
            [X_dyn[:, h, :], X_static],
            axis=1,
        )

    # --------------------------------------------------
    # Labels
    # --------------------------------------------------

    y_bin = df["LOS_label"].astype(int).values
    y_3c = df["LOS_label_3class"].astype(int).values

    X_dyn_train = X_dyn[train_idx]
    X_dyn_valid = X_dyn[valid_idx]
    X_dyn_test = X_dyn[test_idx]

    X_static_train = X_static[train_idx]
    X_static_valid = X_static[valid_idx]
    X_static_test = X_static[test_idx]

    X_seq_train = X_seq[train_idx]
    X_seq_valid = X_seq[valid_idx]
    X_seq_test = X_seq[test_idx]

    y_bin_train = y_bin[train_idx]
    y_bin_valid = y_bin[valid_idx]
    y_bin_test = y_bin[test_idx]

    y_3c_train = y_3c[train_idx]
    y_3c_valid = y_3c[valid_idx]
    y_3c_test = y_3c[test_idx]

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    X_dyn_train = assert_and_clean(X_dyn_train, "X_dyn_train")
    X_dyn_valid = assert_and_clean(X_dyn_valid, "X_dyn_valid")
    X_dyn_test = assert_and_clean(X_dyn_test, "X_dyn_test")

    X_static_train = assert_and_clean(X_static_train, "X_static_train")
    X_static_valid = assert_and_clean(X_static_valid, "X_static_valid")
    X_static_test = assert_and_clean(X_static_test, "X_static_test")

    X_seq_train = assert_and_clean(X_seq_train, "X_seq_train")
    X_seq_valid = assert_and_clean(X_seq_valid, "X_seq_valid")
    X_seq_test = assert_and_clean(X_seq_test, "X_seq_test")

    # --------------------------------------------------
    # Save arrays
    # --------------------------------------------------

    np.save(outdir / "X_seq_train.npy", X_seq_train)
    np.save(outdir / "X_seq_valid.npy", X_seq_valid)
    np.save(outdir / "X_seq_test.npy", X_seq_test)

    np.save(outdir / "X_dyn_train.npy", X_dyn_train)
    np.save(outdir / "X_dyn_valid.npy", X_dyn_valid)
    np.save(outdir / "X_dyn_test.npy", X_dyn_test)

    np.save(outdir / "X_static_train.npy", X_static_train)
    np.save(outdir / "X_static_valid.npy", X_static_valid)
    np.save(outdir / "X_static_test.npy", X_static_test)

    np.save(outdir / "y_bin_train.npy", y_bin_train)
    np.save(outdir / "y_bin_valid.npy", y_bin_valid)
    np.save(outdir / "y_bin_test.npy", y_bin_test)

    np.save(outdir / "y_3c_train.npy", y_3c_train)
    np.save(outdir / "y_3c_valid.npy", y_3c_valid)
    np.save(outdir / "y_3c_test.npy", y_3c_test)

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------

    meta = {
        "dataset": "eICU",
        "input_csv": args.input,
        "split_type": "patient_level_by_uniquepid",
        "base_features_order": base_features,
        "static_columns": static_cols,
        "dynamic_columns": dyn_cols,
        "F_dynamic_per_hour": int(F_dynamic),
        "F_static": int(F_static),
        "N": int(N),
        "train_size": int(len(train_idx)),
        "valid_size": int(len(valid_idx)),
        "test_size": int(len(test_idx)),
        "train_patients": int(df.loc[train_idx, "uniquepid"].nunique()),
        "valid_patients": int(df.loc[valid_idx, "uniquepid"].nunique()),
        "test_patients": int(df.loc[test_idx, "uniquepid"].nunique()),
        "zscore_mean_cols": list(scale_cols),
        "num_static_cols": num_statics,
        "dx_cols": dx_cols,
    }

    with open(outdir / "preprocess_meta_eicu_v1.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved eICU LSTM preprocessing outputs.")
    print(f"Output dir: {outdir}")
    print(f"X_dyn_train: {X_dyn_train.shape}")
    print(f"X_static_train: {X_static_train.shape}")
    print(f"X_seq_train: {X_seq_train.shape}")


if __name__ == "__main__":
    main()