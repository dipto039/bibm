#!/usr/bin/env python3
"""
V12.1_preprocess_script_LSTM.py
-------------------------------
Patient-level split version.

Fix:
- Uses SUBJECT_ID-level train/valid/test split.
- Train-only imputation/scaling preserved.
- Outputs X_seq, X_dyn, X_static, labels.
"""

import os, re, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ---------------- CONFIG ----------------
BASE_DIR   = "/lustre/home/rahas2/mimic_projects/outputs"
DATA_DIR   = "/lustre/home/rahas2/mimic_projects/data"
INPUT_CSV  = f"{BASE_DIR}/mimic_los_timewindows_wide_v12.1_imputed.csv"
OUTPUT_DIR = f"{BASE_DIR}/lstm_data_v12.1_seq"
ADMISSIONS_FILE = f"{DATA_DIR}/ADMISSIONS.csv"
DEFAULT_SEED = 42

REQUIRED_STATIC = {"HADM_ID", "AGE", "GENDER", "ADMISSION_TYPE"}
REQUIRED_LABELS = {"LOS_label", "LOS_label_3class"}
HOUR_SUFFIX_RE = re.compile(r"_(\d+)_(\d+)h$")

# ---------------- HELPERS ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=INPUT_CSV)
    ap.add_argument("--outdir", default=OUTPUT_DIR)
    ap.add_argument("--admissions", default=ADMISSIONS_FILE)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return ap.parse_args()

def extract_hour_start(col: str) -> int:
    m = HOUR_SUFFIX_RE.search(col)
    return int(m.group(1)) if m else -1

def is_dynamic_col(col: str) -> bool:
    return bool(HOUR_SUFFIX_RE.search(col))

def base_name(col: str) -> str:
    m = HOUR_SUFFIX_RE.search(col)
    return col[:m.start()] if m else col

def one_hot(series: pd.Series, vocab, prefix: str) -> pd.DataFrame:
    s = series.fillna("").astype(str).str.upper()
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
        print(f"⚠️  {name} contained NaN/Inf → replaced with 0.0.")
        arr[~np.isfinite(arr)] = 0.0
    else:
        print(f"✅ {name} verified — no NaN/Inf detected.")
    return arr

def blank_series(df: pd.DataFrame):
    return pd.Series("", index=df.index)

def ensure_subject_id(df: pd.DataFrame, admissions_path: str) -> pd.DataFrame:
    if "SUBJECT_ID" in df.columns:
        return df

    if "HADM_ID" not in df.columns:
        raise ValueError("Need HADM_ID to recover SUBJECT_ID.")

    if not os.path.exists(admissions_path):
        raise FileNotFoundError(
            f"SUBJECT_ID missing and ADMISSIONS.csv not found: {admissions_path}"
        )

    print("[preproc V12.1] SUBJECT_ID missing; merging from ADMISSIONS.csv...")
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

# ---------------- MAIN ----------------
def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[preproc V12.1] loading: {args.input}")
    df = pd.read_csv(args.input, low_memory=False)

    df = ensure_subject_id(df, args.admissions)

    required = REQUIRED_STATIC | REQUIRED_LABELS | {"SUBJECT_ID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    dyn_cols = [c for c in df.columns if is_dynamic_col(c)]
    if not dyn_cols:
        raise ValueError("No dynamic hour-based columns detected (*_<h>_<h+1>h).")

    groups = {}
    for c in dyn_cols:
        b = base_name(c)
        groups.setdefault(b, []).append(c)

    for b in groups:
        groups[b].sort(key=extract_hour_start)

    base_features = sorted(groups.keys())

    y_bin_all = pd.to_numeric(df["LOS_label"], errors="coerce").fillna(-1).astype(int)
    mask = y_bin_all != -1
    df = df.loc[mask].reset_index(drop=True)

    train_idx, valid_idx, test_idx = patient_level_split_indices(df, args.seed)

    print(
        f"[preproc V12.1] patient-level row split: "
        f"train={len(train_idx):,}, valid={len(valid_idx):,}, test={len(test_idx):,}"
    )
    print(
        f"[preproc V12.1] unique patients: "
        f"train={df.loc[train_idx, 'SUBJECT_ID'].nunique():,}, "
        f"valid={df.loc[valid_idx, 'SUBJECT_ID'].nunique():,}, "
        f"test={df.loc[test_idx, 'SUBJECT_ID'].nunique():,}"
    )

    df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce").clip(0, 120)

    dyn_train_medians = df.loc[train_idx, dyn_cols].median(skipna=True).fillna(0.0)
    df[dyn_cols] = df[dyn_cols].replace([np.inf, -np.inf], np.nan)
    for split in [train_idx, valid_idx, test_idx]:
        df.loc[split, dyn_cols] = df.loc[split, dyn_cols].fillna(dyn_train_medians)

    # ----- STATIC FEATURES -----
    df["GENDER"] = df["GENDER"].astype(str).str.upper().replace({"MALE": "M", "FEMALE": "F"})

    Xg = one_hot(df["GENDER"], ["M", "F"], prefix="GENDER")

    Xa = one_hot(
        df["ADMISSION_TYPE"].astype(str).str.upper(),
        ["EMERGENCY", "ELECTIVE", "URGENT", "NEWBORN"],
        prefix="ADM"
    )

    Xcu = one_hot(
        df["FIRST_CAREUNIT"] if "FIRST_CAREUNIT" in df.columns else blank_series(df),
        ["MICU", "SICU", "CCU", "CSRU", "TSICU"],
        prefix="CU"
    )

    Xloc = one_hot(
        df["ADMISSION_LOCATION"] if "ADMISSION_LOCATION" in df.columns else blank_series(df),
        ["EMERGENCY ROOM", "PHYS REFERRAL", "CLINIC REFERRAL", "TRANSFER"],
        prefix="ADMLOC"
    )

    Xins = one_hot(
        df["INSURANCE"] if "INSURANCE" in df.columns else blank_series(df),
        ["MEDICARE", "PRIVATE", "MEDICAID", "SELF PAY"],
        prefix="INS"
    )

    Xeth = one_hot(
        df["ETHNICITY"] if "ETHNICITY" in df.columns else blank_series(df),
        ["WHITE", "BLACK", "HISPANIC", "ASIAN"],
        prefix="ETH"
    )

    num_statics = []
    for c in ["ED_TO_ICU_HOURS", "PRIOR_ICU_COUNT_SUBJ"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            num_statics.append(c)

    derived_0_24 = [c for c in [
        "fluid_input_sum_0_24h", "total_output_sum_0_24h", "urine_output_sum_0_24h",
        "fluid_balance_sum_0_24h", "pressor_any_0_24h", "s_f_min_0_24h",
        "antibiotic_any_0_24h", "sedative_any_0_24h", "insulin_any_0_24h", "diuretic_any_0_24h"
    ] if c in df.columns]

    if derived_0_24:
        mu_d = df.loc[train_idx, derived_0_24].mean().fillna(0.0)
        std_d = df.loc[train_idx, derived_0_24].std().replace(0.0, 1.0).fillna(1.0)
        df[derived_0_24] = zscore(df[derived_0_24], mu_d, std_d).fillna(0.0)

    dx_cm_cols = [c for c in df.columns if c.startswith(("dx_", "cm_"))]
    if "charlson_simple_sum" in df.columns:
        dx_cm_cols.append("charlson_simple_sum")

    if dx_cm_cols:
        df[dx_cm_cols] = df[dx_cm_cols].fillna(0).astype(np.float32)

    scale_cols = dyn_cols + ["AGE"] + num_statics
    mu = df.loc[train_idx, scale_cols].mean().fillna(0.0)
    std = df.loc[train_idx, scale_cols].std().replace(0.0, 1.0).fillna(1.0)
    df[scale_cols] = zscore(df[scale_cols], mu, std).fillna(0.0)

    static_pieces = [df["AGE"].astype(np.float32), Xg, Xa, Xcu, Xloc, Xins, Xeth]

    if num_statics:
        static_pieces.append(df[num_statics].astype(np.float32))
    if derived_0_24:
        static_pieces.append(df[derived_0_24].astype(np.float32))
    if dx_cm_cols:
        static_pieces.append(df[dx_cm_cols].astype(np.float32))

    static_block = pd.concat(static_pieces, axis=1).astype(np.float32)
    static_cols = list(static_block.columns)
    F_static = static_block.shape[1]

    # ----- Build 3D dynamic sequences -----
    N = len(df)
    hours = list(range(24))
    F_dynamic_per_hour = len(base_features)

    X_seq = np.zeros((N, 24, F_dynamic_per_hour + F_static), dtype=np.float32)

    for h in hours:
        cols_h = []
        for b in base_features:
            col_h = f"{b}_{h}_{h+1}h"
            if col_h not in df.columns:
                alt = f"{b}{h}_{h+1}h"
                col_h = alt if alt in df.columns else None
            cols_h.append(df[col_h].values if col_h else np.zeros(N, np.float32))

        hour_block = np.vstack(cols_h).T.astype(np.float32)
        X_seq[:, h, :] = np.concatenate([hour_block, static_block.values], axis=1)

    # ----- LABELS -----
    y_bin_train = df.loc[train_idx, "LOS_label"].astype(int).values
    y_bin_valid = df.loc[valid_idx, "LOS_label"].astype(int).values
    y_bin_test  = df.loc[test_idx,  "LOS_label"].astype(int).values

    y_3c_train = df.loc[train_idx, "LOS_label_3class"].astype(int).values
    y_3c_valid = df.loc[valid_idx, "LOS_label_3class"].astype(int).values
    y_3c_test  = df.loc[test_idx,  "LOS_label_3class"].astype(int).values

    X_dyn = X_seq[:, :, :F_dynamic_per_hour]
    X_static = static_block.values.astype(np.float32)

    X_dyn_train = X_dyn[train_idx]
    X_dyn_valid = X_dyn[valid_idx]
    X_dyn_test  = X_dyn[test_idx]

    X_static_train = X_static[train_idx]
    X_static_valid = X_static[valid_idx]
    X_static_test  = X_static[test_idx]

    X_dyn_train = assert_and_clean(X_dyn_train, "X_dyn_train")
    X_seq_train = assert_and_clean(X_seq[train_idx], "X_seq_train")

    np.save(outdir / "X_seq_train.npy", X_seq_train)
    np.save(outdir / "X_seq_valid.npy", X_seq[valid_idx])
    np.save(outdir / "X_seq_test.npy",  X_seq[test_idx])

    np.save(outdir / "X_dyn_train.npy", X_dyn_train)
    np.save(outdir / "X_dyn_valid.npy", X_dyn_valid)
    np.save(outdir / "X_dyn_test.npy",  X_dyn_test)

    np.save(outdir / "X_static_train.npy", X_static_train)
    np.save(outdir / "X_static_valid.npy", X_static_valid)
    np.save(outdir / "X_static_test.npy",  X_static_test)

    np.save(outdir / "y_bin_train.npy", y_bin_train)
    np.save(outdir / "y_bin_valid.npy", y_bin_valid)
    np.save(outdir / "y_bin_test.npy",  y_bin_test)

    np.save(outdir / "y_3c_train.npy", y_3c_train)
    np.save(outdir / "y_3c_valid.npy", y_3c_valid)
    np.save(outdir / "y_3c_test.npy",  y_3c_test)

    meta = {
        "split_type": "patient_level_by_SUBJECT_ID",
        "seed": args.seed,
        "base_features_order": base_features,
        "static_columns": static_cols,
        "F_dynamic_per_hour": F_dynamic_per_hour,
        "F_static": F_static,
        "zscore_mean_cols": list(scale_cols),
        "derived_0_24_included": derived_0_24,
        "dx_cm_cols": dx_cm_cols,
        "n_train_rows": int(len(train_idx)),
        "n_valid_rows": int(len(valid_idx)),
        "n_test_rows": int(len(test_idx)),
        "n_train_subjects": int(df.loc[train_idx, "SUBJECT_ID"].nunique()),
        "n_valid_subjects": int(df.loc[valid_idx, "SUBJECT_ID"].nunique()),
        "n_test_subjects": int(df.loc[test_idx, "SUBJECT_ID"].nunique()),
    }

    with open(outdir / "preprocess_meta_v12_1.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("✓ Saved patient-level dynamic + static splits (V12.1).")
    print(f"Shapes: X_dyn_train {X_dyn_train.shape}, X_static_train {X_static_train.shape}")

if __name__ == "__main__":
    main()