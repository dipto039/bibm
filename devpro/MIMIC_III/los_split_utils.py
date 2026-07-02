"""
los_split_utils.py
-------------------
Shared, leakage-safe patient-level split logic for the MIMIC-III LOS
pipeline. Both V12.1_impute_script.py and V12.1_preprocess_script.py must
agree on EXACTLY which SUBJECT_IDs land in train/valid/test -- if the
imputation statistics are computed over a different "train" than the one
preprocessing later calls train, the leakage these scripts are meant to
prevent comes back in through the side door. Previously this logic was
copy-pasted verbatim in both files; this module is the single source of
truth instead.
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split


def ensure_subject_id(df: pd.DataFrame, admissions_path: str) -> pd.DataFrame:
    """Recover SUBJECT_ID via ADMISSIONS.csv if the wide table doesn't carry it."""
    if "SUBJECT_ID" in df.columns:
        return df

    if "HADM_ID" not in df.columns:
        raise ValueError("Need HADM_ID to recover SUBJECT_ID.")

    if not os.path.exists(admissions_path):
        raise FileNotFoundError(
            f"SUBJECT_ID missing and ADMISSIONS.csv not found: {admissions_path}"
        )

    adm = pd.read_csv(admissions_path, usecols=["HADM_ID", "SUBJECT_ID"])
    adm = adm.drop_duplicates("HADM_ID")

    df = df.merge(adm, on="HADM_ID", how="left")

    if df["SUBJECT_ID"].isna().any():
        n = int(df["SUBJECT_ID"].isna().sum())
        raise ValueError(f"Failed to recover SUBJECT_ID for {n} rows.")

    return df


def patient_level_split_indices(df: pd.DataFrame, seed: int):
    """70/15/15 train/valid/test split, grouped by SUBJECT_ID so no patient's
    admissions cross a split boundary. Stratified on LOS_label_3class (not
    just the binary LOS_label) -- class 2 of the 3-class label is exactly
    LOS_label==1, so stratifying on the finer-grained 3-class label keeps
    both the binary and ordinal class balance consistent across splits.
    """
    y3 = pd.to_numeric(df["LOS_label_3class"], errors="coerce").fillna(-1).astype(int)
    valid_mask = y3 != -1

    work = df.loc[valid_mask, ["SUBJECT_ID"]].copy()
    work["LOS_label_3class"] = y3.loc[valid_mask].values

    subj_labels = (
        work.groupby("SUBJECT_ID")["LOS_label_3class"]
        .max()
        .reset_index()
    )

    train_subj, rest_subj = train_test_split(
        subj_labels["SUBJECT_ID"],
        test_size=0.30,
        stratify=subj_labels["LOS_label_3class"],
        random_state=seed,
    )

    rest_labels = subj_labels[subj_labels["SUBJECT_ID"].isin(rest_subj)]

    valid_subj, test_subj = train_test_split(
        rest_labels["SUBJECT_ID"],
        test_size=0.50,
        stratify=rest_labels["LOS_label_3class"],
        random_state=seed,
    )

    train_idx = df.index[df["SUBJECT_ID"].isin(set(train_subj))].to_numpy()
    valid_idx = df.index[df["SUBJECT_ID"].isin(set(valid_subj))].to_numpy()
    test_idx  = df.index[df["SUBJECT_ID"].isin(set(test_subj))].to_numpy()

    return train_idx, valid_idx, test_idx
