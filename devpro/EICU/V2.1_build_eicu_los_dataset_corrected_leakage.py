#!/usr/bin/env python3
"""
V2.1_build_eicu_los_dataset_corrected_leakage.py
----------------------------

Full eICU LOS wide-table builder for MIMIC-III-style generalizability.

Memory-safe V2 fix: feature blocks are merged immediately after each table is
processed instead of being accumulated in one massive feature_frames list.

This V2.1 corrected-leakage script expands the V1 builder to use all high-value eICU tables available
in /lustre/home/rahas2/mimic_projects/eicu_data.

It is designed to preserve the downstream contract already validated:

    build -> impute -> preprocess -> LSTM/ANN fusion training

Output compatibility is unchanged so downstream scripts run as-is:
    Main output:
        outputs/eicu_los_timewindows_wide_v2.csv

    Compatibility output:
        outputs/eicu_los_timewindows_wide_v1.csv

The compatibility output lets the existing imputation, preprocessing, and
training scripts run unchanged.

Core design:
- ICU stay key: patientunitstayid
- Compatibility key: HADM_ID = patientunitstayid
- First 24h window based on eICU offset columns
- Hourly columns named:
      <feature>_<agg>_<start>_<end>h
- 0-24h summary columns named:
      <feature>_<agg>_0_24h

Feature blocks:
- patient/static
- vitalPeriodic
- vitalAperiodic
- lab / ABG
- nurseCharting
- respiratoryCare
- infusionDrug
- medication
- admissionDrug
- intakeOutput / fluid balance
- treatment/procedure flags
- diagnosis/admissionDx
- microLab
- apacheApsVar
- apachePatientResult
- apachePredVar

Sampling:
    --sample-patients 5000   # sample run
    --sample-patients 0      # full run
"""

import os
import re
import time
import argparse
import logging
import gc
from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================
# CONFIG
# ======================================================

DEFAULT_DATA_DIR = "/lustre/home/rahas2/mimic_projects/eicu_data"
DEFAULT_OUTPUT_DIR = "/lustre/home/rahas2/mimic_projects/outputs"
WINDOW_MINUTES = 24 * 60
SEED = 42


# ======================================================
# LOGGING
# ======================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("eicu_v2_1_corrected_leakage_build")


# ======================================================
# BASIC HELPERS
# ======================================================

def sanitize(x: str) -> str:
    x = "" if pd.isna(x) else str(x)
    x = x.strip().lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x[:80] if x else "unknown"


def find_col(cols, candidates):
    lookup = {c.lower(): c for c in cols}
    for cand in candidates:
        c = lookup.get(cand.lower())
        if c:
            return c
    return None


def read_header(path):
    return pd.read_csv(path, compression="gzip", nrows=0, low_memory=False).columns.tolist()


def file_path(data_dir, name):
    return os.path.join(data_dir, name)


def file_exists(data_dir, name):
    return os.path.exists(file_path(data_dir, name))


def read_csv(path, **kwargs):
    return pd.read_csv(path, compression="gzip", low_memory=False, **kwargs)


def is_leakage_col(col) -> bool:
    """
    Exclude outcome/post-discharge fields from model inputs.

    unitdischargeoffset is allowed only inside build_patient_base()
    to create LOS_DAYS and labels. It must never be retained as an input.
    """
    c = sanitize(col)

    exact_bad = {
        "actualicumortality",
        "actualhospitalmortality",
        "unitdischargeoffset",
        "hospitaldischargeoffset",
    }

    if c in exact_bad:
        return True

    bad_substrings = [
        "actualicu",
        "actualhospital",
        "actual_mortality",
        "actualicumortality",
        "actualhospitalmortality",
        "unit_discharge",
        "hospital_discharge",
        "dischargeoffset",
        "discharge_offset",
        "activeupondischarge",
    ]

    return any(s in c for s in bad_substrings)


def to_hour_bin(s):
    x = pd.to_numeric(s, errors="coerce")
    return (x // 60).astype("Int64")


def restrict_24h(df, offset_col):
    df[offset_col] = pd.to_numeric(df[offset_col], errors="coerce")
    return df[(df[offset_col] >= 0) & (df[offset_col] < WINDOW_MINUTES)].copy()


def filter_ids(df, id_col, ids_set):
    if id_col not in df.columns:
        return df.iloc[0:0].copy()
    return df[df[id_col].isin(ids_set)].copy()


def merge_feature_frames(base, frames):
    for f in frames:
        if f is not None and not f.empty:
            base = base.merge(f, on="HADM_ID", how="left")
    return base


# ======================================================
# CATEGORY RULES
# ======================================================

def med_categories(text):
    x = "" if pd.isna(text) else str(text).lower()
    rules = {
        "pressor": [
            "norepinephrine", "levophed", "epinephrine", "phenylephrine",
            "neo-synephrine", "neosynephrine", "vasopressin", "dopamine",
            "dobutamine", "milrinone"
        ],
        "antibiotic": [
            "vanco", "vancomycin", "cef", "ceftriaxone", "cefepime",
            "zosyn", "piperacillin", "tazobactam", "meropenem",
            "imipenem", "azithro", "cipro", "levo", "metronidazole",
            "flagyl", "linezolid", "daptomycin", "gentamicin",
            "tobramycin", "ampicillin", "unasyn", "rocephin"
        ],
        "sedative": [
            "propofol", "midazolam", "versed", "dexmedetomidine",
            "precedex", "lorazepam", "ativan", "diazepam", "ketamine"
        ],
        "opioid": [
            "fentanyl", "morphine", "hydromorphone", "dilaudid",
            "oxycodone", "methadone"
        ],
        "insulin": ["insulin"],
        "diuretic": ["furosemide", "lasix", "bumetanide", "bumex", "torsemide"],
        "anticoagulant": ["heparin", "warfarin", "enoxaparin", "lovenox", "argatroban"],
        "steroid": ["hydrocortisone", "methylpred", "prednisone", "dexamethasone"],
        "paralytic": ["rocuronium", "vecuronium", "cisatracurium", "nimbex"],
        "fluid": ["normal saline", "lactated", "ringer", "dextrose", "albumin"],
    }
    out = []
    for cat, kws in rules.items():
        if any(k in x for k in kws):
            out.append(cat)
    return out


def treatment_categories(text):
    x = "" if pd.isna(text) else str(text).lower()
    rules = {
        "ventilation": ["ventilation", "ventilator", "mechanical ventilation"],
        "intubation": ["intubation", "endotracheal"],
        "oxygen": ["oxygen", "nasal cannula", "mask", "bipap", "cpap", "high flow"],
        "dialysis": ["dialysis", "hemodialysis", "crrt", "renal replacement"],
        "pressor_support": ["vasopressor", "pressor"],
        "central_line": ["central line", "central venous"],
        "arterial_line": ["arterial line"],
        "transfusion": ["transfusion", "packed red", "platelet", "plasma"],
        "surgery": ["surgery", "operative", "procedure"],
    }
    out = []
    for cat, kws in rules.items():
        if any(k in x for k in kws):
            out.append(cat)
    return out


def dx_categories(text):
    x = "" if pd.isna(text) else str(text).lower()
    rules = {
        "sepsis": ["sepsis", "septic"],
        "respiratory": ["respiratory", "pneumonia", "copd", "ards", "asthma"],
        "cardiac": ["cardiac", "heart", "myocardial", "infarct", "chf", "arrhythmia"],
        "renal": ["renal", "kidney", "aki", "neph"],
        "diabetes": ["diabetes", "diabetic"],
        "liver": ["liver", "hepatic", "cirrhosis"],
        "neuro": ["stroke", "seizure", "neuro", "intracranial", "brain"],
        "trauma": ["trauma", "fracture", "injury"],
        "cancer": ["cancer", "malign", "tumor", "neoplasm"],
    }
    out = []
    for cat, kws in rules.items():
        if any(k in x for k in kws):
            out.append(cat)
    return out


# ======================================================
# WIDE CONVERSION
# ======================================================

def long_to_hourly_wide(long_df, agg_name, agg_func):
    frames = []

    if long_df is None or long_df.empty:
        return frames

    long_df = long_df.dropna(subset=["HADM_ID", "HOUR_BIN", "FEATURE"])
    long_df = long_df[(long_df["HOUR_BIN"] >= 0) & (long_df["HOUR_BIN"] <= 23)]

    if long_df.empty:
        return frames

    if agg_func != "any":
        long_df["VALUE"] = pd.to_numeric(long_df["VALUE"], errors="coerce")
        long_df = long_df.dropna(subset=["VALUE"])
    else:
        long_df["VALUE"] = 1

    if long_df.empty:
        return frames

    if agg_func == "mean":
        g = long_df.groupby(["HADM_ID", "FEATURE", "HOUR_BIN"])["VALUE"].mean().reset_index()
    elif agg_func == "sum":
        g = long_df.groupby(["HADM_ID", "FEATURE", "HOUR_BIN"])["VALUE"].sum().reset_index()
    else:
        g = long_df.groupby(["HADM_ID", "FEATURE", "HOUR_BIN"])["VALUE"].max().reset_index()

    for h in range(24):
        sub = g[g["HOUR_BIN"] == h]
        if sub.empty:
            continue

        wide = sub.pivot_table(
            index="HADM_ID",
            columns="FEATURE",
            values="VALUE",
            aggfunc="max" if agg_func == "any" else agg_func,
            fill_value=0 if agg_func == "any" else None,
        ).reset_index()

        wide.columns = [
            "HADM_ID" if c == "HADM_ID" else f"{c}_{agg_name}_{h}_{h+1}h"
            for c in wide.columns
        ]
        frames.append(wide)

    total = g.groupby(["HADM_ID", "FEATURE"])["VALUE"]
    if agg_func == "mean":
        total = total.mean().reset_index()
    elif agg_func == "sum":
        total = total.sum().reset_index()
    else:
        total = total.max().reset_index()

    wide_total = total.pivot_table(
        index="HADM_ID",
        columns="FEATURE",
        values="VALUE",
        aggfunc="max" if agg_func == "any" else agg_func,
        fill_value=0 if agg_func == "any" else None,
    ).reset_index()

    wide_total.columns = [
        "HADM_ID" if c == "HADM_ID" else f"{c}_{agg_name}_0_24h"
        for c in wide_total.columns
    ]
    frames.append(wide_total)

    return frames


# ======================================================
# PATIENT BASE
# ======================================================

def build_patient_base(data_dir, sample_patients):
    path = file_path(data_dir, "patient.csv.gz")
    logger.info("Loading patient table...")

    patient = read_csv(path)

    patient["HADM_ID"] = patient["patientunitstayid"]

    patient["LOS_DAYS"] = pd.to_numeric(
        patient["unitdischargeoffset"], errors="coerce"
    ) / 1440.0

    patient = patient.dropna(subset=["LOS_DAYS"])
    patient = patient[patient["LOS_DAYS"] >= 0].copy()

    if sample_patients and sample_patients > 0:
        ids = (
            patient["patientunitstayid"]
            .drop_duplicates()
            .sample(min(sample_patients, patient["patientunitstayid"].nunique()), random_state=SEED)
        )
        patient = patient[patient["patientunitstayid"].isin(ids)].copy()
        logger.info(f"Sampling enabled: {len(ids):,} ICU stays")
    else:
        logger.info("Sampling disabled: full cohort")

    patient["LOS_label"] = (patient["LOS_DAYS"] > 7.0).astype("int8")

    patient["LOS_label_3class"] = np.select(
        [
            patient["LOS_DAYS"] <= 3.0,
            (patient["LOS_DAYS"] > 3.0) & (patient["LOS_DAYS"] <= 7.0),
            patient["LOS_DAYS"] > 7.0,
        ],
        [0, 1, 2],
    ).astype("int8")

    out = pd.DataFrame()
    out["HADM_ID"] = patient["HADM_ID"]
    out["AGE"] = pd.to_numeric(patient.get("age"), errors="coerce").clip(0, 120)
    out["GENDER"] = patient.get("gender", "UNKNOWN")
    out["ADMISSION_TYPE"] = patient.get("hospitaladmitsource", "UNKNOWN")
    out["ETHNICITY"] = patient.get("ethnicity", "UNKNOWN")
    out["FIRST_CAREUNIT"] = patient.get("unittype", "UNKNOWN")
    out["LOS_DAYS"] = patient["LOS_DAYS"]
    out["LOS_label"] = patient["LOS_label"]
    out["LOS_label_3class"] = patient["LOS_label_3class"]

    ids_set = set(out["HADM_ID"].astype(int).tolist())

    logger.info(f"Initial cohort shape: {out.shape}")

    return out, ids_set


# ======================================================
# STATIC APACHE FEATURES
# ======================================================

def add_static_numeric_table(base, data_dir, filename, prefix, ids_set):
    if not file_exists(data_dir, filename):
        logger.warning(f"Missing optional table: {filename}")
        return base

    logger.info(f"Processing static numeric table: {filename}")

    df = read_csv(file_path(data_dir, filename))
    if "patientunitstayid" not in df.columns:
        return base

    df = filter_ids(df, "patientunitstayid", ids_set)
    if df.empty:
        return base

    exclude = {
        "patientunitstayid",
        "apacheversion",
        "predictedhospitalmortality",
        "predictedicumortality",
        "actualicumortality",
        "actualhospitalmortality",
        "unitdischargeoffset",
        "hospitaldischargeoffset",
    }

    num_cols = []
    dropped_leakage = []

    for c in df.columns:
        if c in exclude or is_leakage_col(c):
            if c != "patientunitstayid":
                dropped_leakage.append(c)
            continue

        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().sum() > 0:
            df[c] = x
            num_cols.append(c)

    if dropped_leakage:
        logger.info(
            f"Dropped leakage/static outcome columns from {filename}: "
            f"{sorted(set(dropped_leakage))}"
        )

    if not num_cols:
        return base

    keep = ["patientunitstayid"] + num_cols
    tmp = df[keep].groupby("patientunitstayid").mean(numeric_only=True).reset_index()
    tmp.rename(columns={"patientunitstayid": "HADM_ID"}, inplace=True)
    tmp.rename(columns={c: f"{prefix}_{sanitize(c)}" for c in num_cols}, inplace=True)

    return base.merge(tmp, on="HADM_ID", how="left")


# ======================================================
# VITAL PERIODIC
# ======================================================

def process_vital_periodic(data_dir, ids_set, chunk_size):
    filename = "vitalPeriodic.csv.gz"
    if not file_exists(data_dir, filename):
        return []

    logger.info("Processing vitalPeriodic...")

    path = file_path(data_dir, filename)

    feature_map = {
        "heartrate": "heart_rate",
        "systemicsystolic": "systolic_bp",
        "systemicdiastolic": "diastolic_bp",
        "systemicmean": "map",
        "respiration": "resp_rate",
        "spo2": "spo2",
        "temperature": "temperature_c",
    }

    chunks = []
    for i, ch in enumerate(pd.read_csv(path, compression="gzip", chunksize=chunk_size, low_memory=False), start=1):
        if "patientunitstayid" not in ch.columns or "observationoffset" not in ch.columns:
            continue

        ch = filter_ids(ch, "patientunitstayid", ids_set)
        ch = restrict_24h(ch, "observationoffset")
        if ch.empty:
            continue

        ch["HADM_ID"] = ch["patientunitstayid"]
        ch["HOUR_BIN"] = to_hour_bin(ch["observationoffset"])

        existing = [c for c in feature_map if c in ch.columns]
        if not existing:
            continue

        tmp = ch[["HADM_ID", "HOUR_BIN"] + existing].copy()
        tmp = tmp.melt(id_vars=["HADM_ID", "HOUR_BIN"], var_name="FEATURE", value_name="VALUE")
        tmp["FEATURE"] = tmp["FEATURE"].map(feature_map)
        tmp["VALUE"] = pd.to_numeric(tmp["VALUE"], errors="coerce")
        tmp = tmp.dropna(subset=["VALUE"])
        chunks.append(tmp)

        if i % 25 == 0:
            logger.info(f"Processed vitalPeriodic chunks={i:,}")

    if not chunks:
        return []

    return long_to_hourly_wide(pd.concat(chunks, ignore_index=True), "mean", "mean")


# ======================================================
# GENERIC NUMERIC HOURLY TABLE
# ======================================================

def process_numeric_hourly_table(data_dir, filename, offset_candidates, ids_set, chunk_size, prefix):
    if not file_exists(data_dir, filename):
        logger.warning(f"Missing optional table: {filename}")
        return []

    logger.info(f"Processing numeric hourly table: {filename}")

    path = file_path(data_dir, filename)
    cols = read_header(path)
    offset_col = find_col(cols, offset_candidates)

    if not offset_col:
        logger.warning(f"No offset column found in {filename}")
        return []

    exclude = {
        "patientunitstayid", offset_col.lower(), offset_col,
        "hospitalid", "wardid", "uniquepid"
    }

    chunks = []

    for i, ch in enumerate(pd.read_csv(path, compression="gzip", chunksize=chunk_size, low_memory=False), start=1):
        if "patientunitstayid" not in ch.columns or offset_col not in ch.columns:
            continue

        ch = filter_ids(ch, "patientunitstayid", ids_set)
        ch = restrict_24h(ch, offset_col)
        if ch.empty:
            continue

        ch["HADM_ID"] = ch["patientunitstayid"]
        ch["HOUR_BIN"] = to_hour_bin(ch[offset_col])

        numeric_cols = []
        exclude_lc = {str(x).lower() for x in exclude} | {"hadm_id", "hour_bin"}

        dropped_leakage = []

        for c in ch.columns:
            if c.lower() in exclude_lc or is_leakage_col(c):
                if is_leakage_col(c):
                    dropped_leakage.append(c)
                continue

            val = pd.to_numeric(ch[c], errors="coerce")
            if val.notna().sum() > 0:
                ch[c] = val
                numeric_cols.append(c)

        if dropped_leakage:
            logger.info(
                f"Dropped leakage/outcome-like columns from {filename}: "
                f"{sorted(set(dropped_leakage))}"
            )

        if not numeric_cols:
            continue

        tmp = ch[["HADM_ID", "HOUR_BIN"] + numeric_cols].copy()
        tmp = tmp.melt(id_vars=["HADM_ID", "HOUR_BIN"], var_name="FEATURE", value_name="VALUE")
        tmp["FEATURE"] = tmp["FEATURE"].map(lambda x: f"{prefix}_{sanitize(x)}")
        tmp = tmp.dropna(subset=["VALUE"])
        chunks.append(tmp)

        if i % 25 == 0:
            logger.info(f"Processed {filename} chunks={i:,}")

    if not chunks:
        return []

    return long_to_hourly_wide(pd.concat(chunks, ignore_index=True), "mean", "mean")


# ======================================================
# LABS
# ======================================================

def process_lab(data_dir, ids_set, chunk_size):
    filename = "lab.csv.gz"
    if not file_exists(data_dir, filename):
        return []

    logger.info("Processing lab...")

    path = file_path(data_dir, filename)
    chunks = []

    for i, ch in enumerate(pd.read_csv(path, compression="gzip", chunksize=chunk_size, low_memory=False), start=1):
        needed = {"patientunitstayid", "labresultoffset", "labname", "labresult"}
        if not needed.issubset(set(ch.columns)):
            continue

        ch = filter_ids(ch, "patientunitstayid", ids_set)
        ch = restrict_24h(ch, "labresultoffset")
        if ch.empty:
            continue

        tmp = pd.DataFrame()
        tmp["HADM_ID"] = ch["patientunitstayid"]
        tmp["HOUR_BIN"] = to_hour_bin(ch["labresultoffset"])
        tmp["FEATURE"] = ch["labname"].map(sanitize)
        tmp["VALUE"] = pd.to_numeric(ch["labresult"], errors="coerce")
        tmp = tmp.dropna(subset=["VALUE"])
        chunks.append(tmp)

        if i % 10 == 0:
            logger.info(f"Processed lab chunks={i:,}")

    if not chunks:
        return []

    return long_to_hourly_wide(pd.concat(chunks, ignore_index=True), "mean", "mean")


# ======================================================
# NURSE CHARTING
# ======================================================

def process_nurse_charting(data_dir, ids_set, chunk_size):
    filename = "nurseCharting.csv.gz"
    if not file_exists(data_dir, filename):
        return []

    logger.info("Processing nurseCharting...")

    path = file_path(data_dir, filename)
    chunks = []

    for i, ch in enumerate(pd.read_csv(path, compression="gzip", chunksize=chunk_size, low_memory=False), start=1):
        offset_col = find_col(ch.columns, ["nursingchartoffset", "chartoffset"])
        value_col = find_col(ch.columns, ["nursingchartvalue", "chartvalue", "value"])
        label_col = find_col(ch.columns, [
            "nursingchartcelltypevallabel",
            "nursingchartcelltypevalname",
            "nursingchartcelltypecat",
            "nursingchartentryoffset"
        ])

        if not offset_col or not value_col or not label_col or "patientunitstayid" not in ch.columns:
            continue

        ch = filter_ids(ch, "patientunitstayid", ids_set)
        ch = restrict_24h(ch, offset_col)
        if ch.empty:
            continue

        val = pd.to_numeric(ch[value_col], errors="coerce")
        keep = val.notna()
        if keep.sum() == 0:
            continue

        tmp = pd.DataFrame()
        tmp["HADM_ID"] = ch.loc[keep, "patientunitstayid"]
        tmp["HOUR_BIN"] = to_hour_bin(ch.loc[keep, offset_col])
        tmp["FEATURE"] = ch.loc[keep, label_col].map(lambda x: f"nurse_{sanitize(x)}")
        tmp["VALUE"] = val.loc[keep]
        chunks.append(tmp)

        if i % 25 == 0:
            logger.info(f"Processed nurseCharting chunks={i:,}")

    if not chunks:
        return []

    return long_to_hourly_wide(pd.concat(chunks, ignore_index=True), "mean", "mean")


# ======================================================
# RESPIRATORY CARE
# ======================================================

def process_respiratory_care(data_dir, ids_set, chunk_size):
    filename = "respiratoryCare.csv.gz"
    if not file_exists(data_dir, filename):
        return []

    logger.info("Processing respiratoryCare...")

    mean_frames = process_numeric_hourly_table(
        data_dir=data_dir,
        filename=filename,
        offset_candidates=[
            "respcarestatusoffset",
            "respCareStatusOffset",
            "respchartoffset",
            "respchartentryoffset",
            "respchartvaluelabeloffset",
        ],
        ids_set=ids_set,
        chunk_size=chunk_size,
        prefix="resp",
    )

    return mean_frames


# ======================================================
# INFUSION DRUG
# ======================================================

def process_infusion_drug(data_dir, ids_set, chunk_size):
    filename = "infusionDrug.csv.gz"
    if not file_exists(data_dir, filename):
        return []

    logger.info("Processing infusionDrug...")

    path = file_path(data_dir, filename)
    rows_any = []
    rows_rate = []

    for i, ch in enumerate(pd.read_csv(path, compression="gzip", chunksize=chunk_size, low_memory=False), start=1):
        offset_col = find_col(ch.columns, ["infusionoffset", "drugstartoffset"])
        drug_col = find_col(ch.columns, ["drugname", "infusiondrugname"])
        rate_col = find_col(ch.columns, ["drugrate", "infusionrate", "drugamount"])

        if not offset_col or not drug_col or "patientunitstayid" not in ch.columns:
            continue

        ch = filter_ids(ch, "patientunitstayid", ids_set)
        ch = restrict_24h(ch, offset_col)
        if ch.empty:
            continue

        hb = to_hour_bin(ch[offset_col])

        rate_vals = None
        if rate_col:
            rate_vals = pd.to_numeric(ch[rate_col], errors="coerce")

        for idx, drug in ch[drug_col].astype(str).items():
            cats = med_categories(drug)
            if not cats:
                continue

            hadm = ch.loc[idx, "patientunitstayid"]
            h = hb.loc[idx]

            for cat in cats:
                rows_any.append((hadm, h, f"infusion_{cat}"))

                if rate_vals is not None and pd.notna(rate_vals.loc[idx]):
                    rows_rate.append((hadm, h, f"infusion_{cat}_rate", rate_vals.loc[idx]))

        if i % 25 == 0:
            logger.info(f"Processed infusionDrug chunks={i:,}")

    frames = []

    if rows_any:
        any_df = pd.DataFrame(rows_any, columns=["HADM_ID", "HOUR_BIN", "FEATURE"])
        frames.extend(long_to_hourly_wide(any_df, "any", "any"))
        del any_df

    if rows_rate:
        rate_df = pd.DataFrame(rows_rate, columns=["HADM_ID", "HOUR_BIN", "FEATURE", "VALUE"])
        frames.extend(long_to_hourly_wide(rate_df, "mean", "mean"))
        del rate_df

    return frames


# ======================================================
# MEDICATION / ADMISSION DRUG
# ======================================================

def process_medication_like(data_dir, filename, ids_set, chunk_size, prefix):
    if not file_exists(data_dir, filename):
        logger.warning(f"Missing optional table: {filename}")
        return []

    logger.info(f"Processing medication-like table: {filename}")

    path = file_path(data_dir, filename)
    header = read_header(path)

    drug_col = find_col(header, ["drugname", "drugname", "medicationname", "admissiondrugname"])
    offset_col = find_col(header, ["drugstartoffset", "drugoffset", "admissiondrugoffset"])

    if not drug_col:
        logger.warning(f"No drug column found in {filename}")
        return []

    any_chunks = []

    for i, ch in enumerate(pd.read_csv(path, compression="gzip", chunksize=chunk_size, low_memory=False), start=1):
        if "patientunitstayid" not in ch.columns:
            continue

        ch = filter_ids(ch, "patientunitstayid", ids_set)
        if ch.empty:
            continue

        if offset_col and offset_col in ch.columns:
            ch = restrict_24h(ch, offset_col)
            if ch.empty:
                continue
            hb = to_hour_bin(ch[offset_col])
        else:
            # Leakage fix: do not force untimed medication rows into hour 0,
            # except admissionDrug, which is admission-only by design.
            if filename != "admissionDrug.csv.gz":
                logger.warning(f"No offset column found in {filename}; skipping untimed rows to avoid leakage.")
                continue
            hb = pd.Series([0] * len(ch), index=ch.index, dtype="Int64")

        rows = []
        for hadm, h, drug in zip(ch["patientunitstayid"], hb, ch[drug_col]):
            cats = med_categories(drug)
            for cat in cats:
                rows.append((hadm, h, f"{prefix}_{cat}"))

        if rows:
            any_chunks.append(pd.DataFrame(rows, columns=["HADM_ID", "HOUR_BIN", "FEATURE"]))

        if i % 25 == 0:
            logger.info(f"Processed {filename} chunks={i:,}")

    if not any_chunks:
        return []

    return long_to_hourly_wide(pd.concat(any_chunks, ignore_index=True), "any", "any")


# ======================================================
# INTAKE OUTPUT
# ======================================================

def process_intake_output(data_dir, ids_set, chunk_size):
    filename = "intakeOutput.csv.gz"
    if not file_exists(data_dir, filename):
        return []

    logger.info("Processing intakeOutput...")

    path = file_path(data_dir, filename)
    chunks = []

    def io_feat(x):
        x = str(x).lower()

        if "urine" in x:
            return "urine_output"

        if any(k in x for k in ["output", "drain", "stool", "chest tube", "emesis", "gastric"]):
            return "total_output"

        if any(k in x for k in ["intake", "input", "iv", "fluid", "saline", "ringer", "albumin", "tube feed", "tpn"]):
            return "fluid_input"

        return None

    for i, ch in enumerate(pd.read_csv(path, compression="gzip", chunksize=chunk_size, low_memory=False), start=1):
        offset_col = find_col(ch.columns, ["intakeoutputoffset", "cellvaluenisticoffset"])
        value_col = find_col(ch.columns, ["cellvaluenumeric", "cellvalue", "intakeoutputvalue"])
        label_col = find_col(ch.columns, ["celllabel", "cellpath", "intakeoutputcelltype"])

        if not offset_col or not value_col or not label_col or "patientunitstayid" not in ch.columns:
            continue

        ch = filter_ids(ch, "patientunitstayid", ids_set)
        ch = restrict_24h(ch, offset_col)
        if ch.empty:
            continue

        val = pd.to_numeric(ch[value_col], errors="coerce")
        keep = val.notna()
        if keep.sum() == 0:
            continue

        tmp = pd.DataFrame()
        tmp["HADM_ID"] = ch.loc[keep, "patientunitstayid"]
        tmp["HOUR_BIN"] = to_hour_bin(ch.loc[keep, offset_col])
        tmp["FEATURE"] = ch.loc[keep, label_col].map(io_feat)
        tmp["VALUE"] = val.loc[keep]

        tmp = tmp[tmp["FEATURE"].notna()]

        if not tmp.empty:
            chunks.append(tmp)

        if i % 25 == 0:
            logger.info(f"Processed intakeOutput chunks={i:,}")

    if not chunks:
        return []

    long = pd.concat(chunks, ignore_index=True)
    frames = long_to_hourly_wide(long, "sum", "sum")

    bal = long[long["FEATURE"].isin(["fluid_input", "total_output", "urine_output"])].copy()

    if not bal.empty:
        piv = (
            bal.groupby(["HADM_ID", "HOUR_BIN", "FEATURE"])["VALUE"]
            .sum()
            .reset_index()
            .pivot_table(index=["HADM_ID", "HOUR_BIN"], columns="FEATURE", values="VALUE", fill_value=0)
            .reset_index()
        )

        if "fluid_input" not in piv.columns:
            piv["fluid_input"] = 0
        if "total_output" not in piv.columns:
            piv["total_output"] = 0

        piv["FEATURE"] = "fluid_balance"
        piv["VALUE"] = piv["fluid_input"] - piv["total_output"]

        frames.extend(
            long_to_hourly_wide(
                piv[["HADM_ID", "HOUR_BIN", "FEATURE", "VALUE"]],
                "sum",
                "sum",
            )
        )

    return frames
# ======================================================
# TREATMENT
# ======================================================

def process_treatment(data_dir, ids_set, chunk_size):
    filename = "treatment.csv.gz"
    if not file_exists(data_dir, filename):
        return []

    logger.info("Processing treatment...")

    path = file_path(data_dir, filename)
    header = read_header(path)
    offset_col = find_col(header, ["treatmentoffset"])
    text_col = find_col(header, ["treatmentstring"])

    if not text_col:
        return []

    any_chunks = []

    for i, ch in enumerate(pd.read_csv(path, compression="gzip", chunksize=chunk_size, low_memory=False), start=1):
        if "patientunitstayid" not in ch.columns:
            continue

        ch = filter_ids(ch, "patientunitstayid", ids_set)
        if ch.empty:
            continue

        if offset_col and offset_col in ch.columns:
            ch = restrict_24h(ch, offset_col)
            if ch.empty:
                continue
            hb = to_hour_bin(ch[offset_col])
        else:
            logger.warning("No treatment offset found; skipping untimed treatment rows to avoid leakage.")
            continue

        rows = []
        for hadm, h, txt in zip(ch["patientunitstayid"], hb, ch[text_col]):
            for cat in treatment_categories(txt):
                rows.append((hadm, h, f"treatment_{cat}"))

        if rows:
            any_chunks.append(pd.DataFrame(rows, columns=["HADM_ID", "HOUR_BIN", "FEATURE"]))

        if i % 25 == 0:
            logger.info(f"Processed treatment chunks={i:,}")

    if not any_chunks:
        return []

    return long_to_hourly_wide(pd.concat(any_chunks, ignore_index=True), "any", "any")


# ======================================================
# DIAGNOSIS
# ======================================================

def process_diagnosis_like(data_dir, filename, ids_set, prefix):
    if not file_exists(data_dir, filename):
        logger.warning(f"Missing optional table: {filename}")
        return None

    logger.info(f"Processing diagnosis-like table: {filename}")

    df = read_csv(file_path(data_dir, filename))
    if "patientunitstayid" not in df.columns:
        return None

    df = filter_ids(df, "patientunitstayid", ids_set)
    if df.empty:
        return None

    text_col = find_col(df.columns, ["diagnosisstring", "admitdxpath", "admitdxname", "diagnosis"])
    if not text_col:
        return None

    rows = []
    for hadm, txt in zip(df["patientunitstayid"], df[text_col]):
        for cat in dx_categories(txt):
            rows.append((hadm, f"{prefix}_{cat}", 1))

    if not rows:
        return None

    tmp = pd.DataFrame(rows, columns=["HADM_ID", "FEATURE", "VALUE"])
    wide = tmp.pivot_table(index="HADM_ID", columns="FEATURE", values="VALUE", aggfunc="max", fill_value=0).reset_index()
    return wide


# ======================================================
# MICRO LAB
# ======================================================

def process_micro_lab(data_dir, ids_set, chunk_size):
    filename = "microLab.csv.gz"
    if not file_exists(data_dir, filename):
        return []

    logger.info("Processing microLab...")

    path = file_path(data_dir, filename)
    header = read_header(path)
    offset_col = find_col(header, ["culturetakenoffset", "cultureoffset"])
    site_col = find_col(header, ["culturesite", "culturetaken", "specimen"])
    org_col = find_col(header, ["organism", "organismname"])

    if not site_col and not org_col:
        return []

    chunks = []

    for i, ch in enumerate(pd.read_csv(path, compression="gzip", chunksize=chunk_size, low_memory=False), start=1):
        if "patientunitstayid" not in ch.columns:
            continue

        ch = filter_ids(ch, "patientunitstayid", ids_set)
        if ch.empty:
            continue

        if offset_col and offset_col in ch.columns:
            ch = restrict_24h(ch, offset_col)
            if ch.empty:
                continue
            hb = to_hour_bin(ch[offset_col])
        else:
            logger.warning("No microLab offset found; skipping untimed microbiology rows to avoid leakage.")
            continue

        rows = []
        for idx, hadm in ch["patientunitstayid"].items():
            h = hb.loc[idx]
            if site_col:
                rows.append((hadm, h, "micro_site_" + sanitize(ch.loc[idx, site_col])))
            if org_col and pd.notna(ch.loc[idx, org_col]):
                rows.append((hadm, h, "micro_org_" + sanitize(ch.loc[idx, org_col])))

        if rows:
            chunks.append(pd.DataFrame(rows, columns=["HADM_ID", "HOUR_BIN", "FEATURE"]))

        if i % 25 == 0:
            logger.info(f"Processed microLab chunks={i:,}")

    if not chunks:
        return []

    return long_to_hourly_wide(pd.concat(chunks, ignore_index=True), "any", "any")


# ======================================================
# MAIN
# ======================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--sample-patients", type=int, default=5000, help="5000 sample by default. Use 0 for full dataset.")
    ap.add_argument("--chunk-size", type=int, default=500_000)
    args = ap.parse_args()

    t0 = time.time()

    data_dir = args.data_dir
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_v2 = os.path.join(output_dir, "eicu_los_timewindows_wide_v2.csv")
    output_compat = os.path.join(output_dir, "eicu_los_timewindows_wide_v1.csv")

    sample_patients = None if args.sample_patients == 0 else args.sample_patients

    out, ids_set = build_patient_base(data_dir, sample_patients)

    # Static severity: merge directly into output
    out = add_static_numeric_table(out, data_dir, "apacheApsVar.csv.gz", "apache_aps", ids_set)
    # Leakage fix: apachePatientResult/apachePredVar contain outcome/prediction-target-adjacent fields.
    # Keep apacheApsVar only.
    gc.collect()

    def merge_block(block_name, frames):
        nonlocal out
        logger.info(f"Merging block: {block_name}")
        if frames:
            logger.info(f"{block_name} frames: {len(frames):,}")
            for j, f in enumerate(frames, start=1):
                if f is not None and not f.empty:
                    out = out.merge(f, on="HADM_ID", how="left")
                    del f
                    gc.collect()
                    logger.info(f"{block_name}: merged frame {j}/{len(frames)} | shape={out.shape}")
            logger.info(f"Shape after {block_name}: {out.shape}")
        else:
            logger.info(f"No frames for block: {block_name}")
        del frames
        gc.collect()

    # Dynamic numeric tables
    merge_block("vitalPeriodic", process_vital_periodic(data_dir, ids_set, args.chunk_size))

    merge_block(
        "vitalAperiodic",
        process_numeric_hourly_table(
            data_dir, "vitalAperiodic.csv.gz",
            ["observationoffset"],
            ids_set, args.chunk_size, "aperiodic"
        )
    )

    merge_block("lab", process_lab(data_dir, ids_set, args.chunk_size))
    merge_block("nurseCharting", process_nurse_charting(data_dir, ids_set, args.chunk_size))
    merge_block("respiratoryCare", process_respiratory_care(data_dir, ids_set, args.chunk_size))

    # Meds/pressors
    merge_block("infusionDrug", process_infusion_drug(data_dir, ids_set, args.chunk_size))
    merge_block("medication", process_medication_like(data_dir, "medication.csv.gz", ids_set, args.chunk_size, "med"))
    merge_block("admissionDrug", process_medication_like(data_dir, "admissionDrug.csv.gz", ids_set, args.chunk_size, "admitdrug"))

    # IO/fluid
    merge_block("intakeOutput", process_intake_output(data_dir, ids_set, args.chunk_size))

    # Treatment/procedure
    merge_block("treatment", process_treatment(data_dir, ids_set, args.chunk_size))

    # Microbiology
    merge_block("microLab", process_micro_lab(data_dir, ids_set, args.chunk_size))

    # Static diagnoses
    # Leakage fix: skip diagnosis.csv because rows may be added after the first 24h.
    # Keep admissionDx only because it is admission-time information.
    dx_frames = []
    dx2 = process_diagnosis_like(data_dir, "admissionDx.csv.gz", ids_set, "admitdx")
    if dx2 is not None:
        dx_frames.append(dx2)
    merge_block("admissionDx", dx_frames)

    out = out.replace([np.inf, -np.inf], np.nan)

    logger.info(f"Final shape: {out.shape}")

    logger.info(f"Saving V2: {output_v2}")
    out.to_csv(output_v2, index=False)

    logger.info(f"Saving compatibility copy: {output_compat}")
    out.to_csv(output_compat, index=False)

    logger.info(f"Done. Elapsed sec: {round(time.time() - t0, 1)}")


if __name__ == "__main__":
    main()