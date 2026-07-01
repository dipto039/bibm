"""
V12.1_build_script.py
------------------
Extends V8 with additional **time-variant** features (first 24h) for ICU LOS prediction (MIMIC-III),
keeping V8 logic intact and only adding what’s necessary.

WHAT’S NEW (on top of V8)
- Fluid inputs (INPUTEVENTS_MV + INPUTEVENTS_CV): per-hour sums (+ optional mean RATE); 0–24h totals
- Extended pressors: add phenylephrine and include CV events; per-hour binary; 0–24h any flag
- Fluid balance: per-hour (inputs − outputs_total); 0–24h total
- Procedure flags (PROCEDUREEVENTS_MV): per-hour binary for intubation, extubation, central line
- 0–24h totals for outputs (urine, total) derived from existing hourly sums

All naming, chunking, guards, and merge style follow V8.

Column naming convention for new features (examples):
- fluid_input_sum_7_8h, fluid_input_rate_mean_7_8h
- phenylephrine_any_7_8h
- fluid_balance_sum_7_8h
- intubation_any_7_8h, extubation_any_7_8h, central_line_any_7_8h
- *_0_24h (derived totals/flags)

NOTE: No imputation here; a separate preprocessing step handles missingness.
"""

import os
import time
import logging
import re
from collections import defaultdict

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

# ====================== CONFIG ======================
SAMPLE_ADMISSIONS = None      # e.g., 2000 for a quick sample; None for FULL
MAX_CHUNKS = None           # e.g., 30 to stop early while sampling; None for FULL
CHUNK_SIZE = 500_000
WINDOW_HOURS = list(range(0, 25))   # edges for hourly bins 0–24
CONVERT_TEMPS_TO_C = True

DATA_DIR = "/lustre/home/rahas2/mimic_projects/data"
OUTPUT_DIR = "/lustre/home/rahas2/mimic_projects/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = f"{OUTPUT_DIR}/mimic_los_timewindows_wide_v12.1.csv"
LOG_FILE   = f"{OUTPUT_DIR}/build_mimic_los_dataset_v12.1.log"

# --- Core files (same as V8) ---
ICUSTAYS_FILE       = f"{DATA_DIR}/ICUSTAYS.csv"
ADMISSIONS_FILE     = f"{DATA_DIR}/ADMISSIONS.csv"
PATIENTS_FILE       = f"{DATA_DIR}/PATIENTS.csv"
CHARTEVENTS_FILE    = f"{DATA_DIR}/CHARTEVENTS.csv.gz"
LABEVENTS_FILE      = f"{DATA_DIR}/LABEVENTS.csv"
D_LABITEMS_FILE     = f"{DATA_DIR}/D_LABITEMS.csv"
D_ITEMS_FILE        = f"{DATA_DIR}/D_ITEMS.csv"

# --- V9 additional event files ---
OUTPUTEVENTS_FILE        = f"{DATA_DIR}/OUTPUTEVENTS.csv.gz"
INPUTEVENTS_MV_FILE      = f"{DATA_DIR}/INPUTEVENTS_MV.csv.gz"
INPUTEVENTS_CV_FILE      = f"{DATA_DIR}/INPUTEVENTS_CV.csv.gz"
PROCEDUREEVENTS_MV_FILE  = f"{DATA_DIR}/PROCEDUREEVENTS_MV.csv.gz"

# ================== V7/V8 Vital ITEMIDs (kept) ===================
vital_itemids = {
    "Heart Rate":   [211, 220045],
    "Systolic BP":  [51, 220179],
    "Diastolic BP": [8368, 220180],
    "Mean BP":      [52, 220181],
    "Resp Rate":    [618, 220210],
    "Temperature":  [223761, 678],  # 223761=Fahrenheit, 678=Celsius
    "SpO2":         [646, 220277],
}
ITEMID_TO_LABEL = {iid: label for label, ids in vital_itemids.items() for iid in ids}
ALL_CHART_ITEMIDS = set(ITEMID_TO_LABEL.keys())
TEMP_F_ITEMID = 223761

# ===================== LOGGING ======================
logger = logging.getLogger("los_build_v9")
logger.setLevel(logging.INFO)
logger.handlers = []
_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
_sh = logging.StreamHandler();  _sh.setFormatter(_fmt)
_fh = logging.FileHandler(LOG_FILE); _fh.setFormatter(_fmt)
logger.addHandler(_sh); logger.addHandler(_fh)

# ===================== HELPERS ======================

def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None
    
def window_pairs(ws):
    return list(zip(ws[:-1], ws[1:]))

def wname(a, b):
    return f"{int(a)}_{int(b)}h"

def to_celsius_if_needed(itemid_series, val_series):
    if not CONVERT_TEMPS_TO_C:
        return val_series
    mask = (itemid_series.values == TEMP_F_ITEMID)
    out = val_series.copy()
    if mask.any():
        out.loc[mask] = (out.loc[mask] - 32.0) * (5.0 / 9.0)
    return out

def sanitize_label(s: str) -> str:
    """Make a safe column label fragment (letters, numbers, underscores)."""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed"

def hour_bin_from_delta_hours(hours: pd.Series) -> pd.Series:
    """Convert hours since INTIME to integer hour bins clipped to [0, 23]."""
    hb = hours.astype(np.float64).astype(int)
    return hb.clip(0, 23)

def safe_to_datetime(df, cols):
    for c in cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def _read_csv_maybe_gz(path, **kwargs):
    """Robust reader: tries as-is, then tries with compression='gzip' if needed."""
    try:
        return pd.read_csv(path, **kwargs)
    except ValueError:
        return pd.read_csv(path, compression="gzip", **kwargs)
    except FileNotFoundError:
        # also try opposite extension (csv <-> csv.gz)
        alt = path + (".gz" if not path.endswith(".gz") else "").replace(".csv.gz.gz", ".csv.gz")
        if os.path.exists(alt):
            try:
                return pd.read_csv(alt, **kwargs)
            except ValueError:
                return pd.read_csv(alt, compression="gzip", **kwargs)
        raise

# ============== LOAD DICTIONARIES (labels & ITEMIDs) =========
def load_lab_label_map():
    """Load D_LABITEMS to map LABEVENTS.ITEMID -> human-readable label."""
    if not os.path.exists(D_LABITEMS_FILE):
        logger.warning("D_LABITEMS not found; using ITEMID as label for labs.")
        return {}
    dlab = pd.read_csv(D_LABITEMS_FILE, usecols=["ITEMID", "LABEL"])
    dlab["LABEL"] = dlab["LABEL"].astype(str)
    return dict(zip(dlab["ITEMID"].astype(int), dlab["LABEL"]))

def load_pressor_itemids():
    """
    Identify INPUTEVENTS* ITEMIDs for key vasopressors via D_ITEMS labels.
    Returns dict: { 'dopamine': set(...), 'dobutamine': set(...), 'epinephrine': set(...),
                    'norepinephrine': set(...), 'phenylephrine': set(...) }
    """
    wanted = {
        "dopamine": {"dopamine"},
        "dobutamine": {"dobutamine"},
        "epinephrine": {"epinephrine", "adrenaline"},
        "norepinephrine": {"norepinephrine", "noradrenaline"},
        # === NEW (V9) ===
        "phenylephrine": {"phenylephrine"},
    }
    out = {k: set() for k in wanted.keys()}
    if not os.path.exists(D_ITEMS_FILE):
        logger.warning("D_ITEMS not found; vasopressor detection will be limited.")
        return out
    di = pd.read_csv(D_ITEMS_FILE, usecols=["ITEMID", "LABEL"])
    di["LABEL"] = di["LABEL"].astype(str).str.lower()
    for _, r in di.iterrows():
        label = r["LABEL"]
        iid = int(r["ITEMID"])
        for drug, keys in wanted.items():
            if any(k in label for k in keys):
                out[drug].add(iid)
    return out

def load_output_itemid_map():
    """Load D_ITEMS for OUTPUTEVENTS to label urine vs total outputs."""
    if not os.path.exists(D_ITEMS_FILE):
        logger.warning("D_ITEMS not found; urine detection will fallback to total output only.")
        return {}
    di = pd.read_csv(D_ITEMS_FILE, usecols=["ITEMID", "LABEL"])
    di["LABEL"] = di["LABEL"].astype(str).str.lower()
    return dict(zip(di["ITEMID"].astype(int), di["LABEL"]))

def load_gcs_itemids():
    """
    Discover CHARTEVENTS ITEMIDs for GCS components from D_ITEMS by label text.
    Returns dict: {itemid -> 'gcs_eye'|'gcs_verbal'|'gcs_motor'|'gcs_total'}
    """
    if not os.path.exists(D_ITEMS_FILE):
        logger.warning("D_ITEMS not found; GCS detection skipped.")
        return {}
    di = pd.read_csv(D_ITEMS_FILE, usecols=["ITEMID","LABEL"])
    s = di["LABEL"].astype(str).str.lower()

    eye_mask    = s.str.contains(r"glasgow.*eye|gcs.*eye|eye opening", regex=True)
    verbal_mask = s.str.contains(r"glasgow.*verbal|gcs.*verbal", regex=True)
    motor_mask  = s.str.contains(r"glasgow.*motor|gcs.*motor", regex=True)
    total_mask  = s.str.contains(r"glasgow coma scale.*total|gcs total|gcs\s*-\s*total", regex=True)

    out = {}
    for iid in di.loc[eye_mask, "ITEMID"].astype(int):    out[iid] = "gcs_eye"
    for iid in di.loc[verbal_mask, "ITEMID"].astype(int): out[iid] = "gcs_verbal"
    for iid in di.loc[motor_mask, "ITEMID"].astype(int):  out[iid] = "gcs_motor"
    for iid in di.loc[total_mask, "ITEMID"].astype(int):  out[iid] = "gcs_total"
    return out


# ===================== MAIN ======================
def main():
    start_t = time.time()
    logger.info("Building LOS dataset....")

    # Discover and merge GCS ITEMIDs (after logger exists)
    global ITEMID_TO_LABEL, ALL_CHART_ITEMIDS
    gcs_map = load_gcs_itemids()
    if gcs_map:
        ITEMID_TO_LABEL.update(gcs_map)
        ALL_CHART_ITEMIDS |= set(gcs_map.keys())
        logger.info(f"GCS ITEMIDs discovered: {len(gcs_map)} (E/V/M/T)")
    else:
        logger.warning("No GCS ITEMIDs discovered; proceeding without GCS.")

    # 1) Load ICU, Admissions, Patients (safe parsing)  [UNCHANGED from V8, except logger name/paths]
    icu_raw = pd.read_csv(
        ICUSTAYS_FILE,
        usecols=["ICUSTAY_ID","HADM_ID","INTIME","OUTTIME","FIRST_CAREUNIT"],
        parse_dates=["INTIME","OUTTIME"]
    ).dropna(subset=["INTIME","OUTTIME"])

    admissions = pd.read_csv(
        ADMISSIONS_FILE,
        usecols=[
            "HADM_ID","SUBJECT_ID","ADMITTIME","DISCHTIME",
            "ADMISSION_TYPE","ADMISSION_LOCATION","INSURANCE","ETHNICITY"
        ],
        parse_dates=["ADMITTIME","DISCHTIME"]
    ).dropna(subset=["ADMITTIME","DISCHTIME"])

    patients = pd.read_csv(
        PATIENTS_FILE,
        usecols=["SUBJECT_ID","GENDER","DOB"],
        parse_dates=["DOB"]
    ).dropna(subset=["DOB"])

    # Keep FIRST ICU stay per HADM_ID (earliest INTIME)  [UNCHANGED]
    icustays = (
        icu_raw.sort_values(["HADM_ID","INTIME"])
               .groupby("HADM_ID", as_index=False).first()
    )

    # Compute ICU LOS labels  [UNCHANGED]
    icustays["LOS_DAYS"] = (icustays["OUTTIME"] - icustays["INTIME"]).dt.total_seconds() / (24*3600)
    icustays["LOS_label"] = (icustays["LOS_DAYS"] > 7.0).astype("int8")
    conds = [
        icustays["LOS_DAYS"] <= 3.0,
        (icustays["LOS_DAYS"] > 3.0) & (icustays["LOS_DAYS"] <= 7.0),
        icustays["LOS_DAYS"] > 7.0,
    ]
    icustays["LOS_label_3class"] = np.select(conds, [0,1,2]).astype("int8")

    # Merge SUBJECT + ADMISSION_TYPE onto ICU rows  [UNCHANGED]
    icux = icustays.merge(
        admissions[["HADM_ID","SUBJECT_ID","ADMISSION_TYPE"]],
        on="HADM_ID", how="left"
    )

    # === Sampling (optional) ===  [UNCHANGED]
    if SAMPLE_ADMISSIONS is not None:
        n_avail = icux["HADM_ID"].nunique()
        take = min(int(SAMPLE_ADMISSIONS), n_avail)
        hadm_sample = icux["HADM_ID"].drop_duplicates().sample(n=take, random_state=42)
        icux = icux[icux["HADM_ID"].isin(hadm_sample)]
        icustays = icustays[icustays["HADM_ID"].isin(hadm_sample)]
        admissions = admissions[admissions["HADM_ID"].isin(hadm_sample)]
        patients = patients[patients["SUBJECT_ID"].isin(admissions["SUBJECT_ID"].unique())]

    # === SAFE AGE (calendar arithmetic; no timedelta) ===  [UNCHANGED]
    adm_pat = admissions.merge(patients, on="SUBJECT_ID", how="left")
    adm_pat = safe_to_datetime(adm_pat, ["ADMITTIME", "DOB"])
    mask = (
        adm_pat["ADMITTIME"].notna() &
        adm_pat["DOB"].notna() &
        (adm_pat["ADMITTIME"] >= pd.Timestamp("1900-01-01")) &
        (adm_pat["DOB"]       >= pd.Timestamp("1900-01-01")) &
        (adm_pat["ADMITTIME"] >= adm_pat["DOB"])
    )
    adm_pat["AGE"] = np.nan
    y = adm_pat.loc[mask, "ADMITTIME"].dt.year - adm_pat.loc[mask, "DOB"].dt.year
    adm_md  = adm_pat.loc[mask, "ADMITTIME"].dt.month * 100 + adm_pat.loc[mask, "ADMITTIME"].dt.day
    dob_md  = adm_pat.loc[mask, "DOB"].dt.month * 100 + adm_pat.loc[mask, "DOB"].dt.day
    y = y - (adm_md < dob_md).astype(int)
    adm_pat.loc[mask, "AGE"] = y.astype(float).clip(lower=0, upper=120)

    # Normalize GENDER to single letter if needed  [UNCHANGED]
    if adm_pat["GENDER"].dtype == object:
        adm_pat["GENDER"] = adm_pat["GENDER"].astype(str).str.strip().str.upper()

    static_cols = adm_pat[["HADM_ID","AGE","GENDER","ADMISSION_TYPE"]].drop_duplicates("HADM_ID")
    
    # === Add new static features ===
    logger.info("Adding new static features...")

    # 1) ED → ICU time gap (hours)
    ed_icu = icustays[["HADM_ID", "INTIME"]].merge(
        admissions[["HADM_ID", "ADMITTIME"]], on="HADM_ID", how="left"
    )
    ed_icu["ED_TO_ICU_HOURS"] = (
        (ed_icu["INTIME"] - ed_icu["ADMITTIME"]).dt.total_seconds() / 3600.0
    ).clip(lower=0, upper=72)
    ed_icu = ed_icu[["HADM_ID", "ED_TO_ICU_HOURS"]]

    # 2) First ICU care unit
    careunit = (
        icustays.sort_values(["HADM_ID", "INTIME"])
        .groupby("HADM_ID", as_index=False)
        .first()[["HADM_ID", "FIRST_CAREUNIT"]]
    )

    # 3) Admission location, insurance, ethnicity
    adm_aux = admissions[["HADM_ID", "ADMISSION_LOCATION", "INSURANCE", "ETHNICITY"]].copy()
    for c in ["ADMISSION_LOCATION", "INSURANCE", "ETHNICITY"]:
        adm_aux[c] = adm_aux[c].astype(str).str.strip().str.upper()

    # 4) Prior ICU count for subject
    prior = icustays.merge(
        icustays[["HADM_ID", "INTIME"]].rename(columns={"INTIME": "INDEX_INTIME"}),
        on="HADM_ID",
        how="left",
    )
    prior = prior[prior["OUTTIME"] < prior["INDEX_INTIME"]]
    prior_cnt = (
        prior.groupby("HADM_ID")["ICUSTAY_ID"]
        .count()
        .reset_index(name="PRIOR_ICU_COUNT_SUBJ")
    )

    # --- Merge all into static_cols ---
    static_cols = (
        static_cols.merge(ed_icu, on="HADM_ID", how="left")
        .merge(careunit, on="HADM_ID", how="left")
        .merge(adm_aux, on="HADM_ID", how="left")
        .merge(prior_cnt, on="HADM_ID", how="left")
    )

    logger.info("Completed adding new static features.")

        # === Add diagnosis-based features ===
    # logger.info("Adding diagnosis features...")

    # diag = pd.read_csv(f"{DATA_DIR}/DIAGNOSES_ICD.csv", usecols=["HADM_ID", "ICD9_CODE"])
    # dmap = pd.read_csv(f"{DATA_DIR}/D_ICD_DIAGNOSES.csv.gz", usecols=["ICD9_CODE", "SHORT_TITLE"])
    # diag = diag.merge(dmap, on="ICD9_CODE", how="left")

    # # --- Normalize diagnosis text ---
    # diag["SHORT_TITLE"] = diag["SHORT_TITLE"].astype(str).str.lower()

    # # --- Top 10 broad groups ---
    # top_groups = {
    #     "sepsis": ["sepsis", "septic"],
    #     "respiratory": ["pneumonia", "respiratory", "copd", "asthma"],
    #     "cardiac": ["cardiac", "heart", "myocard", "infarct", "angina"],
    #     "renal": ["renal", "kidney", "uremia"],
    #     "neuro": ["stroke", "cerebral", "neuro", "seizure"],
    #     "trauma": ["fracture", "injury", "trauma"],
    #     "infection": ["infection"],
    #     "liver": ["liver", "hepatic"],
    #     "cancer": ["cancer", "malignan"],
    #     "metabolic": ["diabetes", "obesity", "metabolic"]
    # }

    # diag_flags = []
    # for name, kws in top_groups.items():
    #     m = diag["SHORT_TITLE"].str.contains("|".join(kws), case=False, na=False)
    #     subset = diag.loc[m, ["HADM_ID"]].drop_duplicates().assign(**{f"dx_{name}": 1})
    #     diag_flags.append(subset)

    # diag_flags_df = static_cols[["HADM_ID"]].copy()
    # for df_flag in diag_flags:
    #     diag_flags_df = diag_flags_df.merge(df_flag, on="HADM_ID", how="left")

    # diag_flags_df = diag_flags_df.fillna(0).astype(int)

    # # --- Charlson Comorbidity Index (simplified count proxy) ---
    # cci = diag.groupby("HADM_ID")["ICD9_CODE"].nunique().reset_index(name="CHARLSON_INDEX")
    # diag_flags_df = diag_flags_df.merge(cci, on="HADM_ID", how="left")

    # # --- Merge into static_cols ---
    # static_cols = static_cols.merge(diag_flags_df, on="HADM_ID", how="left")

    # logger.info("Completed adding diagnosis features.")

    # # === NEW: True comorbidity flags (Charlson-style, ICD-9 prefixes) ===
    # diag["ICD9_CODE"] = diag["ICD9_CODE"].astype(str).str.replace(r"\D", "", regex=True)

    # charlson_map = {
    #     "mi":        [r"^410", r"^412"],
    #     "chf":       [r"^428"],
    #     "pvd":       [r"^4439", r"^441", r"^7854", r"^V434", r"^V433"],
    #     "cerebro":   [r"^430", r"^431", r"^432", r"^433", r"^434", r"^435", r"^436", r"^437", r"^438"],
    #     "dementia":  [r"^290"],
    #     "copd":      [r"^490", r"^491", r"^492", r"^494", r"^496"],
    #     "rheum":     [r"^710", r"^714", r"^725"],
    #     "pud":       [r"^531", r"^532", r"^533", r"^534"],
    #     "mild_liver":[r"^5712", r"^5714", r"^5715", r"^5716", r"^5718", r"^5719"],
    #     "diab_wo":   [r"^2500", r"^2501", r"^2502", r"^2503", r"^2508", r"^2509"],
    #     "diab_w":    [r"^2504", r"^2505", r"^2506", r"^2507"],
    #     "paralysis": [r"^342", r"^343", r"^344"],
    #     "renal":     [r"^582", r"^5830", r"^5831", r"^5832", r"^5834", r"^5836", r"^585", r"^586", r"^5880"],
    #     "any_cancer":[r"^140", r"^141", r"^142", r"^143", r"^144", r"^145", r"^146", r"^147", r"^148", r"^149",
    #                 r"^150", r"^151", r"^152", r"^153", r"^154", r"^155", r"^156", r"^157", r"^158", r"^159",
    #                 r"^160", r"^161", r"^162", r"^163", r"^164", r"^165", r"^170", r"^171", r"^172", r"^174",
    #                 r"^175", r"^176", r"^179", r"^180", r"^181", r"^182", r"^183", r"^184", r"^185", r"^186",
    #                 r"^187", r"^188", r"^189", r"^190", r"^191", r"^192", r"^193", r"^194", r"^195", r"^196",
    #                 r"^197", r"^198", r"^199"],
    #     "modsev_liver":[r"^5722", r"^5723", r"^5724", r"^5728"],
    #     "metastatic":[r"^196", r"^197", r"^198", r"^199"],
    #     "aids":      [r"^042", r"^043", r"^044"],
    # }

    # def icd_match(series, patterns):
    #     pat = "|".join(patterns)
    #     return series.str.match(pat, na=False)

    # # Per-HADM flags
    # cm_flags = []
    # for name, pats in charlson_map.items():
    #     m = icd_match(diag["ICD9_CODE"], pats)
    #     cm_flags.append(diag.loc[m, ["HADM_ID"]].drop_duplicates().assign(**{f"cm_{name}": 1}))

    # cm_df = static_cols[["HADM_ID"]].drop_duplicates().copy()
    # for f in cm_flags:
    #     cm_df = cm_df.merge(f, on="HADM_ID", how="left")
    # cm_df = cm_df.fillna(0).astype(int)

    # # Optional: simple Charlson unweighted sum (presence-based)
    # cm_cols = [c for c in cm_df.columns if c.startswith("cm_")]
    # cm_df["charlson_simple_sum"] = cm_df[cm_cols].sum(axis=1).astype("int16")

    # # Merge into static_cols in the same block where you merge diag_flags_df
    # static_cols = static_cols.merge(cm_df, on="HADM_ID", how="left")
    # logger.info("Added comorbidity flags (Charlson-style) and charlson_simple_sum.")




    # Maps  [UNCHANGED]
    icu_map = icux.set_index("ICUSTAY_ID")[["HADM_ID","INTIME"]].to_dict(orient="index")
    hadm_to_intime = icux.set_index("HADM_ID")["INTIME"].to_dict()

    win_pairs = window_pairs(WINDOW_HOURS)

    # ============================================================
    # A) CHARTEVENTS (same as V8) — hourly mean for selected vitals
    # ============================================================
    vit_sum_acc = defaultdict(lambda: defaultdict(float))  # wn -> {(hadm, vital): sum}
    vit_cnt_acc = defaultdict(lambda: defaultdict(int))    # wn -> {(hadm, vital): cnt}

    try:
        chunk_iter = pd.read_csv(
            CHARTEVENTS_FILE,
            usecols=["ICUSTAY_ID","CHARTTIME","ITEMID","VALUENUM"],
            parse_dates=["CHARTTIME"],
            chunksize=CHUNK_SIZE,
            compression="gzip",
            low_memory=False,
        )
    except ValueError:
        chunk_iter = pd.read_csv(
            CHARTEVENTS_FILE,
            usecols=["ICUSTAY_ID","CHARTTIME","ITEMID","VALUENUM"],
            parse_dates=["CHARTTIME"],
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )

    rows_seen = 0
    for i, ch in enumerate(chunk_iter, start=1):
        rows_seen += len(ch)
        ch = ch.dropna(subset=["ICUSTAY_ID","ITEMID","CHARTTIME","VALUENUM"])
        ch = ch[ch["ITEMID"].isin(ALL_CHART_ITEMIDS)]
        if ch.empty:
            if MAX_CHUNKS and i >= MAX_CHUNKS: break
            continue

        ch["HADM_ID"] = ch["ICUSTAY_ID"].map(lambda x: icu_map.get(int(x), {}).get("HADM_ID", np.nan))
        ch["INTIME"]  = ch["ICUSTAY_ID"].map(lambda x: icu_map.get(int(x), {}).get("INTIME", pd.NaT))
        ch = ch.dropna(subset=["HADM_ID","INTIME"])

        ch["VALUENUM"] = to_celsius_if_needed(ch["ITEMID"], ch["VALUENUM"])

        ch["hours_since_intime"] = (ch["CHARTTIME"] - ch["INTIME"]).dt.total_seconds() / 3600.0
        ch = ch[(ch["hours_since_intime"] >= 0) & (ch["hours_since_intime"] < 24)]
        if ch.empty:
            if MAX_CHUNKS and i >= MAX_CHUNKS: break
            continue

        ch["VITAL"] = ch["ITEMID"].map(ITEMID_TO_LABEL)
        ch["HOUR_BIN"] = hour_bin_from_delta_hours(ch["hours_since_intime"])

        g = ch.groupby(["HADM_ID","VITAL","HOUR_BIN"])["VALUENUM"].agg(["sum","count"]).reset_index()
        for _, r in g.iterrows():
            hadm = int(r["HADM_ID"]); vital = r["VITAL"]; hb = int(r["HOUR_BIN"])
            wn = wname(hb, hb+1)
            vit_sum_acc[wn][(hadm, vital)] += float(r["sum"])
            vit_cnt_acc[wn][(hadm, vital)] += int(r["count"])

        if i % 5 == 0:
            logger.info(f"[CHARTEVENTS] chunks={i:,} rows≈{rows_seen:,}")
        if MAX_CHUNKS and i >= MAX_CHUNKS:
            break

    chart_frames = []
    for wn in vit_sum_acc.keys():
        sdict, cdict = vit_sum_acc[wn], vit_cnt_acc[wn]
        if not sdict:
            continue
        keys = list(sdict.keys())
        dfw = pd.DataFrame({
            "HADM_ID": [k[0] for k in keys],
            "VITAL":   [k[1] for k in keys],
            "MEAN":    [sdict[k] / max(cdict[k], 1) for k in keys],
        })
        wide = dfw.pivot_table(index="HADM_ID", columns="VITAL", values="MEAN").reset_index()
        wide.columns = ["HADM_ID"] + [f"{c}_mean_{wn}" for c in wide.columns if c != "HADM_ID"]
        chart_frames.append(wide)

        # ============================================================
    # (NEW) RESPIRATORY SUPPORT — FiO2, PEEP, Ventilation flag (from CHARTEVENTS)
    # ============================================================
    logger.info("Adding respiratory support features (FiO2, PEEP, Ventilation)...")

    resp_itemids = {
        "FiO2": [3420, 190, 223835],
        "PEEP": [505, 220339],
        "VentMode": [720, 223849, 467, 640, 3459, 224688],
    }

    resp_sum_acc = defaultdict(lambda: defaultdict(float))
    resp_cnt_acc = defaultdict(lambda: defaultdict(int))
    vent_flag_acc = defaultdict(lambda: defaultdict(int))

    resp_iter = _read_csv_maybe_gz(
        CHARTEVENTS_FILE,
        usecols=["ICUSTAY_ID","CHARTTIME","ITEMID","VALUENUM"],
        parse_dates=["CHARTTIME"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for i, ch in enumerate(resp_iter, start=1):
        ch = ch.dropna(subset=["ICUSTAY_ID","ITEMID","CHARTTIME"])
        ch = ch[ch["ITEMID"].isin(sum(resp_itemids.values(), []))]
        if ch.empty:
            if MAX_CHUNKS and i >= MAX_CHUNKS: break
            continue

        ch["HADM_ID"] = ch["ICUSTAY_ID"].map(lambda x: icu_map.get(int(x), {}).get("HADM_ID", np.nan))
        ch["INTIME"] = ch["ICUSTAY_ID"].map(lambda x: icu_map.get(int(x), {}).get("INTIME", pd.NaT))
        ch = ch.dropna(subset=["HADM_ID","INTIME"])

        ch["hours_since_intime"] = (ch["CHARTTIME"] - ch["INTIME"]).dt.total_seconds() / 3600.0
        ch = ch[(ch["hours_since_intime"] >= 0) & (ch["hours_since_intime"] < 24)]
        if ch.empty:
            if MAX_CHUNKS and i >= MAX_CHUNKS: break
            continue

        ch["HOUR_BIN"] = hour_bin_from_delta_hours(ch["hours_since_intime"])
        for _, r in ch.iterrows():
            hadm = int(r["HADM_ID"]); itemid = int(r["ITEMID"]); hb = int(r["HOUR_BIN"])
            wn = wname(hb, hb+1)
            val = r.get("VALUENUM", np.nan)
            if itemid in resp_itemids["FiO2"] and pd.notna(val):
                resp_sum_acc[wn][(hadm, "FiO2_mean")] += float(val)
                resp_cnt_acc[wn][(hadm, "FiO2_mean")] += 1
            elif itemid in resp_itemids["PEEP"] and pd.notna(val):
                resp_sum_acc[wn][(hadm, "PEEP_mean")] += float(val)
                resp_cnt_acc[wn][(hadm, "PEEP_mean")] += 1
            elif itemid in resp_itemids["VentMode"]:
                vent_flag_acc[wn][(hadm, "ventilation_any")] = 1
        if MAX_CHUNKS and i >= MAX_CHUNKS: break

    # --- Build wide frames ---
    resp_frames = []
    for wn in resp_sum_acc.keys():
        sdict, cdict = resp_sum_acc[wn], resp_cnt_acc[wn]
        keys = list(sdict.keys())
        dfw = pd.DataFrame({
            "HADM_ID": [k[0] for k in keys],
            "MEASURE": [k[1] for k in keys],
            "MEAN": [sdict[k] / max(cdict[k], 1) for k in keys],
        })
        wide = dfw.pivot_table(index="HADM_ID", columns="MEASURE", values="MEAN").reset_index()
        wide.columns = ["HADM_ID"] + [f"{c}_{wn}" for c in wide.columns if c != "HADM_ID"]
        resp_frames.append(wide)

    for wn in vent_flag_acc.keys():
        sdict = vent_flag_acc[wn]
        keys = list(sdict.keys())
        dfw = pd.DataFrame({
            "HADM_ID": [k[0] for k in keys],
            "FLAG": [k[1] for k in keys],
            "ANY": [sdict[k] for k in keys],
        })
        wide = dfw.pivot_table(index="HADM_ID", columns="FLAG", values="ANY", aggfunc="max", fill_value=0).reset_index()
        wide.columns = ["HADM_ID"] + [f"{c}_{wn}" for c in wide.columns if c != "HADM_ID"]
        resp_frames.append(wide)
    logger.info("Completed adding respiratory support features.")

        # ============================================================
    # (NEW) Ventilator Mechanics — VT, PIP, Plateau, MAP, Vent RR
    # ============================================================
    logger.info("Adding ventilator mechanics features...")

    vent_itemids = {
        "TidalVolume": [398, 683, 224684],
        "PIP": [224695],
        "Plateau": [224696],
        "MAP": [224697],
        "VentRate": [224689],
    }

    vent_sum_acc = defaultdict(lambda: defaultdict(float))
    vent_cnt_acc = defaultdict(lambda: defaultdict(int))

    vent_iter = _read_csv_maybe_gz(
        CHARTEVENTS_FILE,
        usecols=["ICUSTAY_ID","CHARTTIME","ITEMID","VALUENUM"],
        parse_dates=["CHARTTIME"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for i, ch in enumerate(vent_iter, start=1):
        ch = ch.dropna(subset=["ICUSTAY_ID","ITEMID","CHARTTIME","VALUENUM"])
        ch = ch[ch["ITEMID"].isin(sum(vent_itemids.values(), []))]
        if ch.empty:
            if MAX_CHUNKS and i >= MAX_CHUNKS: break
            continue

        ch["HADM_ID"] = ch["ICUSTAY_ID"].map(lambda x: icu_map.get(int(x), {}).get("HADM_ID", np.nan))
        ch["INTIME"] = ch["ICUSTAY_ID"].map(lambda x: icu_map.get(int(x), {}).get("INTIME", pd.NaT))
        ch = ch.dropna(subset=["HADM_ID","INTIME"])

        ch["hours_since_intime"] = (ch["CHARTTIME"] - ch["INTIME"]).dt.total_seconds() / 3600.0
        ch = ch[(ch["hours_since_intime"] >= 0) & (ch["hours_since_intime"] < 24)]
        if ch.empty:
            if MAX_CHUNKS and i >= MAX_CHUNKS: break
            continue

        ch["HOUR_BIN"] = hour_bin_from_delta_hours(ch["hours_since_intime"])
        for _, r in ch.iterrows():
            hadm = int(r["HADM_ID"]); val = r["VALUENUM"]; iid = int(r["ITEMID"]); hb = int(r["HOUR_BIN"])
            wn = wname(hb, hb+1)
            for label, ids in vent_itemids.items():
                if iid in ids:
                    vent_sum_acc[wn][(hadm, label)] += float(val)
                    vent_cnt_acc[wn][(hadm, label)] += 1
        if i % 10 == 0:
            logger.info(f"[VENTILATOR] chunks={i:,}")
        if MAX_CHUNKS and i >= MAX_CHUNKS:
            break

    vent_frames = []
    for wn in vent_sum_acc.keys():
        sdict, cdict = vent_sum_acc[wn], vent_cnt_acc[wn]
        if not sdict: continue
        keys = list(sdict.keys())
        dfw = pd.DataFrame({
            "HADM_ID": [k[0] for k in keys],
            "MEASURE": [k[1] for k in keys],
            "MEAN": [sdict[k] / max(cdict[k],1) for k in keys],
        })
        wide = dfw.pivot_table(index="HADM_ID", columns="MEASURE", values="MEAN").reset_index()
        wide.columns = ["HADM_ID"] + [f"{c}_mean_{wn}" for c in wide.columns if c != "HADM_ID"]
        vent_frames.append(wide)

    logger.info("Completed adding ventilator mechanics features.")


    # ============================================================
    # Medication categories (0–24h ANY): antibiotics, sedatives, insulin, diuretics
    # ============================================================
    logger.info("Adding medication category flags (0–24h any)...")

    def load_med_itemids():
        # Map D_ITEMS labels -> category via substring match (lowercased)
        cats = {
            "antibiotic": [
                "penicillin","ampicillin","amoxicillin","piperacillin",
                "tazobactam","zosyn","cef","ceph","cefazolin","ceftriaxone",
                "cefepime","ceftazidime","meropenem","ertapenem","imipenem",
                "vancomycin","linezolid","azith","clarith","clindamycin",
                "metronidazole","levofloxacin","ciprofloxacin","moxifloxacin",
            ],
            "sedative": ["midazolam","lorazepam","propofol","dexmedetomidine"],
            "insulin": ["insulin"],
            "diuretic": ["furosemide","lasix","bumetanide","torsemide"]
        }
        out = {k: set() for k in cats}
        if not os.path.exists(D_ITEMS_FILE):
            logger.warning("D_ITEMS not found; medication categorization limited.")
            return out
        di = pd.read_csv(D_ITEMS_FILE, usecols=["ITEMID","LABEL"])
        di["LABEL_L"] = di["LABEL"].astype(str).str.lower()
        for _, r in di.iterrows():
            iid = int(r["ITEMID"]); lab = r["LABEL_L"]
            for k, kws in cats.items():
                if any(kw in lab for kw in kws):
                    out[k].add(iid)
        return out

    med_itemids = load_med_itemids()
    has_meds = any(len(s) for s in med_itemids.values())
    med_acc = defaultdict(lambda: defaultdict(int))  # wn -> {(hadm, 'antibiotic_any'): 0/1}

    def _mark_medications_from_inputs(path, label):
        if not (os.path.exists(path) and has_meds):
            return
        header = _read_csv_maybe_gz(path, nrows=0)
        cols_int = [c for c in ["ICUSTAY_ID","HADM_ID","ITEMID","STARTTIME","ENDTIME","AMOUNT","RATE"] if c in header.columns]
        cols_inst = [c for c in ["ICUSTAY_ID","HADM_ID","ITEMID","CHARTTIME","AMOUNT","RATE"] if c in header.columns]
        parse_int = [c for c in ["STARTTIME","ENDTIME"] if c in cols_int]
        parse_inst = [c for c in ["CHARTTIME"] if c in cols_inst]

        def _attach_times(df):
            # map INTIME / HADM_ID like you do elsewhere
            if "ICUSTAY_ID" in df.columns:
                df["HADM_MAP"] = df["ICUSTAY_ID"].map(lambda x: icu_map.get(_safe_int(x), {}).get("HADM_ID", np.nan))
                df["INTIME"]   = df["ICUSTAY_ID"].map(lambda x: icu_map.get(_safe_int(x), {}).get("INTIME", pd.NaT))
                msk = df["INTIME"].isna()
                if "HADM_ID" in df.columns:
                    df.loc[msk, "INTIME"]   = df.loc[msk, "HADM_ID"].map(hadm_to_intime)
                    df.loc[msk, "HADM_MAP"] = df.loc[msk, "HADM_ID"]
                df.rename(columns={"HADM_MAP":"HADM_ID"}, inplace=True)
            else:
                df["INTIME"] = df["HADM_ID"].map(hadm_to_intime)
            # dedupe cols and coerce
            df = df.loc[:, ~df.columns.duplicated()].copy()
            df["HADM_ID"] = pd.to_numeric(df["HADM_ID"], errors="coerce")
            return df.dropna(subset=["HADM_ID","INTIME"]).assign(HADM_ID=lambda x: x["HADM_ID"].astype("int64"))

        # ----- Interval rows (START/END) -----
        if parse_int:
            it = _read_csv_maybe_gz(path, usecols=cols_int, parse_dates=parse_int, chunksize=CHUNK_SIZE, low_memory=False)
            for i, df in enumerate(it, 1):
                df = _attach_times(df)
                if df.empty: 
                    if MAX_CHUNKS and i >= MAX_CHUNKS: break
                    continue
                df = df.dropna(subset=["ITEMID","STARTTIME"])
                df["ENDTIME"] = pd.to_datetime(df["ENDTIME"], errors="coerce").fillna(df["STARTTIME"])
                df["start_delta_h"] = (df["STARTTIME"] - df["INTIME"]).dt.total_seconds()/3600.0
                df["end_delta_h"]   = (df["ENDTIME"]   - df["INTIME"]).dt.total_seconds()/3600.0
                df["start_h"] = np.floor(np.maximum(df["start_delta_h"], 0)).astype(int)
                df["end_h"]   = np.ceil(np.minimum(df["end_delta_h"], 24)).astype(int)
                df = df[(df["end_h"] > 0) & (df["start_h"] < 24)]
                if df.empty: 
                    if MAX_CHUNKS and i >= MAX_CHUNKS: break
                    continue
                for _, r in df.iterrows():
                    hadm = int(r["HADM_ID"]); itemid = int(r["ITEMID"])
                    for cat, idset in med_itemids.items():
                        if itemid in idset:
                            for hb in range(max(0,int(r["start_h"])), min(24,int(r["end_h"]))):
                                wn = wname(hb, hb+1)
                                med_acc[wn][(hadm, f"{cat}_any")] = 1
                if MAX_CHUNKS and i >= MAX_CHUNKS: break

        # ----- Instant rows (CHARTTIME) -----
        if cols_inst and "CHARTTIME" in cols_inst:
            it = _read_csv_maybe_gz(path, usecols=cols_inst, parse_dates=parse_inst, chunksize=CHUNK_SIZE, low_memory=False)
            for i, df in enumerate(it, 1):
                df = _attach_times(df)
                if df.empty: 
                    if MAX_CHUNKS and i >= MAX_CHUNKS: break
                    continue
                df = df.dropna(subset=["ITEMID","CHARTTIME"])
                df["hours_since_intime"] = (df["CHARTTIME"] - df["INTIME"]).dt.total_seconds()/3600.0
                df = df[(df["hours_since_intime"] >= 0) & (df["hours_since_intime"] < 24)]
                if df.empty: 
                    if MAX_CHUNKS and i >= MAX_CHUNKS: break
                    continue
                df["HOUR_BIN"] = hour_bin_from_delta_hours(df["hours_since_intime"])
                for _, r in df.iterrows():
                    hadm = int(r["HADM_ID"]); hb = int(r["HOUR_BIN"]); wn = wname(hb, hb+1)
                    itemid = int(r["ITEMID"])
                    for cat, idset in med_itemids.items():
                        if itemid in idset:
                            med_acc[wn][(hadm, f"{cat}_any")] = 1
                if MAX_CHUNKS and i >= MAX_CHUNKS: break

    # Run on both input sources
    _mark_medications_from_inputs(INPUTEVENTS_MV_FILE, "INPUTS_MV")
    _mark_medications_from_inputs(INPUTEVENTS_CV_FILE, "INPUTS_CV")

    # Derive 0–24h any flags (one column per category) and merge later like you do for pressors
    def derive_med_any_0_24(acc):
        per_cat = defaultdict(lambda: defaultdict(int))  # cat -> hadm -> 0/1
        for sdict in acc.values():
            for (hadm, cat_any), v in sdict.items():
                if v:
                    cat = cat_any.replace("_any","")
                    per_cat[cat][hadm] = 1
        if not per_cat:
            return None
        rows = {}
        for cat, hm in per_cat.items():
            for h, v in hm.items():
                rows.setdefault(h, {})[f"{cat}_any_0_24h"] = v
        df = pd.DataFrame.from_dict(rows, orient="index").reset_index().rename(columns={"index":"HADM_ID"})
        for c in ["antibiotic_any_0_24h","sedative_any_0_24h","insulin_any_0_24h","diuretic_any_0_24h"]:
            if c not in df.columns: df[c] = 0
        return df

    med_any_0_24_df = derive_med_any_0_24(med_acc)
    logger.info("Completed adding medication category flags (0–24h).")



    # === Oxygenation indices (S/F ratio) ===
    logger.info("Adding oxygenation indices (S/F ratio)...")
    ox_frames = []

    def _fi02_to_fraction(x):
        if pd.isna(x): 
            return np.nan
        try:
            x = float(x)
        except Exception:
            return np.nan
        # if recorded as percent, convert to fraction
        if x > 1.0:
            x = x / 100.0
        # clamp physiologic range
        x = float(np.clip(x, 0.21, 1.0))
        return x

    # Build per-hour S/F from the same accumulators we already have in-memory
    # vit_sum_acc / vit_cnt_acc hold SpO2 means per window (label 'SpO2')
    # resp_sum_acc / resp_cnt_acc hold FiO2 means per window (key 'FiO2_mean')
    for wn in sorted(set(list(vit_sum_acc.keys()) + list(resp_sum_acc.keys()))):
        # Gather SpO2 means
        spo2_map = {}
        sdict_v = vit_sum_acc.get(wn, {})
        cdict_v = vit_cnt_acc.get(wn, {})
        for (hadm, vital), s in sdict_v.items():
            if vital == "SpO2":
                cnt = max(cdict_v.get((hadm, vital), 1), 1)
                spo2_map[hadm] = float(s) / float(cnt)

        # Gather FiO2 means
        fio2_map = {}
        sdict_r = resp_sum_acc.get(wn, {})
        cdict_r = resp_cnt_acc.get(wn, {})
        for (hadm, meas), s in sdict_r.items():
            if meas == "FiO2_mean":
                cnt = max(cdict_r.get((hadm, meas), 1), 1)
                fio2_map[hadm] = float(s) / float(cnt)

        if not spo2_map and not fio2_map:
            continue

        hadms = set(spo2_map.keys()) | set(fio2_map.keys())
        rows = []
        for h in hadms:
            spo2 = spo2_map.get(h, np.nan)
            fio2f = _fi02_to_fraction(fio2_map.get(h, np.nan))
            sfr = np.nan
            if pd.notna(spo2) and pd.notna(fio2f) and fio2f > 0:
                sfr = spo2 / fio2f
            rows.append((h, sfr))

        dfo = pd.DataFrame(rows, columns=["HADM_ID", f"SFratio_{wn}"])
        ox_frames.append(dfo)

    # Derive worst S/F in 0–24h
    if ox_frames:
        tmp = ox_frames[0][["HADM_ID"]].copy()
        for df in ox_frames:
            tmp = tmp.merge(df, on="HADM_ID", how="outer")
        s_cols = [c for c in tmp.columns if c.startswith("SFratio_")]
        if s_cols:
            tmp["s_f_min_0_24h"] = tmp[s_cols].min(axis=1, skipna=True)
            # Keep only HADM_ID + derived; per-hour S/F will be merged via ox_frames anyway
            s_f_min_0_24_df = tmp[["HADM_ID", "s_f_min_0_24h"]]
    else:
        s_f_min_0_24_df = None
    logger.info("Completed adding oxygenation indices (S/F ratio).")

    # ============================================================
    # (NEW) PaO2/FiO2 (P/F) ratio features — mean, min, max, worst 0–24h
    # ============================================================
    logger.info("Adding PaO2/FiO2 (P/F) ratio features...")

    # Ensure both PaO2 and FiO2 hourly columns exist
    pao2_cols = [c for c in df.columns if c.startswith("PaO2_") and c.endswith("h")]
    fio2_cols = [c for c in df.columns if c.startswith("FiO2_") and c.endswith("h")]

    if pao2_cols and fio2_cols:
        fio2_mean = df[fio2_cols].replace(0, np.nan).mean(axis=1, skipna=True)
        pao2_mean = df[pao2_cols].mean(axis=1, skipna=True)
        pf_ratio = pao2_mean / fio2_mean
        df["PF_ratio_mean_0_24h"] = pf_ratio

        df["PF_ratio_min_0_24h"] = (df[pao2_cols].min(axis=1, skipna=True) /
                                    df[fio2_cols].max(axis=1, skipna=True).replace(0, np.nan))
        df["PF_ratio_max_0_24h"] = (df[pao2_cols].max(axis=1, skipna=True) /
                                    df[fio2_cols].min(axis=1, skipna=True).replace(0, np.nan))
        df["PF_ratio_worst_0_24h"] = df["PF_ratio_min_0_24h"]  # Worst = lowest

        logger.info("Completed adding PaO2/FiO2 (P/F) ratio features.")
    else:
        logger.warning("Missing PaO2 or FiO2 columns — skipping P/F ratio.")


    # ============================================================
    # (NEW) 24-hour variability features (std, coefficient of variation)
    # ============================================================
    logger.info("Adding 24-hour variability features...")

    vitals_for_var = ["Heart_Rate", "MAP", "Resp_Rate", "Temperature_C", "SpO2", "FiO2", "PEEP"]
    for v in vitals_for_var:
        cols = [c for c in df.columns if c.startswith(v + "_mean_") and c.endswith("h")]
        if not cols:
            continue
        df[f"{v}_std_0_24h"] = df[cols].std(axis=1, skipna=True)
        df[f"{v}_cv_0_24h"] = df[f"{v}_std_0_24h"] / (df[cols].mean(axis=1, skipna=True) + 1e-6)

    logger.info("Completed adding 24-hour variability features.")

    # ============================================================
    # B) LABEVENTS — hourly mean per lab label  [UNCHANGED]
    # ============================================================
    lab_label_map = load_lab_label_map()
    lab_sum_acc = defaultdict(lambda: defaultdict(float))  # wn -> {(hadm, lab_label): sum}
    lab_cnt_acc = defaultdict(lambda: defaultdict(int))

    if os.path.exists(LABEVENTS_FILE):
        lab_iter = _read_csv_maybe_gz(
            LABEVENTS_FILE,
            usecols=["HADM_ID","ITEMID","CHARTTIME","VALUENUM"],
            parse_dates=["CHARTTIME"],
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )
        lab_rows = 0
        for i, lb in enumerate(lab_iter, start=1):
            lab_rows += len(lb)
            lb = lb.dropna(subset=["HADM_ID","ITEMID","CHARTTIME","VALUENUM"])
            lb = lb.copy()
            lb.loc[:, "INTIME"] = lb["HADM_ID"].map(hadm_to_intime)
            lb = lb.dropna(subset=["INTIME"])

            lb["hours_since_intime"] = (lb["CHARTTIME"] - lb["INTIME"]).dt.total_seconds() / 3600.0
            lb = lb[(lb["hours_since_intime"] >= 0) & (lb["hours_since_intime"] < 24)]
            if lb.empty:
                if MAX_CHUNKS and i >= MAX_CHUNKS: break
                continue

            lb["LAB"] = lb["ITEMID"].astype(int).map(lab_label_map)
            lb["LAB"] = lb["LAB"].fillna(lb["ITEMID"].astype(int).astype(str).radd("item_"))
            lb["LAB"] = lb["LAB"].map(sanitize_label)

            lb["HOUR_BIN"] = hour_bin_from_delta_hours(lb["hours_since_intime"])
            g = lb.groupby(["HADM_ID","LAB","HOUR_BIN"])["VALUENUM"].agg(["sum","count"]).reset_index()
            for _, r in g.iterrows():
                hadm = int(r["HADM_ID"]); lab = r["LAB"]; hb = int(r["HOUR_BIN"])
                wn = wname(hb, hb+1)
                lab_sum_acc[wn][(hadm, lab)] += float(r["sum"])
                lab_cnt_acc[wn][(hadm, lab)] += int(r["count"])

            if i % 10 == 0:
                logger.info(f"[LABEVENTS] chunks={i:,} rows≈{lab_rows:,}")
            if MAX_CHUNKS and i >= MAX_CHUNKS:
                break
    else:
        logger.warning("LABEVENTS not found; skipping labs.")

    lab_frames = []
    for wn in lab_sum_acc.keys():
        sdict, cdict = lab_sum_acc[wn], lab_cnt_acc[wn]
        if not sdict:
            continue
        keys = list(sdict.keys())
        dfw = pd.DataFrame({
            "HADM_ID": [k[0] for k in keys],
            "LAB":     [k[1] for k in keys],
            "MEAN":    [sdict[k] / max(cdict[k], 1) for k in keys],
        })
        wide = dfw.pivot_table(index="HADM_ID", columns="LAB", values="MEAN").reset_index()
        wide.columns = ["HADM_ID"] + [f"{c}_mean_{wn}" for c in wide.columns if c != "HADM_ID"]
        lab_frames.append(wide)

    # (NEW) ARTERIAL BLOOD GAS (ABG) — pH, PaO2, PaCO2, HCO3, Base Excess, Lactate
    # ============================================================
    logger.info("Adding arterial blood gas (ABG) features...")

    abg_keywords = ["ph", "po2", "pco2", "hco3", "base excess", "lactate"]
    abg_itemids = {
        k: [iid for iid, label in lab_label_map.items() if any(w in label.lower() for w in [k])]
        for k in abg_keywords
    }

    abg_sum_acc = defaultdict(lambda: defaultdict(float))
    abg_cnt_acc = defaultdict(lambda: defaultdict(int))

    abg_iter = _read_csv_maybe_gz(
        LABEVENTS_FILE,
        usecols=["HADM_ID","ITEMID","CHARTTIME","VALUENUM"],
        parse_dates=["CHARTTIME"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for i, lb in enumerate(abg_iter, start=1):
        lb = lb.dropna(subset=["HADM_ID","ITEMID","CHARTTIME","VALUENUM"])
        lb["INTIME"] = lb["HADM_ID"].map(hadm_to_intime)
        lb = lb.dropna(subset=["INTIME"])
        lb["hours_since_intime"] = (lb["CHARTTIME"] - lb["INTIME"]).dt.total_seconds() / 3600.0
        lb = lb[(lb["hours_since_intime"] >= 0) & (lb["hours_since_intime"] < 24)]
        if lb.empty: continue

        lb["HOUR_BIN"] = hour_bin_from_delta_hours(lb["hours_since_intime"])
        for key, ids in abg_itemids.items():
            dfk = lb[lb["ITEMID"].isin(ids)]
            g = dfk.groupby(["HADM_ID","HOUR_BIN"])["VALUENUM"].agg(["sum","count"]).reset_index()
            for _, r in g.iterrows():
                hadm = int(r["HADM_ID"]); hb = int(r["HOUR_BIN"])
                wn = wname(hb, hb+1)
                abg_sum_acc[wn][(hadm, key)] += float(r["sum"])
                abg_cnt_acc[wn][(hadm, key)] += int(r["count"])

    abg_frames = []
    for wn in abg_sum_acc.keys():
        sdict, cdict = abg_sum_acc[wn], abg_cnt_acc[wn]
        if not sdict: continue
        keys = list(sdict.keys())
        dfw = pd.DataFrame({
            "HADM_ID": [k[0] for k in keys],
            "LAB":     [k[1] for k in keys],
            "MEAN":    [sdict[k] / max(cdict[k],1) for k in keys],
        })
        wide = dfw.pivot_table(index="HADM_ID", columns="LAB", values="MEAN").reset_index()
        wide.columns = ["HADM_ID"] + [f"{c}_mean_{wn}" for c in wide.columns if c != "HADM_ID"]
        abg_frames.append(wide)

    logger.info("Completed adding arterial blood gas (ABG) features.")

    # ============================================================
    # C) OUTPUTEVENTS — hourly sums; urine-specific & total  [UNCHANGED]
    # ============================================================
    output_itemid_to_label = load_output_itemid_map()
    out_sum_acc_total = defaultdict(lambda: defaultdict(float))   # wn -> {(hadm, "total_output_sum"): sum}
    out_sum_acc_urine = defaultdict(lambda: defaultdict(float))   # wn -> {(hadm, "urine_output_sum"): sum}

    if os.path.exists(OUTPUTEVENTS_FILE):
        out_iter = _read_csv_maybe_gz(
            OUTPUTEVENTS_FILE,
            usecols=["HADM_ID","ITEMID","CHARTTIME","VALUE"],  # ← no VALUENUM
            parse_dates=["CHARTTIME"],
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )

        out_rows = 0
        for i, oe in enumerate(out_iter, start=1):
            logger.info(f"[OUTPUTEVENTS] chunks={i:,} rows≈{i * CHUNK_SIZE:,}")

            # print(f"DEBUG: OUTPUTEVENTS chunk {i} rows={len(oe)}")
            # print("DEBUG: Columns ->", oe.columns.tolist()[:10])
            # print("DEBUG: Columns ->", list(oe.columns))
            # print("DEBUG: Sample ITEMIDs ->", oe["ITEMID"].dropna().unique()[:10])

            out_rows += len(oe)

            # one-pass: coerce + drop infs
            oe["VAL"] = pd.to_numeric(oe["VALUE"], errors="coerce")
            oe.loc[:, "VAL"] = oe["VAL"].replace([np.inf, -np.inf], np.nan)

            oe = oe.dropna(subset=["HADM_ID","CHARTTIME","VAL"])
            oe["INTIME"] = oe["HADM_ID"].map(hadm_to_intime)
            oe = oe.dropna(subset=["INTIME"])

            oe["hours_since_intime"] = (oe["CHARTTIME"] - oe["INTIME"]).dt.total_seconds() / 3600.0
            oe = oe[(oe["hours_since_intime"] >= 0) & (oe["hours_since_intime"] < 24)]
            if oe.empty:
                if MAX_CHUNKS and i >= MAX_CHUNKS: break
                continue
            oe["HOUR_BIN"] = hour_bin_from_delta_hours(oe["hours_since_intime"])

            if output_itemid_to_label:
                oe["is_urine"] = (
                    oe["ITEMID"].astype(float).fillna(-1).astype(int)
                    .map(output_itemid_to_label).fillna("").str.contains("urine", case=False)
                )
            else:
                oe["is_urine"] = False

            g = oe.groupby(["HADM_ID","HOUR_BIN"])["VAL"].sum().reset_index()
            for _, r in g.iterrows():
                hadm = int(r["HADM_ID"]); hb = int(r["HOUR_BIN"]); wn = wname(hb, hb+1)
                out_sum_acc_total[wn][(hadm, "total_output_sum")] += float(r["VAL"])

            gu = oe[oe["is_urine"]].groupby(["HADM_ID","HOUR_BIN"])["VAL"].sum().reset_index()
            for _, r in gu.iterrows():
                hadm = int(r["HADM_ID"]); hb = int(r["HOUR_BIN"]); wn = wname(hb, hb+1)
                out_sum_acc_urine[wn][(hadm, "urine_output_sum")] += float(r["VAL"])

            if i % 10 == 0:
                logger.info(f"[OUTPUTEVENTS] chunks={i:,} rows≈{out_rows:,}")
            if MAX_CHUNKS and i >= MAX_CHUNKS:
                break

    else:
        logger.warning("OUTPUTEVENTS not found; skipping outputs.")

    def make_out_frame(acc_dict):
        frames = []
        for wn in acc_dict.keys():
            sdict = acc_dict[wn]
            if not sdict:
                continue
            keys = list(sdict.keys())
            dfw = pd.DataFrame({
                "HADM_ID": [k[0] for k in keys],
                "OUTLBL":  [k[1] for k in keys],   # total_output_sum / urine_output_sum
                "SUM":     [sdict[k] for k in keys],
            })
            wide = dfw.pivot_table(index="HADM_ID", columns="OUTLBL", values="SUM").reset_index()
            wide.columns = ["HADM_ID"] + [f"{c}_{wn}" for c in wide.columns if c != "HADM_ID"]
            frames.append(wide)
        return frames

    out_frames_total = make_out_frame(out_sum_acc_total)
    out_frames_urine = make_out_frame(out_sum_acc_urine)

    # === derive 0–24h totals for outputs ===
    def derive_0_24_total_from_acc(acc_dict, out_name):
        rows = []
        for wn, sdict in acc_dict.items():
            pass  # just to touch wn
        # aggregate across all hour-bins per HADM_ID
        total_map = defaultdict(float)
        for sdict in acc_dict.values():
            for (hadm, lbl), val in sdict.items():
                if lbl == out_name:
                    total_map[hadm] += float(val)
        if not total_map:
            return None
        df = pd.DataFrame({"HADM_ID": list(total_map.keys()),
                           f"{out_name}_0_24h": list(total_map.values())})
        return df

    total_outputs_0_24 = derive_0_24_total_from_acc(out_sum_acc_total, "total_output_sum")
    urine_outputs_0_24 = derive_0_24_total_from_acc(out_sum_acc_urine, "urine_output_sum")

    # ============================================================
    # D) INPUTEVENTS_MV — vasopressors per-hour presence 
    # ============================================================
    pressor_itemids = load_pressor_itemids()
    any_pressors = any(len(s) > 0 for s in pressor_itemids.values())

    pressor_acc = defaultdict(lambda: defaultdict(int))  # wn -> {(hadm, 'dopamine_any'): 0/1}

    if os.path.exists(INPUTEVENTS_MV_FILE) and any_pressors:
        header_mv = _read_csv_maybe_gz(INPUTEVENTS_MV_FILE, nrows=0)
        cols = ["ICUSTAY_ID","HADM_ID","ITEMID","STARTTIME","ENDTIME","AMOUNT","RATE"]
        usecols = [c for c in cols if c in header_mv.columns]
        parse_dates = [c for c in ["STARTTIME","ENDTIME"] if c in header_mv.columns]

        inp_iter = _read_csv_maybe_gz(
            INPUTEVENTS_MV_FILE,
            usecols=usecols,
            parse_dates=parse_dates,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )
        inp_rows = 0
        for i, iv in enumerate(inp_iter, start=1):
            inp_rows += len(iv)
            icustay_ok = "ICUSTAY_ID" in iv.columns
            if icustay_ok:

                iv["HADM_MAP"] = iv["ICUSTAY_ID"].map(
                    lambda x: icu_map.get(int(x), {}).get("HADM_ID", np.nan) if pd.notna(x) else np.nan
                )
                iv["INTIME"] = iv["ICUSTAY_ID"].map(
                    lambda x: icu_map.get(int(x), {}).get("INTIME", pd.NaT) if pd.notna(x) else pd.NaT
                )

                msk = iv["INTIME"].isna()
                if "HADM_ID" in iv.columns:
                    iv.loc[msk, "INTIME"] = iv.loc[msk, "HADM_ID"].map(hadm_to_intime)
                    iv.loc[msk, "HADM_MAP"] = iv.loc[msk, "HADM_ID"]
                iv.rename(columns={"HADM_MAP":"HADM_ID"}, inplace=True)
            else:
                iv["INTIME"] = iv["HADM_ID"].map(hadm_to_intime)

            iv = safe_to_datetime(iv, [c for c in ["STARTTIME","ENDTIME"] if c in iv.columns])
            iv = iv.dropna(subset=["HADM_ID","INTIME","ITEMID"])
            if "STARTTIME" not in iv.columns or "ENDTIME" not in iv.columns:
                logger.warning("INPUTEVENTS_MV missing STARTTIME/ENDTIME; skipping pressors in MV.")
            else:
                iv["any_amt"] = False
                if "AMOUNT" in iv.columns:
                    iv["any_amt"] |= pd.to_numeric(iv["AMOUNT"], errors="coerce").fillna(0) > 0
                if "RATE" in iv.columns:
                    iv["any_amt"] |= pd.to_numeric(iv["RATE"], errors="coerce").fillna(0) > 0
                iv = iv[iv["any_amt"]]
                if iv.empty:
                    if MAX_CHUNKS and i >= MAX_CHUNKS: break
                    continue

                iv["start_delta_h"] = (iv["STARTTIME"] - iv["INTIME"]).dt.total_seconds() / 3600.0
                iv["end_delta_h"]   = (iv["ENDTIME"]   - iv["INTIME"]).dt.total_seconds() / 3600.0
                iv["start_h"] = np.floor(np.maximum(iv["start_delta_h"], 0)).astype(int)
                iv["end_h"]   = np.ceil(np.minimum(iv["end_delta_h"], 24)).astype(int)
                iv = iv[(iv["end_h"] > 0) & (iv["start_h"] < 24)]
                if iv.empty:
                    if MAX_CHUNKS and i >= MAX_CHUNKS: break
                    continue

                # after all preprocessing of iv (mapping INTIME, HADM_ID, etc.)
                iv = iv.loc[:, ~iv.columns.duplicated()]   # ✅ remove duplicate columns safely

                def mark_presence(row):
                    hadm_val = row.get("HADM_ID")
                    hadm = int(hadm_val) if pd.notna(hadm_val) and not isinstance(hadm_val, (pd.Series, list)) else None
                    start_h = max(0, int(row["start_h"]))
                    end_h   = min(24, int(row["end_h"]))
                    itemid = int(row["ITEMID"])
                    for drug, idset in pressor_itemids.items():
                        if itemid in idset and hadm is not None:
                            for hb in range(start_h, end_h):
                                wn = wname(hb, hb+1)
                                pressor_acc[wn][(hadm, f"{drug}_any")] = 1

                # now safely apply
                iv.apply(mark_presence, axis=1)


            if i % 10 == 0:
                logger.info(f"[INPUTEVENTS_MV pressors] chunks={i:,} rows≈{inp_rows:,}")
            if MAX_CHUNKS and i >= MAX_CHUNKS:
                break
    else:
        if not os.path.exists(INPUTEVENTS_MV_FILE):
            logger.warning("INPUTEVENTS_MV not found; skipping MV pressors.")
        elif not any_pressors:
            logger.warning("No vasopressor ITEMIDs identified; skipping MV pressors.")

    # === include CV pressors (esp. phenylephrine) ===
    if os.path.exists(INPUTEVENTS_CV_FILE) and any_pressors:
        header_cv = _read_csv_maybe_gz(INPUTEVENTS_CV_FILE, nrows=0)
        cols = ["HADM_ID","ICUSTAY_ID","ITEMID","CHARTTIME","AMOUNT","RATE","STARTTIME","ENDTIME"]
        usecols = [c for c in cols if c in header_cv.columns]
        parse_dates = [c for c in ["CHARTTIME","STARTTIME","ENDTIME"] if c in header_cv.columns]

        cv_iter = _read_csv_maybe_gz(
            INPUTEVENTS_CV_FILE,
            usecols=usecols,
            parse_dates=parse_dates,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )
        cv_rows = 0
        for i, cv in enumerate(cv_iter, start=1):
            cv_rows += len(cv)
            # Attach INTIME
            if "ICUSTAY_ID" in cv.columns:
                cv["HADM_MAP"] = cv["ICUSTAY_ID"].map(
                    lambda x: icu_map.get(int(x), {}).get("HADM_ID", np.nan)
                    if pd.notna(x) else np.nan
                )
                
                cv["INTIME"] = cv["ICUSTAY_ID"].map(
                    lambda x: icu_map.get(int(x), {}).get("INTIME", pd.NaT)
                    if pd.notna(x) else pd.NaT
                )

                msk = cv["INTIME"].isna()
                if "HADM_ID" in cv.columns:
                    cv.loc[msk, "INTIME"] = cv.loc[msk, "HADM_ID"].map(hadm_to_intime)
                    cv.loc[msk, "HADM_MAP"] = cv.loc[msk, "HADM_ID"]
                cv.rename(columns={"HADM_MAP":"HADM_ID"}, inplace=True)
            else:
                cv["INTIME"] = cv["HADM_ID"].map(hadm_to_intime)

            cv = cv.dropna(subset=["HADM_ID","INTIME","ITEMID"])

            # Use START/END if present; else use CHARTTIME as instantaneous
            has_interval = ("STARTTIME" in cv.columns) and ("ENDTIME" in cv.columns) and cv["STARTTIME"].notna().any() and cv["ENDTIME"].notna().any()
            if has_interval:
                cv = safe_to_datetime(cv, ["STARTTIME","ENDTIME"])
                cv = cv.dropna(subset=["STARTTIME","ENDTIME"])
                cv["any_amt"] = False
                if "AMOUNT" in cv.columns:
                    cv["any_amt"] |= pd.to_numeric(cv["AMOUNT"], errors="coerce").fillna(0) > 0
                if "RATE" in cv.columns:
                    cv["any_amt"] |= pd.to_numeric(cv["RATE"], errors="coerce").fillna(0) > 0
                cv = cv[cv["any_amt"]]
                if cv.empty:
                    if MAX_CHUNKS and i >= MAX_CHUNKS: break
                    continue

                cv["start_delta_h"] = (cv["STARTTIME"] - cv["INTIME"]).dt.total_seconds() / 3600.0
                cv["end_delta_h"]   = (cv["ENDTIME"]   - cv["INTIME"]).dt.total_seconds() / 3600.0
                cv["start_h"] = np.floor(np.maximum(cv["start_delta_h"], 0)).astype(int)
                cv["end_h"]   = np.ceil(np.minimum(cv["end_delta_h"], 24)).astype(int)
                cv = cv[(cv["end_h"] > 0) & (cv["start_h"] < 24)]
                if cv.empty:
                    if MAX_CHUNKS and i >= MAX_CHUNKS: break
                    continue

                for _, row in cv.iterrows():
                    hadm_val = row["HADM_ID"]
                    if isinstance(hadm_val, (pd.Series, list)) or pd.isna(hadm_val):
                        continue
                    hadm = int(hadm_val)

                    itemid = int(row["ITEMID"])
                    start_h = max(0, int(row["start_h"]))
                    end_h   = min(24, int(row["end_h"]))
                    for drug, idset in pressor_itemids.items():
                        if itemid in idset:
                            for hb in range(start_h, end_h):
                                wn = wname(hb, hb+1)
                                pressor_acc[wn][(hadm, f"{drug}_any")] = 1
            else:
                # Fallback instantaneous at CHARTTIME (mark that hour)
                if "CHARTTIME" not in cv.columns:
                    if MAX_CHUNKS and i >= MAX_CHUNKS: break
                    continue
                cv = safe_to_datetime(cv, ["CHARTTIME"])
                cv["hours_since_intime"] = (cv["CHARTTIME"] - cv["INTIME"]).dt.total_seconds() / 3600.0
                cv = cv[(cv["hours_since_intime"] >= 0) & (cv["hours_since_intime"] < 24)]
                if cv.empty:
                    if MAX_CHUNKS and i >= MAX_CHUNKS: break
                    continue
                cv["HOUR_BIN"] = hour_bin_from_delta_hours(cv["hours_since_intime"])


                for _, row in cv.iterrows():
                    hadm_val = row.get("HADM_ID")
                    if isinstance(hadm_val, (pd.Series, list)) or (hadm_val is None) or (isinstance(hadm_val, float) and np.isnan(hadm_val)):
                        continue
                    hadm = int(hadm_val)
                    itemid = int(row["ITEMID"])
                    hb = int(row["HOUR_BIN"])
                    wn = wname(hb, hb + 1)
                    for drug, idset in pressor_itemids.items():
                        if itemid in idset:
                            pressor_acc[wn][(hadm, f"{drug}_any")] = 1

                    # wn = wname(hb, hb+1)
                    # for drug, idset in pressor_itemids.items():
                    #     if itemid in idset:
                    #         pressor_acc[wn][(hadm, f"{drug}_any")] = 1

            if i % 10 == 0:
                logger.info(f"[INPUTEVENTS_CV pressors] chunks={i:,} rows≈{cv_rows:,}")
            if MAX_CHUNKS and i >= MAX_CHUNKS:
                break
    else:
        if not os.path.exists(INPUTEVENTS_CV_FILE):
            logger.warning("INPUTEVENTS_CV not found; skipping CV pressors.")

    pressor_frames = []
    for wn in pressor_acc.keys():
        sdict = pressor_acc[wn]
        if not sdict:
            continue
        keys = list(sdict.keys())
        dfw = pd.DataFrame({
            "HADM_ID": [k[0] for k in keys],
            "DRUG":    [k[1] for k in keys],  # e.g., dopamine_any
            "ANY":     [sdict[k] for k in keys],
        })
        wide = dfw.pivot_table(index="HADM_ID", columns="DRUG", values="ANY", aggfunc="max", fill_value=0).reset_index()
        wide.columns = ["HADM_ID"] + [f"{c}_{wn}" for c in wide.columns if c != "HADM_ID"]
        pressor_frames.append(wide)

    # === 0–24h derived any-pressor flag ===
    def derive_any_pressor_0_24(pressor_acc):
        any_map = defaultdict(int)
        for sdict in pressor_acc.values():
            for (hadm, drug_any), v in sdict.items():
                if v:
                    any_map[hadm] = 1
        if not any_map:
            return None
        return pd.DataFrame({"HADM_ID": list(any_map.keys()), "pressor_any_0_24h": list(any_map.values())})

    any_pressor_0_24_df = derive_any_pressor_0_24(pressor_acc)

    # ============================================================
    # FLUID INPUTS from INPUTEVENTS_MV + INPUTEVENTS_CV
    # ============================================================
    # Accumulate per-hour sums and mean rates; MV and CV combined
    input_sum_acc = defaultdict(lambda: defaultdict(float))  # wn -> {(hadm, "fluid_input_sum"): sum_ml}
    input_rate_acc_sum = defaultdict(lambda: defaultdict(float))  # wn -> {(hadm, "fluid_input_rate_mean"): sum_rates}
    input_rate_acc_cnt = defaultdict(lambda: defaultdict(int))    # wn -> {(hadm, "fluid_input_rate_mean"): count}

    def _accumulate_inputs_interval(df, source_label):
        # df must have: HADM_ID, INTIME, STARTTIME, ENDTIME, AMOUNT?, RATE?
        if df.empty: return
        df = df.copy()
        df["any_amt"] = False
        if "AMOUNT" in df.columns:
            df["any_amt"] |= pd.to_numeric(df["AMOUNT"], errors="coerce").fillna(0) > 0
        if "RATE" in df.columns:
            df["any_amt"] |= pd.to_numeric(df["RATE"], errors="coerce").fillna(0) > 0
        df = df[df["any_amt"]]
        if df.empty: return

        df["start_delta_h"] = (df["STARTTIME"] - df["INTIME"]).dt.total_seconds() / 3600.0
        df["end_delta_h"]   = (df["ENDTIME"]   - df["INTIME"]).dt.total_seconds() / 3600.0
        df["start_h"] = np.floor(np.maximum(df["start_delta_h"], 0)).astype(int)
        df["end_h"]   = np.ceil(np.minimum(df["end_delta_h"], 24)).astype(int)
        df = df[(df["end_h"] > 0) & (df["start_h"] < 24)]
        if df.empty: return

        for _, row in df.iterrows():
            hadm_val = row.get("HADM_ID")

            # robust skip guard
            if isinstance(hadm_val, (pd.Series, list)):
                continue
            if hadm_val is None or (isinstance(hadm_val, float) and np.isnan(hadm_val)):
                continue

            hadm = int(hadm_val)

            start_h = max(0, int(row["start_h"]))
            end_h   = min(24, int(row["end_h"]))
            dur_h = max(0.0, float(row["end_delta_h"] - row["start_delta_h"]))

            amt = 0.0
            if "AMOUNT" in df.columns and pd.notna(row.get("AMOUNT", np.nan)):
                amt = float(pd.to_numeric(row["AMOUNT"], errors="coerce") or 0.0)
            elif "RATE" in df.columns and pd.notna(row.get("RATE", np.nan)) and dur_h > 0:
                rate = float(pd.to_numeric(row["RATE"], errors="coerce") or 0.0)
                amt = max(0.0, rate * dur_h)
            if amt < 0 or dur_h <= 0:
                continue

            n_hours = max(1, end_h - start_h)
            per_hour = amt / n_hours
            for hb in range(start_h, end_h):
                wn = wname(hb, hb+1)
                input_sum_acc[wn][(hadm, "fluid_input_sum")] += per_hour
                if "RATE" in df.columns and pd.notna(row.get("RATE", np.nan)):
                    rate_val = float(pd.to_numeric(row["RATE"], errors="coerce") or 0.0)
                    input_rate_acc_sum[wn][(hadm, "fluid_input_rate_mean")] += rate_val
                    input_rate_acc_cnt[wn][(hadm, "fluid_input_rate_mean")] += 1


    def _accumulate_inputs_from_file(path, label):
        if not os.path.exists(path):
            logger.warning(f"{label} not found; skipping inputs from {label}.")
            return

        header = _read_csv_maybe_gz(path, nrows=0)

        # Helper: safe int for ICUSTAY_ID mapping
        def _safe_int(x):
            try:
                return int(x)
            except Exception:
                return None

        # Helper: flatten any weird HADM_ID cells (lists/Series/dicts)
        def _flatten_hadm(x):
            if isinstance(x, pd.Series):
                return x.iloc[0] if not x.empty else np.nan
            if isinstance(x, (list, tuple, np.ndarray)):
                return x[0] if len(x) > 0 else np.nan
            if isinstance(x, dict):
                return x.get("HADM_ID", np.nan)
            return x


        # ==================== INTERVAL PATH ====================
        if set(["STARTTIME", "ENDTIME"]).issubset(header.columns):
            usecols = [c for c in ["ICUSTAY_ID", "HADM_ID", "ITEMID", "STARTTIME", "ENDTIME", "AMOUNT", "RATE"] if c in header.columns]
            parse_dates = [c for c in ["STARTTIME", "ENDTIME"] if c in header.columns]
            it = _read_csv_maybe_gz(path, usecols=usecols, parse_dates=parse_dates,
                                    chunksize=CHUNK_SIZE, low_memory=False)
            rows = 0

            for i, df in enumerate(it, start=1):
                rows += len(df)

                # Attach INTIME/HADM via ICUSTAY_ID when present
                if "ICUSTAY_ID" in df.columns:
                    df["HADM_MAP"] = df["ICUSTAY_ID"].map(
                        lambda x: icu_map.get(_safe_int(x), {}).get("HADM_ID", np.nan)
                    )
                    df["INTIME"] = df["ICUSTAY_ID"].map(
                        lambda x: icu_map.get(_safe_int(x), {}).get("INTIME", pd.NaT)
                    )
                    msk = df["INTIME"].isna()
                    if "HADM_ID" in df.columns:
                        df.loc[msk, "INTIME"] = df.loc[msk, "HADM_ID"].map(hadm_to_intime)
                        df.loc[msk, "HADM_MAP"] = df.loc[msk, "HADM_ID"]
                    df.rename(columns={"HADM_MAP": "HADM_ID"}, inplace=True)
                else:
                    df["INTIME"] = df["HADM_ID"].map(hadm_to_intime)

                # --- FIX HADM_ID DIMENSION ISSUES (robust flatten) ---
                if "HADM_ID" in df.columns:
                    df["HADM_ID"] = [_flatten_hadm(v) for v in df["HADM_ID"].squeeze().to_numpy()]

                # Remove duplicate column names before any further ops
                df = df.loc[:, ~df.columns.duplicated()].copy()

                # Ensure HADM_ID is clean numeric int64
                df["HADM_ID"] = pd.to_numeric(df["HADM_ID"], errors="coerce")
                df = df.dropna(subset=["HADM_ID", "INTIME", "STARTTIME", "ENDTIME"]).copy()
                df["HADM_ID"] = df["HADM_ID"].astype("int64")

                df = safe_to_datetime(df, [c for c in ["STARTTIME", "ENDTIME"] if c in df.columns])

                _accumulate_inputs_interval(df, label)

                if i % 10 == 0:
                    logger.info(f"[{label} inputs] chunks={i:,} rows≈{rows:,}")
                if MAX_CHUNKS and i >= MAX_CHUNKS:
                    break

        # ==================== FALLBACK PATH ====================
        else:
            # CHARTTIME-only rows: treat as instantaneous 1-hour events
            usecols = [c for c in ["ICUSTAY_ID", "HADM_ID", "ITEMID", "CHARTTIME", "AMOUNT", "RATE"] if c in header.columns]
            parse_dates = [c for c in ["CHARTTIME"] if c in header.columns]
            it = _read_csv_maybe_gz(path, usecols=usecols, parse_dates=parse_dates,
                                    chunksize=CHUNK_SIZE, low_memory=False)
            rows = 0

            for i, df in enumerate(it, start=1):
                rows += len(df)

                if "ICUSTAY_ID" in df.columns:
                    df["HADM_MAP"] = df["ICUSTAY_ID"].map(
                        lambda x: icu_map.get(_safe_int(x), {}).get("HADM_ID", np.nan)
                    )
                    df["INTIME"] = df["ICUSTAY_ID"].map(
                        lambda x: icu_map.get(_safe_int(x), {}).get("INTIME", pd.NaT)
                    )
                    msk = df["INTIME"].isna()
                    if "HADM_ID" in df.columns:
                        df.loc[msk, "INTIME"] = df.loc[msk, "HADM_ID"].map(hadm_to_intime)
                        df.loc[msk, "HADM_MAP"] = df.loc[msk, "HADM_ID"]
                    df.rename(columns={"HADM_MAP": "HADM_ID"}, inplace=True)
                else:
                    df["INTIME"] = df["HADM_ID"].map(hadm_to_intime)

                # --- FIX HADM_ID DIMENSION ISSUES (robust flatten) ---
                if "HADM_ID" in df.columns:
                    df["HADM_ID"] = [_flatten_hadm(v) for v in df["HADM_ID"].squeeze().to_numpy()]


                # Remove duplicate column names before any further ops
                df = df.loc[:, ~df.columns.duplicated()].copy()

                # Ensure HADM_ID is clean numeric int64
                df["HADM_ID"] = pd.to_numeric(df["HADM_ID"], errors="coerce")
                df = df.dropna(subset=["HADM_ID", "INTIME", "CHARTTIME"]).copy()
                df["HADM_ID"] = df["HADM_ID"].astype("int64")

                # Hours and binning
                df["hours_since_intime"] = (df["CHARTTIME"] - df["INTIME"]).dt.total_seconds() / 3600.0
                df = df[(df["hours_since_intime"] >= 0) & (df["hours_since_intime"] < 24)]
                if df.empty:
                    if MAX_CHUNKS and i >= MAX_CHUNKS:
                        break
                    continue
                df["HOUR_BIN"] = hour_bin_from_delta_hours(df["hours_since_intime"])

                # Amount: prefer AMOUNT; else RATE (proxy for 1 hour)
                df["AMT_EFF"] = (
                    pd.to_numeric(df.get("AMOUNT", np.nan), errors="coerce")
                    .fillna(pd.to_numeric(df.get("RATE", np.nan), errors="coerce"))
                    .fillna(0.0)
                    .clip(lower=0.0)
                )

                # Per-hour input sum
                g = df.groupby(["HADM_ID", "HOUR_BIN"])["AMT_EFF"].sum().reset_index()
                for _, r in g.iterrows():
                    hadm = int(r["HADM_ID"]); hb = int(r["HOUR_BIN"]); wn = wname(hb, hb + 1)
                    input_sum_acc[wn][(hadm, "fluid_input_sum")] += float(r["AMT_EFF"])

                # Optional mean RATE
                if "RATE" in df.columns:
                    dfr = df.dropna(subset=["RATE"]).copy()
                    if not dfr.empty:
                        dfr["RATE_NUM"] = pd.to_numeric(dfr["RATE"], errors="coerce")
                        dfr = dfr.dropna(subset=["RATE_NUM"])
                        if not dfr.empty:
                            gr = dfr.groupby(["HADM_ID", "HOUR_BIN"])["RATE_NUM"].agg(["sum", "count"]).reset_index()
                            for _, r in gr.iterrows():
                                hadm = int(r["HADM_ID"]); hb = int(r["HOUR_BIN"]); wn = wname(hb, hb + 1)
                                input_rate_acc_sum[wn][(hadm, "fluid_input_rate_mean")] += float(r["sum"])
                                input_rate_acc_cnt[wn][(hadm, "fluid_input_rate_mean")] += int(r["count"])

                if i % 10 == 0:
                    logger.info(f"[{label} inputs fallback] chunks={i:,} rows≈{rows:,}")
                if MAX_CHUNKS and i >= MAX_CHUNKS:
                    break



    _accumulate_inputs_from_file(INPUTEVENTS_MV_FILE, "INPUTS_MV")
    _accumulate_inputs_from_file(INPUTEVENTS_CV_FILE, "INPUTS_CV")

    # Build input frames (per hour)
    input_frames = []
    for wn in input_sum_acc.keys():
        sdict = input_sum_acc[wn]
        if sdict:
            keys = list(sdict.keys())
            dfw = pd.DataFrame({
                "HADM_ID": [k[0] for k in keys],
                "LBL":     [k[1] for k in keys],  # "fluid_input_sum"
                "SUM":     [sdict[k] for k in keys],
            })
            wide = dfw.pivot_table(index="HADM_ID", columns="LBL", values="SUM").reset_index()
            wide.columns = ["HADM_ID"] + [f"{c}_{wn}" for c in wide.columns if c != "HADM_ID"]
            input_frames.append(wide)
        # rate means
        rsum = input_rate_acc_sum.get(wn, {})
        rcnt = input_rate_acc_cnt.get(wn, {})
        if rsum and rcnt:
            keys = list(rsum.keys())
            dfw = pd.DataFrame({
                "HADM_ID": [k[0] for k in keys],
                "LBL":     [k[1] for k in keys],  # "fluid_input_rate_mean"
                "MEAN":    [ (rsum[k] / max(rcnt.get(k, 1), 1)) for k in keys ],
            })
            wide = dfw.pivot_table(index="HADM_ID", columns="LBL", values="MEAN").reset_index()
            wide.columns = ["HADM_ID"] + [f"{c}_{wn}" for c in wide.columns if c != "HADM_ID"]
            input_frames.append(wide)

    # 0–24h total inputs ===
    def derive_inputs_0_24(acc_dict):
        total_map = defaultdict(float)
        for sdict in acc_dict.values():
            for (hadm, lbl), val in sdict.items():
                if lbl == "fluid_input_sum":
                    total_map[hadm] += float(val)
        if not total_map:
            return None
        return pd.DataFrame({"HADM_ID": list(total_map.keys()),
                             "fluid_input_sum_0_24h": list(total_map.values())})
    total_inputs_0_24_df = derive_inputs_0_24(input_sum_acc)

    # ============================================================
    # FLUID BALANCE per-hour and 0–24h
    # ============================================================
    # Compute after we have inputs and outputs hourly accumulators.
    balance_acc = defaultdict(lambda: defaultdict(float))  # wn -> {(hadm, "fluid_balance_sum"): sum}

    # Build quick dict of outputs per hour for total_output_sum
    outputs_total_by_wn_hadm = {}
    for wn, sdict in out_sum_acc_total.items():
        for (hadm, lbl), val in sdict.items():
            if lbl == "total_output_sum":
                outputs_total_by_wn_hadm[(wn, hadm)] = outputs_total_by_wn_hadm.get((wn, hadm), 0.0) + float(val)

    for wn, sdict in input_sum_acc.items():
        for (hadm, lbl), val in sdict.items():
            if lbl != "fluid_input_sum":
                continue
            in_val = float(val)
            out_val = float(outputs_total_by_wn_hadm.get((wn, hadm), 0.0))
            balance_acc[wn][(hadm, "fluid_balance_sum")] += (in_val - out_val)

    balance_frames = []
    for wn in balance_acc.keys():
        sdict = balance_acc[wn]
        if not sdict:
            continue
        keys = list(sdict.keys())
        dfw = pd.DataFrame({
            "HADM_ID": [k[0] for k in keys],
            "LBL":     [k[1] for k in keys],  # "fluid_balance_sum"
            "SUM":     [sdict[k] for k in keys],
        })
        wide = dfw.pivot_table(index="HADM_ID", columns="LBL", values="SUM").reset_index()
        wide.columns = ["HADM_ID"] + [f"{c}_{wn}" for c in wide.columns if c != "HADM_ID"]
        balance_frames.append(wide)

    # 0–24h fluid balance
    def derive_balance_0_24(balance_acc):
        total_map = defaultdict(float)
        for sdict in balance_acc.values():
            for (hadm, lbl), val in sdict.items():
                if lbl == "fluid_balance_sum":
                    total_map[hadm] += float(val)
        if not total_map:
            return None
        return pd.DataFrame({"HADM_ID": list(total_map.keys()),
                             "fluid_balance_sum_0_24h": list(total_map.values())})
    balance_0_24_df = derive_balance_0_24(balance_acc)

    # ============================================================
    # PROCEDUREEVENTS_MV — hourly procedure category flags
    # ============================================================
    proc_frames = []
    if os.path.exists(PROCEDUREEVENTS_MV_FILE):
        header_p = _read_csv_maybe_gz(PROCEDUREEVENTS_MV_FILE, nrows=0)
        cols = ["ICUSTAY_ID","HADM_ID","STARTTIME","ENDTIME","ORDERCATEGORYNAME"]
        usecols = [c for c in cols if c in header_p.columns]
        parse_dates = [c for c in ["STARTTIME","ENDTIME"] if c in header_p.columns]

        pit = _read_csv_maybe_gz(
            PROCEDUREEVENTS_MV_FILE,
            usecols=usecols,
            parse_dates=parse_dates,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )

        proc_acc = defaultdict(lambda: defaultdict(int))  # wn -> {(hadm, flag): 0/1}
        n_chunks, n_rows = 0, 0
        for i, pv in enumerate(pit, start=1):
            n_chunks += 1
            n_rows += len(pv)
            logger.info(f"[PROCEDUREEVENTS_MV] chunks={n_chunks} rows≈{n_rows}")

            # print(f"DEBUG: PROCEDURE chunk {i} rows={len(pv)}")
            # print("DEBUG: Columns ->", pv.columns.tolist()[:10])

            
            # === Combine text sources for robust tagging ===
            pv["INTIME"] = pv["HADM_ID"].map(hadm_to_intime)
            pv = pv.dropna(subset=["HADM_ID","INTIME"])


            title = (
                pv.get("ORDERCATEGORYDESCRIPTION", pd.Series("", index=pv.index)).astype(str) + " " +
                pv.get("ORDERCATEGORYNAME", pd.Series("", index=pv.index)).astype(str)
            ).str.lower()

            # print("DEBUG: sample TITLEs ->", title.head(10).tolist())


            pv["tag_vent"]       = title.str.contains(r"\bventilation\b", na=False)
            pv["tag_invasive"]   = title.str.contains("invasive", na=False)
            pv["tag_peripheral"] = title.str.contains("peripheral", na=False)
            pv["tag_imaging"]    = title.str.contains("imaging", na=False)

            # print(f"DEBUG: tag sums -> vent={pv['tag_vent'].sum()}, inv={pv['tag_invasive'].sum()}, periph={pv['tag_peripheral'].sum()}, img={pv['tag_imaging'].sum()}")


            pv = safe_to_datetime(pv, ["STARTTIME","ENDTIME"])

            # === Handle rows with/without ENDTIME ===
            with_end = pv["ENDTIME"].notna()

            # --- interval path ---
            pvi = pv[with_end].copy()
            if not pvi.empty:
                pvi["start_delta_h"] = (pvi["STARTTIME"] - pvi["INTIME"]).dt.total_seconds() / 3600.0
                pvi["end_delta_h"]   = (pvi["ENDTIME"]   - pvi["INTIME"]).dt.total_seconds() / 3600.0
                pvi["start_h"] = np.floor(np.maximum(pvi["start_delta_h"], 0)).astype(int)
                pvi["end_h"]   = np.ceil(np.minimum(pvi["end_delta_h"], 24)).astype(int)
                pvi = pvi[(pvi["end_h"] > 0) & (pvi["start_h"] < 24)]

                # print(f"DEBUG: interval usable rows = {len(pvi)}")

                for _, row in pvi.iterrows():
                    hadm = int(row["HADM_ID"])
                    for hb in range(max(0,int(row["start_h"])), min(24,int(row["end_h"]))):
                        wn = wname(hb, hb+1)
                        if row["tag_vent"]:       proc_acc[wn][(hadm, "ventilation_any")] = 1
                        if row["tag_invasive"]:   proc_acc[wn][(hadm, "invasive_line_any")] = 1
                        if row["tag_peripheral"]: proc_acc[wn][(hadm, "peripheral_line_any")] = 1
                        if row["tag_imaging"]:    proc_acc[wn][(hadm, "imaging_any")] = 1

            # --- instantaneous path (no ENDTIME) ---
            pvs = pv[~with_end].copy()
            if not pvs.empty:
                pvs["hours_since_intime"] = (pvs["STARTTIME"] - pvs["INTIME"]).dt.total_seconds() / 3600.0
                pvs = pvs[(pvs["hours_since_intime"] >= 0) & (pvs["hours_since_intime"] < 24)]

                # print(f"DEBUG: instant usable rows = {len(pvs)}")

                pvs["HOUR_BIN"] = hour_bin_from_delta_hours(pvs["hours_since_intime"])
                for _, row in pvs.iterrows():
                    hadm = int(row["HADM_ID"]); hb = int(row["HOUR_BIN"]); wn = wname(hb, hb+1)
                    if row["tag_vent"]:       proc_acc[wn][(hadm, "ventilation_any")] = 1
                    if row["tag_invasive"]:   proc_acc[wn][(hadm, "invasive_line_any")] = 1
                    if row["tag_peripheral"]: proc_acc[wn][(hadm, "peripheral_line_any")] = 1
                    if row["tag_imaging"]:    proc_acc[wn][(hadm, "imaging_any")] = 1


            if i % 10 == 0:
                logger.info(f"[PROCEDUREEVENTS_MV] chunks={i:,}")
            if MAX_CHUNKS and i >= MAX_CHUNKS:
                break
        
        # === DEBUG: check if proc_acc is populated ===
        # print("DEBUG: proc_acc time windows (wn):", list(proc_acc.keys())[:5])
        for wn in list(proc_acc.keys())[:2]:  # limit output
            sample_keys = list(proc_acc[wn].keys())[:5]
            # print(f"DEBUG: [{wn}] sample keys:", sample_keys)
            # print(f"DEBUG: [{wn}] tag values:", [proc_acc[wn][k] for k in sample_keys])


        # Convert accumulated dicts to per-hour dataframes
        for wn in proc_acc.keys():
            sdict = proc_acc[wn]
            if not sdict: continue
            keys = list(sdict.keys())
            dfw = pd.DataFrame({
                "HADM_ID": [k[0] for k in keys],
                "PROC":    [k[1] for k in keys],
                "ANY":     [sdict[k] for k in keys],
            })
            wide = dfw.pivot_table(index="HADM_ID", columns="PROC", values="ANY", aggfunc="max", fill_value=0).reset_index()
            wide.columns = ["HADM_ID"] + [f"{c}_{wn}" for c in wide.columns if c != "HADM_ID"]
            proc_frames.append(wide)

    else:
        logger.warning("PROCEDUREEVENTS_MV not found; skipping procedures.")

   

    # ============================================================
    # Merge all hourly feature frames  [EXTENDS V8 with new frames]
    # ============================================================
    feature_frames = []
    # A) Chart vitals
    feature_frames.extend(chart_frames)
    # Respiratory (FiO2, PEEP, Ventilation)
    feature_frames.extend(resp_frames)
    # Oxygenation indices (S/F)
    feature_frames.extend(ox_frames)

    feature_frames.extend(vent_frames)


    # B) Labs
    feature_frames.extend(lab_frames)
    # C) Outputs (total + urine)
    feature_frames.extend(out_frames_total)
    feature_frames.extend(out_frames_urine)
    # D) Pressors (MV + CV, extended)
    feature_frames.extend(pressor_frames)
    # Inputs, Balance, Procedures
    feature_frames.extend(input_frames)
    feature_frames.extend(balance_frames)
    feature_frames.extend(proc_frames)

    features = feature_frames[0] if feature_frames else pd.DataFrame(columns=["HADM_ID"])
    for w in feature_frames[1:]:
        features = features.merge(w, on="HADM_ID", how="outer")

    # === Merge static + labels ===  [UNCHANGED]
    # === Merge static + labels (use static_cols as base to guarantee rows) ===
    final = static_cols.merge(features, on="HADM_ID", how="left")
    final = final.merge(
        icux[["HADM_ID","LOS_label","LOS_label_3class","LOS_DAYS"]],
        on="HADM_ID", how="left"
    )


    # append 0–24h derived totals/flags
    # inputs
    if total_inputs_0_24_df is not None:
        final = final.merge(total_inputs_0_24_df, on="HADM_ID", how="left")
    # outputs
    if total_outputs_0_24 is not None:
        final = final.merge(total_outputs_0_24, on="HADM_ID", how="left")
    if urine_outputs_0_24 is not None:
        final = final.merge(urine_outputs_0_24, on="HADM_ID", how="left")
    # pressor any
    if any_pressor_0_24_df is not None:
        final = final.merge(any_pressor_0_24_df, on="HADM_ID", how="left")
    # balance
    if balance_0_24_df is not None:
        final = final.merge(balance_0_24_df, on="HADM_ID", how="left")
    # medications any 0–24h
    if med_any_0_24_df is not None:
        final = final.merge(med_any_0_24_df, on="HADM_ID", how="left")
    # oxygenation
    if s_f_min_0_24_df is not None:
        final = final.merge(s_f_min_0_24_df, on="HADM_ID", how="left")

    # === Derived: total hours ventilated in first 24h ===
    logger.info("Computing total ventilator hours (0–24h)...")
    vent_cols = [c for c in final.columns if c.startswith("ventilation_any_") and c.endswith("h")]
    if vent_cols:
        final["vent_hours_0_24h"] = final[vent_cols].sum(axis=1).astype("int16")
        logger.info(f"Added vent_hours_0_24h from {len(vent_cols)} hourly flags.")
    else:
        logger.warning("No ventilation_any_* hourly columns found — skipped vent_hours_0_24h.")


    # Reorder columns: static, time features, labels  [FOLLOW V8]
    label_cols   = ["LOS_label","LOS_label_3class","LOS_DAYS"]
    static_order = ["HADM_ID","AGE","GENDER","ADMISSION_TYPE"]
    other_cols   = [c for c in final.columns if c not in (static_order + label_cols)]
    final = final[static_order + other_cols + label_cols]

    # Clean NaN/Inf to keep downstream readers happy
    final.replace([np.inf, -np.inf], np.nan, inplace=True)


    # Save
    final.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Saved: {OUTPUT_CSV}  Rows={len(final):,}  Cols={final.shape[1]}")
    logger.info(f"---- LOS build V12.1 done ----  Elapsed={time.time()-start_t:.1f}s")

if __name__ == "__main__":
    main()
