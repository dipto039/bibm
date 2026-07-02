#!/usr/bin/env python3
"""
dataset_stats.py
-----------------
Reports train/valid/test split sizes and per-class counts (binary + 3-class
LOS labels) from the preprocessed MIMIC-III tensors -- a quick sanity check
on cohort size and class balance without needing to reload the full X_dyn/
X_static arrays.

Run after V12.1_preprocess_script.py has produced
processed_data/structured_LOS_dynamic/lstm_data_v12.1_seq/:

    python dataset_stats.py
"""

import os
import argparse
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
_DEFAULT_DATA_DIR = os.path.join(
    _PROJECT_ROOT, "processed_data", "structured_LOS_dynamic", "lstm_data_v12.1_seq"
)

BINARY_LABELS = {0: "LOS<=7 days", 1: "LOS>7 days"}
MULTICLASS_LABELS = {0: "<3 days", 1: "3-7 days", 2: ">=7 days"}

SPLITS = ["train", "valid", "test"]


def load_split_labels(data_dir: str, label_prefix: str, split: str) -> np.ndarray:
    path = os.path.join(data_dir, f"{label_prefix}_{split}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path} -- did V12.1_preprocess_script.py finish?")
    return np.load(path)


def print_class_table(label_prefix: str, label_names: dict, data_dir: str):
    print(f"\n=== {label_prefix} labels ===")
    n_classes = len(label_names)

    header = f"{'split':<8}{'total':>10}"
    for c in range(n_classes):
        header += f"{label_names[c]:>16}"
    print(header)
    print("-" * len(header))

    grand_total = 0
    grand_counts = np.zeros(n_classes, dtype=np.int64)

    for split in SPLITS:
        y = load_split_labels(data_dir, label_prefix, split)
        total = len(y)
        counts = np.bincount(y, minlength=n_classes)
        grand_total += total
        grand_counts += counts

        row = f"{split:<8}{total:>10,}"
        for c in range(n_classes):
            pct = 100.0 * counts[c] / total if total else 0.0
            row += f"{counts[c]:>9,} ({pct:4.1f}%)"
        print(row)

    row = f"{'ALL':<8}{grand_total:>10,}"
    for c in range(n_classes):
        pct = 100.0 * grand_counts[c] / grand_total if grand_total else 0.0
        row += f"{grand_counts[c]:>9,} ({pct:4.1f}%)"
    print("-" * len(header))
    print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=_DEFAULT_DATA_DIR)
    args = ap.parse_args()

    print(f"Reading from: {args.data_dir}")

    # Split sizes (admissions, not unique patients -- a patient can have
    # multiple admissions, all routed to the same split, see
    # los_split_utils.patient_level_split_indices).
    print("\n=== Split sizes (admissions) ===")
    total_n = 0
    for split in SPLITS:
        y = load_split_labels(args.data_dir, "y_bin", split)
        total_n += len(y)
        print(f"  {split:<8}{len(y):>10,}")
    print(f"  {'ALL':<8}{total_n:>10,}")

    print_class_table("y_bin", BINARY_LABELS, args.data_dir)
    print_class_table("y_3c", MULTICLASS_LABELS, args.data_dir)


if __name__ == "__main__":
    main()
