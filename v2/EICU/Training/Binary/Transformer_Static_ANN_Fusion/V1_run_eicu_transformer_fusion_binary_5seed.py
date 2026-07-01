#!/usr/bin/env python3
"""
V1_run_eicu_transformer_fusion_binary_5seed.py

5-seed runner for Transformer + Static ANN Fusion Binary.

Reuses:
    V1_train_eicu_transformer_fusion_binary_auc.py

Creates seed-specific output folders:
    /lustre/home/rahas2/mimic_projects/outputs/eicu_transformer_fusion_binary_auc_5seed_models_v1/seed_<seed>

Reports:
    Acc, Macro F1, Weighted F1, AUROC, AUPRC

Run:
    python V1_run_eicu_transformer_fusion_binary_5seed.py --workers 0 --batch 128
"""

import os
import sys
import json
import glob
import argparse
import subprocess
from pathlib import Path

import numpy as np


BASE_OUT = "/lustre/home/rahas2/mimic_projects/outputs"
RUN_ROOT = f"{BASE_OUT}/eicu_transformer_fusion_binary_auc_5seed_models_v1"
SOURCE_SCRIPT = "V1_train_eicu_transformer_fusion_binary_auc.py"


def latest_metrics_json(folder: str) -> str:
    files = glob.glob(os.path.join(folder, "metrics_*.json"))
    if not files:
        raise FileNotFoundError(f"No metrics_*.json found in {folder}")
    return max(files, key=os.path.getmtime)


def patch_script(src_text: str, seed: int, seed_out_dir: str) -> str:
    text = src_text

    text = text.replace(
        'OUT_DIR = f"{BASE}/eicu_transformer_fusion_binary_auc_models_v1"',
        f'OUT_DIR = "{seed_out_dir}"'
    )

    text = text.replace(
        "seed_everything(42)",
        f"seed_everything({seed})"
    )

    text = text.replace(
        'print("USING eICU TRANSFORMER + ANN FUSION BINARY WITH AUROC/AUPRC")',
        f'print("USING eICU TRANSFORMER + ANN FUSION BINARY WITH AUROC/AUPRC | SEED {seed}")'
    )

    return text


def summarize(all_metrics, summary_dir):
    weighted_vals = [
        m["classification_report"]["weighted avg"]["f1-score"]
        for m in all_metrics
    ]

    summary = {
        "model": "Transformer + Static ANN Fusion Binary",
        "task": "eICU binary LOS >7d",
        "seeds": [m["seed"] for m in all_metrics],
        "per_seed": all_metrics,
        "mean_std": {},
    }

    for key in [
        "test_acc",
        "test_f1_macro",
        "test_auroc",
        "test_auprc",
    ]:
        vals = np.array([m[key] for m in all_metrics], dtype=float)
        summary["mean_std"][key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)),
            "values": vals.tolist(),
        }

    vals = np.array(weighted_vals, dtype=float)
    summary["mean_std"]["test_f1_weighted"] = {
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)),
        "values": vals.tolist(),
    }

    json_path = os.path.join(summary_dir, "summary_transformer_binary_5seed.json")
    txt_path = os.path.join(summary_dir, "summary_transformer_binary_5seed.txt")

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "Transformer + Static ANN Fusion Binary | 5-Seed Summary",
        "",
    ]

    for key, obj in summary["mean_std"].items():
        lines.append(f"{key}: {obj['mean']:.4f} ± {obj['std']:.4f}")

    lines.append("")
    lines.append("Per-seed:")

    for m in all_metrics:
        weighted_f1 = m["classification_report"]["weighted avg"]["f1-score"]
        lines.append(
            f"seed={m['seed']} | "
            f"acc={m['test_acc']:.4f} | "
            f"f1_macro={m['test_f1_macro']:.4f} | "
            f"f1_weighted={weighted_f1:.4f} | "
            f"auroc={m['test_auroc']:.4f} | "
            f"auprc={m['test_auprc']:.4f}"
        )

    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return json_path, txt_path, summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    source_path = script_dir / SOURCE_SCRIPT

    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing {SOURCE_SCRIPT} in {script_dir}. "
            f"Put this runner in the same folder as {SOURCE_SCRIPT}."
        )

    os.makedirs(RUN_ROOT, exist_ok=True)

    src_text = source_path.read_text()
    all_metrics = []

    for seed in args.seeds:
        seed_out = os.path.join(RUN_ROOT, f"seed_{seed}")
        os.makedirs(seed_out, exist_ok=True)

        tmp_script = script_dir / f"_tmp_transformer_binary_auc_seed{seed}.py"
        tmp_script.write_text(patch_script(src_text, seed, seed_out))

        cmd = [
            sys.executable,
            str(tmp_script),
            "--workers", str(args.workers),
            "--batch", str(args.batch),
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--lr", str(args.lr),
        ]

        if args.amp:
            cmd.append("--amp")

        print("\n" + "=" * 80)
        print(f"RUNNING SEED {seed}")
        print("=" * 80)
        print(" ".join(cmd))

        try:
            subprocess.run(cmd, check=True)
        finally:
            try:
                tmp_script.unlink()
            except FileNotFoundError:
                pass

        metrics_path = latest_metrics_json(seed_out)

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        metrics["seed"] = int(seed)
        metrics["metrics_path"] = metrics_path
        all_metrics.append(metrics)

    json_path, txt_path, summary = summarize(all_metrics, RUN_ROOT)

    print("\n" + "=" * 80)
    print("FINAL 5-SEED SUMMARY")
    print("=" * 80)

    for key, obj in summary["mean_std"].items():
        print(f"{key}: {obj['mean']:.4f} ± {obj['std']:.4f}")

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved TXT:  {txt_path}")


if __name__ == "__main__":
    main()
