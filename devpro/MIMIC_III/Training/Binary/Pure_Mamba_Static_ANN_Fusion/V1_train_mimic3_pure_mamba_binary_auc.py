#!/usr/bin/env python3
"""
V1_train_mimic3_pure_mamba_binary_auc.py

MIMIC-III Pure BiMamba + Static ANN Fusion Binary LOS prediction.
"""

import os
import json
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from mamba_ssm.modules.mamba_simple import Mamba

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from model_common import build_standard_head, build_standard_optimizer, build_standard_scheduler, STANDARD_HPARAMS

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", "..", "..", ".."))

BASE = os.path.join(_PROJECT_ROOT, "processed_data", "structured_LOS_dynamic")
DATA_DIR = f"{BASE}/lstm_data_v12.1_seq"
OUT_DIR = os.path.join(_PROJECT_ROOT, "results_LOS_dynamic", "mimic3_pure_mamba_binary_auc_models_v1")
os.makedirs(OUT_DIR, exist_ok=True)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def finite_np(x) -> bool:
    return x is not None and np.isfinite(x).all()


class DynStaticBinaryDataset(Dataset):
    def __init__(self, X_dyn, X_static, y_bin):
        self.Xd = X_dyn
        self.Xs = X_static
        self.y = y_bin.astype(np.int64)

    def __len__(self):
        return self.Xd.shape[0]

    def __getitem__(self, idx):
        xd = torch.from_numpy(np.array(self.Xd[idx], dtype=np.float32, copy=True))
        xs = torch.from_numpy(np.array(self.Xs[idx], dtype=np.float32, copy=True))
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return xd, xs, y


def make_loaders(Xd_tr, Xs_tr, y_tr, Xd_va, Xs_va, y_va, Xd_te, Xs_te, y_te, batch, workers):
    common = dict(batch_size=batch, num_workers=workers, pin_memory=True, persistent_workers=(workers > 0))
    return (
        DataLoader(DynStaticBinaryDataset(Xd_tr, Xs_tr, y_tr), shuffle=True, **common),
        DataLoader(DynStaticBinaryDataset(Xd_va, Xs_va, y_va), shuffle=False, **common),
        DataLoader(DynStaticBinaryDataset(Xd_te, Xs_te, y_te), shuffle=False, **common),
    )


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w


class BiMambaEncoder(nn.Module):
    def __init__(self, f_dyn, d_model=192, layers=3, d_state=16, d_conv=4, expand=2, resid_scale=0.5):
        super().__init__()
        self.proj = nn.Linear(f_dyn, d_model)
        self.blocks = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(layers)
        ])
        self.norm = RMSNorm(d_model)
        self.resid_scale = float(resid_scale)

    def encode(self, x):
        h = self.proj(x)
        for block in self.blocks:
            h = h + self.resid_scale * block(h)
            h = self.norm(h)
        return h

    def forward(self, x):
        f = self.encode(x)
        b = self.encode(torch.flip(x, dims=[1]))
        return torch.cat([f[:, -1], b[:, -1], f.mean(dim=1), b.mean(dim=1)], dim=1)


class PureMambaBinaryModel(nn.Module):
    def __init__(self, f_dyn, f_static, d_model=192, layers=3):
        super().__init__()

        self.dyn = BiMambaEncoder(
            f_dyn=f_dyn,
            d_model=d_model,
            layers=layers,
            d_state=16,
            d_conv=4,
            expand=2,
            resid_scale=0.5,
        )

        self.stat = nn.Sequential(
            nn.Linear(f_static, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )

        self.head = build_standard_head(4 * d_model + 128, n_classes=2)

    def forward(self, xd, xs):
        h = torch.cat([self.dyn(xd), self.stat(xs)], dim=1)
        return self.head(h)


def tune_binary_threshold(y_true, prob_pos):
    best = {"threshold": 0.5, "f1": -1.0, "acc": 0.0}
    for t in np.arange(0.05, 0.96, 0.01):
        pred = (prob_pos >= t).astype(np.int64)
        f1 = f1_score(y_true, pred, average="macro", zero_division=0)
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": float(f1), "acc": float(accuracy_score(y_true, pred))}
    return best


def run_epoch(model, loader, device, criterion, optimizer=None, grad_clip=1.0, amp=False, scaler=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    loss_sum = 0.0
    n = 0
    probs_all = []
    y_all = []
    invalid_eval = False

    for xd, xs, y in loader:
        xd = xd.to(device, non_blocking=True)
        xs = xs.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        if amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(xd, xs)
                loss = criterion(logits, y)
        else:
            logits = model(xd, xs)
            loss = criterion(logits, y)

        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            if training:
                raise RuntimeError("Training produced NaN/Inf logits or loss.")
            invalid_eval = True
            break

        if training:
            if amp and device.type == "cuda" and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        bs = y.size(0)
        loss_sum += float(loss.detach().cpu().item()) * bs
        n += bs

        if not training:
            probs = torch.softmax(logits, dim=1)
            if not torch.isfinite(probs).all():
                invalid_eval = True
                break
            probs_all.append(probs.detach().cpu().numpy())
            y_all.append(y.detach().cpu().numpy())

    if training:
        return loss_sum / max(n, 1), None, None, True

    if invalid_eval or not probs_all:
        return float("nan"), np.full((0, 2), np.nan), np.array([]), False

    probs = np.concatenate(probs_all)
    y_true = np.concatenate(y_all)

    if not np.isfinite(probs).all():
        return float("nan"), probs, y_true, False

    return loss_sum / max(n, 1), probs, y_true, True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=STANDARD_HPARAMS["max_epochs"])
    parser.add_argument("--batch", type=int, default=STANDARD_HPARAMS["batch_size"])
    parser.add_argument("--lr", type=float, default=STANDARD_HPARAMS["lr"])
    parser.add_argument("--patience", type=int, default=STANDARD_HPARAMS["patience"])
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=STANDARD_HPARAMS["grad_clip"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=3)

    args = parser.parse_args()

    seed_everything(42)

    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")

    print("USING MIMIC-III PURE BIMAMBA + STATIC ANN FUSION BINARY WITH AUROC/AUPRC")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"Device: {device}")

    Xd_tr = np.load(f"{DATA_DIR}/X_dyn_train.npy", mmap_mode="r")
    Xd_va = np.load(f"{DATA_DIR}/X_dyn_valid.npy", mmap_mode="r")
    Xd_te = np.load(f"{DATA_DIR}/X_dyn_test.npy", mmap_mode="r")

    Xs_tr = np.load(f"{DATA_DIR}/X_static_train.npy")
    Xs_va = np.load(f"{DATA_DIR}/X_static_valid.npy")
    Xs_te = np.load(f"{DATA_DIR}/X_static_test.npy")

    y_tr = np.load(f"{DATA_DIR}/y_bin_train.npy")
    y_va = np.load(f"{DATA_DIR}/y_bin_valid.npy")
    y_te = np.load(f"{DATA_DIR}/y_bin_test.npy")

    print(f"Loaded dyn: train={Xd_tr.shape}, valid={Xd_va.shape}, test={Xd_te.shape}")
    print(f"Loaded static: train={Xs_tr.shape}, valid={Xs_va.shape}, test={Xs_te.shape}")

    class_counts = np.bincount(y_tr, minlength=2)
    print("Train class counts:", class_counts)

    weights = class_counts.sum() / (class_counts + 1e-9)
    weights = weights / weights.sum()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    print("Using weighted CrossEntropy with validation threshold tuning.")
    print("Class weights:", weights.tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    tr_loader, va_loader, te_loader = make_loaders(
        Xd_tr, Xs_tr, y_tr,
        Xd_va, Xs_va, y_va,
        Xd_te, Xs_te, y_te,
        batch=args.batch,
        workers=args.workers,
    )

    model = PureMambaBinaryModel(
        f_dyn=Xd_tr.shape[2],
        f_static=Xs_tr.shape[1],
        d_model=args.d_model,
        layers=args.layers,
    ).to(device)

    optimizer = build_standard_optimizer(model, lr=args.lr)
    scheduler = build_standard_scheduler(optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_val_f1 = -1.0
    best_val_loss = None
    best_state = None
    best_epoch = -1
    best_threshold = 0.5

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, _, _, _ = run_epoch(
            model, tr_loader, device, criterion,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
            amp=args.amp,
            scaler=scaler,
        )

        va_loss, va_probs, va_true, valid_eval = run_epoch(
            model, va_loader, device, criterion,
            optimizer=None,
            grad_clip=args.grad_clip,
            amp=args.amp,
            scaler=None,
        )

        if (
            not valid_eval
            or not np.isfinite(va_loss)
            or va_probs.shape[0] == 0
            or not finite_np(va_probs)
            or not finite_np(va_true)
        ):
            print(
                f"Epoch {epoch:02d} | train {tr_loss:.4f} | "
                f"val NaN/Inf detected — skipping checkpoint/threshold update"
            )

            if best_epoch != -1 and (epoch - best_epoch) >= args.patience:
                print("Early stopping.")
                break
            continue

        info = tune_binary_threshold(va_true, va_probs[:, 1])
        scheduler.step(info["f1"])

        print(
            f"Epoch {epoch:02d} | train {tr_loss:.4f} | val {va_loss:.4f} | "
            f"F1 {info['f1']:.4f} | Acc {info['acc']:.4f} | thr={info['threshold']:.2f}"
        )

        if info["f1"] > best_val_f1:
            best_val_f1 = float(info["f1"])
            best_val_loss = float(va_loss)
            best_epoch = epoch
            best_threshold = float(info["threshold"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        elif best_epoch != -1 and (epoch - best_epoch) >= args.patience:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError(
            "No valid checkpoint produced. Mamba eval stayed NaN/Inf. "
            "Try: --lr 3e-4 --grad_clip 0.5 --layers 2"
        )

    model.load_state_dict(best_state)

    te_loss, te_probs, te_true, valid_test = run_epoch(
        model, te_loader, device, criterion,
        optimizer=None,
        grad_clip=args.grad_clip,
        amp=args.amp,
        scaler=None,
    )

    if (
        not valid_test
        or not np.isfinite(te_loss)
        or te_probs.shape[0] == 0
        or not finite_np(te_probs)
        or not finite_np(te_true)
    ):
        raise RuntimeError(
            "Test produced NaN/Inf probabilities. "
            "Do not report this run. Try: --lr 3e-4 --grad_clip 0.5 --layers 2"
        )

    te_pred = (te_probs[:, 1] >= best_threshold).astype(np.int64)

    te_acc = accuracy_score(te_true, te_pred)
    te_f1_macro = f1_score(te_true, te_pred, average="macro", zero_division=0)
    te_f1_weighted = f1_score(te_true, te_pred, average="weighted", zero_division=0)
    te_auroc = roc_auc_score(te_true, te_probs[:, 1])
    te_auprc = average_precision_score(te_true, te_probs[:, 1])

    rep = classification_report(te_true, te_pred, digits=4, output_dict=True, zero_division=0)
    cm = confusion_matrix(te_true, te_pred).tolist()

    stamp = (
        f"mimic3_pure_mamba_binary_"
        f"ep{best_epoch}_D{args.d_model}_L{args.layers}_"
        f"Fdyn{Xd_tr.shape[2]}_Fstat{Xs_tr.shape[1]}"
    )

    model_path = os.path.join(OUT_DIR, f"pure_mamba_binary_{stamp}.pt")
    metrics_path = os.path.join(OUT_DIR, f"metrics_{stamp}.json")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dyn_dim": int(Xd_tr.shape[2]),
            "input_static_dim": int(Xs_tr.shape[1]),
            "d_model": int(args.d_model),
            "layers": int(args.layers),
            "num_classes": 2,
            "label": "binary",
            "best_epoch": int(best_epoch),
            "best_val_f1": float(best_val_f1),
            "best_val_loss": float(best_val_loss),
            "best_threshold": float(best_threshold),
            "model": "Pure BiMamba + static ANN fusion binary",
            "test_auroc": float(te_auroc),
            "test_auprc": float(te_auprc),
        },
        model_path,
    )

    metrics = {
        "dataset": "MIMIC-III",
        "model": "Pure BiMamba + static ANN fusion binary",
        "label": "binary",
        "test_loss": float(te_loss),
        "test_acc": float(te_acc),
        "test_f1_macro": float(te_f1_macro),
        "test_f1_weighted": float(te_f1_weighted),
        "test_auroc": float(te_auroc),
        "test_auprc": float(te_auprc),
        "classification_report": rep,
        "confusion_matrix": cm,
        "best_epoch": int(best_epoch),
        "best_val_f1": float(best_val_f1),
        "best_val_loss": float(best_val_loss),
        "best_threshold": float(best_threshold),
        "elapsed_sec": round(time.time() - t0, 1),
        "class_counts_train": class_counts.tolist(),
        "train_shape_dyn": list(Xd_tr.shape),
        "valid_shape_dyn": list(Xd_va.shape),
        "test_shape_dyn": list(Xd_te.shape),
        "train_shape_static": list(Xs_tr.shape),
        "input_dyn_dim": int(Xd_tr.shape[2]),
        "input_static_dim": int(Xs_tr.shape[1]),
        "seq_len": int(Xd_tr.shape[1]),
        "d_model": int(args.d_model),
        "layers": int(args.layers),
        "checkpoint_metric": "validation_macro_f1",
        "binary_threshold_tuned_on": "validation_macro_f1",
        "output_dir": OUT_DIR,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model:   {model_path}")
    print(f"Saved metrics: {metrics_path}")

    print(
        f"Test Acc: {te_acc:.4f} | "
        f"F1 macro: {te_f1_macro:.4f} | "
        f"F1 weighted: {te_f1_weighted:.4f} | "
        f"AUROC: {te_auroc:.4f} | "
        f"AUPRC: {te_auprc:.4f} | "
        f"Test loss: {te_loss:.4f} | "
        f"Best epoch: {best_epoch} | "
        f"Best val F1: {best_val_f1:.4f} | "
        f"Threshold: {best_threshold:.2f}"
    )


if __name__ == "__main__":
    main()