#!/usr/bin/env python3
"""
V1_train_eicu_hybrid_lstm_mamba_ordinal_auc.py
------------------------------------------

Ordinal eICU 3-class LOS training script using the same dynamic/static architecture
family as the eICU Hybrid BiLSTM + BiMamba + static ANN fusion model.

This script is ONLY for the ordered 3-class LOS task:
    class 0: LOS <= 3 days
    class 1: 3 < LOS <= 7 days
    class 2: LOS > 7 days

Ordinal formulation:
    logit 0 learns P(LOS > 3 days)
    logit 1 learns P(LOS > 7 days)

Targets:
    y = 0 -> [0, 0]
    y = 1 -> [1, 0]
    y = 2 -> [1, 1]

Prediction:
    class = I(sigmoid(logit0) >= t0) + I(sigmoid(logit1) >= t1)

The script tunes t0 and t1 on validation macro-F1 and applies the best thresholds
to the test set.

Reads the SAME files as the other eICU scripts:
    /lustre/home/rahas2/mimic_projects/outputs/eicu_lstm_data_v1

Required files:
    X_dyn_train.npy / valid / test
    X_static_train.npy / valid / test
    y_3c_train.npy / valid / test

Requires:
    mamba-ssm
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

from mamba_ssm.modules.mamba_simple import Mamba  # hard fail if missing


# ======================================================
# CONFIG
# ======================================================

BASE = "/lustre/home/rahas2/mimic_projects/outputs"
DATA_DIR = f"{BASE}/eicu_lstm_data_v1"
OUT_DIR = f"{BASE}/eicu_hybrid_lstm_mamba_ordinal_auc_models_v1"
os.makedirs(OUT_DIR, exist_ok=True)


# ======================================================
# SEED
# ======================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ======================================================
# DATASET
# ======================================================

class DynStaticOrdinalDataset(Dataset):
    def __init__(self, X_dyn, X_static, y_3c, copy_mmap: bool = True):
        self.Xd = X_dyn
        self.Xs = X_static
        self.y = y_3c.astype(np.int64)
        self.copy_mmap = copy_mmap

    def __len__(self):
        return self.Xd.shape[0]

    def __getitem__(self, idx):
        if self.copy_mmap:
            xd = torch.from_numpy(np.array(self.Xd[idx], dtype=np.float32, copy=True))
        else:
            xd = torch.from_numpy(self.Xd[idx]).float()

        xs = torch.from_numpy(np.array(self.Xs[idx], dtype=np.float32, copy=False))
        y = int(self.y[idx])

        # Ordinal targets: [LOS > 3d, LOS > 7d]
        y_ord = torch.tensor([1.0 if y >= 1 else 0.0, 1.0 if y >= 2 else 0.0], dtype=torch.float32)
        y_cls = torch.tensor(y, dtype=torch.long)

        return xd, xs, y_ord, y_cls


def make_loaders(
    Xd_tr, Xs_tr, y_tr,
    Xd_va, Xs_va, y_va,
    Xd_te, Xs_te, y_te,
    batch: int,
    workers: int,
    copy_mmap: bool = True,
):
    common = dict(
        batch_size=batch,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
    )

    tr = DataLoader(
        DynStaticOrdinalDataset(Xd_tr, Xs_tr, y_tr, copy_mmap=copy_mmap),
        shuffle=True,
        **common,
    )
    va = DataLoader(
        DynStaticOrdinalDataset(Xd_va, Xs_va, y_va, copy_mmap=copy_mmap),
        shuffle=False,
        **common,
    )
    te = DataLoader(
        DynStaticOrdinalDataset(Xd_te, Xs_te, y_te, copy_mmap=copy_mmap),
        shuffle=False,
        **common,
    )
    return tr, va, te


# ======================================================
# MODEL
# ======================================================

class RMSNorm(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w


class BiMamba(nn.Module):
    def __init__(self, f_in: int, d_model: int = 128, layers: int = 2, resid_scale: float = 0.5):
        super().__init__()
        self.proj = nn.Linear(f_in, d_model)
        self.blocks = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
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


class HybridDynEncoder(nn.Module):
    def __init__(self, f_dyn: int, lstm_hidden: int = 256, mamba_dim: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=f_dyn,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.mamba = BiMamba(f_dyn, d_model=mamba_dim, layers=2, resid_scale=0.5)

    def forward(self, x):
        out, (hn, _) = self.lstm(x)
        h_lstm = torch.cat([hn[-2], hn[-1], out.mean(dim=1)], dim=1)  # 4*256? actually 2H + 2H
        h_mamba = self.mamba(x)  # 4*mamba_dim
        h = torch.cat([h_lstm, h_mamba], dim=1)
        return F.layer_norm(h, (h.size(1),))


class HybridOrdinalModel(nn.Module):
    def __init__(self, f_dyn: int, f_static: int):
        super().__init__()
        self.dyn = HybridDynEncoder(f_dyn, lstm_hidden=256, mamba_dim=128)

        self.stat = nn.Sequential(
            nn.Linear(f_static, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )

        # Dynamic encoder output = LSTM: 4*256? In code above: hn[-2]+hn[-1]+out.mean => 256+256+512=1024.
        # Mamba output = 4*128=512. Static=128. Total=1664.
        self.head = nn.Sequential(
            nn.Linear(4 * 256 + 4 * 128 + 128, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),  # ordinal logits: >3d and >7d
        )

    def forward(self, xd, xs):
        h = torch.cat([self.dyn(xd), self.stat(xs)], dim=1)
        return self.head(h)


# ======================================================
# LOSS / ORDINAL UTILS
# ======================================================

def compute_pos_weight(y_train_3c: np.ndarray, device):
    y0 = (y_train_3c >= 1).astype(np.float32)
    y1 = (y_train_3c >= 2).astype(np.float32)

    targets = np.stack([y0, y1], axis=1)
    pos = targets.sum(axis=0)
    neg = targets.shape[0] - pos
    weights = neg / np.maximum(pos, 1.0)

    # Avoid absurd values, but keep long-stay emphasis.
    weights = np.clip(weights, 1.0, 12.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def ordinal_logits_to_class(logits, thresholds=(0.5, 0.5)):
    probs = torch.sigmoid(logits)
    t0, t1 = thresholds
    pred = (probs[:, 0] >= t0).long() + (probs[:, 1] >= t1).long()
    return pred


def ordinal_probs_to_class_np(probs: np.ndarray, thresholds=(0.5, 0.5)):
    t0, t1 = thresholds
    return ((probs[:, 0] >= t0).astype(np.int64) + (probs[:, 1] >= t1).astype(np.int64))


def tune_thresholds(y_true: np.ndarray, probs: np.ndarray):
    best_f1 = -1.0
    best_t = (0.5, 0.5)

    # Keep grid small enough to be fast but useful.
    grid0 = np.arange(0.20, 0.81, 0.02)
    grid1 = np.arange(0.10, 0.91, 0.02)

    for t0 in grid0:
        p0 = probs[:, 0] >= t0
        for t1 in grid1:
            pred = p0.astype(np.int64) + (probs[:, 1] >= t1).astype(np.int64)
            f1 = f1_score(y_true, pred, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = float(f1)
                best_t = (float(t0), float(t1))

    return best_t, best_f1


# ======================================================
# TRAIN / EVAL
# ======================================================

def run_epoch(model, loader, device, criterion, optimizer=None, grad_clip=1.0, amp=False, scaler=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    loss_sum = 0.0
    n = 0
    probs_all = []
    ycls_all = []

    for xd, xs, y_ord, y_cls in loader:
        xd = xd.to(device, non_blocking=True)
        xs = xs.to(device, non_blocking=True)
        y_ord = y_ord.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        if amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(xd, xs)
                loss = criterion(logits, y_ord)
        else:
            logits = model(xd, xs)
            loss = criterion(logits, y_ord)

        if training:
            if not torch.isfinite(loss):
                raise RuntimeError("Loss became NaN/Inf. Aborting.")

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

        bs = y_ord.size(0)
        loss_sum += float(loss.detach().cpu().item()) * bs
        n += bs

        if not training:
            probs_all.append(torch.sigmoid(logits).detach().cpu().numpy())
            ycls_all.append(y_cls.numpy())

    if training:
        return loss_sum / max(n, 1), None, None

    probs = np.concatenate(probs_all) if probs_all else np.empty((0, 2))
    y_true = np.concatenate(ycls_all) if ycls_all else np.array([])
    return loss_sum / max(n, 1), probs, y_true


# ======================================================
# MAIN
# ======================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--copy_mmap", action="store_true")
    args = ap.parse_args()

    seed_everything(42)

    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")

    print("USING eICU HYBRID LSTM + MAMBA ORDINAL 3-CLASS WITH AUROC/AUPRC")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"Device: {device}")

    Xd_tr = np.load(f"{DATA_DIR}/X_dyn_train.npy", mmap_mode="r")
    Xd_va = np.load(f"{DATA_DIR}/X_dyn_valid.npy", mmap_mode="r")
    Xd_te = np.load(f"{DATA_DIR}/X_dyn_test.npy", mmap_mode="r")

    Xs_tr = np.load(f"{DATA_DIR}/X_static_train.npy")
    Xs_va = np.load(f"{DATA_DIR}/X_static_valid.npy")
    Xs_te = np.load(f"{DATA_DIR}/X_static_test.npy")

    y_tr = np.load(f"{DATA_DIR}/y_3c_train.npy")
    y_va = np.load(f"{DATA_DIR}/y_3c_valid.npy")
    y_te = np.load(f"{DATA_DIR}/y_3c_test.npy")

    print(f"Loaded dyn: train={Xd_tr.shape}, valid={Xd_va.shape}, test={Xd_te.shape}")
    print(f"Loaded static: train={Xs_tr.shape}, valid={Xs_va.shape}, test={Xs_te.shape}")
    print("Train class counts:", np.bincount(y_tr, minlength=3))
    print("Using ordinal BCE loss with validation threshold tuning.")

    tr_loader, va_loader, te_loader = make_loaders(
        Xd_tr, Xs_tr, y_tr,
        Xd_va, Xs_va, y_va,
        Xd_te, Xs_te, y_te,
        batch=args.batch,
        workers=args.workers,
        copy_mmap=True,
    )

    model = HybridOrdinalModel(f_dyn=Xd_tr.shape[2], f_static=Xs_tr.shape[1]).to(device)

    pos_weight = compute_pos_weight(y_tr, device=device)
    print("Ordinal pos_weight:", pos_weight.detach().cpu().numpy().tolist())

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_val_f1 = -1.0
    best_val_loss = None
    best_state = None
    best_epoch = -1
    best_thresholds = (0.5, 0.5)

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, _, _ = run_epoch(
            model,
            tr_loader,
            device,
            criterion,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
            amp=args.amp,
            scaler=scaler,
        )

        va_loss, va_probs, va_true = run_epoch(
            model,
            va_loader,
            device,
            criterion,
            optimizer=None,
            grad_clip=args.grad_clip,
            amp=args.amp,
            scaler=None,
        )

        thresholds, va_f1 = tune_thresholds(va_true, va_probs)
        va_pred = ordinal_probs_to_class_np(va_probs, thresholds)
        va_acc = accuracy_score(va_true, va_pred)

        print(
            f"Epoch {epoch:02d} | train {tr_loss:.4f} | val {va_loss:.4f} | "
            f"F1 {va_f1:.4f} | Acc {va_acc:.4f} | t={thresholds}"
        )

        if va_f1 > best_val_f1:
            best_val_f1 = float(va_f1)
            best_val_loss = float(va_loss)
            best_epoch = epoch
            best_thresholds = thresholds
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif best_epoch != -1 and (epoch - best_epoch) >= args.patience:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint produced.")

    model.load_state_dict(best_state)

    te_loss, te_probs, te_true = run_epoch(
        model,
        te_loader,
        device,
        criterion,
        optimizer=None,
        grad_clip=args.grad_clip,
        amp=args.amp,
        scaler=None,
    )

    te_pred = ordinal_probs_to_class_np(te_probs, best_thresholds)

    te_acc = accuracy_score(te_true, te_pred)
    te_f1 = f1_score(te_true, te_pred, average="macro", zero_division=0)

    y_te_ord = np.stack(
        [
            (te_true > 0).astype(int),
            (te_true > 1).astype(int),
        ],
        axis=1,
    )

    te_auroc_macro = roc_auc_score(y_te_ord, te_probs, average="macro")
    te_auprc_macro = average_precision_score(y_te_ord, te_probs, average="macro")

    te_auroc_gt3 = roc_auc_score(y_te_ord[:, 0], te_probs[:, 0])
    te_auroc_gt7 = roc_auc_score(y_te_ord[:, 1], te_probs[:, 1])

    te_auprc_gt3 = average_precision_score(y_te_ord[:, 0], te_probs[:, 0])
    te_auprc_gt7 = average_precision_score(y_te_ord[:, 1], te_probs[:, 1])

    rep = classification_report(te_true, te_pred, digits=4, output_dict=True, zero_division=0)
    cm = confusion_matrix(te_true, te_pred).tolist()

    stamp = f"eicu_v1_ordinal_mamba_ep{best_epoch}_Fdyn{Xd_tr.shape[2]}_Fstat{Xs_tr.shape[1]}"
    model_path = os.path.join(OUT_DIR, f"hybrid_ordinal_{stamp}.pt")
    metrics_path = os.path.join(OUT_DIR, f"metrics_{stamp}.json")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dyn_dim": int(Xd_tr.shape[2]),
            "input_static_dim": int(Xs_tr.shape[1]),
            "num_classes": 3,
            "ordinal_logits": 2,
            "best_epoch": int(best_epoch),
            "best_thresholds": list(best_thresholds),
            "test_auroc_macro": float(te_auroc_macro),
            "test_auprc_macro": float(te_auprc_macro),
            "test_auroc_gt3": float(te_auroc_gt3),
            "test_auroc_gt7": float(te_auroc_gt7),
            "test_auprc_gt3": float(te_auprc_gt3),
            "test_auprc_gt7": float(te_auprc_gt7),
            "model": "Hybrid BiLSTM + BiMamba + static ANN fusion ordinal",
        },
        model_path,
    )

    metrics = {
        "dataset": "eICU",
        "model": "Hybrid BiLSTM + BiMamba + static ANN fusion ordinal",
        "label": "3class_ordinal",
        "test_loss": float(te_loss),
        "test_acc": float(te_acc),
        "test_f1_macro": float(te_f1),
        "test_auroc_macro": float(te_auroc_macro),
        "test_auprc_macro": float(te_auprc_macro),
        "test_auroc_gt3": float(te_auroc_gt3),
        "test_auroc_gt7": float(te_auroc_gt7),
        "test_auprc_gt3": float(te_auprc_gt3),
        "test_auprc_gt7": float(te_auprc_gt7),
        "classification_report": rep,
        "confusion_matrix": cm,
        "best_epoch": int(best_epoch),
        "best_val_f1": float(best_val_f1),
        "best_val_loss": float(best_val_loss),
        "best_thresholds": list(best_thresholds),
        "elapsed_sec": round(time.time() - t0, 1),
        "class_counts_train": np.bincount(y_tr, minlength=3).tolist(),
        "train_shape_dyn": list(Xd_tr.shape),
        "valid_shape_dyn": list(Xd_va.shape),
        "test_shape_dyn": list(Xd_te.shape),
        "train_shape_static": list(Xs_tr.shape),
        "input_dyn_dim": int(Xd_tr.shape[2]),
        "input_static_dim": int(Xs_tr.shape[1]),
        "checkpoint_metric": "validation_macro_f1",
        "ordinal_targets": ["LOS_gt_3_days", "LOS_gt_7_days"],
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model:   {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(
        f"Test Acc: {te_acc:.4f} | "
        f"F1 macro: {te_f1:.4f} | "
        f"AUROC macro: {te_auroc_macro:.4f} | "
        f"AUPRC macro: {te_auprc_macro:.4f} | "
        f"Test loss: {te_loss:.4f} | "
        f"Best epoch: {best_epoch} | thresholds={best_thresholds}"
    )

    print(
        f"Ordinal AUROC: gt3={te_auroc_gt3:.4f}, gt7={te_auroc_gt7:.4f} | "
        f"Ordinal AUPRC: gt3={te_auprc_gt3:.4f}, gt7={te_auprc_gt7:.4f}"
    )


if __name__ == "__main__":
    main()
