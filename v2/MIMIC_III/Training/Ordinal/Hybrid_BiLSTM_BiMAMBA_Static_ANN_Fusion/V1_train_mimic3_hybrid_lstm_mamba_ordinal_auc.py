#!/usr/bin/env python3
"""
V1_train_mimic3_hybrid_lstm_mamba_ordinal_auc.py
------------------------------------------------

MIMIC-III Hybrid BiLSTM + BiMamba + Static ANN Fusion Ordinal.

Equivalent to the eICU hybrid ordinal AUC script, but for MIMIC-III.

Task:
    3-class LOS ordinal:
        0 = LOS <= 3 days
        1 = 3 < LOS <= 7 days
        2 = LOS > 7 days

Ordinal targets:
    logit 0 = P(LOS > 3 days)
    logit 1 = P(LOS > 7 days)

Reads:
    /lustre/home/rahas2/mimic_projects/outputs/lstm_data_v12.1_seq

Expected files:
    X_dyn_train.npy / valid / test
    X_static_train.npy / valid / test
    y_3c_train.npy / valid / test

Outputs:
    /lustre/home/rahas2/mimic_projects/outputs/mimic3_hybrid_lstm_mamba_ordinal_auc_models_v1

Run:
    python V1_train_mimic3_hybrid_lstm_mamba_ordinal_auc.py --workers 0 --batch 128
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


# ======================================================
# CONFIG
# ======================================================

BASE = "/lustre/home/rahas2/mimic_projects/outputs"
DATA_DIR = f"{BASE}/lstm_data_v12.1_seq"
OUT_DIR = f"{BASE}/mimic3_hybrid_lstm_mamba_ordinal_auc_models_v1"

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

        xs = torch.from_numpy(np.array(self.Xs[idx], dtype=np.float32, copy=True))
        y = torch.tensor(self.y[idx], dtype=torch.long)

        return xd, xs, y


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
# ORDINAL UTILS
# ======================================================

def ordinal_targets(y: torch.Tensor) -> torch.Tensor:
    y = y.long()
    return torch.stack(
        [
            (y > 0).float(),
            (y > 1).float(),
        ],
        dim=1,
    )


class OrdinalBCELoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()

        if pos_weight is not None:
            self.register_buffer(
                "pos_weight",
                torch.tensor(pos_weight, dtype=torch.float32),
            )
        else:
            self.pos_weight = None

    def forward(self, logits, y_class):
        y_ord = ordinal_targets(y_class).to(logits.device)

        return F.binary_cross_entropy_with_logits(
            logits,
            y_ord,
            pos_weight=self.pos_weight,
            reduction="mean",
        )


def ordinal_probs_to_class(prob_gt3, prob_gt7, thr1, thr2):
    pred = np.ones_like(prob_gt3, dtype=np.int64)
    pred[prob_gt3 < thr1] = 0
    pred[prob_gt7 >= thr2] = 2
    return pred


def tune_ordinal_thresholds(y_true, prob_gt3, prob_gt7):
    best = {
        "f1": -1.0,
        "acc": 0.0,
        "thr1": 0.5,
        "thr2": 0.5,
    }

    for t1 in np.arange(0.30, 0.86, 0.02):
        for t2 in np.arange(0.30, 0.91, 0.02):
            pred = ordinal_probs_to_class(prob_gt3, prob_gt7, t1, t2)
            f1 = f1_score(y_true, pred, average="macro", zero_division=0)

            if f1 > best["f1"]:
                best["f1"] = float(f1)
                best["acc"] = float(accuracy_score(y_true, pred))
                best["thr1"] = float(t1)
                best["thr2"] = float(t2)

    return best


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
    def __init__(
        self,
        f_in: int,
        d_model: int = 128,
        layers: int = 2,
        resid_scale: float = 0.5,
    ):
        super().__init__()

        self.proj = nn.Linear(f_in, d_model)

        self.blocks = nn.ModuleList(
            [
                Mamba(
                    d_model=d_model,
                    d_state=16,
                    d_conv=4,
                    expand=2,
                )
                for _ in range(layers)
            ]
        )

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

        return torch.cat(
            [
                f[:, -1],
                b[:, -1],
                f.mean(dim=1),
                b.mean(dim=1),
            ],
            dim=1,
        )


class HybridDynEncoder(nn.Module):
    def __init__(
        self,
        f_dyn: int,
        lstm_hidden: int = 256,
        mamba_dim: int = 128,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=f_dyn,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        self.mamba = BiMamba(
            f_dyn,
            d_model=mamba_dim,
            layers=2,
            resid_scale=0.5,
        )

    def forward(self, x):
        out, (hn, _) = self.lstm(x)

        h_lstm = torch.cat(
            [
                hn[-2],
                hn[-1],
                out.mean(dim=1),
            ],
            dim=1,
        )

        h_mamba = self.mamba(x)

        h = torch.cat([h_lstm, h_mamba], dim=1)

        return F.layer_norm(h, (h.size(1),))


class HybridOrdinalModel(nn.Module):
    def __init__(self, f_dyn: int, f_static: int):
        super().__init__()

        self.dyn = HybridDynEncoder(
            f_dyn=f_dyn,
            lstm_hidden=256,
            mamba_dim=128,
        )

        self.stat = nn.Sequential(
            nn.Linear(f_static, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )

        # Dynamic = LSTM 1024 + Mamba 512 = 1536
        # Static = 128
        # Total = 1664
        self.head = nn.Sequential(
            nn.Linear(4 * 256 + 4 * 128 + 128, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, xd, xs):
        h = torch.cat([self.dyn(xd), self.stat(xs)], dim=1)
        return self.head(h)


# ======================================================
# TRAIN / EVAL
# ======================================================

def run_epoch(
    model,
    loader,
    device,
    criterion,
    optimizer=None,
    grad_clip=1.0,
    amp=False,
    scaler=None,
):
    training = optimizer is not None
    model.train() if training else model.eval()

    loss_sum = 0.0
    n = 0

    probs_all = []
    y_all = []

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

        if training:
            if not torch.isfinite(loss):
                raise RuntimeError("Loss became NaN/Inf.")

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
            probs_all.append(torch.sigmoid(logits).detach().cpu().numpy())
            y_all.append(y.detach().cpu().numpy())

    if training:
        return loss_sum / max(n, 1), None, None

    probs = np.concatenate(probs_all) if probs_all else np.empty((0, 2))
    y_true = np.concatenate(y_all) if y_all else np.array([])

    return loss_sum / max(n, 1), probs, y_true


# ======================================================
# MAIN
# ======================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--copy_mmap", action="store_true")

    args = parser.parse_args()

    seed_everything(42)

    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")

    print("USING MIMIC-III HYBRID LSTM + MAMBA ORDINAL WITH AUROC/AUPRC")
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

    class_counts = np.bincount(y_tr, minlength=3)
    print("Train class counts:", class_counts)

    y_ord_train = np.stack(
        [
            (y_tr > 0).astype(np.float32),
            (y_tr > 1).astype(np.float32),
        ],
        axis=1,
    )

    pos = y_ord_train.sum(axis=0)
    neg = y_ord_train.shape[0] - pos

    pos_weight = neg / np.maximum(pos, 1.0)
    pos_weight = np.minimum(pos_weight, np.array([6.0, 12.0], dtype=np.float32))

    print("Using ordinal BCE loss with validation threshold tuning.")
    print("Ordinal pos_weight:", pos_weight.tolist())

    criterion = OrdinalBCELoss(pos_weight=pos_weight).to(device)

    tr_loader, va_loader, te_loader = make_loaders(
        Xd_tr, Xs_tr, y_tr,
        Xd_va, Xs_va, y_va,
        Xd_te, Xs_te, y_te,
        batch=args.batch,
        workers=args.workers,
        copy_mmap=True,
    )

    model = HybridOrdinalModel(
        f_dyn=Xd_tr.shape[2],
        f_static=Xs_tr.shape[1],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

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

        threshold_info = tune_ordinal_thresholds(
            va_true,
            va_probs[:, 0],
            va_probs[:, 1],
        )

        va_f1 = threshold_info["f1"]
        va_acc = threshold_info["acc"]

        print(
            f"Epoch {epoch:02d} | "
            f"train {tr_loss:.4f} | val {va_loss:.4f} | "
            f"F1 {va_f1:.4f} | Acc {va_acc:.4f} | "
            f"t=({threshold_info['thr1']:.2f}, {threshold_info['thr2']:.2f})"
        )

        if va_f1 > best_val_f1:
            best_val_f1 = float(va_f1)
            best_val_loss = float(va_loss)
            best_epoch = epoch
            best_thresholds = (
                float(threshold_info["thr1"]),
                float(threshold_info["thr2"]),
            )
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

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

    te_pred = ordinal_probs_to_class(
        te_probs[:, 0],
        te_probs[:, 1],
        best_thresholds[0],
        best_thresholds[1],
    )

    te_acc = accuracy_score(te_true, te_pred)
    te_f1_macro = f1_score(te_true, te_pred, average="macro", zero_division=0)
    te_f1_weighted = f1_score(te_true, te_pred, average="weighted", zero_division=0)

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

    rep = classification_report(
        te_true,
        te_pred,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(te_true, te_pred).tolist()

    stamp = (
        f"mimic3_hybrid_lstm_mamba_ordinal_3class_"
        f"ep{best_epoch}_Fdyn{Xd_tr.shape[2]}_Fstat{Xs_tr.shape[1]}"
    )

    model_path = os.path.join(OUT_DIR, f"hybrid_ordinal_{stamp}.pt")
    metrics_path = os.path.join(OUT_DIR, f"metrics_{stamp}.json")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dyn_dim": int(Xd_tr.shape[2]),
            "input_static_dim": int(Xs_tr.shape[1]),
            "num_classes_original": 3,
            "num_ordinal_logits": 2,
            "label": "3class_ordinal",
            "best_epoch": int(best_epoch),
            "best_val_f1": float(best_val_f1),
            "best_val_loss": float(best_val_loss),
            "best_thresholds": list(best_thresholds),
            "model": "Hybrid BiLSTM + BiMamba + static ANN fusion ordinal",
            "test_auroc_macro": float(te_auroc_macro),
            "test_auprc_macro": float(te_auprc_macro),
            "test_auroc_gt3": float(te_auroc_gt3),
            "test_auroc_gt7": float(te_auroc_gt7),
            "test_auprc_gt3": float(te_auprc_gt3),
            "test_auprc_gt7": float(te_auprc_gt7),
        },
        model_path,
    )

    metrics = {
        "dataset": "MIMIC-III",
        "model": "Hybrid BiLSTM + BiMamba + static ANN fusion ordinal",
        "label": "3class_ordinal",
        "test_loss": float(te_loss),
        "test_acc": float(te_acc),
        "test_f1_macro": float(te_f1_macro),
        "test_f1_weighted": float(te_f1_weighted),
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
        "class_counts_train": class_counts.tolist(),
        "train_shape_dyn": list(Xd_tr.shape),
        "valid_shape_dyn": list(Xd_va.shape),
        "test_shape_dyn": list(Xd_te.shape),
        "train_shape_static": list(Xs_tr.shape),
        "input_dyn_dim": int(Xd_tr.shape[2]),
        "input_static_dim": int(Xs_tr.shape[1]),
        "checkpoint_metric": "validation_macro_f1",
        "ordinal_targets": ["LOS_gt_3_days", "LOS_gt_7_days"],
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
        f"AUROC macro: {te_auroc_macro:.4f} | "
        f"AUPRC macro: {te_auprc_macro:.4f} | "
        f"Test loss: {te_loss:.4f} | "
        f"Best epoch: {best_epoch} | "
        f"Thresholds: {best_thresholds}"
    )

    print(
        f"Ordinal AUROC: gt3={te_auroc_gt3:.4f}, gt7={te_auroc_gt7:.4f} | "
        f"Ordinal AUPRC: gt3={te_auprc_gt3:.4f}, gt7={te_auprc_gt7:.4f}"
    )


if __name__ == "__main__":
    main()
