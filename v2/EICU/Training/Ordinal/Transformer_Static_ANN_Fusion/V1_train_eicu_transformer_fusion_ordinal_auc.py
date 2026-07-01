#!/usr/bin/env python3
"""
V1_train_eicu_transformer_fusion_ordinal_auc.py
-------------------------------------------

eICU Transformer + static ANN fusion model for 3-class ORDINAL LOS prediction.

This script is designed for the eICU pipeline already built in this project.

Reads the SAME files as the other eICU training scripts:

    /lustre/home/rahas2/mimic_projects/outputs/eicu_lstm_data_v1

Expected files:
    X_dyn_train.npy
    X_dyn_valid.npy
    X_dyn_test.npy
    X_static_train.npy
    X_static_valid.npy
    X_static_test.npy
    y_3c_train.npy
    y_3c_valid.npy
    y_3c_test.npy

Task:
    3-class ordinal LOS:
        0 = LOS <= 3 days
        1 = 3 < LOS <= 7 days
        2 = LOS > 7 days

Ordinal targets:
    class 0 -> [0, 0]
    class 1 -> [1, 0]
    class 2 -> [1, 1]

Ordinal logits:
    logit 0 -> P(LOS > 3 days)
    logit 1 -> P(LOS > 7 days)

Architecture:
    - dynamic feature LayerNorm
    - dynamic projection MLP per timestep
    - learned positional embeddings
    - static ANN -> CLS/context token
    - Transformer encoder over [static CLS + 24 hourly dynamic tokens]
    - CLS pooled ordinal head
    - validation threshold tuning for macro F1

Output folder:
    /lustre/home/rahas2/mimic_projects/outputs/eicu_transformer_fusion_ordinal_auc_models_v1

This folder is intentionally separate from:
    eicu_lstm_models_v1
    eicu_lstm_fusion_ordinal_models_v1
    eicu_hybrid_lstm_mamba_models_v1
    eicu_hybrid_lstm_mamba_ordinal_models_v1
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


# ======================================================
# CONFIG
# ======================================================

BASE = "/lustre/home/rahas2/mimic_projects/outputs"
DATA_DIR = f"{BASE}/eicu_lstm_data_v1"
OUT_DIR = f"{BASE}/eicu_transformer_fusion_ordinal_auc_models_v1"

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
    def __init__(self, X_dyn, X_static, y_3c):
        self.Xd = X_dyn
        self.Xs = X_static
        self.y = y_3c.astype(np.int64)

    def __len__(self):
        return self.Xd.shape[0]

    def __getitem__(self, idx):
        xd = torch.from_numpy(np.array(self.Xd[idx], dtype=np.float32, copy=True))
        xs = torch.from_numpy(np.array(self.Xs[idx], dtype=np.float32, copy=True))
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return xd, xs, y


def make_loaders(
    X_dyn_train,
    X_static_train,
    y_train,
    X_dyn_valid,
    X_static_valid,
    y_valid,
    X_dyn_test,
    X_static_test,
    y_test,
    batch=256,
    workers=0,
):
    kwargs = dict(
        batch_size=batch,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
    )

    train_loader = DataLoader(
        DynStaticOrdinalDataset(X_dyn_train, X_static_train, y_train),
        shuffle=True,
        **kwargs,
    )

    valid_loader = DataLoader(
        DynStaticOrdinalDataset(X_dyn_valid, X_static_valid, y_valid),
        shuffle=False,
        **kwargs,
    )

    test_loader = DataLoader(
        DynStaticOrdinalDataset(X_dyn_test, X_static_test, y_test),
        shuffle=False,
        **kwargs,
    )

    return train_loader, valid_loader, test_loader


# ======================================================
# ORDINAL UTILS
# ======================================================

def ordinal_targets(y: torch.Tensor) -> torch.Tensor:
    """
    Convert class labels 0/1/2 into cumulative ordinal targets:
        0 -> [0, 0]
        1 -> [1, 0]
        2 -> [1, 1]
    """
    y = y.long()
    return torch.stack(
        [
            (y > 0).float(),
            (y > 1).float(),
        ],
        dim=1,
    )


def ordinal_probs_to_class(prob_gt3, prob_gt7, thr1, thr2):
    """
    Convert ordinal probabilities into class predictions.

    Rule:
        class 0 if P(LOS > 3d) < thr1
        class 2 if P(LOS > 7d) >= thr2
        else class 1
    """
    pred = np.ones_like(prob_gt3, dtype=np.int64)
    pred[prob_gt3 < thr1] = 0
    pred[prob_gt7 >= thr2] = 2
    return pred


def tune_ordinal_thresholds(y_true, prob_gt3, prob_gt7):
    """
    Tune thresholds on validation macro F1.

    Grid is moderate and fast enough for repeated epochs.
    """
    best = {
        "f1": -1.0,
        "acc": 0.0,
        "thr1": 0.5,
        "thr2": 0.5,
    }

    grid1 = np.arange(0.30, 0.86, 0.02)
    grid2 = np.arange(0.30, 0.91, 0.02)

    for t1 in grid1:
        for t2 in grid2:
            pred = ordinal_probs_to_class(prob_gt3, prob_gt7, t1, t2)
            f1 = f1_score(y_true, pred, average="macro", zero_division=0)

            if f1 > best["f1"]:
                best["f1"] = float(f1)
                best["acc"] = float(accuracy_score(y_true, pred))
                best["thr1"] = float(t1)
                best["thr2"] = float(t2)

    return best


# ======================================================
# LOSS
# ======================================================

class OrdinalBCELoss(nn.Module):
    """
    Weighted BCE loss for cumulative ordinal targets.

    logits[:, 0] => LOS > 3 days
    logits[:, 1] => LOS > 7 days
    """
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


# ======================================================
# MODEL
# ======================================================

class TransformerStaticOrdinalFusion(nn.Module):
    """
    Transformer temporal encoder with static ANN CLS token.

    Dynamic path:
        X_dyn: (B, 24, F_dyn)
        LayerNorm(F_dyn)
        MLP projection to d_model

    Static path:
        X_static: (B, F_static)
        LayerNorm(F_static)
        MLP projection to d_model
        used as CLS/context token

    Sequence:
        [static_CLS, dyn_t0, ..., dyn_t23]

    Output:
        two ordinal logits:
            LOS > 3 days
            LOS > 7 days
    """

    def __init__(
        self,
        dyn_dim: int,
        static_dim: int,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 3,
        ff_dim: int = 384,
        dropout: float = 0.15,
        max_timesteps: int = 24,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.max_timesteps = max_timesteps
        self.d_model = d_model

        self.dyn_norm = nn.LayerNorm(dyn_dim)
        self.static_norm = nn.LayerNorm(static_dim)

        self.dyn_proj = nn.Sequential(
            nn.Linear(dyn_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        self.pos_embed = nn.Embedding(max_timesteps + 1, d_model)

        # Learnable temporal decay, initialized mildly.
        self.time_decay = nn.Parameter(torch.tensor(0.03))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.final_ln = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, x_dyn, x_static):
        B, T, _ = x_dyn.shape

        if T > self.max_timesteps:
            raise ValueError(f"Unexpected T={T}, max_timesteps={self.max_timesteps}")

        x_dyn = self.dyn_norm(x_dyn)
        x_static = self.static_norm(x_static)

        dyn_tokens = self.dyn_proj(x_dyn)  # (B,T,D)

        # Temporal decay: later tokens are not destroyed, but early trajectory is emphasized.
        time_idx = torch.arange(T, device=x_dyn.device, dtype=torch.float32)
        decay = torch.exp(-torch.clamp(self.time_decay, 0.0, 1.0) * time_idx)
        dyn_tokens = dyn_tokens * decay.view(1, T, 1)

        cls_token = self.static_proj(x_static).unsqueeze(1)  # (B,1,D)

        tokens = torch.cat([cls_token, dyn_tokens], dim=1)  # (B,1+T,D)

        pos_ids = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        tokens = tokens + self.pos_embed(pos_ids)

        encoded = self.encoder(tokens)

        cls_out = self.final_ln(encoded[:, 0, :])

        return self.head(cls_out)


# ======================================================
# TRAIN / EVAL
# ======================================================

def train_epoch(
    model,
    loader,
    device,
    criterion,
    optimizer,
    scheduler=None,
    scaler=None,
    amp: bool = False,
    grad_clip: float = 1.0,
):
    model.train()

    loss_sum = 0.0
    n = 0

    for Xd, Xs, y in loader:
        Xd = Xd.to(device, non_blocking=True)
        Xs = Xs.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if amp and scaler is not None and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(Xd, Xs)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        else:
            logits = model(Xd, Xs)
            loss = criterion(logits, y)

            if not torch.isfinite(loss):
                raise RuntimeError("Loss became NaN/Inf.")

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        bs = y.size(0)
        loss_sum += loss.item() * bs
        n += bs

    return loss_sum / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device, criterion, amp: bool = False):
    model.eval()

    loss_sum = 0.0
    n = 0

    probs_all = []
    y_all = []

    for Xd, Xs, y in loader:
        Xd = Xd.to(device, non_blocking=True)
        Xs = Xs.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(Xd, Xs)
                loss = criterion(logits, y)
        else:
            logits = model(Xd, Xs)
            loss = criterion(logits, y)

        probs = torch.sigmoid(logits)

        bs = y.size(0)
        loss_sum += loss.item() * bs
        n += bs

        probs_all.append(probs.detach().cpu().numpy())
        y_all.append(y.detach().cpu().numpy())

    probs_all = np.concatenate(probs_all) if probs_all else np.array([])
    y_all = np.concatenate(y_all) if y_all else np.array([])

    return loss_sum / max(n, 1), probs_all, y_all


# ======================================================
# MAIN
# ======================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--ff_dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()

    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("USING eICU TRANSFORMER + ANN FUSION ORDINAL 3-CLASS WITH AUROC/AUPRC")
    print(f"Device: {device}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"OUT_DIR: {OUT_DIR}")

    X_dyn_train = np.load(f"{DATA_DIR}/X_dyn_train.npy", mmap_mode="r")
    X_dyn_valid = np.load(f"{DATA_DIR}/X_dyn_valid.npy", mmap_mode="r")
    X_dyn_test = np.load(f"{DATA_DIR}/X_dyn_test.npy", mmap_mode="r")

    X_static_train = np.load(f"{DATA_DIR}/X_static_train.npy")
    X_static_valid = np.load(f"{DATA_DIR}/X_static_valid.npy")
    X_static_test = np.load(f"{DATA_DIR}/X_static_test.npy")

    y_train = np.load(f"{DATA_DIR}/y_3c_train.npy")
    y_valid = np.load(f"{DATA_DIR}/y_3c_valid.npy")
    y_test = np.load(f"{DATA_DIR}/y_3c_test.npy")

    print(f"Loaded dyn: train={X_dyn_train.shape}, valid={X_dyn_valid.shape}, test={X_dyn_test.shape}")
    print(f"Loaded static: train={X_static_train.shape}, valid={X_static_valid.shape}, test={X_static_test.shape}")

    class_counts = np.bincount(y_train, minlength=3)
    print("Train class counts:", class_counts)

    y_ord_train = np.stack(
        [
            (y_train > 0).astype(np.float32),
            (y_train > 1).astype(np.float32),
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
        X_dyn_train,
        X_static_train,
        y_train,
        X_dyn_valid,
        X_static_valid,
        y_valid,
        X_dyn_test,
        X_static_test,
        y_test,
        batch=args.batch,
        workers=args.workers,
    )

    dyn_dim = X_dyn_train.shape[2]
    static_dim = X_static_train.shape[1]

    model = TransformerStaticOrdinalFusion(
        dyn_dim=dyn_dim,
        static_dim=static_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        max_timesteps=X_dyn_train.shape[1],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # Warmup then flat LR.
    total_steps = args.epochs * max(len(tr_loader), 1)
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    scaler = torch.cuda.amp.GradScaler(
        enabled=(args.amp and device.type == "cuda")
    )

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_thresholds = (0.5, 0.5)
    best_state = None
    best_epoch = -1

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            tr_loader,
            device,
            criterion,
            optimizer,
            scheduler=scheduler,
            scaler=scaler,
            amp=(args.amp and device.type == "cuda"),
            grad_clip=args.grad_clip,
        )

        val_loss, val_probs, val_true = eval_epoch(
            model,
            va_loader,
            device,
            criterion,
            amp=(args.amp and device.type == "cuda"),
        )

        threshold_info = tune_ordinal_thresholds(
            val_true,
            val_probs[:, 0],
            val_probs[:, 1],
        )

        val_f1 = threshold_info["f1"]
        val_acc = threshold_info["acc"]

        print(
            f"Epoch {epoch:02d} | "
            f"train {train_loss:.4f} | "
            f"val {val_loss:.4f} | "
            f"F1 {val_f1:.4f} | "
            f"Acc {val_acc:.4f} | "
            f"t=({threshold_info['thr1']:.2f}, {threshold_info['thr2']:.2f})"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = float(val_f1)
            best_val_loss = float(val_loss)
            best_thresholds = (
                float(threshold_info["thr1"]),
                float(threshold_info["thr2"]),
            )
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            best_epoch = epoch

        elif best_epoch != -1 and (epoch - best_epoch) >= args.patience:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint produced.")

    model.load_state_dict(best_state)

    test_loss, test_probs, test_true = eval_epoch(
        model,
        te_loader,
        device,
        criterion,
        amp=(args.amp and device.type == "cuda"),
    )

    test_pred = ordinal_probs_to_class(
        test_probs[:, 0],
        test_probs[:, 1],
        best_thresholds[0],
        best_thresholds[1],
    )

    test_acc = accuracy_score(test_true, test_pred)

    test_f1_macro = f1_score(
        test_true,
        test_pred,
        average="macro",
        zero_division=0,
    )

    # Ordinal AUROC/AUPRC:
    # column 0 = LOS > 3 days
    # column 1 = LOS > 7 days
    y_test_ord = np.stack(
        [
            (test_true > 0).astype(int),
            (test_true > 1).astype(int),
        ],
        axis=1,
    )

    test_auroc_macro = roc_auc_score(
        y_test_ord,
        test_probs,
        average="macro",
    )

    test_auprc_macro = average_precision_score(
        y_test_ord,
        test_probs,
        average="macro",
    )

    test_auroc_gt3 = roc_auc_score(
        y_test_ord[:, 0],
        test_probs[:, 0],
    )

    test_auroc_gt7 = roc_auc_score(
        y_test_ord[:, 1],
        test_probs[:, 1],
    )

    test_auprc_gt3 = average_precision_score(
        y_test_ord[:, 0],
        test_probs[:, 0],
    )

    test_auprc_gt7 = average_precision_score(
        y_test_ord[:, 1],
        test_probs[:, 1],
    )

    report = classification_report(
        test_true,
        test_pred,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(test_true, test_pred).tolist()

    stamp = (
        f"eicu_v1_transformer_ordinal_3class_"
        f"D{args.d_model}_H{args.nhead}_L{args.layers}_FF{args.ff_dim}_"
        f"Fdyn{dyn_dim}_Fstat{static_dim}"
    )

    model_path = os.path.join(OUT_DIR, f"transformer_ordinal_{stamp}.pt")
    metrics_path = os.path.join(OUT_DIR, f"metrics_{stamp}.json")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "dyn_dim": int(dyn_dim),
            "static_dim": int(static_dim),
            "d_model": int(args.d_model),
            "nhead": int(args.nhead),
            "layers": int(args.layers),
            "ff_dim": int(args.ff_dim),
            "dropout": float(args.dropout),
            "num_classes_original": 3,
            "num_ordinal_logits": 2,
            "label": "3class_ordinal",
            "best_epoch": int(best_epoch),
            "best_val_f1": float(best_val_f1),
            "best_val_loss": float(best_val_loss),
            "best_thresholds": list(best_thresholds),
            "test_auroc_macro": float(test_auroc_macro),
            "test_auprc_macro": float(test_auprc_macro),
            "test_auroc_gt3": float(test_auroc_gt3),
            "test_auroc_gt7": float(test_auroc_gt7),
            "test_auprc_gt3": float(test_auprc_gt3),
            "test_auprc_gt7": float(test_auprc_gt7),
        },
        model_path,
    )

    metrics = {
        "dataset": "eICU",
        "model": "Transformer + static ANN fusion ordinal",
        "label": "3class_ordinal",
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_f1_macro": float(test_f1_macro),
        "test_auroc_macro": float(test_auroc_macro),
        "test_auprc_macro": float(test_auprc_macro),
        "test_auroc_gt3": float(test_auroc_gt3),
        "test_auroc_gt7": float(test_auroc_gt7),
        "test_auprc_gt3": float(test_auprc_gt3),
        "test_auprc_gt7": float(test_auprc_gt7),
        "classification_report": report,
        "confusion_matrix": cm,
        "best_epoch": int(best_epoch),
        "best_val_f1": float(best_val_f1),
        "best_val_loss": float(best_val_loss),
        "best_thresholds": list(best_thresholds),
        "elapsed_sec": round(time.time() - t0, 1),
        "class_counts_train": class_counts.tolist(),
        "train_shape_dyn": list(X_dyn_train.shape),
        "valid_shape_dyn": list(X_dyn_valid.shape),
        "test_shape_dyn": list(X_dyn_test.shape),
        "train_shape_static": list(X_static_train.shape),
        "input_dyn_dim": int(dyn_dim),
        "input_static_dim": int(static_dim),
        "checkpoint_metric": "validation_macro_f1",
        "ordinal_targets": [
            "LOS_gt_3_days",
            "LOS_gt_7_days",
        ],
        "output_dir": OUT_DIR,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model:   {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(
        f"Test Acc: {test_acc:.4f} | "
        f"F1 macro: {test_f1_macro:.4f} | "
        f"AUROC macro: {test_auroc_macro:.4f} | "
        f"AUPRC macro: {test_auprc_macro:.4f} | "
        f"Test loss: {test_loss:.4f} | "
        f"Best epoch: {best_epoch} | "
        f"Best val F1: {best_val_f1:.4f} | "
        f"Thresholds: {best_thresholds}"
    )

    print(
        f"Ordinal AUROC: gt3={test_auroc_gt3:.4f}, gt7={test_auroc_gt7:.4f} | "
        f"Ordinal AUPRC: gt3={test_auprc_gt3:.4f}, gt7={test_auprc_gt7:.4f}"
    )


if __name__ == "__main__":
    main()
