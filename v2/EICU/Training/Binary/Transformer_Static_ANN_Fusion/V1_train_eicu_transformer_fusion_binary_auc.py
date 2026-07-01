#!/usr/bin/env python3
"""
V1_train_eicu_transformer_fusion_binary_auc.py
------------------------------------------

eICU Transformer + static ANN fusion model for BINARY LOS prediction.

Reads the SAME files as the other eICU training scripts:

    /lustre/home/rahas2/mimic_projects/outputs/eicu_lstm_data_v1

Expected files:
    X_dyn_train.npy
    X_dyn_valid.npy
    X_dyn_test.npy
    X_static_train.npy
    X_static_valid.npy
    X_static_test.npy
    y_bin_train.npy
    y_bin_valid.npy
    y_bin_test.npy

Task:
    Binary LOS:
        0 = LOS <= 7 days
        1 = LOS > 7 days

Architecture:
    - dynamic feature LayerNorm
    - dynamic projection MLP per timestep
    - learned positional embeddings
    - static ANN -> CLS/context token
    - Transformer encoder over [static CLS + 24 hourly dynamic tokens]
    - CLS pooled binary classifier head
    - validation macro-F1 checkpointing
    - validation threshold tuning for positive class

Output folder:
    /lustre/home/rahas2/mimic_projects/outputs/eicu_transformer_fusion_binary_auc_models_v1

This folder is intentionally separate from:
    eicu_lstm_models_v1
    eicu_lstm_fusion_ordinal_models_v1
    eicu_hybrid_lstm_mamba_models_v1
    eicu_hybrid_lstm_mamba_ordinal_models_v1
    eicu_transformer_fusion_ordinal_models_v1
"""

import os
import json
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
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
OUT_DIR = f"{BASE}/eicu_transformer_fusion_binary_auc_models_v1"

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
        DynStaticBinaryDataset(X_dyn_train, X_static_train, y_train),
        shuffle=True,
        **kwargs,
    )

    valid_loader = DataLoader(
        DynStaticBinaryDataset(X_dyn_valid, X_static_valid, y_valid),
        shuffle=False,
        **kwargs,
    )

    test_loader = DataLoader(
        DynStaticBinaryDataset(X_dyn_test, X_static_test, y_test),
        shuffle=False,
        **kwargs,
    )

    return train_loader, valid_loader, test_loader


# ======================================================
# THRESHOLD TUNING
# ======================================================

def tune_binary_threshold(y_true, prob_pos):
    """
    Tune positive-class probability threshold on validation macro F1.
    """
    best_t = 0.5
    best_f1 = -1.0
    best_acc = 0.0

    for t in np.arange(0.05, 0.96, 0.01):
        pred = (prob_pos >= t).astype(np.int64)
        f1 = f1_score(y_true, pred, average="macro", zero_division=0)

        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
            best_acc = float(accuracy_score(y_true, pred))

    return {
        "threshold": best_t,
        "f1": best_f1,
        "acc": best_acc,
    }


# ======================================================
# MODEL
# ======================================================

class TransformerStaticBinaryFusion(nn.Module):
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
        two binary logits for classes:
            0 = LOS <= 7 days
            1 = LOS > 7 days
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

        # Mild learnable temporal decay.
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

        dyn_tokens = self.dyn_proj(x_dyn)

        time_idx = torch.arange(T, device=x_dyn.device, dtype=torch.float32)
        decay = torch.exp(-torch.clamp(self.time_decay, 0.0, 1.0) * time_idx)
        dyn_tokens = dyn_tokens * decay.view(1, T, 1)

        cls_token = self.static_proj(x_static).unsqueeze(1)

        tokens = torch.cat([cls_token, dyn_tokens], dim=1)

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

        probs = torch.softmax(logits, dim=1)

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

    print("USING eICU TRANSFORMER + ANN FUSION BINARY WITH AUROC/AUPRC")
    print(f"Device: {device}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"OUT_DIR: {OUT_DIR}")

    X_dyn_train = np.load(f"{DATA_DIR}/X_dyn_train.npy", mmap_mode="r")
    X_dyn_valid = np.load(f"{DATA_DIR}/X_dyn_valid.npy", mmap_mode="r")
    X_dyn_test = np.load(f"{DATA_DIR}/X_dyn_test.npy", mmap_mode="r")

    X_static_train = np.load(f"{DATA_DIR}/X_static_train.npy")
    X_static_valid = np.load(f"{DATA_DIR}/X_static_valid.npy")
    X_static_test = np.load(f"{DATA_DIR}/X_static_test.npy")

    y_train = np.load(f"{DATA_DIR}/y_bin_train.npy")
    y_valid = np.load(f"{DATA_DIR}/y_bin_valid.npy")
    y_test = np.load(f"{DATA_DIR}/y_bin_test.npy")

    print(f"Loaded dyn: train={X_dyn_train.shape}, valid={X_dyn_valid.shape}, test={X_dyn_test.shape}")
    print(f"Loaded static: train={X_static_train.shape}, valid={X_static_valid.shape}, test={X_static_test.shape}")

    class_counts = np.bincount(y_train, minlength=2)
    print("Train class counts:", class_counts)

    weights = class_counts.sum() / (class_counts + 1e-9)
    weights = weights / weights.sum()
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    print("Using weighted CrossEntropy with validation threshold tuning.")
    print("Class weights:", weights.tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights)

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

    model = TransformerStaticBinaryFusion(
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
    best_threshold = 0.5
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

        threshold_info = tune_binary_threshold(
            val_true,
            val_probs[:, 1],
        )

        val_pred = (val_probs[:, 1] >= threshold_info["threshold"]).astype(np.int64)
        val_f1 = threshold_info["f1"]
        val_acc = threshold_info["acc"]

        print(
            f"Epoch {epoch:02d} | "
            f"train {train_loss:.4f} | "
            f"val {val_loss:.4f} | "
            f"F1 {val_f1:.4f} | "
            f"Acc {val_acc:.4f} | "
            f"thr={threshold_info['threshold']:.2f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = float(val_f1)
            best_val_loss = float(val_loss)
            best_threshold = float(threshold_info["threshold"])
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

    test_pred = (test_probs[:, 1] >= best_threshold).astype(np.int64)

    test_acc = accuracy_score(test_true, test_pred)

    test_f1_macro = f1_score(
        test_true,
        test_pred,
        average="macro",
        zero_division=0,
    )

    test_auroc = roc_auc_score(
        test_true,
        test_probs[:, 1],
    )

    test_auprc = average_precision_score(
        test_true,
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
        f"eicu_v1_transformer_binary_"
        f"D{args.d_model}_H{args.nhead}_L{args.layers}_FF{args.ff_dim}_"
        f"Fdyn{dyn_dim}_Fstat{static_dim}"
    )

    model_path = os.path.join(OUT_DIR, f"transformer_binary_{stamp}.pt")
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
            "num_classes": 2,
            "label": "binary",
            "best_epoch": int(best_epoch),
            "best_val_f1": float(best_val_f1),
            "best_val_loss": float(best_val_loss),
            "best_threshold": float(best_threshold),
            "test_auroc": float(test_auroc),
            "test_auprc": float(test_auprc),
        },
        model_path,
    )

    metrics = {
        "dataset": "eICU",
        "model": "Transformer + static ANN fusion binary",
        "label": "binary",
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_f1_macro": float(test_f1_macro),
        "test_auroc": float(test_auroc),
        "test_auprc": float(test_auprc),
        "classification_report": report,
        "confusion_matrix": cm,
        "best_epoch": int(best_epoch),
        "best_val_f1": float(best_val_f1),
        "best_val_loss": float(best_val_loss),
        "best_threshold": float(best_threshold),
        "elapsed_sec": round(time.time() - t0, 1),
        "class_counts_train": class_counts.tolist(),
        "train_shape_dyn": list(X_dyn_train.shape),
        "valid_shape_dyn": list(X_dyn_valid.shape),
        "test_shape_dyn": list(X_dyn_test.shape),
        "train_shape_static": list(X_static_train.shape),
        "input_dyn_dim": int(dyn_dim),
        "input_static_dim": int(static_dim),
        "checkpoint_metric": "validation_macro_f1",
        "binary_threshold_tuned_on": "validation_macro_f1",
        "output_dir": OUT_DIR,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model:   {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(
        f"Test Acc: {test_acc:.4f} | "
        f"F1 macro: {test_f1_macro:.4f} | "
        f"AUROC: {test_auroc:.4f} | "
        f"AUPRC: {test_auprc:.4f} | "
        f"Test loss: {test_loss:.4f} | "
        f"Best epoch: {best_epoch} | "
        f"Best val F1: {best_val_f1:.4f} | "
        f"Threshold: {best_threshold:.2f}"
    )


if __name__ == "__main__":
    main()
