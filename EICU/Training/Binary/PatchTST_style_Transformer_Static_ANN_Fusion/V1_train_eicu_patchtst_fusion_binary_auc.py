#!/usr/bin/env python3
"""
V1_train_eicu_patchtst_fusion_binary_auc.py
-------------------------------------------

PatchTST-style Transformer + static ANN fusion model for eICU binary LOS prediction
with AUROC/AUPRC reporting and validation macro-F1 threshold tuning.

Task:
    Binary LOS:
        0 = LOS <= 7 days
        1 = LOS > 7 days

Reads:
    /lustre/home/rahas2/mimic_projects/outputs/eicu_lstm_data_v1

Expected files:
    X_dyn_train.npy / valid / test
    X_static_train.npy / valid / test
    y_bin_train.npy / valid / test

Outputs:
    /lustre/home/rahas2/mimic_projects/outputs/eicu_patchtst_fusion_binary_auc_models_v1

Run:
    python V1_train_eicu_patchtst_fusion_binary_auc.py --workers 0 --batch 128
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

BASE = "/lustre/home/rahas2/mimic_projects/outputs"
DATA_DIR = f"{BASE}/eicu_lstm_data_v1"
OUT_DIR = f"{BASE}/eicu_patchtst_fusion_binary_auc_models_v1"
os.makedirs(OUT_DIR, exist_ok=True)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def make_loaders(Xd_tr, Xs_tr, y_tr, Xd_va, Xs_va, y_va, Xd_te, Xs_te, y_te, batch=256, workers=0):
    kw = dict(batch_size=batch, num_workers=workers, pin_memory=True, persistent_workers=(workers > 0))
    tr = DataLoader(DynStaticBinaryDataset(Xd_tr, Xs_tr, y_tr), shuffle=True, **kw)
    va = DataLoader(DynStaticBinaryDataset(Xd_va, Xs_va, y_va), shuffle=False, **kw)
    te = DataLoader(DynStaticBinaryDataset(Xd_te, Xs_te, y_te), shuffle=False, **kw)
    return tr, va, te


def tune_binary_threshold(y_true, prob_pos):
    best = {"threshold": 0.5, "f1": -1.0, "acc": 0.0}

    for t in np.arange(0.05, 0.96, 0.01):
        pred = (prob_pos >= t).astype(np.int64)
        f1 = f1_score(y_true, pred, average="macro", zero_division=0)

        if f1 > best["f1"]:
            best = {
                "threshold": float(t),
                "f1": float(f1),
                "acc": float(accuracy_score(y_true, pred)),
            }

    return best


class PatchTSTBinaryFusion(nn.Module):
    """PatchTST-style encoder for multivariate ICU time series plus static ANN fusion."""

    def __init__(
        self,
        dyn_dim: int,
        static_dim: int,
        seq_len: int = 24,
        patch_len: int = 4,
        stride: int = 2,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 3,
        ff_dim: int = 384,
        dropout: float = 0.15,
        static_hidden: int = 128,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        if patch_len > seq_len:
            raise ValueError("patch_len cannot exceed seq_len.")

        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = ((seq_len - patch_len) // stride) + 1

        self.dyn_norm = nn.LayerNorm(dyn_dim)
        self.static_norm = nn.LayerNorm(static_dim)

        self.patch_proj = nn.Sequential(
            nn.Linear(patch_len * dyn_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        self.pos_embed = nn.Embedding(self.num_patches, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dyn_pool_ln = nn.LayerNorm(d_model)

        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, static_hidden),
            nn.GELU(),
            nn.LayerNorm(static_hidden),
            nn.Dropout(dropout),
            nn.Linear(static_hidden, static_hidden),
            nn.GELU(),
            nn.LayerNorm(static_hidden),
        )

        fusion_dim = d_model + static_hidden

        self.fusion_ln = nn.LayerNorm(fusion_dim)
        self.fusion_gate = nn.Linear(fusion_dim, fusion_dim)

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def make_patches(self, x_dyn):
        patches = x_dyn.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        return patches.view(patches.size(0), patches.size(1), -1)

    def forward(self, x_dyn, x_static):
        _, T, _ = x_dyn.shape

        if T != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got T={T}")

        x_dyn = self.dyn_norm(x_dyn)
        x_static = self.static_norm(x_static)

        patches = self.make_patches(x_dyn)
        tokens = self.patch_proj(patches)

        pos_ids = torch.arange(self.num_patches, device=x_dyn.device).unsqueeze(0)
        tokens = tokens + self.pos_embed(pos_ids)

        encoded = self.encoder(tokens)

        h_dyn = self.dyn_pool_ln(encoded.mean(dim=1))
        h_static = self.static_mlp(x_static)

        fused = torch.cat([h_dyn, h_static], dim=1)
        fused = self.fusion_ln(fused)
        fused = fused * torch.sigmoid(self.fusion_gate(fused))

        return self.head(fused)


def train_epoch(model, loader, device, criterion, optimizer, scheduler=None, scaler=None, amp=False, grad_clip=1.0):
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
        loss_sum += float(loss.detach().cpu().item()) * bs
        n += bs

    return loss_sum / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device, criterion, amp=False):
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
        loss_sum += float(loss.detach().cpu().item()) * bs
        n += bs

        probs_all.append(probs.detach().cpu().numpy())
        y_all.append(y.detach().cpu().numpy())

    return loss_sum / max(n, 1), np.concatenate(probs_all), np.concatenate(y_all)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--patch_len", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2)
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

    print("USING eICU PATCHTST-STYLE TRANSFORMER + ANN FUSION BINARY WITH AUROC/AUPRC")
    print(f"Device: {device}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"OUT_DIR: {OUT_DIR}")

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

    print("Using weighted CrossEntropyLoss with validation macro-F1 threshold tuning.")
    print("Class weights:", weights.tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    tr, va, te = make_loaders(
        Xd_tr, Xs_tr, y_tr,
        Xd_va, Xs_va, y_va,
        Xd_te, Xs_te, y_te,
        batch=args.batch,
        workers=args.workers,
    )

    dyn_dim = Xd_tr.shape[2]
    static_dim = Xs_tr.shape[1]
    seq_len = Xd_tr.shape[1]

    model = PatchTSTBinaryFusion(
        dyn_dim=dyn_dim,
        static_dim=static_dim,
        seq_len=seq_len,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        static_hidden=128,
    ).to(device)

    print(
        f"Patch config: seq_len={seq_len}, patch_len={args.patch_len}, "
        f"stride={args.stride}, num_patches={model.num_patches}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    total_steps = args.epochs * max(len(tr), 1)
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_lambda(step):
        return float(step + 1) / float(warmup_steps) if step < warmup_steps else 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_threshold = 0.5
    best_state = None
    best_epoch = -1

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            tr,
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
            va,
            device,
            criterion,
            amp=(args.amp and device.type == "cuda"),
        )

        threshold_info = tune_binary_threshold(val_true, val_probs[:, 1])
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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        elif best_epoch != -1 and (epoch - best_epoch) >= args.patience:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint produced.")

    model.load_state_dict(best_state)

    test_loss, test_probs, test_true = eval_epoch(
        model,
        te,
        device,
        criterion,
        amp=(args.amp and device.type == "cuda"),
    )

    prob_pos = test_probs[:, 1]
    test_pred = (prob_pos >= best_threshold).astype(np.int64)

    test_acc = accuracy_score(test_true, test_pred)
    test_f1_macro = f1_score(test_true, test_pred, average="macro", zero_division=0)
    test_f1_weighted = f1_score(test_true, test_pred, average="weighted", zero_division=0)

    test_auroc = roc_auc_score(test_true, prob_pos)
    test_auprc = average_precision_score(test_true, prob_pos)

    report = classification_report(test_true, test_pred, digits=4, output_dict=True, zero_division=0)
    cm = confusion_matrix(test_true, test_pred).tolist()

    stamp = (
        f"eicu_v1_patchtst_binary_auc_"
        f"P{args.patch_len}_S{args.stride}_D{args.d_model}_H{args.nhead}_L{args.layers}_"
        f"FF{args.ff_dim}_Fdyn{dyn_dim}_Fstat{static_dim}"
    )

    model_path = os.path.join(OUT_DIR, f"patchtst_binary_auc_{stamp}.pt")
    metrics_path = os.path.join(OUT_DIR, f"metrics_{stamp}.json")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "dyn_dim": int(dyn_dim),
            "static_dim": int(static_dim),
            "seq_len": int(seq_len),
            "patch_len": int(args.patch_len),
            "stride": int(args.stride),
            "num_patches": int(model.num_patches),
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
        "model": "PatchTST-style Transformer + static ANN fusion binary",
        "label": "binary",
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_f1_macro": float(test_f1_macro),
        "test_f1_weighted": float(test_f1_weighted),
        "test_auroc": float(test_auroc),
        "test_auprc": float(test_auprc),
        "classification_report": report,
        "confusion_matrix": cm,
        "best_epoch": int(best_epoch),
        "best_val_f1": float(best_val_f1),
        "best_val_loss": float(best_val_loss),
        "best_threshold": float(best_threshold),
        "threshold_grid": "np.arange(0.05, 0.96, 0.01)",
        "elapsed_sec": round(time.time() - t0, 1),
        "class_counts_train": class_counts.tolist(),
        "class_weights": weights.tolist(),
        "train_shape_dyn": list(Xd_tr.shape),
        "valid_shape_dyn": list(Xd_va.shape),
        "test_shape_dyn": list(Xd_te.shape),
        "train_shape_static": list(Xs_tr.shape),
        "input_dyn_dim": int(dyn_dim),
        "input_static_dim": int(static_dim),
        "seq_len": int(seq_len),
        "patch_len": int(args.patch_len),
        "stride": int(args.stride),
        "num_patches": int(model.num_patches),
        "checkpoint_metric": "validation_macro_f1",
        "binary_threshold_tuned_on": "validation_macro_f1",
        "positive_class": "LOS_gt_7_days",
        "output_dir": OUT_DIR,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\\nSaved model:   {model_path}")
    print(f"Saved metrics: {metrics_path}")

    print(
        f"Test Acc: {test_acc:.4f} | "
        f"F1 macro: {test_f1_macro:.4f} | "
        f"F1 weighted: {test_f1_weighted:.4f} | "
        f"AUROC: {test_auroc:.4f} | "
        f"AUPRC: {test_auprc:.4f} | "
        f"Test loss: {test_loss:.4f} | "
        f"Best epoch: {best_epoch} | "
        f"Best val F1: {best_val_f1:.4f} | "
        f"Threshold: {best_threshold:.2f}"
    )


if __name__ == "__main__":
    main()
