#!/usr/bin/env python3
"""
V1_train_mimic3_patchtst_fusion_binary_auc.py
----------------------------------------

PatchTST-style Transformer + static ANN fusion model for MIMIC-III binary LOS prediction with AUROC/AUPRC.

Reads the same tensors as the other MIMIC-III scripts:
    /lustre/home/rahas2/mimic_projects/outputs/lstm_data_v12.1_seq

Expected files:
    X_dyn_train.npy / valid / test
    X_static_train.npy / valid / test
    y_bin_train.npy / valid / test

Task:
    0 = LOS <= 7 days
    1 = LOS > 7 days

Architecture:
    - PatchTST-style temporal patches over 24 hourly dynamic sequence
    - overlapping patches, default patch_len=4, stride=2
    - patch projection to d_model
    - learned positional embeddings
    - Transformer encoder over patch tokens
    - static ANN branch
    - gated fusion head with 2 class logits
    - validation threshold tuning for macro F1

Output folder:
    /lustre/home/rahas2/mimic_projects/outputs/mimic3_patchtst_fusion_binary_auc_models_v1

Note:
    This is PatchTST-style, adapted for short ICU 24-hour sequences.
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

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score, average_precision_score

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from model_common import build_standard_head, build_standard_optimizer, build_standard_scheduler, STANDARD_HPARAMS

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", "..", "..", ".."))

BASE = os.path.join(_PROJECT_ROOT, "processed_data", "structured_LOS_dynamic")
DATA_DIR = f"{BASE}/lstm_data_v12.1_seq"
OUT_DIR = os.path.join(_PROJECT_ROOT, "results_LOS_dynamic", "mimic3_patchtst_fusion_binary_auc_models_v1")
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


def binary_probs_to_class(prob_pos, threshold):
    return (prob_pos >= threshold).astype(np.int64)


def tune_binary_threshold(y_true, prob_pos):
    best = {"f1": -1.0, "acc": 0.0, "threshold": 0.5}
    for t in np.arange(0.05, 0.96, 0.01):
        pred = binary_probs_to_class(prob_pos, t)
        f1 = f1_score(y_true, pred, average="macro", zero_division=0)
        if f1 > best["f1"]:
            best = {
                "f1": float(f1),
                "acc": float(accuracy_score(y_true, pred)),
                "threshold": float(t),
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
        self.head = build_standard_head(fusion_dim, n_classes=2)

    def make_patches(self, x_dyn):
        # x_dyn: (B,T,F) -> (B,num_patches,patch_len*F)
        patches = x_dyn.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # PyTorch gives (B, num_patches, F, patch_len). Move to (B,num_patches,patch_len,F)
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
    loss_sum, n = 0.0, 0
    for Xd, Xs, y in loader:
        Xd, Xs, y = Xd.to(device, non_blocking=True), Xs.to(device, non_blocking=True), y.to(device, non_blocking=True)
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

        bs = y.size(0)
        loss_sum += loss.item() * bs
        n += bs
    return loss_sum / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device, criterion, amp=False):
    model.eval()
    loss_sum, n = 0.0, 0
    probs_all, y_all = [], []
    for Xd, Xs, y in loader:
        Xd, Xs, y = Xd.to(device, non_blocking=True), Xs.to(device, non_blocking=True), y.to(device, non_blocking=True)
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
    return loss_sum / max(n, 1), np.concatenate(probs_all), np.concatenate(y_all)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=STANDARD_HPARAMS["max_epochs"])
    parser.add_argument("--batch", type=int, default=STANDARD_HPARAMS["batch_size"])
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=STANDARD_HPARAMS["lr"])
    parser.add_argument("--patience", type=int, default=STANDARD_HPARAMS["patience"])
    parser.add_argument("--patch_len", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--ff_dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--grad_clip", type=float, default=STANDARD_HPARAMS["grad_clip"])
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("USING MIMIC-III PATCHTST-STYLE TRANSFORMER + ANN FUSION BINARY WITH AUROC/AUPRC")
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

    class_counts = np.bincount(y_tr.astype(np.int64), minlength=2)
    print("Train class counts:", class_counts)

    n_total = class_counts.sum()
    class_weights = n_total / np.maximum(2.0 * class_counts, 1.0)
    class_weights = class_weights.astype(np.float32)

    print("Using CrossEntropyLoss with validation threshold tuning.")
    print("Class weights:", class_weights.tolist())

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

    tr, va, te = make_loaders(Xd_tr, Xs_tr, y_tr, Xd_va, Xs_va, y_va, Xd_te, Xs_te, y_te, batch=args.batch, workers=args.workers)

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

    print(f"Patch config: seq_len={seq_len}, patch_len={args.patch_len}, stride={args.stride}, num_patches={model.num_patches}")

    optimizer = build_standard_optimizer(model, lr=args.lr)
    scheduler = build_standard_scheduler(optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_threshold = 0.5
    best_state = None
    best_epoch = -1
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, tr, device, criterion, optimizer, scheduler=scheduler, scaler=scaler, amp=(args.amp and device.type == "cuda"), grad_clip=args.grad_clip)
        val_loss, val_probs, val_true = eval_epoch(model, va, device, criterion, amp=(args.amp and device.type == "cuda"))
        threshold_info = tune_binary_threshold(val_true, val_probs[:, 1])
        val_f1 = threshold_info["f1"]
        val_acc = threshold_info["acc"]
        scheduler.step(val_f1)
        print(
            f"Epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f} | "
            f"F1 {val_f1:.4f} | Acc {val_acc:.4f} | thr={threshold_info['threshold']:.2f}"
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
    test_loss, test_probs, test_true = eval_epoch(model, te, device, criterion, amp=(args.amp and device.type == "cuda"))

    test_prob_pos = test_probs[:, 1]
    test_pred = binary_probs_to_class(test_prob_pos, best_threshold)

    test_acc = accuracy_score(test_true, test_pred)
    test_f1_macro = f1_score(test_true, test_pred, average="macro", zero_division=0)
    test_f1_weighted = f1_score(test_true, test_pred, average="weighted", zero_division=0)

    test_auroc = roc_auc_score(test_true, test_prob_pos)
    test_auprc = average_precision_score(test_true, test_prob_pos)

    report = classification_report(test_true, test_pred, digits=4, output_dict=True, zero_division=0)
    cm = confusion_matrix(test_true, test_pred).tolist()

    stamp = f"mimic3_patchtst_binary_P{args.patch_len}_S{args.stride}_D{args.d_model}_H{args.nhead}_L{args.layers}_FF{args.ff_dim}_Fdyn{dyn_dim}_Fstat{static_dim}"
    model_path = os.path.join(OUT_DIR, f"patchtst_binary_{stamp}.pt")
    metrics_path = os.path.join(OUT_DIR, f"metrics_{stamp}.json")

    torch.save({
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
    }, model_path)

    metrics = {
        "dataset": "MIMIC-III",
        "model": "PatchTST-style Transformer + static ANN fusion binary",
        "label": "binary",
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_f1_macro": float(test_f1_macro),
        "test_f1_weighted": float(test_f1_weighted),
        "classification_report": report,
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
        "test_auroc": float(test_auroc),
        "test_auprc": float(test_auprc)
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model:   {model_path}")
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
