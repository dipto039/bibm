#!/usr/bin/env python3
"""
V1_train_eicu_hybrid_lstm_mamba_binary_auc.py

Hybrid BiLSTM + BiMamba + Static ANN Fusion for eICU binary LOS prediction.

Task:
    0 = LOS <= 7 days
    1 = LOS > 7 days

Reads:
    /lustre/home/rahas2/mimic_projects/outputs/eicu_lstm_data_v1

Expected files:
    X_dyn_train.npy / valid / test
    X_static_train.npy / valid / test
    y_bin_train.npy / valid / test

Outputs:
    /lustre/home/rahas2/mimic_projects/outputs/eicu_hybrid_lstm_mamba_binary_auc_models_v1

Run:
    python V1_train_eicu_hybrid_lstm_mamba_binary_auc.py --workers 0 --batch 128
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


BASE = "/lustre/home/rahas2/mimic_projects/outputs"
DATA_DIR = f"{BASE}/eicu_lstm_data_v1"
OUT_DIR = f"{BASE}/eicu_hybrid_lstm_mamba_binary_auc_models_v1"
os.makedirs(OUT_DIR, exist_ok=True)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DynStaticBinaryDataset(Dataset):
    def __init__(self, X_dyn, X_static, y_bin, copy_mmap: bool = True):
        self.Xd = X_dyn
        self.Xs = X_static
        self.y = y_bin.astype(np.int64)
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
        DynStaticBinaryDataset(Xd_tr, Xs_tr, y_tr, copy_mmap=copy_mmap),
        shuffle=True,
        **common,
    )

    va = DataLoader(
        DynStaticBinaryDataset(Xd_va, Xs_va, y_va, copy_mmap=copy_mmap),
        shuffle=False,
        **common,
    )

    te = DataLoader(
        DynStaticBinaryDataset(Xd_te, Xs_te, y_te, copy_mmap=copy_mmap),
        shuffle=False,
        **common,
    )

    return tr, va, te


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

        self.mamba = BiMamba(
            f_dyn,
            d_model=mamba_dim,
            layers=2,
            resid_scale=0.5,
        )

    def forward(self, x):
        out, (hn, _) = self.lstm(x)

        # hn[-2], hn[-1] = final forward/backward states: 256 + 256
        # out.mean = bidirectional hidden: 512
        h_lstm = torch.cat([hn[-2], hn[-1], out.mean(dim=1)], dim=1)

        # 4 * mamba_dim = forward last, backward last, forward mean, backward mean
        h_mamba = self.mamba(x)

        h = torch.cat([h_lstm, h_mamba], dim=1)

        return F.layer_norm(h, (h.size(1),))


class HybridBinaryModel(nn.Module):
    def __init__(self, f_dyn: int, f_static: int):
        super().__init__()

        self.dyn = HybridDynEncoder(
            f_dyn,
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


def tune_binary_threshold(y_true: np.ndarray, prob_pos: np.ndarray):
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

    return best_t, best_f1, best_acc


def run_epoch(model, loader, device, criterion, optimizer=None, grad_clip=1.0, amp=False, scaler=None):
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
            probs_all.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
            y_all.append(y.detach().cpu().numpy())

    if training:
        return loss_sum / max(n, 1), None, None

    probs = np.concatenate(probs_all) if probs_all else np.empty((0, 2))
    y_true = np.concatenate(y_all) if y_all else np.array([])

    return loss_sum / max(n, 1), probs, y_true


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

    print("USING eICU HYBRID LSTM + MAMBA BINARY WITH AUROC/AUPRC")
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

    tr_loader, va_loader, te_loader = make_loaders(
        Xd_tr, Xs_tr, y_tr,
        Xd_va, Xs_va, y_va,
        Xd_te, Xs_te, y_te,
        batch=args.batch,
        workers=args.workers,
        copy_mmap=True,
    )

    model = HybridBinaryModel(
        f_dyn=Xd_tr.shape[2],
        f_static=Xs_tr.shape[1],
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
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
    best_threshold = 0.5

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

        threshold, va_f1, va_acc = tune_binary_threshold(
            va_true,
            va_probs[:, 1],
        )

        print(
            f"Epoch {epoch:02d} | train {tr_loss:.4f} | val {va_loss:.4f} | "
            f"F1 {va_f1:.4f} | Acc {va_acc:.4f} | thr={threshold:.2f}"
        )

        if va_f1 > best_val_f1:
            best_val_f1 = float(va_f1)
            best_val_loss = float(va_loss)
            best_epoch = epoch
            best_threshold = float(threshold)
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

    te_pred = (te_probs[:, 1] >= best_threshold).astype(np.int64)

    te_acc = accuracy_score(te_true, te_pred)
    te_f1 = f1_score(te_true, te_pred, average="macro", zero_division=0)
    te_auroc = roc_auc_score(te_true, te_probs[:, 1])
    te_auprc = average_precision_score(te_true, te_probs[:, 1])

    rep = classification_report(
        te_true,
        te_pred,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(te_true, te_pred).tolist()

    stamp = f"eicu_v1_binary_mamba_ep{best_epoch}_Fdyn{Xd_tr.shape[2]}_Fstat{Xs_tr.shape[1]}"

    model_path = os.path.join(OUT_DIR, f"hybrid_binary_{stamp}.pt")
    metrics_path = os.path.join(OUT_DIR, f"metrics_{stamp}.json")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dyn_dim": int(Xd_tr.shape[2]),
            "input_static_dim": int(Xs_tr.shape[1]),
            "num_classes": 2,
            "best_epoch": int(best_epoch),
            "best_threshold": float(best_threshold),
            "model": "Hybrid BiLSTM + BiMamba + static ANN fusion binary",
            "test_auroc": float(te_auroc),
            "test_auprc": float(te_auprc),
        },
        model_path,
    )

    metrics = {
        "dataset": "eICU",
        "model": "Hybrid BiLSTM + BiMamba + static ANN fusion binary",
        "label": "binary",
        "test_loss": float(te_loss),
        "test_acc": float(te_acc),
        "test_f1_macro": float(te_f1),
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
        f"F1 macro: {te_f1:.4f} | "
        f"AUROC: {te_auroc:.4f} | "
        f"AUPRC: {te_auprc:.4f} | "
        f"Test loss: {te_loss:.4f} | "
        f"Best epoch: {best_epoch} | "
        f"Threshold: {best_threshold:.2f}"
    )


if __name__ == "__main__":
    main()
