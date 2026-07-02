#!/usr/bin/env python3
"""
V1_train_eicu_lstm_fusion_binary_auc.py
---------------------------------------

eICU BiLSTM + Static ANN Fusion Binary LOS training script
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
    /lustre/home/rahas2/mimic_projects/outputs/eicu_lstm_fusion_binary_auc_models_v1

Run:
    python V1_train_eicu_lstm_fusion_binary_auc.py --workers 0 --batch 128
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
OUT_DIR = f"{BASE}/eicu_lstm_fusion_binary_auc_models_v1"

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


def make_loaders(Xd_tr, Xs_tr, y_tr, Xd_va, Xs_va, y_va, Xd_te, Xs_te, y_te, batch, workers):
    kw = dict(
        batch_size=batch,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
    )

    return (
        DataLoader(DynStaticBinaryDataset(Xd_tr, Xs_tr, y_tr), shuffle=True, **kw),
        DataLoader(DynStaticBinaryDataset(Xd_va, Xs_va, y_va), shuffle=False, **kw),
        DataLoader(DynStaticBinaryDataset(Xd_te, Xs_te, y_te), shuffle=False, **kw),
    )


class FusionLSTMBinary(nn.Module):
    def __init__(
        self,
        input_dyn_dim: int,
        input_static_dim: int,
        hidden: int = 384,
        layers: int = 2,
        dropout: float = 0.3,
        static_hidden: int = 128,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dyn_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )

        dyn_out_dim = hidden * 2
        fused_dyn_dim = dyn_out_dim * 2

        self.static_mlp = nn.Sequential(
            nn.Linear(input_static_dim, static_hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(static_hidden),
            nn.Dropout(dropout),
            nn.Linear(static_hidden, static_hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(static_hidden),
        )

        fusion_in_dim = fused_dyn_dim + static_hidden

        self.fusion_ln = nn.LayerNorm(fusion_in_dim)
        self.fusion_gate = nn.Linear(fusion_in_dim, fusion_in_dim)

        self.head = nn.Sequential(
            nn.Linear(fusion_in_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x_dyn, x_static):
        out, (hn, _) = self.lstm(x_dyn)

        h_forward = hn[-2]
        h_backward = hn[-1]

        h_last = torch.cat([h_forward, h_backward], dim=1)
        h_mean = out.mean(dim=1)
        h_dyn = torch.cat([h_last, h_mean], dim=1)

        h_static = self.static_mlp(x_static)

        fused = torch.cat([h_dyn, h_static], dim=1)
        fused = self.fusion_ln(fused)

        gate = torch.sigmoid(self.fusion_gate(fused))
        return self.head(fused * gate)


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


def train_epoch(model, loader, device, criterion, optimizer, scaler=None, grad_clip=1.0, amp=False):
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
    targets = []

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

        bs = y.size(0)
        loss_sum += float(loss.detach().cpu().item()) * bs
        n += bs

        probs_all.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())

    return loss_sum / max(n, 1), np.concatenate(probs_all), np.concatenate(targets)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=1.0)

    args = ap.parse_args()

    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("USING eICU LSTM + ANN FUSION BINARY WITH AUROC/AUPRC")
    print(f"Using device: {device}")
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
        args.batch,
        args.workers,
    )

    model = FusionLSTMBinary(
        input_dyn_dim=Xd_tr.shape[2],
        input_static_dim=Xs_tr.shape[1],
        hidden=args.hidden,
        layers=args.layers,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_threshold = 0.5
    best_state = None
    best_epoch = -1

    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(
            model,
            tr,
            device,
            criterion,
            optimizer,
            scaler=scaler,
            grad_clip=args.grad_clip,
            amp=(args.amp and device.type == "cuda"),
        )

        va_loss, va_probs, va_true = eval_epoch(
            model,
            va,
            device,
            criterion,
            amp=(args.amp and device.type == "cuda"),
        )

        info = tune_binary_threshold(va_true, va_probs[:, 1])
        scheduler.step(info["f1"])

        print(
            f"Epoch {ep:02d} | "
            f"train {tr_loss:.4f} | "
            f"val {va_loss:.4f} | "
            f"F1 {info['f1']:.4f} | "
            f"Acc {info['acc']:.4f} | "
            f"thr={info['threshold']:.2f}"
        )

        if info["f1"] > best_val_f1:
            best_val_f1 = float(info["f1"])
            best_val_loss = float(va_loss)
            best_threshold = float(info["threshold"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep

        elif ep - best_epoch >= args.patience:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError("No valid model checkpoint was produced.")

    model.load_state_dict(best_state)

    te_loss, te_probs, te_true = eval_epoch(
        model,
        te,
        device,
        criterion,
        amp=(args.amp and device.type == "cuda"),
    )

    prob_pos = te_probs[:, 1]
    te_pred = (prob_pos >= best_threshold).astype(np.int64)

    te_acc = accuracy_score(te_true, te_pred)
    te_f1_macro = f1_score(te_true, te_pred, average="macro", zero_division=0)
    te_f1_weighted = f1_score(te_true, te_pred, average="weighted", zero_division=0)

    te_auroc = roc_auc_score(te_true, prob_pos)
    te_auprc = average_precision_score(te_true, prob_pos)

    rep = classification_report(
        te_true,
        te_pred,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(te_true, te_pred).tolist()

    stamp = (
        f"eicu_v1_lstm_ann_binary_auc_"
        f"H{args.hidden}_L{args.layers}_Fdyn{Xd_tr.shape[2]}_Fstat{Xs_tr.shape[1]}"
    )

    model_path = os.path.join(OUT_DIR, f"lstm_ann_binary_auc_{stamp}.pt")
    metrics_path = os.path.join(OUT_DIR, f"metrics_{stamp}.json")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dyn_dim": int(Xd_tr.shape[2]),
            "input_static_dim": int(Xs_tr.shape[1]),
            "hidden": int(args.hidden),
            "layers": int(args.layers),
            "num_classes": 2,
            "label": "binary",
            "best_epoch": int(best_epoch),
            "best_val_f1": float(best_val_f1),
            "best_val_loss": float(best_val_loss),
            "best_threshold": float(best_threshold),
            "model": "BiLSTM + static ANN fusion binary",
            "test_auroc": float(te_auroc),
            "test_auprc": float(te_auprc),
        },
        model_path,
    )

    metrics = {
        "dataset": "eICU",
        "model": "BiLSTM + static ANN fusion binary",
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
        "threshold_grid": "np.arange(0.05, 0.96, 0.01)",
        "elapsed_sec": round(time.time() - t0, 1),
        "class_counts_train": class_counts.tolist(),
        "class_weights": weights.tolist(),
        "train_shape_dyn": list(Xd_tr.shape),
        "valid_shape_dyn": list(Xd_va.shape),
        "test_shape_dyn": list(Xd_te.shape),
        "train_shape_static": list(Xs_tr.shape),
        "input_dyn_dim": int(Xd_tr.shape[2]),
        "input_static_dim": int(Xs_tr.shape[1]),
        "checkpoint_metric": "validation_macro_f1",
        "binary_threshold_tuned_on": "validation_macro_f1",
        "positive_class": "LOS_gt_7_days",
        "output_dir": OUT_DIR,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model:   {model_path}")
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
