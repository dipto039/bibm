#!/usr/bin/env python3
"""
V1_train_mimic3_pure_mamba_ordinal_auc.py

MIMIC-III Pure BiMamba + Static ANN Fusion Ordinal 3-class LOS prediction.

Run:
    python V1_train_mimic3_pure_mamba_ordinal_auc.py --workers 0 --batch 128
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
DATA_DIR = f"{BASE}/lstm_data_v12.1_seq"
OUT_DIR = f"{BASE}/mimic3_pure_mamba_ordinal_auc_models_v1"
os.makedirs(OUT_DIR, exist_ok=True)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def make_loaders(Xd_tr, Xs_tr, y_tr, Xd_va, Xs_va, y_va, Xd_te, Xs_te, y_te, batch, workers):
    common = dict(batch_size=batch, num_workers=workers, pin_memory=True, persistent_workers=(workers > 0))
    return (
        DataLoader(DynStaticOrdinalDataset(Xd_tr, Xs_tr, y_tr), shuffle=True, **common),
        DataLoader(DynStaticOrdinalDataset(Xd_va, Xs_va, y_va), shuffle=False, **common),
        DataLoader(DynStaticOrdinalDataset(Xd_te, Xs_te, y_te), shuffle=False, **common),
    )


def ordinal_targets(y: torch.Tensor) -> torch.Tensor:
    y = y.long()
    return torch.stack([(y > 0).float(), (y > 1).float()], dim=1)


class OrdinalBCELoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float32))
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
    best = {"f1": -1.0, "acc": 0.0, "thr1": 0.5, "thr2": 0.5}
    for t1 in np.arange(0.30, 0.86, 0.02):
        for t2 in np.arange(0.30, 0.91, 0.02):
            pred = ordinal_probs_to_class(prob_gt3, prob_gt7, t1, t2)
            f1 = f1_score(y_true, pred, average="macro", zero_division=0)
            if f1 > best["f1"]:
                best = {
                    "f1": float(f1),
                    "acc": float(accuracy_score(y_true, pred)),
                    "thr1": float(t1),
                    "thr2": float(t2),
                }
    return best


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


class PureMambaOrdinalModel(nn.Module):
    def __init__(self, f_dyn, f_static, d_model=192, layers=3):
        super().__init__()
        self.dyn = BiMambaEncoder(f_dyn=f_dyn, d_model=d_model, layers=layers)
        self.stat = nn.Sequential(
            nn.Linear(f_static, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )
        self.head = nn.Sequential(
            nn.Linear(4 * d_model + 128, 256),
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
            probs_all.append(torch.sigmoid(logits).detach().cpu().numpy())
            y_all.append(y.detach().cpu().numpy())

    if training:
        return loss_sum / max(n, 1), None, None

    return loss_sum / max(n, 1), np.concatenate(probs_all), np.concatenate(y_all)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=3)
    args = parser.parse_args()

    seed_everything(42)
    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")

    print("USING MIMIC-III PURE BIMAMBA + STATIC ANN FUSION ORDINAL WITH AUROC/AUPRC")
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

    y_ord_train = np.stack([(y_tr > 0).astype(np.float32), (y_tr > 1).astype(np.float32)], axis=1)
    pos = y_ord_train.sum(axis=0)
    neg = y_ord_train.shape[0] - pos
    pos_weight = neg / np.maximum(pos, 1.0)
    pos_weight = np.minimum(pos_weight, np.array([6.0, 12.0], dtype=np.float32))
    print("Using ordinal BCE loss with validation threshold tuning.")
    print("Ordinal pos_weight:", pos_weight.tolist())
    criterion = OrdinalBCELoss(pos_weight=pos_weight).to(device)

    tr_loader, va_loader, te_loader = make_loaders(
        Xd_tr, Xs_tr, y_tr, Xd_va, Xs_va, y_va, Xd_te, Xs_te, y_te,
        batch=args.batch, workers=args.workers
    )

    model = PureMambaOrdinalModel(
        f_dyn=Xd_tr.shape[2],
        f_static=Xs_tr.shape[1],
        d_model=args.d_model,
        layers=args.layers,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_val_f1 = -1.0
    best_val_loss = None
    best_state = None
    best_epoch = -1
    best_thresholds = (0.5, 0.5)
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, _, _ = run_epoch(model, tr_loader, device, criterion, optimizer=optimizer, grad_clip=args.grad_clip, amp=args.amp, scaler=scaler)
        va_loss, va_probs, va_true = run_epoch(model, va_loader, device, criterion, optimizer=None, grad_clip=args.grad_clip, amp=args.amp, scaler=None)
        info = tune_ordinal_thresholds(va_true, va_probs[:, 0], va_probs[:, 1])
        print(
            f"Epoch {epoch:02d} | train {tr_loss:.4f} | val {va_loss:.4f} | "
            f"F1 {info['f1']:.4f} | Acc {info['acc']:.4f} | "
            f"t=({info['thr1']:.2f}, {info['thr2']:.2f})"
        )
        if info["f1"] > best_val_f1:
            best_val_f1 = float(info["f1"])
            best_val_loss = float(va_loss)
            best_epoch = epoch
            best_thresholds = (float(info["thr1"]), float(info["thr2"]))
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif best_epoch != -1 and (epoch - best_epoch) >= args.patience:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint produced.")

    model.load_state_dict(best_state)
    te_loss, te_probs, te_true = run_epoch(model, te_loader, device, criterion, optimizer=None, grad_clip=args.grad_clip, amp=args.amp, scaler=None)
    te_pred = ordinal_probs_to_class(te_probs[:, 0], te_probs[:, 1], best_thresholds[0], best_thresholds[1])

    te_acc = accuracy_score(te_true, te_pred)
    te_f1_macro = f1_score(te_true, te_pred, average="macro", zero_division=0)
    te_f1_weighted = f1_score(te_true, te_pred, average="weighted", zero_division=0)

    y_te_ord = np.stack([(te_true > 0).astype(int), (te_true > 1).astype(int)], axis=1)
    te_auroc_macro = roc_auc_score(y_te_ord, te_probs, average="macro")
    te_auprc_macro = average_precision_score(y_te_ord, te_probs, average="macro")
    te_auroc_gt3 = roc_auc_score(y_te_ord[:, 0], te_probs[:, 0])
    te_auroc_gt7 = roc_auc_score(y_te_ord[:, 1], te_probs[:, 1])
    te_auprc_gt3 = average_precision_score(y_te_ord[:, 0], te_probs[:, 0])
    te_auprc_gt7 = average_precision_score(y_te_ord[:, 1], te_probs[:, 1])

    rep = classification_report(te_true, te_pred, digits=4, output_dict=True, zero_division=0)
    cm = confusion_matrix(te_true, te_pred).tolist()

    stamp = f"mimic3_pure_mamba_ordinal_3class_ep{best_epoch}_D{args.d_model}_L{args.layers}_Fdyn{Xd_tr.shape[2]}_Fstat{Xs_tr.shape[1]}"
    model_path = os.path.join(OUT_DIR, f"pure_mamba_ordinal_{stamp}.pt")
    metrics_path = os.path.join(OUT_DIR, f"metrics_{stamp}.json")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dyn_dim": int(Xd_tr.shape[2]),
            "input_static_dim": int(Xs_tr.shape[1]),
            "d_model": int(args.d_model),
            "layers": int(args.layers),
            "num_classes_original": 3,
            "num_ordinal_logits": 2,
            "label": "3class_ordinal",
            "best_epoch": int(best_epoch),
            "best_val_f1": float(best_val_f1),
            "best_val_loss": float(best_val_loss),
            "best_thresholds": list(best_thresholds),
            "model": "Pure BiMamba + static ANN fusion ordinal",
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
        "model": "Pure BiMamba + static ANN fusion ordinal",
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
        "seq_len": int(Xd_tr.shape[1]),
        "d_model": int(args.d_model),
        "layers": int(args.layers),
        "checkpoint_metric": "validation_macro_f1",
        "ordinal_targets": ["LOS_gt_3_days", "LOS_gt_7_days"],
        "output_dir": OUT_DIR,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model:   {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(
        f"Test Acc: {te_acc:.4f} | F1 macro: {te_f1_macro:.4f} | "
        f"F1 weighted: {te_f1_weighted:.4f} | AUROC macro: {te_auroc_macro:.4f} | "
        f"AUPRC macro: {te_auprc_macro:.4f} | Test loss: {te_loss:.4f} | "
        f"Best epoch: {best_epoch} | Best val F1: {best_val_f1:.4f} | "
        f"Thresholds: {best_thresholds}"
    )
    print(
        f"Ordinal AUROC: gt3={te_auroc_gt3:.4f}, gt7={te_auroc_gt7:.4f} | "
        f"Ordinal AUPRC: gt3={te_auprc_gt3:.4f}, gt7={te_auprc_gt7:.4f}"
    )


if __name__ == "__main__":
    main()
