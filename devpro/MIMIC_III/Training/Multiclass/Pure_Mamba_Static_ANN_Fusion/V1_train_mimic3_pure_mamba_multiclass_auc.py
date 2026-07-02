#!/usr/bin/env python3
"""
V1_train_mimic3_pure_mamba_multiclass_auc.py

MIMIC-III Pure BiMamba + Static ANN Fusion plain 3-class (multiclass) LOS
prediction. Plain softmax/CrossEntropy over 3 classes -- NOT the
cumulative-logit ordinal formulation under Training/Ordinal/.
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

from sklearn.metrics import classification_report, confusion_matrix

from mamba_ssm.modules.mamba_simple import Mamba

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from model_common import (
    build_standard_head,
    build_standard_optimizer,
    build_standard_scheduler,
    compute_multiclass_metrics,
    STANDARD_HPARAMS,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", "..", "..", ".."))

BASE = os.path.join(_PROJECT_ROOT, "processed_data", "structured_LOS_dynamic")
DATA_DIR = f"{BASE}/lstm_data_v12.1_seq"
OUT_DIR = os.path.join(_PROJECT_ROOT, "results_timeseries", "mimic3_pure_mamba_multiclass_auc_models_v1")
os.makedirs(OUT_DIR, exist_ok=True)

N_CLASSES = 3


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def finite_np(x) -> bool:
    return x is not None and np.isfinite(x).all()


class DynStaticMulticlassDataset(Dataset):
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
        DataLoader(DynStaticMulticlassDataset(Xd_tr, Xs_tr, y_tr), shuffle=True, **common),
        DataLoader(DynStaticMulticlassDataset(Xd_va, Xs_va, y_va), shuffle=False, **common),
        DataLoader(DynStaticMulticlassDataset(Xd_te, Xs_te, y_te), shuffle=False, **common),
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


class PureMambaMulticlassModel(nn.Module):
    def __init__(self, f_dyn, f_static, d_model=192, layers=3):
        super().__init__()

        self.dyn = BiMambaEncoder(
            f_dyn=f_dyn, d_model=d_model, layers=layers,
            d_state=16, d_conv=4, expand=2, resid_scale=0.5,
        )

        self.stat = nn.Sequential(
            nn.Linear(f_static, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )

        self.head = build_standard_head(4 * d_model + 128, n_classes=N_CLASSES)

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
        return float("nan"), np.full((0, N_CLASSES), np.nan), np.array([]), False

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

    print("USING MIMIC-III PURE BIMAMBA + STATIC ANN FUSION MULTICLASS (3-CLASS SOFTMAX) WITH AUROC/AUPRC")
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

    class_counts = np.bincount(y_tr, minlength=N_CLASSES)
    print("Train class counts:", class_counts)

    weights = class_counts.sum() / (class_counts + 1e-9)
    weights = weights / weights.sum()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    print("Using weighted CrossEntropy (plain 3-way softmax, no ordinal structure).")
    print("Class weights:", weights.tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    tr_loader, va_loader, te_loader = make_loaders(
        Xd_tr, Xs_tr, y_tr,
        Xd_va, Xs_va, y_va,
        Xd_te, Xs_te, y_te,
        batch=args.batch,
        workers=args.workers,
    )

    model = PureMambaMulticlassModel(
        f_dyn=Xd_tr.shape[2],
        f_static=Xs_tr.shape[1],
        d_model=args.d_model,
        layers=args.layers,
    ).to(device)

    optimizer = build_standard_optimizer(model, lr=args.lr)
    scheduler = build_standard_scheduler(optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_val_f1_macro = -1.0
    best_val_loss = None
    best_state = None
    best_epoch = -1

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, _, _, _ = run_epoch(
            model, tr_loader, device, criterion,
            optimizer=optimizer, grad_clip=args.grad_clip, amp=args.amp, scaler=scaler,
        )

        va_loss, va_probs, va_true, valid_eval = run_epoch(
            model, va_loader, device, criterion,
            optimizer=None, grad_clip=args.grad_clip, amp=args.amp, scaler=None,
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
                f"val NaN/Inf detected — skipping checkpoint update"
            )
            if best_epoch != -1 and (epoch - best_epoch) >= args.patience:
                print("Early stopping.")
                break
            continue

        va_metrics, _ = compute_multiclass_metrics(va_true, va_probs, n_classes=N_CLASSES)
        scheduler.step(va_metrics["test_f1_macro"])

        print(
            f"Epoch {epoch:02d} | train {tr_loss:.4f} | val {va_loss:.4f} | "
            f"F1_macro {va_metrics['test_f1_macro']:.4f} | F1_micro {va_metrics['test_f1_micro']:.4f} | "
            f"F1_weighted {va_metrics['test_f1_weighted']:.4f} | Acc {va_metrics['test_acc']:.4f}"
        )

        if va_metrics["test_f1_macro"] > best_val_f1_macro:
            best_val_f1_macro = float(va_metrics["test_f1_macro"])
            best_val_loss = float(va_loss)
            best_epoch = epoch
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
        optimizer=None, grad_clip=args.grad_clip, amp=args.amp, scaler=None,
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

    te_metrics, te_pred = compute_multiclass_metrics(te_true, te_probs, n_classes=N_CLASSES)

    rep = classification_report(te_true, te_pred, digits=4, output_dict=True, zero_division=0)
    cm = confusion_matrix(te_true, te_pred).tolist()

    stamp = (
        f"mimic3_pure_mamba_multiclass_"
        f"ep{best_epoch}_D{args.d_model}_L{args.layers}_"
        f"Fdyn{Xd_tr.shape[2]}_Fstat{Xs_tr.shape[1]}"
    )

    model_path = os.path.join(OUT_DIR, f"pure_mamba_multiclass_{stamp}.pt")
    metrics_path = os.path.join(OUT_DIR, f"metrics_{stamp}.json")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dyn_dim": int(Xd_tr.shape[2]),
            "input_static_dim": int(Xs_tr.shape[1]),
            "d_model": int(args.d_model),
            "layers": int(args.layers),
            "num_classes": N_CLASSES,
            "label": "multiclass_3class",
            "best_epoch": int(best_epoch),
            "best_val_f1_macro": float(best_val_f1_macro),
            "best_val_loss": float(best_val_loss),
            "model": "Pure BiMamba + static ANN fusion multiclass",
            **te_metrics,
        },
        model_path,
    )

    metrics = {
        "dataset": "MIMIC-III",
        "model": "Pure BiMamba + static ANN fusion multiclass",
        "label": "multiclass_3class",
        "test_loss": float(te_loss),
        **te_metrics,
        "classification_report": rep,
        "confusion_matrix": cm,
        "best_epoch": int(best_epoch),
        "best_val_f1_macro": float(best_val_f1_macro),
        "best_val_loss": float(best_val_loss),
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
        "output_dir": OUT_DIR,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model:   {model_path}")
    print(f"Saved metrics: {metrics_path}")

    print(
        f"Test Acc: {te_metrics['test_acc']:.4f} | F1 micro: {te_metrics['test_f1_micro']:.4f} | "
        f"F1 macro: {te_metrics['test_f1_macro']:.4f} | F1 weighted: {te_metrics['test_f1_weighted']:.4f} | "
        f"AUROC macro: {te_metrics['test_auroc_macro']:.4f} | AUPRC macro: {te_metrics['test_auprc_macro']:.4f} | "
        f"Test loss: {te_loss:.4f} | Best epoch: {best_epoch}"
    )


if __name__ == "__main__":
    main()
