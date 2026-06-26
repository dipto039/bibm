#!/usr/bin/env python3
"""
V1_train_mimic3_lstm_fusion_ordinal_auc.py
----------------------------------------

MIMIC-III BiLSTM + Static ANN Fusion Ordinal 3-class LOS training script
with AUROC/AUPRC reporting.

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

Outputs:
    /lustre/home/rahas2/mimic_projects/outputs/mimic3_lstm_fusion_ordinal_auc_models_v1

Run:
    python V1_train_mimic3_lstm_fusion_ordinal_auc.py --workers 0 --batch 128
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

BASE = "/lustre/home/rahas2/mimic_projects/outputs"
DATA_DIR = f"{BASE}/lstm_data_v12.1_seq"
OUT_DIR = f"{BASE}/mimic3_lstm_fusion_ordinal_auc_models_v1"
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
    kw = dict(batch_size=batch, num_workers=workers, pin_memory=True, persistent_workers=(workers > 0))
    return (
        DataLoader(DynStaticOrdinalDataset(Xd_tr, Xs_tr, y_tr), shuffle=True, **kw),
        DataLoader(DynStaticOrdinalDataset(Xd_va, Xs_va, y_va), shuffle=False, **kw),
        DataLoader(DynStaticOrdinalDataset(Xd_te, Xs_te, y_te), shuffle=False, **kw),
    )


def ordinal_targets(y: torch.Tensor) -> torch.Tensor:
    y = y.long()
    return torch.stack([(y > 0).float(), (y > 1).float()], dim=1)


def ordinal_probs_to_class(prob_gt3, prob_gt7, thr1, thr2):
    pred = np.ones_like(prob_gt3, dtype=np.int64)
    pred[prob_gt3 < thr1] = 0
    pred[prob_gt7 >= thr2] = 2
    return pred


def tune_ordinal_thresholds(y_true, prob_gt3, prob_gt7):
    best = {"f1": -1.0, "thr1": 0.5, "thr2": 0.5, "acc": 0.0}
    for t1 in np.arange(0.30, 0.86, 0.02):
        for t2 in np.arange(0.30, 0.91, 0.02):
            pred = ordinal_probs_to_class(prob_gt3, prob_gt7, float(t1), float(t2))
            f1 = f1_score(y_true, pred, average="macro", zero_division=0)
            if f1 > best["f1"]:
                best = {
                    "f1": float(f1),
                    "thr1": float(t1),
                    "thr2": float(t2),
                    "acc": float(accuracy_score(y_true, pred)),
                }
    return best


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


class FusionLSTMOrdinal(nn.Module):
    def __init__(self, input_dyn_dim, input_static_dim, hidden=384, layers=2, dropout=0.3, static_hidden=128):
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
        h_last = torch.cat([hn[-2], hn[-1]], dim=1)
        h_mean = out.mean(dim=1)
        h_dyn = torch.cat([h_last, h_mean], dim=1)
        h_static = self.static_mlp(x_static)
        fused = torch.cat([h_dyn, h_static], dim=1)
        fused = self.fusion_ln(fused)
        gate = torch.sigmoid(self.fusion_gate(fused))
        return self.head(fused * gate)


def train_epoch(model, loader, device, criterion, optimizer, scaler=None, grad_clip=1.0, amp=False):
    model.train()
    loss_sum, n = 0.0, 0
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
    loss_sum, n = 0.0, 0
    probs, targets = [], []
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
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())
    return loss_sum / max(n, 1), np.concatenate(probs), np.concatenate(targets)


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

    print("USING MIMIC-III LSTM + ANN FUSION ORDINAL 3-CLASS WITH AUROC/AUPRC")
    print(f"Using device: {device}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"OUT_DIR: {OUT_DIR}")

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

    counts = np.bincount(y_tr, minlength=3)
    print("Train class counts:", counts)

    y_ord = np.stack([(y_tr > 0).astype(np.float32), (y_tr > 1).astype(np.float32)], axis=1)
    pos = y_ord.sum(axis=0)
    neg = y_ord.shape[0] - pos
    pos_weight = neg / np.maximum(pos, 1.0)
    pos_weight = np.minimum(pos_weight, np.array([6.0, 12.0], dtype=np.float32))
    print("Using ordinal BCE loss with validation threshold tuning.")
    print("Ordinal pos_weight:", pos_weight.tolist())

    criterion = OrdinalBCELoss(pos_weight=pos_weight).to(device)
    tr, va, te = make_loaders(Xd_tr, Xs_tr, y_tr, Xd_va, Xs_va, y_va, Xd_te, Xs_te, y_te, args.batch, args.workers)

    model = FusionLSTMOrdinal(
        input_dyn_dim=Xd_tr.shape[2],
        input_static_dim=Xs_tr.shape[1],
        hidden=args.hidden,
        layers=args.layers,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_thresholds = (0.5, 0.5)
    best_state = None
    best_epoch = -1
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, tr, device, criterion, opt, scaler=scaler, grad_clip=args.grad_clip, amp=(args.amp and device.type == "cuda"))
        va_loss, va_probs, va_true = eval_epoch(model, va, device, criterion, amp=(args.amp and device.type == "cuda"))
        info = tune_ordinal_thresholds(va_true, va_probs[:, 0], va_probs[:, 1])
        scheduler.step(info["f1"])
        print(f"Epoch {ep:02d} | train {tr_loss:.4f} | val {va_loss:.4f} | F1 {info['f1']:.4f} | Acc {info['acc']:.4f} | t=({info['thr1']:.2f}, {info['thr2']:.2f})")
        if info["f1"] > best_val_f1:
            best_val_f1 = float(info["f1"])
            best_val_loss = float(va_loss)
            best_thresholds = (float(info["thr1"]), float(info["thr2"]))
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep
        elif ep - best_epoch >= args.patience:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError("No valid model checkpoint was produced.")

    model.load_state_dict(best_state)
    te_loss, te_probs, te_true = eval_epoch(model, te, device, criterion, amp=(args.amp and device.type == "cuda"))
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

    stamp = f"mimic3_v12_1_lstm_ann_ordinal_auc_3class_H{args.hidden}_L{args.layers}_Fdyn{Xd_tr.shape[2]}_Fstat{Xs_tr.shape[1]}"
    model_path = os.path.join(OUT_DIR, f"lstm_ann_ordinal_auc_{stamp}.pt")
    metrics_path = os.path.join(OUT_DIR, f"metrics_{stamp}.json")

    torch.save({
        "state_dict": model.state_dict(),
        "input_dyn_dim": int(Xd_tr.shape[2]),
        "input_static_dim": int(Xs_tr.shape[1]),
        "hidden": int(args.hidden),
        "layers": int(args.layers),
        "num_classes_original": 3,
        "num_ordinal_logits": 2,
        "label": "3class_ordinal",
        "best_epoch": int(best_epoch),
        "best_val_f1": float(best_val_f1),
        "best_val_loss": float(best_val_loss),
        "best_thresholds": list(best_thresholds),
        "test_auroc_macro": float(te_auroc_macro),
        "test_auprc_macro": float(te_auprc_macro),
        "test_auroc_gt3": float(te_auroc_gt3),
        "test_auroc_gt7": float(te_auroc_gt7),
        "test_auprc_gt3": float(te_auprc_gt3),
        "test_auprc_gt7": float(te_auprc_gt7),
    }, model_path)

    metrics = {
        "dataset": "MIMIC-III",
        "model": "BiLSTM + static ANN fusion ordinal",
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
        "class_counts_train": counts.tolist(),
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

    print(f"Saved model:   {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Test Acc: {te_acc:.4f} | F1 macro: {te_f1_macro:.4f} | F1 weighted: {te_f1_weighted:.4f} | AUROC macro: {te_auroc_macro:.4f} | AUPRC macro: {te_auprc_macro:.4f} | Test loss: {te_loss:.4f} | Best epoch: {best_epoch} | Best val F1: {best_val_f1:.4f} | Thresholds: {best_thresholds}")
    print(f"Ordinal AUROC: gt3={te_auroc_gt3:.4f}, gt7={te_auroc_gt7:.4f} | Ordinal AUPRC: gt3={te_auprc_gt3:.4f}, gt7={te_auprc_gt7:.4f}")


if __name__ == "__main__":
    main()
