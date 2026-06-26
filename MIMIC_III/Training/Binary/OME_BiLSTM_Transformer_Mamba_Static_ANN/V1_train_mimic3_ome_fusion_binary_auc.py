#!/usr/bin/env python3
"""
V1_train_mimic3_ome_fusion_binary_auc.py
-----------------------------------------

OME-Fusion: Binary Multi-Encoder Fusion for MIMIC-III LOS prediction.

Dynamic 24h ICU sequence is encoded in parallel by:
    1) BiLSTM encoder
    2) Transformer encoder
    3) Pure Mamba encoder

Static features are encoded by:
    4) Static ANN

Then:
    [LSTM + Transformer + Mamba + Static]
    -> gated fusion
    -> binary 2-logit head

Expected files:
    /lustre/home/rahas2/mimic_projects/outputs/lstm_data_v12.1_seq

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

Output folder:
    /lustre/home/rahas2/mimic_projects/outputs/mimic3_ome_fusion_binary_auc_models_v1

Requires:
    mamba_ssm

Run:
    python V1_train_mimic3_ome_fusion_binary_auc.py --workers 0 --batch 128

Optional if stable and you want speed:
    python V1_train_mimic3_ome_fusion_binary_auc.py --workers 0 --batch 128 --amp
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

try:
    from mamba_ssm.modules.mamba_simple import Mamba
except Exception as e:
    raise ImportError(
        "Could not import mamba_ssm. Test with:\n"
        "python -c \"from mamba_ssm.modules.mamba_simple import Mamba; print('OK')\"\n"
        f"Original error: {repr(e)}"
    )


# ======================================================
# CONFIG
# ======================================================

BASE = "/lustre/home/rahas2/mimic_projects/outputs"
DATA_DIR = f"{BASE}/lstm_data_v12.1_seq"
OUT_DIR = f"{BASE}/mimic3_ome_fusion_binary_auc_models_v1"

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
# BINARY HELPERS
# ======================================================

def binary_probs_to_class(prob_pos, threshold):
    return (prob_pos >= threshold).astype(np.int64)


def tune_binary_threshold(y_true, prob_pos):
    best = {
        "f1": -1.0,
        "acc": 0.0,
        "threshold": 0.5,
    }

    for t in np.arange(0.05, 0.96, 0.01):
        pred = binary_probs_to_class(prob_pos, t)
        f1 = f1_score(y_true, pred, average="macro", zero_division=0)

        if f1 > best["f1"]:
            best["f1"] = float(f1)
            best["acc"] = float(accuracy_score(y_true, pred))
            best["threshold"] = float(t)

    return best


# ======================================================
# ENCODER BLOCKS
# ======================================================

class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.mamba(self.norm(x)))


class BiLSTMEncoder(nn.Module):
    def __init__(
        self,
        dyn_dim: int,
        hidden: int = 192,
        layers: int = 2,
        dropout: float = 0.15,
        out_dim: int = 192,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dyn_dim)

        self.lstm = nn.LSTM(
            input_size=dyn_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )

        self.proj = nn.Sequential(
            nn.Linear(hidden * 4, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.norm(x)
        out, (hn, _) = self.lstm(x)

        h_fwd = hn[-2]
        h_bwd = hn[-1]
        h_last = torch.cat([h_fwd, h_bwd], dim=1)
        h_mean = out.mean(dim=1)

        return self.proj(torch.cat([h_last, h_mean], dim=1))


class TransformerEncoderBranch(nn.Module):
    def __init__(
        self,
        dyn_dim: int,
        seq_len: int = 24,
        d_model: int = 192,
        nhead: int = 6,
        layers: int = 2,
        ff_dim: int = 384,
        dropout: float = 0.15,
        out_dim: int = 192,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.seq_len = seq_len
        self.norm = nn.LayerNorm(dyn_dim)

        self.proj = nn.Sequential(
            nn.Linear(dyn_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        self.pos_embed = nn.Embedding(seq_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)

        self.out = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, T, _ = x.shape

        if T != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got T={T}")

        x = self.norm(x)
        tokens = self.proj(x)

        pos_ids = torch.arange(T, device=x.device).unsqueeze(0)
        tokens = tokens + self.pos_embed(pos_ids)

        enc = self.encoder(tokens)

        h_last = enc[:, -1, :]
        h_mean = enc.mean(dim=1)

        return self.out(torch.cat([h_last, h_mean], dim=1))


class PureMambaEncoderBranch(nn.Module):
    def __init__(
        self,
        dyn_dim: int,
        seq_len: int = 24,
        d_model: int = 192,
        layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.15,
        out_dim: int = 192,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.norm = nn.LayerNorm(dyn_dim)

        self.proj = nn.Sequential(
            nn.Linear(dyn_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        self.pos_embed = nn.Embedding(seq_len, d_model)

        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )

        self.out = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, T, _ = x.shape

        if T != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got T={T}")

        x = self.norm(x)
        tokens = self.proj(x)

        pos_ids = torch.arange(T, device=x.device).unsqueeze(0)
        tokens = tokens + self.pos_embed(pos_ids)

        for block in self.blocks:
            tokens = block(tokens)

        h_last = tokens[:, -1, :]
        h_mean = tokens.mean(dim=1)

        return self.out(torch.cat([h_last, h_mean], dim=1))


# ======================================================
# OME-FUSION MODEL
# ======================================================

class OMEFusionBinary(nn.Module):
    """
    Binary Multi-Encoder Fusion:
        BiLSTM branch
        Transformer branch
        Mamba branch
        Static ANN branch

    Then gated fusion -> binary 2-logit head.
    """

    def __init__(
        self,
        dyn_dim: int,
        static_dim: int,
        seq_len: int = 24,
        branch_dim: int = 192,
        lstm_hidden: int = 192,
        lstm_layers: int = 2,
        transformer_layers: int = 2,
        mamba_layers: int = 2,
        nhead: int = 6,
        ff_dim: int = 384,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.15,
        static_hidden: int = 128,
    ):
        super().__init__()

        self.lstm_branch = BiLSTMEncoder(
            dyn_dim=dyn_dim,
            hidden=lstm_hidden,
            layers=lstm_layers,
            dropout=dropout,
            out_dim=branch_dim,
        )

        self.transformer_branch = TransformerEncoderBranch(
            dyn_dim=dyn_dim,
            seq_len=seq_len,
            d_model=branch_dim,
            nhead=nhead,
            layers=transformer_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            out_dim=branch_dim,
        )

        self.mamba_branch = PureMambaEncoderBranch(
            dyn_dim=dyn_dim,
            seq_len=seq_len,
            d_model=branch_dim,
            layers=mamba_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            out_dim=branch_dim,
        )

        self.static_norm = nn.LayerNorm(static_dim)

        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, static_hidden),
            nn.GELU(),
            nn.LayerNorm(static_hidden),
            nn.Dropout(dropout),
            nn.Linear(static_hidden, static_hidden),
            nn.GELU(),
            nn.LayerNorm(static_hidden),
        )

        fusion_dim = branch_dim * 3 + static_hidden

        self.fusion_ln = nn.LayerNorm(fusion_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )

        # Small learned branch attention over LSTM / Transformer / Mamba.
        self.branch_score = nn.Sequential(
            nn.Linear(branch_dim, branch_dim // 2),
            nn.Tanh(),
            nn.Linear(branch_dim // 2, 1),
        )

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 384),
            nn.GELU(),
            nn.LayerNorm(384),
            nn.Dropout(dropout),
            nn.Linear(384, 192),
            nn.GELU(),
            nn.LayerNorm(192),
            nn.Dropout(dropout),
            nn.Linear(192, 2),
        )

    def forward(self, x_dyn, x_static):
        h_lstm = self.lstm_branch(x_dyn)
        h_trf = self.transformer_branch(x_dyn)
        h_mamba = self.mamba_branch(x_dyn)

        # Branch attention rescales dynamic branches but keeps all branches available.
        branches = torch.stack([h_lstm, h_trf, h_mamba], dim=1)  # (B,3,D)
        scores = self.branch_score(branches).squeeze(-1)        # (B,3)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)    # (B,3,1)
        branches = branches * weights

        h_lstm = branches[:, 0, :]
        h_trf = branches[:, 1, :]
        h_mamba = branches[:, 2, :]

        x_static = self.static_norm(x_static)
        h_static = self.static_mlp(x_static)

        fused = torch.cat([h_lstm, h_trf, h_mamba, h_static], dim=1)
        fused = self.fusion_ln(fused)

        gate = self.fusion_gate(fused)
        fused = fused * gate

        return self.head(fused)




# ======================================================
# TRAIN / EVAL
# ======================================================

def train_epoch(
    model,
    loader,
    device,
    criterion,
    optimizer,
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
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=8)

    parser.add_argument("--branch_dim", type=int, default=192)
    parser.add_argument("--lstm_hidden", type=int, default=192)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--mamba_layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--ff_dim", type=int, default=384)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)

    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()

    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("USING MIMIC-III OME-FUSION BINARY WITH AUROC/AUPRC")
    print("Binary Multi-Encoder Fusion: BiLSTM + Transformer + Mamba + Static ANN")
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

    class_counts = np.bincount(y_train.astype(np.int64), minlength=2)
    print("Train class counts:", class_counts)

    n_total = class_counts.sum()
    class_weights = n_total / np.maximum(2.0 * class_counts, 1.0)
    class_weights = class_weights.astype(np.float32)

    print("Using CrossEntropyLoss with validation threshold tuning.")
    print("Class weights:", class_weights.tolist())

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

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
    seq_len = X_dyn_train.shape[1]

    model = OMEFusionBinary(
        dyn_dim=dyn_dim,
        static_dim=static_dim,
        seq_len=seq_len,
        branch_dim=args.branch_dim,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        transformer_layers=args.transformer_layers,
        mamba_layers=args.mamba_layers,
        nhead=args.nhead,
        ff_dim=args.ff_dim,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        dropout=args.dropout,
        static_hidden=128,
    ).to(device)

    print(
        "OME config: "
        f"branch_dim={args.branch_dim}, "
        f"lstm_layers={args.lstm_layers}, "
        f"transformer_layers={args.transformer_layers}, "
        f"mamba_layers={args.mamba_layers}, "
        f"nhead={args.nhead}, d_state={args.d_state}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

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

        val_f1 = threshold_info["f1"]
        val_acc = threshold_info["acc"]

        scheduler.step(val_f1)

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

    test_prob_pos = test_probs[:, 1]
    test_pred = binary_probs_to_class(test_prob_pos, best_threshold)

    test_acc = accuracy_score(test_true, test_pred)

    test_f1_macro = f1_score(
        test_true,
        test_pred,
        average="macro",
        zero_division=0,
    )

    test_f1_weighted = f1_score(
        test_true,
        test_pred,
        average="weighted",
        zero_division=0,
    )

    test_auroc = roc_auc_score(test_true, test_prob_pos)
    test_auprc = average_precision_score(test_true, test_prob_pos)

    report = classification_report(
        test_true,
        test_pred,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(test_true, test_pred).tolist()

    stamp = (
        f"mimic3_ome_fusion_binary_"
        f"B{args.branch_dim}_"
        f"LSTM{args.lstm_layers}_TRF{args.transformer_layers}_MAMBA{args.mamba_layers}_"
        f"Fdyn{dyn_dim}_Fstat{static_dim}"
    )

    model_path = os.path.join(OUT_DIR, f"ome_fusion_binary_{stamp}.pt")
    metrics_path = os.path.join(OUT_DIR, f"metrics_{stamp}.json")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "dyn_dim": int(dyn_dim),
            "static_dim": int(static_dim),
            "seq_len": int(seq_len),
            "branch_dim": int(args.branch_dim),
            "lstm_hidden": int(args.lstm_hidden),
            "lstm_layers": int(args.lstm_layers),
            "transformer_layers": int(args.transformer_layers),
            "mamba_layers": int(args.mamba_layers),
            "nhead": int(args.nhead),
            "ff_dim": int(args.ff_dim),
            "d_state": int(args.d_state),
            "d_conv": int(args.d_conv),
            "expand": int(args.expand),
            "dropout": float(args.dropout),
            "num_classes": 2,
            "label": "binary",
            "class_weights": class_weights.tolist(),
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
        "dataset": "MIMIC-III",
        "model": "OME-Fusion binary: BiLSTM + Transformer + Mamba + static ANN",
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
        "class_weights": class_weights.tolist(),
        "train_shape_dyn": list(X_dyn_train.shape),
        "valid_shape_dyn": list(X_dyn_valid.shape),
        "test_shape_dyn": list(X_dyn_test.shape),
        "train_shape_static": list(X_static_train.shape),
        "input_dyn_dim": int(dyn_dim),
        "input_static_dim": int(static_dim),
        "seq_len": int(seq_len),
        "branch_dim": int(args.branch_dim),
        "lstm_layers": int(args.lstm_layers),
        "transformer_layers": int(args.transformer_layers),
        "mamba_layers": int(args.mamba_layers),
        "checkpoint_metric": "validation_macro_f1",
        "positive_class": "LOS_gt_7_days",
        "output_dir": OUT_DIR,
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
