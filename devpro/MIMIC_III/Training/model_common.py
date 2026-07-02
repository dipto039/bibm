"""
model_common.py
----------------
Shared, standardized training configuration for the BIBM MIMIC-III
6-architecture comparison sweep (Binary + Ordinal). Before this module
existed, each of the 12 train scripts defined its own optimizer settings,
LR schedule, and classifier head inline -- they had drifted apart (different
learning rates, batch sizes, early-stopping patience, head width/depth, and
even whether an LR schedule existed at all), which confounds "architecture A
beats architecture B" with "architecture A happened to get a more generous
training budget." Centralizing it here means every architecture gets the
same optimizer, the same LR schedule, and a head that differs only in its
input width (the fusion layer's output size) and output size (2 for binary,
3 for ordinal) -- so the comparison is actually about the encoder, not the
training recipe wrapped around it.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

STANDARD_HPARAMS = dict(
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=128,
    patience=10,
    max_epochs=40,
    grad_clip=1.0,
)


def build_standard_head(in_dim: int, n_classes: int, dropout: float = 0.2) -> nn.Sequential:
    """Classifier head shared by all 6 architectures. Only in_dim (the
    fusion layer's output width) and n_classes (2 binary / 3 ordinal) vary
    between models -- the head shape itself does not."""
    return nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.GELU(),
        nn.LayerNorm(256),
        nn.Dropout(dropout),
        nn.Linear(256, 128),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(128, n_classes),
    )


def build_standard_optimizer(model: nn.Module, lr: float = None, weight_decay: float = None) -> torch.optim.Optimizer:
    """lr/weight_decay default to STANDARD_HPARAMS but can still be
    overridden per-run (e.g. --lr) for deliberate one-off experiments --
    only the DEFAULT needs to be consistent across architectures, not the
    ability to override it."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr if lr is not None else STANDARD_HPARAMS["lr"],
        weight_decay=weight_decay if weight_decay is not None else STANDARD_HPARAMS["weight_decay"],
    )


def build_standard_scheduler(optimizer: torch.optim.Optimizer):
    """Adaptive, not a fixed warmup schedule, so it doesn't need total step
    count computed up front -- stepped once per epoch via
    scheduler.step(val_macro_f1), matching the convention already used by
    the two reference models (BiLSTM, OME-Fusion) this standard is based on.
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )


def compute_multiclass_metrics(y_true: np.ndarray, probs: np.ndarray, n_classes: int = 3) -> dict:
    """Plain (non-ordinal) multi-class metrics: argmax predictions, plus
    micro/macro/weighted F1 and one-vs-rest AUROC/AUPRC. Shared across all 6
    multiclass scripts so the metric definitions can't drift between them.
    """
    y_pred = probs.argmax(axis=1)
    y_true_onehot = np.eye(n_classes)[y_true]

    metrics = {
        "test_acc": float(accuracy_score(y_true, y_pred)),
        "test_f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "test_f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    try:
        metrics["test_auroc_macro"] = float(
            roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
        )
        metrics["test_auroc_weighted"] = float(
            roc_auc_score(y_true, probs, multi_class="ovr", average="weighted")
        )
    except ValueError:
        metrics["test_auroc_macro"] = float("nan")
        metrics["test_auroc_weighted"] = float("nan")

    metrics["test_auprc_macro"] = float(
        average_precision_score(y_true_onehot, probs, average="macro")
    )
    metrics["test_auprc_weighted"] = float(
        average_precision_score(y_true_onehot, probs, average="weighted")
    )

    return metrics, y_pred
