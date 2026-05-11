"""Metrics for segmentation and OOD evaluation."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def prepare_segmentation_labels(
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Map labels outside the trained class set to ignore_index."""

    labels = labels.clone()
    labels[labels >= num_classes] = ignore_index
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels = labels.squeeze(1)
    return labels.long()


def segmentation_confusion_matrix(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Compute confusion matrix with rows as targets and columns as predictions."""

    preds = logits.argmax(dim=1).flatten()
    labels = labels.flatten()
    valid = labels != ignore_index
    if not valid.any():
        return torch.zeros((num_classes, num_classes), dtype=torch.float64, device=logits.device)

    labels = labels[valid]
    preds = preds[valid]
    encoded = labels * num_classes + preds
    conf = torch.bincount(encoded, minlength=num_classes * num_classes)
    return conf.reshape(num_classes, num_classes).to(torch.float64)


def iou_from_confusion(confusion: torch.Tensor) -> torch.Tensor:
    """Compute per-class IoU from a confusion matrix."""

    true_positive = confusion.diag()
    false_positive = confusion.sum(dim=0) - true_positive
    false_negative = confusion.sum(dim=1) - true_positive
    denom = true_positive + false_positive + false_negative
    iou = true_positive / denom.clamp_min(1.0)
    iou[denom == 0] = torch.nan
    return iou


class MeanIoUAccumulator:
    """Small torch-only mIoU accumulator."""

    def __init__(self, num_classes: int, ignore_index: int = 255, device: str | torch.device = "cpu") -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = torch.device(device)
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=self.device)

    def reset(self) -> None:
        self.confusion.zero_()

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        labels = prepare_segmentation_labels(labels, self.num_classes, self.ignore_index)
        self.confusion += segmentation_confusion_matrix(logits, labels, self.num_classes, self.ignore_index)

    def per_class_iou(self) -> torch.Tensor:
        return iou_from_confusion(self.confusion)

    def compute(self) -> torch.Tensor:
        return torch.nanmean(self.per_class_iou())


def binary_auroc_score(id_scores: torch.Tensor, ood_scores: torch.Tensor) -> float:
    """Compute AUROC where higher scores should indicate OOD."""

    if id_scores.numel() == 0 or ood_scores.numel() == 0:
        return math.nan
    y_true = np.concatenate(
        [
            np.zeros(id_scores.numel(), dtype=np.int64),
            np.ones(ood_scores.numel(), dtype=np.int64),
        ]
    )
    y_score = np.concatenate(
        [
            id_scores.detach().cpu().float().numpy().reshape(-1),
            ood_scores.detach().cpu().float().numpy().reshape(-1),
        ]
    )
    if np.unique(y_true).size < 2:
        return math.nan
    return float(roc_auc_score(y_true, y_score))


def fpr_at_tpr(
    y_true: Iterable[int] | np.ndarray,
    y_score: Iterable[float] | np.ndarray,
    target_tpr: float = 0.95,
) -> float:
    """Interpolated FPR at a target TPR."""

    fpr, tpr, _ = roc_curve(np.asarray(y_true), np.asarray(y_score))
    order = np.argsort(tpr)
    tpr = tpr[order]
    fpr = fpr[order]
    if target_tpr <= tpr.min():
        return float(fpr[0])
    if target_tpr >= tpr.max():
        return float(fpr[-1])
    return float(np.interp(target_tpr, tpr, fpr))

