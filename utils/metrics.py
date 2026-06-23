"""Evaluation metrics for zero-shot anomaly detection.

image-level : AUROC, AP
pixel-level : AUROC, AP, PRO  (PRO = per-region overlap, the standard localization metric)
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from skimage import measure


def image_auroc(labels, scores):
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores).ravel()
    if len(np.unique(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, scores) * 100.0


def image_ap(labels, scores):
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores).ravel()
    if len(np.unique(labels)) < 2:
        return float("nan")
    return average_precision_score(labels, scores) * 100.0


def pixel_auroc(masks, score_maps):
    masks = np.asarray(masks).ravel().astype(np.int32)
    score_maps = np.asarray(score_maps).ravel()
    if len(np.unique(masks)) < 2:
        return float("nan")
    return roc_auc_score(masks, score_maps) * 100.0


def pixel_ap(masks, score_maps):
    masks = np.asarray(masks).ravel().astype(np.int32)
    score_maps = np.asarray(score_maps).ravel()
    if len(np.unique(masks)) < 2:
        return float("nan")
    return average_precision_score(masks, score_maps) * 100.0


def compute_pro(masks, score_maps, num_th=200):
    """Per-Region-Overlap (AUPRO), integrated up to FPR=0.3 then normalized.

    masks      : (N, H, W) binary ground-truth
    score_maps : (N, H, W) anomaly scores
    """
    masks = np.asarray(masks)
    score_maps = np.asarray(score_maps)
    if masks.sum() == 0:
        return float("nan")

    mn, mx = score_maps.min(), score_maps.max()
    deltas = (mx - mn) / num_th
    pros, fprs = [], []
    binary = np.zeros_like(score_maps, dtype=bool)

    inverse_gt = 1 - masks  # negatives for FPR
    inverse_sum = inverse_gt.sum()

    th = mx
    for _ in range(num_th):
        th -= deltas
        binary = score_maps >= th
        # PRO: mean overlap over connected components of the GT
        overlaps = []
        for m, b in zip(masks, binary):
            if m.sum() == 0:
                continue
            labeled = measure.label(m, connectivity=2)
            for region in measure.regionprops(labeled):
                coords = region.coords
                tp = b[coords[:, 0], coords[:, 1]].sum()
                overlaps.append(tp / region.area)
        if not overlaps:
            continue
        pro = np.mean(overlaps)
        fpr = (binary & (inverse_gt == 1)).sum() / (inverse_sum + 1e-8)
        pros.append(pro)
        fprs.append(fpr)

    pros = np.array(pros)
    fprs = np.array(fprs)
    keep = fprs <= 0.3
    if keep.sum() < 2:
        return float("nan")
    order = np.argsort(fprs[keep])
    x, y = fprs[keep][order], pros[keep][order]
    auc = np.trapz(y, x) / 0.3
    return auc * 100.0


def evaluate_class(labels, image_scores, masks, score_maps, with_pro=False):
    res = {
        "image AUC": image_auroc(labels, image_scores),
        "image AP":  image_ap(labels, image_scores),
        "pixel AUC": pixel_auroc(masks, score_maps),
        "pixel AP":  pixel_ap(masks, score_maps),
    }
    if with_pro:
        res["pixel PRO"] = compute_pro(masks, score_maps)
    return res
