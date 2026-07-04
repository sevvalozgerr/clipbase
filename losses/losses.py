"""Segmentation losses for anomaly localization (focal + dice).

Tuned for EXTREME foreground/background imbalance: anomaly masks here cover
~0.1-0.3% of pixels. Key choices that prevent the "predict-normal-everywhere,
loss->0" collapse:
  * binary focal applied directly to the anomaly probability, with alpha
    weighting the RARE anomaly class UP (alpha=0.75, not 0.25);
  * dice on the anomaly channel only (the normal-channel dice is trivially
    satisfied and only drags the loss to zero, so it is removed);
  * return_parts=True lets you print focal vs dice per step to verify the
    loss is actually alive.
"""
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Binary focal loss on the anomaly probability map.

    alpha = weight on the anomaly (positive, rare) class. Higher = the model
    is penalized more for missing anomalies. For ~0.1% masks, 0.75 is a sane
    start; push to 0.9 if anomalies are still ignored.
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, anomaly_prob, mask):
        # anomaly_prob: (B,H,W) in [0,1]; mask: (B,H,W) in {0,1}
        p = anomaly_prob.clamp(1e-6, 1 - 1e-6)
        mask = mask.float()
        pt = p * mask + (1 - p) * (1 - mask)               # prob of the true class
        alpha_t = self.alpha * mask + (1 - self.alpha) * (1 - mask)
        loss = -alpha_t * (1 - pt).pow(self.gamma) * torch.log(pt)
        return loss.mean()


class DiceLoss(nn.Module):
    """Soft dice on the anomaly channel. Per-image, smoothed."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, prob, target):
        prob = prob.reshape(prob.size(0), -1)
        target = target.reshape(target.size(0), -1).float()
        inter = (prob * target).sum(1)
        union = prob.sum(1) + target.sum(1)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return (1 - dice).mean()


class SegLoss(nn.Module):
    """focal + dice on the anomaly map. Signature kept compatible with train.py:
    SegLoss(lambda_focal, lambda_dice) still works; alpha/gamma are optional.
    """
    def __init__(self, lambda_focal=1.0, lambda_dice=1.0, alpha=0.75, gamma=2.0):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.dice = DiceLoss()
        self.lf = lambda_focal
        self.ld = lambda_dice

    def forward(self, anomaly_prob, mask, return_parts=False):
        f = self.focal(anomaly_prob, mask)
        d = self.dice(anomaly_prob, mask)          # anomaly-only dice (rare class)
        total = self.lf * f + self.ld * d
        if return_parts:
            return total, {"focal": float(f.detach()), "dice": float(d.detach())}
        return total