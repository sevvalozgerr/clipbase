"""Segmentation losses for anomaly localization (focal + dice)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, prob, target):
        # prob: (B, 2, H, W) softmax probabilities; target: (B, H, W) in {0,1}
        target = target.long()
        p = prob.clamp(1e-6, 1 - 1e-6)
        ce = F.nll_loss(torch.log(p), target, reduction="none")  # (B,H,W)
        pt = p.gather(1, target.unsqueeze(1)).squeeze(1)         # prob of true class
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * (1 - pt) ** self.gamma * ce
        return loss.mean()


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, prob_fg, target):
        # prob_fg: (B, H, W) prob of anomaly class; target: (B, H, W) {0,1}
        prob_fg = prob_fg.contiguous().view(prob_fg.size(0), -1)
        target = target.contiguous().view(target.size(0), -1).float()
        inter = (prob_fg * target).sum(1)
        union = prob_fg.sum(1) + target.sum(1)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return (1 - dice).mean()


class SegLoss(nn.Module):
    """Combined focal + dice over the anomaly probability map."""
    def __init__(self, lambda_focal=1.0, lambda_dice=1.0):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = BinaryDiceLoss()
        self.lf = lambda_focal
        self.ld = lambda_dice

    def forward(self, anomaly_prob, mask):
        # anomaly_prob: (B, H, W) in [0,1]; mask: (B, H, W) {0,1}
        normal_prob = 1 - anomaly_prob
        prob2 = torch.stack([normal_prob, anomaly_prob], dim=1)  # (B,2,H,W)
        return self.lf * self.focal(prob2, mask) + self.ld * (
            self.dice(anomaly_prob, mask) + self.dice(normal_prob, 1 - mask)
        )
