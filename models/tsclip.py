"""TS-CLIP model.

Baseline (routing='none'): object-agnostic prompts + frozen CLIP multi-scale
patch features + a small trainable projection adapter. Patch anomaly score =
softmax over {normal, anomaly} text similarities.

routing='spatial' is a stub for your architecture step (filled in later).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_backbone import CLIPBackbone
from .prompts import ObjectAgnosticPrompts


class PatchAdapter(nn.Module):
    """Small trainable head that refines frozen patch features (one per scale)."""
    def __init__(self, dim, n_scales):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(n_scales)])
        for m in self.proj:
            nn.init.eye_(m.weight)  # start as identity -> stable

    def forward(self, feats):
        out = []
        for i, f in enumerate(feats):
            r = self.proj[i](f)
            r = r / r.norm(dim=-1, keepdim=True)
            out.append(r)
        return out


class TSCLIP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = CLIPBackbone(
            cfg["backbone"], cfg["pretrained"],
            feature_layers=cfg["feature_layers"], img_size=cfg["img_size"],
        )
        self.prompts = ObjectAgnosticPrompts(self.backbone, n_ctx=cfg["n_ctx"])
        self.adapter = PatchAdapter(self.backbone.embed_dim, len(cfg["feature_layers"]))
        self.routing = cfg.get("routing", "none")
        self.img_size = cfg["img_size"]
        if self.routing == "spatial":
            # TODO(step-2): SpatialAwareTokenRouting goes here (your architecture)
            pass

    def _score_map(self, patch_feat, text):
        # patch_feat: (B,N,d), text: (2,d) -> anomaly prob map (B, h, w)
        B, N, d = patch_feat.shape
        h = w = int(math.sqrt(N))
        logits = patch_feat @ text.t()            # (B,N,2)
        prob = (logits / 0.07).softmax(dim=-1)    # temperature
        anom = prob[..., 1].reshape(B, h, w)
        return anom

    def forward(self, x):
        text = self.prompts()                     # (2, d)
        cls, patch_feats = self.backbone.encode_image(x)
        patch_feats = self.adapter(patch_feats)

        # image-level score from cls vs text
        img_logits = cls @ text.t()               # (B,2)
        img_score = (img_logits / 0.07).softmax(-1)[:, 1]

        # multi-scale pixel score (mean of upsampled maps)
        maps = []
        for f in patch_feats:
            m = self._score_map(f, text).unsqueeze(1)            # (B,1,h,w)
            m = F.interpolate(m, size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False)
            maps.append(m)
        anomaly_map = torch.cat(maps, 1).mean(1)                 # (B,H,W)
        return {"img_score": img_score, "anomaly_map": anomaly_map}

    def trainable_parameters(self):
        return list(self.prompts.parameters()) + list(self.adapter.parameters())
