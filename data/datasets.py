"""Dataset loading for ZSAD.

Two ways to use it:
  (A) jsonl metadata (recommended, robust): reuse the metadata you already
      generated in the AA-CLIP repo (dataset/metadata/<DS>/*.jsonl). Each line:
      {"img_path": ..., "mask_path": ... or "", "anomaly": 0/1, "cls_name": ...}
      -> set meta_path in get_loaders(). Sidesteps per-dataset folder quirks.
  (B) standard MVTec-style folder walk (works for MVTec and MPDD).

For step zero you need: VisA (train) + MVTec (test). Use (A) with your existing
AA-CLIP jsonl files; it is the fastest correct path.
"""
import os, json, glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def build_transforms(img_size, train=False):
    img_t = transforms.Compose([
        transforms.Resize((img_size, img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    mask_t = transforms.Compose([
        transforms.Resize((img_size, img_size), Image.NEAREST),
        transforms.ToTensor(),
    ])
    return img_t, mask_t


class JsonlAD(Dataset):
    """Reads a jsonl metadata file. Keys are configurable; defaults match common
    AA-CLIP/AnomalyCLIP metadata. Adjust KEYS if your jsonl differs."""
    KEYS = {"img": "image_path", "mask": "mask_path",
            "label": "anomaly", "cls": "class_name"}

    def __init__(self, meta_path, base_path, img_size, train=False):
        self.base = base_path
        self.items = []
        with open(meta_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))
        self.img_t, self.mask_t = build_transforms(img_size, train)
        self.img_size = img_size

    def __len__(self):
        return len(self.items)

    def _path(self, p):
        if not p:
            return ""
        return p if os.path.isabs(p) else os.path.join(self.base, p)

    def __getitem__(self, i):
        it = self.items[i]
        img_p = self._path(it[self.KEYS["img"]])
        label = int(it.get(self.KEYS["label"], 0))
        cls = it.get(self.KEYS["cls"], "unknown")
        img = Image.open(img_p).convert("RGB")
        img = self.img_t(img)

        mask_p = self._path(it.get(self.KEYS["mask"], ""))
        if label == 1 and mask_p and os.path.exists(mask_p):
            mask = Image.open(mask_p).convert("L")
            mask = (self.mask_t(mask) > 0.5).float()[0]
        else:
            mask = torch.zeros(self.img_size, self.img_size)
        return {"image": img, "mask": mask, "label": label, "cls_name": cls}


def get_loaders_from_jsonl(meta_path, base_path, img_size, batch_size,
                           train=False, num_workers=4):
    ds = JsonlAD(meta_path, base_path, img_size, train)
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      num_workers=num_workers, pin_memory=True)


# ---- per-class loaders for evaluation (needed for per-class metrics) ----
def per_class_jsonl_loaders(meta_path, base_path, img_size, batch_size=16):
    """Splits a test jsonl into one DataLoader per class for per-class metrics."""
    by_cls = {}
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            it = json.loads(line)
            by_cls.setdefault(it.get("cls_name", "unknown"), []).append(it)
    loaders = {}
    for cls, items in by_cls.items():
        ds = JsonlAD.__new__(JsonlAD)
        ds.base = base_path
        ds.items = items
        ds.img_t, ds.mask_t = build_transforms(img_size)
        ds.img_size = img_size
        loaders[cls] = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                  num_workers=4, pin_memory=True)
    return loaders
