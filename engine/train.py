"""Training loop. Colab-safe: checkpoints to Drive every epoch + --resume."""
import os, argparse, yaml, random
import numpy as np
import torch
from tqdm import tqdm

from models.tsclip import TSCLIP
from losses.losses import SegLoss
from data.datasets import get_loaders_from_jsonl


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_cfg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--routing", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--train_meta", default=None,
                    help="jsonl metadata for the TRAIN set (e.g. AA-CLIP VisA full-shot.jsonl)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.routing:
        cfg["routing"] = args.routing
    cfg["_resume"] = args.resume
    cfg["_train_meta"] = args.train_meta
    return cfg


def main():
    cfg = load_cfg()
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["save_path"], exist_ok=True)

    model = TSCLIP(cfg).to(device)
    opt = torch.optim.Adam(model.trainable_parameters(), lr=cfg["lr"])
    seg_loss = SegLoss(cfg["lambda_focal"], cfg["lambda_dice"])

    assert cfg["_train_meta"], "Pass --train_meta <path to train jsonl> (e.g. VisA full-shot.jsonl)"
    loader = get_loaders_from_jsonl(
        cfg["_train_meta"], cfg["data_root"], cfg["img_size"],
        cfg["batch_size"], train=True,
    )

    start_epoch = 0
    ckpt_path = os.path.join(cfg["save_path"], "last.pth")
    if cfg["_resume"] and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        running = 0.0
        for batch in tqdm(loader, desc=f"epoch {epoch}"):
            img = batch["image"].to(device)
            mask = batch["mask"].to(device)
            out = model(img)
            loss = seg_loss(out["anomaly_map"], mask)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        print(f"epoch {epoch}: loss={running/len(loader):.4f}")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch, "cfg": cfg}, ckpt_path)
    print("training done ->", ckpt_path)


if __name__ == "__main__":
    main()
