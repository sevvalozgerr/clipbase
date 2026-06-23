"""Evaluation: per-class + per-dataset metrics, writes results.json."""
import os, argparse, yaml, json
import numpy as np
import torch
from tqdm import tqdm

from models.tsclip import TSCLIP
from data.datasets import per_class_jsonl_loaders
from utils.metrics import evaluate_class


def load_cfg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--routing", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--test_meta", nargs="+", required=True,
                    help="one jsonl per test dataset, e.g. MVTec/test.jsonl ...")
    ap.add_argument("--with_pro", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.routing:
        cfg["routing"] = args.routing
    cfg["_ckpt"] = args.ckpt or os.path.join(cfg["save_path"], "last.pth")
    cfg["_test_meta"] = args.test_meta
    cfg["_with_pro"] = args.with_pro
    return cfg


@torch.no_grad()
def eval_dataset(model, loaders, device, img_size, with_pro):
    per_class = []
    for cls, loader in loaders.items():
        labels, iscores, masks, smaps = [], [], [], []
        for batch in tqdm(loader, desc=cls, leave=False):
            img = batch["image"].to(device)
            out = model(img)
            iscores.append(out["img_score"].cpu().numpy())
            smaps.append(out["anomaly_map"].cpu().numpy())
            labels.append(batch["label"].numpy())
            masks.append(batch["mask"].numpy())
        labels = np.concatenate(labels)
        iscores = np.concatenate(iscores)
        masks = np.concatenate(masks)
        smaps = np.concatenate(smaps)
        res = evaluate_class(labels, iscores, masks, smaps, with_pro)
        res["class name"] = cls
        per_class.append(res)
        print(f"  {cls:14s} P-AUC={res['pixel AUC']:.1f} I-AUC={res['image AUC']:.1f}")
    agg = {k: float(np.nanmean([r[k] for r in per_class]))
           for k in ["pixel AUC", "image AUC", "pixel AP", "image AP"]}
    return agg, per_class


def main():
    cfg = load_cfg()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TSCLIP(cfg).to(device)
    ck = torch.load(cfg["_ckpt"], map_location=device)
    model.load_state_dict(ck["model"]); model.eval()

    all_results = []
    for ds_name, meta in zip(cfg["test_datasets"], cfg["_test_meta"]):
        print(f"=== {ds_name} ===")
        loaders = per_class_jsonl_loaders(meta, cfg["data_root"], cfg["img_size"])
        agg, per_class = eval_dataset(model, loaders, device, cfg["img_size"], cfg["_with_pro"])
        print(f"[{ds_name}] P-AUROC={agg['pixel AUC']:.2f} I-AUROC={agg['image AUC']:.2f}")
        all_results.append({"dataset": ds_name,
                            "p_auroc": agg["pixel AUC"], "i_auroc": agg["image AUC"],
                            "p_ap": agg["pixel AP"], "i_ap": agg["image AP"],
                            "per_class": per_class})
    out = os.path.join(cfg["save_path"], "results.json")
    json.dump(all_results, open(out, "w"), indent=2)
    print("saved ->", out)


if __name__ == "__main__":
    main()
