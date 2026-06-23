# TS-CLIP — own pipeline (step zero: baseline)

Own AnomalyCLIP-style ZSAD baseline. No dependency on AA-CLIP's repo.
Colab-A100 friendly: code lives here on GitHub, Colab only launches it.

## Repo layout
```
config.yaml            all hyperparams + dataset paths
models/clip_backbone.py frozen CLIP (open_clip) + multi-scale patch features (hooks)
models/prompts.py       object-agnostic learnable prompts (normal / damaged object)
models/tsclip.py        main model (routing='none' = baseline; 'spatial' = your arch later)
losses/losses.py        focal + dice
data/datasets.py        jsonl-driven loaders (reuse AA-CLIP metadata) + per-class eval
engine/train.py         training loop (--resume, checkpoints to Drive)
engine/evaluate.py      per-class + per-dataset metrics -> results.json
utils/metrics.py        AUROC / AP / PRO
```

## Workflow (Colab = launcher only)
1. Edit code on GitHub (github.dev / Codespace / local VS Code) -> commit.
2. In Colab: `git pull` + `!python -m engine.train ...`. Never paste model code in cells.
3. Checkpoints + results go to Drive (`save_path`), so disconnects don't lose work.

## Data
Symlink your Drive data once per session:
```
ln -sfn /content/drive/MyDrive/AA-CLIP/data data_root
```
For loaders, reuse the jsonl metadata you already generated in the AA-CLIP repo
(`dataset/metadata/<DS>/full-shot.jsonl` for train, the test jsonl for eval).
Copy those jsonl files into this repo (e.g. `data/meta/`) or pass absolute paths.
If your jsonl keys differ, edit `JsonlAD.KEYS` in `data/datasets.py`.

## Run (step zero)
```
# train on VisA
python -m engine.train  --config config.yaml --routing none \
    --train_meta data/meta/VisA_full-shot.jsonl
# test on MVTec/BTAD/MPDD
python -m engine.evaluate --config config.yaml --routing none \
    --test_meta data/meta/MVTec_test.jsonl data/meta/BTAD_test.jsonl data/meta/MPDD_test.jsonl
```

## STEP-ZERO SUCCESS CRITERION
Baseline trained VisA->MVTec should reach **MVTec pixel-AUROC ~90-91,
image-AUROC ~91**. Do NOT add routing until this matches the published baseline.

## Things to validate once (flagged in code)
- `clip_backbone.py`: open_clip attribute names (`ln_post`, `proj`, `attn_mask`).
  Print shapes once; if your open_clip version differs, adjust the 2-3 flagged lines.
- `datasets.py`: jsonl key names (`JsonlAD.KEYS`).
- If the baseline pixel number is short of ~90, add **DPAM (V-V attention)** in the
  last visual block — this is the known trick that lifts AnomalyCLIP localization.
  (Port it from AnomalyCLIP's official repo into `clip_backbone.py`.)

## Then: your architecture (step 2+)
Set `routing: spatial` and implement `SpatialAwareTokenRouting` in `tsclip.py`:
learned spatial-prior embedding -> per-token visual-adapter modulation ->
multi-scale subspace routing, with orthogonality + load-balancing losses.
Each addition = one ablation row.
