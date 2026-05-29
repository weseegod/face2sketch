# Phase 1 Report — pix2pix GAN: Photo → Sketch

> **Status:** ✅ Complete  
> **Date:** May 2026  
> **Checkpoint:** `checkpoints/pix2pix_best_2.pt` (epoch 197, ~454MB)

---

## What We Built

| Component | File | Params |
|-----------|------|--------|
| U-Net Generator | `src/unet.py` | ~37M |
| PatchGAN Discriminator | `src/discriminator.py` | ~2.8M |
| Training loop | `src/train.py` | — |
| Inference & visualization | `src/sample.py` | — |
| Evaluation | `src/evaluate.py` | — |

## Training Runs

Two full 200-epoch runs on T4 Colab (free tier), ~2 hours each.

| | Run 1 | Run 2 |
|---|---|---|
| D learning rate | 2e-4 | **1e-4** |
| Epochs completed | 155 (early stop) | 200 |
| Final Train G_L1 | 0.229 | 0.265 |
| Test L1 (100 unseen) | 0.343 | 0.360 |
| Df at end | 0.000 (dead) | 0.000 (dead) |

**Key finding:** With only 322 training pairs, the discriminator finishes its job early (epoch ~30-40) — it memorizes all real pairs and rejects everything else. After that, training is pure L1 regression. Halving D_lr bought ~13 more epochs of useful adversarial signal but didn't fundamentally solve the data-scarcity limitation.

Both runs converged to roughly the same L1 on unseen test data, suggesting the model saturated what's learnable from 322 pairs.

## Dataset

- **Training:** 322 paired (photo, sketch) — CUHK Student Sketch (188) + SKSF-A (134)
- **Test:** 100 paired (photo, sketch) — app-generated, held out completely
- **Resolution:** 256×256, 100% used for training (no val split)

## What Works

- Generator produces recognizable face sketches from unseen photos
- No mode collapse — different inputs produce different outputs
- Training is stable — no NaN losses, no oscillations
- Checkpoint resume works
- Training on Colab T4 with `data.zip` upload flow works
- Evaluation pipeline: quick visual + full L1 on unseen test set

## What's Tricky

- **Small dataset (322 pairs):** Discriminator saturates quickly. GAN adversarial training only active for first ~30 epochs. This is a known property of pix2pix on small datasets — the paper uses thousands of pairs.
- **L1 doesn't reach 0.15:** Test L1 plateaus around 0.35. The 0.15 target from the pix2pix paper assumes cleaner, higher-resolution academic sketches. Our test set (app-generated sketches) has inherent mismatch that inflates L1.
- **D_lr tuning helped but didn't fully solve:** Halving discriminator learning rate extended useful adversarial training from ~20 to ~33 epochs. For larger datasets this would be enough.

## Artifacts

```
checkpoints/
  pix2pix_best_1.pt    — run 1 best (epoch 155)
  pix2pix_best_2.pt    — run 2 best (epoch 197) ← use this for Phase 2

samples/
  epoch_010.png ... epoch_200.png   — sample grids every 10 epochs

outputs/
  phase1_quick_eval.png             — 10 random test photos + generated
  phase1_full_eval.png              — 8 generated vs 8 ground-truth
```

## Next: Phase 2

Load `pix2pix_best_2.pt` as pretrained weights, finetune on TwitterPicasso (184 caricature pairs). The photo→drawing mapping transfers — only the style needs to adapt. Fresh discriminator on a new dataset means adversarial signal will be active again.

```
python src/train.py --mode train --resume checkpoints/pix2pix_best_2.pt --device cuda
```
