# Phase 1 Report — pix2pix GAN: Photo → Sketch

> **Status:** ✅ Complete — v5 accepted  
> **Date:** May 2026  
> **Best checkpoint:** `checkpoints/pix2pix_v5_best.pt` (epoch 200, ~441MB)

---

## What We Built

| Component | File | Params |
|-----------|------|--------|
| U-Net Generator | `src/unet.py` | ~37M |
| PatchGAN Discriminator | `src/discriminator.py` | ~1.5M (ndf=48) |
| Training loop | `src/train.py` | — |
| Inference & visualization | `src/sample.py` | — |
| Evaluation | `src/evaluate.py` | — |

## The Problem: D Dies on Small Datasets

With only 322 training pairs, the discriminator memorizes all real images by epoch ~30 and rejects everything with 99.99% confidence (Df=0.000). The generator receives zero adversarial gradient — the loss that adds pencil-stroke texture simply stops working.

**Symptom:** Outputs look like "black-and-white photocopies" — structurally correct, but no pencil texture.

## The Fix: Spectral Norm + Noise Injection + Smaller D

Three changes to the discriminator that keep it alive:

| Technique | v2 (before) | v5 (final) | Why |
|-----------|-------------|------------|-----|
| **Spectral Normalization** | ❌ | ✅ | Bounds D's weight norms — can't get 99.99% confident |
| **Noise Injection** | 0.00 | 0.07 | Gaussian noise on D inputs — prevents pixel-perfect memorization |
| **Discriminator size** | ndf=64 (~2.8M) | ndf=48 (~1.5M) | Smaller D has less capacity to memorize 322 pairs |

## Version History

| Run | ndf | noise | Epoch | Train L1 | Test L1 | Visual |
|-----|-----|-------|-------|----------|---------|--------|
| v2 | 64 | 0.00 | 197 | 0.265 | 0.360 | Photocopy — no texture |
| v3 | 64 | 0.05 | 199 | 0.265 | 0.349 | Better — some strokes visible |
| v4 | 32 | 0.10 | 199 | 0.262 | 0.361 | Sharp strokes, lost details (hair, face) |
| **v5** | **48** | **0.07** | **200** | **0.263** | **0.326** | **Best — sharp strokes, full detail** |

**Key insight:** L1 numbers are identical across all runs (0.26-0.27). L1 measures pixel accuracy, not artistic quality. The improvement from v2 → v5 is entirely visual — D staying alive longer gives adversarial loss time to add pencil-drawing texture without sacrificing structural accuracy.

## Dataset

- **Training:** 322 paired (photo, sketch) — CUHK Student Sketch (188) + SKSF-A (134)
- **Test:** 100 paired (photo, sketch) — app-generated, held out completely
- **Resolution:** 256×256, 100% used for training

## What Works

- Generator produces recognizable face sketches with visible pencil strokes
- D stays alive across the full 200-epoch training run
- Adversarial loss contributes texture throughout, not just first 30 epochs
- No mode collapse, stable training
- Multi-checkpoint comparison tool (`--checkpoints v3.pt v4.pt v5.pt --mode compare`)

## Artifacts

```
checkpoints/
  pix2pix_v3_best.pt   — ndf=64, noise=0.05 (full detail, soft strokes)
  pix2pix_v4_best.pt   — ndf=32, noise=0.10 (sharp, lost details)
  pix2pix_v5_best.pt   — ndf=48, noise=0.07 ✅ BEST

outputs/
  phase1_vs_phase2.png  — 4-row comparison: photos | v3 | v4 | v5
```

## Next: Phase 2 — Fine-tune on Caricatures

Load `pix2pix_v5_best.pt`, finetune on TwitterPicasso (184 pairs). Same D stabilization (ndf=48, noise=0.07) applied.

```bash
python src/train.py --mode finetune \
    --config configs/pix2pix_phase2.yaml \
    --finetune checkpoints/pix2pix_v5_best.pt \
    --device cuda --name phase2_v3_
```
