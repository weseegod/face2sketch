# Phase 2 Report — Finetune pix2pix on Caricatures

> **Status:** ❌ Incomplete — GAN approach insufficient for caricature style transfer  
> **Date:** May 2026

---

## Goal

Fine-tune Phase 1 pix2pix model (photo→sketch) on TwitterPicasso (184 caricature pairs) to produce exaggerated, bold-line caricatures instead of realistic sketches.

## What We Tried

### Approach: Transfer Learning

Load Phase 1 generator weights (which know face geometry and photo→drawing mapping), train on caricature data with a fresh discriminator. The hypothesis: structural knowledge transfers, only style needs to adapt.

### Version History

| Run | λ_L1 | LR | D strategy | Epochs | Train L1 | Test L1 | Result |
|-----|------|-----|-----------|--------|----------|---------|--------|
| v1 | 50 | 5e-5 | ndf=64, no stabilization | 49 | 0.430 | — | Style didn't transfer |
| v2 | 200 | 1e-4 | ndf=64, 50-epoch L1 warmup | 69 | 0.400 | 0.507 | Some caricature structure, early stop killed it |
| v3 | 200 | 1e-4 | ndf=48, noise 0.07, spectral norm | 92 | 0.417 | 0.450 | Slightly better, but outputs nearly identical to Phase 1 sketches |

### v3 Configuration (best attempt)

```
G: Phase 1 v5 weights (37M params)
D: ndf=48 (~1.5M), spectral norm, noise σ=0.07
λ_L1=200, λ_adv=1.0
LR=1e-4, 50-epoch L1-only warmup, patience=0
```

## Why It Didn't Work

**184 pairs is too few for GAN-based style transfer.** The discriminator's job is to learn "what makes a caricature" — exaggerated proportions, bolder strokes, cartoon-like features. With only 184 examples, it can't learn this. Instead it memorizes all real pairs and dies early, just like Phase 1 did on 322 pairs.

Without a functioning discriminator, training is pure L1 regression. But L1 is pulling toward the caricature ground-truth while the generator's Phase 1 weights are pulling toward sketch style. The result: a compromise that produces outputs nearly indistinguishable from Phase 1.

Three separate strategies (different λ_L1, LR, warmup, D size, noise levels) all converged to the same outcome. The limitation is architectural: **adversarial training needs enough data for the discriminator to learn meaningful features, and 184 pairs isn't enough for caricatures.**

## What We Learned

1. **GAN transfer learning works for similar domains, not style jumps.** Photo→sketch to photo→caricature is too far. The discriminator can't bridge the gap on limited data.

2. **L1 loss alone can't change artistic style.** L1 minimizes pixel difference. A caricature deliberately differs from a sketch in structural ways. High L1 weight just forces the model toward an L1-optimal compromise that loses the caricature character.

3. **D stabilization helps but doesn't solve.** Spectral norm + noise injection kept D alive longer, but not long enough to learn caricature features from 184 pairs.

## Artifacts

```
checkpoints/
  phase2_v3_best.pt   — epoch 92, G_L1=0.417 (v3 final)

outputs/
  phase1_vs_phase2.png  — v5 sketch vs v3 "caricature" (nearly identical)
```

## Next: Phase 3 — Conditional DDPM from Scratch

### Why DDPM (not ControlNet yet)

The learning roadmap now splits into two diffusion phases:

- **Phase 3 (Path B):** Build DDPM/DDIM from scratch. No Stable Diffusion, no HuggingFace. Understand every line of the forward noising process, reverse denoising, noise prediction, and fast sampling. Use the same U-Net architecture from Phase 1, modified for noise prediction with timestep conditioning.

- **Phase 4 (Path A):** ControlNet + LoRA on Stable Diffusion 1.5. Production-grade conditional generation. Leverage pretrained SD for quality, train only adapters on 184 pairs.

Phase 3 comes first because: understanding diffusion internals before using production wrappers. Same philosophy as Phase 1/2 — build from scratch, then upgrade.

### Why DDPM Might Work Where GAN Failed

- **No discriminator.** Loss is simple MSE on predicted noise — no adversarial balance, no D dying on small data.
- **Distribution learning.** DDPM learns a distribution over outputs, so 184 pairs teach it the full style space rather than point estimates.
- **Conditioning is explicit.** Photo is concatenated as input channel — direct structural guidance without adversarial dynamics.

### What to Build

| Component | File | 
|-----------|------|
| **DDPM Scheduler** | `src/diffusion.py` |
| **DDIM Sampler** | `src/diffusion.py` |
| **Noise Predictor U-Net** | `src/unet.py` (extended) |
| **Training Loop** | `src/train_diffusion.py` |
| **Inference** | `src/sample.py` (extended) |
