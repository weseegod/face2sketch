# Phase 1 Report — pix2pix GAN: Photo → Sketch

> **Status:** ⚠️ Retraining needed (D stabilization)  
> **Date:** May 2026

---

## What We Built

| Component | File | Params |
|-----------|------|--------|
| U-Net Generator | `src/unet.py` | ~37M |
| PatchGAN Discriminator | `src/discriminator.py` | ~2.8M |
| Training loop | `src/train.py` | — |
| Inference & visualization | `src/sample.py` | — |
| Evaluation | `src/evaluate.py` | — |

## Training History (3 attempts — all share the same root problem)

| Run | D LR | λ_L1 | Epochs | Train L1 | Test L1 | D died at |
|-----|------|------|--------|----------|---------|-----------|
| v1 | 2e-4 | 100 | 155 | 0.229 | 0.343 | epoch ~30 |
| v2 | 1e-4 | 100 | 197 | 0.265 | 0.360 | epoch ~33 |
| v3 | — | — | pending | — | — | — |

**Phase 2** (caricature finetune): Two attempts, both failed to transfer style. Root cause: Phase 1 base weights lack texture/sharpness, and 184 pairs is even fewer → D dies faster. Run v2 reached Train L1=0.40 at epoch 69.

## The Problem: D Dies on Small Datasets

Every run follows the same pattern:

```
Epochs 1-30:   D alive   → adversarial signal contributes texture
Epochs 30+:    D dead    → pure L1 regression only
```

With only 322 training pairs, the discriminator **memorizes all real images** by epoch ~30 and rejects everything else with 99.99% confidence (Df=0.000). The generator receives zero adversarial gradient — the loss that's supposed to add pencil-stroke texture and sharp lines simply stops working.

**Visual symptom:** Outputs look like "black-and-white photocopies" — structurally correct faces, but no pencil-drawing texture, no sharp strokes, no artistic quality. This is the classic L1-only blur effect. L1 loss produces the *average* sketch; adversarial loss adds the *texture*.

Lowering D's learning rate (v2: 1e-4 vs 2e-4) bought ~3 more epochs. Not enough.

## The Fix: Spectral Norm + Noise Injection

Two light-weight additions to the discriminator that prevent memorization:

| Technique | What it does | Why it helps |
|-----------|-------------|--------------|
| **Spectral Normalization** | Bounds each D layer's weight norm by its largest singular value | Prevents D from getting too confident — limits its capacity to memorize |
| **Noise Injection** | Adds Gaussian noise (σ=0.05) to D's inputs during training | Forces D to generalize — can't rely on exact pixel values to discriminate |

Both are standard practice for small-dataset GANs. Combined, they should keep Df above 0.0 for 100+ epochs instead of 30, giving the adversarial loss time to contribute pencil-drawing texture.

### Code changes

```python
# discriminator.py — two new constructor params (both default True)
PatchGANDiscriminator(
    ...,
    use_spectral_norm=True,   # wraps every Conv2d in nn.utils.spectral_norm
    noise_std=0.05,           # adds randn_like(x) * 0.05 to D's forward()
)

# train.py — config defaults
"d_spectral_norm": True,
"d_noise_std": 0.05,
```

## What Stays the Same

- Architecture: UNetGenerator (ngf=64, 5 levels) + PatchGAN (ndf=64, 3 layers)
- Loss: λ_L1=100, λ_adv=1, BCE + L1
- Optimizer: Adam β1=0.5, G_lr=2e-4, D_lr=1e-4
- Data: 322 pairs, 256×256, 100% training
- Epochs: 200

## Dataset

- **Training:** 322 paired (photo, sketch) — CUHK Student Sketch (188) + SKSF-A (134)
- **Test:** 100 paired (photo, sketch) — app-generated, held out completely
- **Resolution:** 256×256

## Success Criteria

- [ ] Generator produces recognizable face sketches (already ✅ from v1/v2)
- [ ] Outputs have visible pencil-stroke texture, not just blurred photocopy
- [ ] D survives 100+ epochs (Df stays in 0.1–0.4 range)
- [ ] Adversarial training contributes throughout, not just first 30 epochs

## Commands

```bash
# Colab
python src/train.py --mode train --device cuda --name pix2pix_v3_

# Kaggle (on P100 or T4x2)
python src/train.py --mode train --device cuda --name pix2pix_v3_ --batch-size 24
```

## Next After Fix

If Phase 1 produces sharp pencil-drawing outputs → proceed to Phase 2 finetune on caricatures with the same D stabilization fixes applied.
