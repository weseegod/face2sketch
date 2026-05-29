# 🔮 Face Caricature — Learning Roadmap

> Build image-to-image models from scratch in raw PyTorch. Zero diffusers, zero HuggingFace wrappers.
> Goal: user uploads face photo → model outputs funny cartoon drawing.
> Approach: GAN first (industry baseline), then diffusion (modern SOTA).

---

## 🎯 End Goal

```
Input                          Output
┌──────────┐                   ┌──────────────┐
│  selfie  │                   │  🤪          │
│  photo   │  ──── model ──►  │  funny       │
│  😊      │                   │  caricature  │
│          │                   │  drawing     │
└──────────┘                   └──────────────┘
```

Build everything yourself, understand every line. Same philosophy: no black boxes.

---

## 🗺️ Three Phases

```
Phase 1 ──────► Phase 2 ──────► Phase 3
pix2pix GAN    Finetune to    Upgrade to
Photo→Sketch   Caricature     Diffusion
1-2 weeks      1 week         2-3 weeks
```

---

## 📋 Phase 1: pix2pix GAN — Photo → Realistic Sketch (1-2 weeks)

### Why GAN first?

GANs (Generative Adversarial Networks) have been the dominant paradigm for image-to-image translation since 2014. pix2pix (Isola et al. 2017) is the canonical paired-translation architecture. Understanding GAN dynamics — adversarial training, mode collapse, discriminator-generator balance — is essential industry knowledge even if diffusion is now SOTA.

### Why this specific task?

CUHK Student Sketch + SKSF-A provides 322 paired (photo, sketch) images. This is a well-studied benchmark. The task is simple enough to train quickly but complex enough to teach you:

- How to structure a conditional generator
- How adversarial loss improves output sharpness beyond L1 alone
- Why PatchGAN works better than global discrimination
- GAN training stability tricks

### Goal

Train a model that takes any face photo and outputs a recognizable pencil sketch.

### What You Build

| Component | File | What You Learn |
|-----------|------|----------------|
| **U-Net Generator** | `src/unet.py` | Encoder-decoder, skip connections, conditional input via channel concatenation |
| **PatchGAN Discriminator** | `src/discriminator.py` | Strided convolutions, patch-level real/fake classification, receptive field math |
| **GAN Training Loop** | `src/train.py` | Alternating optimizer updates, L1 reconstruction + adversarial loss, loss weighting |
| **Inference** | `src/sample.py` | Single forward pass through generator, no iterative sampling needed |

### Architecture

```
Generator (U-Net, ~50M params):
  Input: photo (3 channels)
  Encoder: 5 levels of downsampling (128→256→512→512→512 ch)
  Bottleneck: lowest resolution features
  Decoder: 5 levels of upsampling + skip connections from encoder
  Output: sketch (3 channels)
  Activation: tanh on output (range [-1, 1])

Discriminator (PatchGAN, ~3M params):
  Input: concatenated(photo, sketch_or_fake) → 6 channels
  5 strided conv layers outputting N×N patch predictions
  Each patch prediction: probability that this 70×70 region is "real"
  Output: 30×30 feature map (each cell = one patch judgment)
```

### Loss Function

```
L_G = λ_L1 * L1(real_sketch, fake_sketch)    ← reconstruction
    + λ_adv * BCE(D(photo, fake_sketch), 1)   ← fool discriminator

L_D = 0.5 * BCE(D(photo, real_sketch), 1)    ← real pairs
    + 0.5 * BCE(D(photo, fake_sketch), 0)    ← fake pairs
```

Key insight: Without adversarial loss (λ_adv = 0), the generator just minimizes L1 — outputs are blurry (the "average" sketch). The adversarial loss pushes outputs to look REAL — sharp lines, correct texture. The tradeoff is controlled by λ_L1 (typically 100).

### Schedule

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Implement U-Net generator (`unet.py`) | Forward pass works, unit tests pass |
| 3 | Implement PatchGAN discriminator (`discriminator.py`) | Forward pass, receptive field verified |
| 4-5 | Implement GAN training loop | Loss curves show generator and discriminator competing |
| 6-7 | Train on CUHK + SKSF-A (~200 epochs) | Recognizable face sketches from photos |
| 8-10 | Debug, tune hyperparameters, add perceptual loss | Sharp, clean sketches |
| 11-14 | Polish: checkpointing, visualization, evaluation metrics | Clean code, good samples |

### Success Criteria

- [ ] U-Net forward pass: photo (B,3,256,256) → sketch (B,3,256,256)
- [ ] Discriminator forward pass: (photo, sketch) → (B,1,30,30) patch predictions
- [ ] Training: generator loss decreases, discriminator stays around 0.5-0.7 (balanced)
- [ ] Inference on validation photos produces recognizable face sketches
- [ ] No mode collapse (different input photos → different output sketches)
- [ ] Checkpoint save/resume works

### Colab L4 Budget

- Epochs: 200
- Batch: 16
- Resolution: 256×256
- Time per epoch: ~20-30s
- **Total: ~1.5 hours**
- VRAM: ~8GB / 24GB (comfortable)

---

## 📋 Phase 2: Finetune on Caricatures (1 week)

### Goal

Take the Phase 1 model (which can draw clean realistic sketches) and teach it to draw in the TwitterPicasso exaggerated caricature style.

### Why This Works (Transfer Learning)

Phase 1 taught the model:
- Face geometry (where eyes, nose, mouth go)
- How to map photo → drawing
- Edge detection, shading

Phase 2 only needs to teach:
- Exaggerate features (bigger nose, wider eyes, funnier proportions)
- Adapt to the TwitterPicasso artistic style

This is the same principle as fine-tuning a pretrained LLM — the base knowledge transfers, only the style needs to adapt.

### What Changes from Phase 1

| | Phase 1 | Phase 2 |
|---|---|---|
| Data | CUHK + SKSF-A (322 pairs) | TwitterPicasso (184 pairs) |
| Initialization | Random | Phase 1 checkpoint |
| Learning rate | 2e-4 | 5e-5 (lower for fine-tuning) |
| Augmentation | Moderate | Aggressive (more rotations, elastic transforms) |
| Epochs | 200 | 100 |
| L1 weight (λ_L1) | 100 | 50 (lean more on adversarial for style) |

### Schedule

| Day | Task |
|-----|------|
| 1 | Load Phase 1 checkpoint, verify it works |
| 2-3 | Train on TwitterPicasso (100 epochs) |
| 4-5 | Evaluate quality, tune λ_L1 vs λ_adv balance |
| 6-7 | Full run, generate comparison grid (Phase 1 vs Phase 2 outputs) |

---

## 📋 Phase 3: Upgrade to Conditional Diffusion (2-3 weeks)

### Why Upgrade?

- **Diversity:** GANs can produce only one output per input. Diffusion can generate multiple different caricatures from the same photo.
- **Quality at scale:** Diffusion scales better with compute and data.
- **Industry SOTA:** Most production image generation systems (Midjourney, DALL-E, SD) use diffusion.

### Two Sub-Paths

#### Path A: ControlNet + LoRA on Stable Diffusion (Production)

```
  Face Photo ──► ControlNet (frozen) ──► Guides diffusion process
                                           │
  Random Noise ──► SD U-Net + LoRA ────────┤
                                           ▼
                                    Funny Caricature
```

- Use pretrained Stable Diffusion 1.5 (~860M params)
- Add ControlNet conditioned on face photo
- Train only ControlNet + LoRA weights (~20M params)
- **Pros:** Works with 184 pairs, production quality, fast training
- **Cons:** Less "from scratch" understanding, depends on external model

#### Path B: Conditional DDPM from Scratch (Educational)

```
  Photo + Noisy Sketch ──► Conditional U-Net ──► Predicted Noise
  (concatenated)
```

- Build DDPM/DDIM scheduler from scratch
- Modify Phase 1 U-Net to accept 6 input channels (photo + noisy target)
- **Pros:** Deep understanding of diffusion internals
- **Cons:** Needs 5K-20K images for decent quality, long training time

### Recommendation

Do **both** — start with Path B on a small scale to understand the fundamentals (train on ~500 generated pairs), then do Path A for actual production-quality results. This teaches you both: the internals AND how production systems actually work.

---

## 📊 Full Training Budget (L4 Colab)

| Phase | Approach | Epochs | Batch | Resolution | Time |
|-------|----------|--------|-------|------------|------|
| 1: pix2pix | GAN from scratch | 200 | 16 | 256×256 | ~1.5h |
| 2: Finetune | GAN transfer | 100 | 16 | 256×256 | ~45min |
| 3: SD+ControlNet | Diffusion adapter | ~50 | 4 | 512×512 | ~4-6h |
| 3 alt: DDPM | Diffusion from scratch | ~1000 | 8 | 256×256 | ~20h+ |
| **Total** | | | | | **~7h (GAN path)** |

---

## 📚 Reading List (by phase)

### Phase 1
- [pix2pix (Isola et al. 2017)](https://arxiv.org/abs/1611.07004) — the canonical paired image translation paper
- [GANs in 50 lines of PyTorch](https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f) — intuition
- [U-Net (Ronneberger et al. 2015)](https://arxiv.org/abs/1505.04597) — original U-Net paper for segmentation
- [PatchGAN explained](https://machinelearningmastery.com/a-gentle-introduction-to-pix2pix-generative-adversarial-network/) — blog post

### Phase 2
- [Transfer Learning in GANs](https://arxiv.org/abs/1812.04948) — few-shot GAN adaptation
- [pix2pixHD (Wang et al. 2018)](https://arxiv.org/abs/1711.11585) — multi-scale, perceptual loss

### Phase 3
- [DDPM (Ho et al. 2020)](https://arxiv.org/abs/2006.11239) — the diffusion bible
- [DDIM (Song et al. 2021)](https://arxiv.org/abs/2010.02502) — fast sampling
- [ControlNet (Zhang et al. 2023)](https://arxiv.org/abs/2302.05543) — conditional control for diffusion
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) — line-by-line code

---

## 🚦 Phase Gates

### Phase 1 → Phase 2: GO when
- [ ] Generator produces recognizable sketches from validation photos
- [ ] Discriminator accuracy ~50-70% (not too strong, not too weak)
- [ ] Training is stable across 200 epochs (no mode collapse)
- [ ] L1 loss < 0.1 on validation set

### Phase 2 → Phase 3: GO when
- [ ] Fine-tuned model produces caricatures in TwitterPicasso style
- [ ] Face identity is preserved (you can tell who the person is)
- [ ] Outputs are noticeably different from Phase 1 (funnier, more exaggerated)

### Phase 3 → Ship: GO when
- [ ] Caricature style transfers consistently
- [ ] Diverse outputs from same input (if using diffusion)
- [ ] Fun factor is there — images make you smile 😄

---

## 🏗️ Project Structure (Final)

```
face2sketch/
├── src/
│   ├── unet.py              # U-Net generator (all phases)
│   ├── discriminator.py     # PatchGAN discriminator (Phase 1-2)
│   ├── gan_trainer.py       # pix2pix GAN training loop (Phase 1-2)
│   ├── diffusion.py         # DDPM/DDIM scheduler (Phase 3)
│   ├── data_loader.py       # ✅ Paired data loading (all phases)
│   ├── train.py             # Main entry point
│   └── sample.py            # Generation & visualization
├── configs/
│   └── pix2pix_phase1.yaml  # Phase 1 config
├── docs/
│   ├── ROADMAP.md           # This file
│   ├── ARCHITECTURE.md      # U-Net + PatchGAN + loss functions
│   └── DATA.md              # Dataset guide
├── checkpoints/
├── data/
│   ├── dataset/             # 322 pairs (CUHK + SKSF-A)
│   └── finetune/            # 184 pairs (TwitterPicasso)
├── notebooks/
├── samples/
├── requirements.txt
└── README.md
```

---

## 💡 Key Industry Lessons You'll Learn

1. **GAN dynamics are subtle.** Generator and discriminator are in an arms race. If one gets too strong, the other can't learn. Balance is everything.

2. **L1 reconstruction + adversarial loss** is a universal pattern. The L1 loss gives the "content" (structure), the adversarial loss gives the "style" (texture, sharpness). This separation of content and style is fundamental.

3. **Patch-level discrimination** is more effective than global. Judging an entire image as real/fake is too coarse — the discriminator overfits. Judging small patches forces the discriminator to learn local texture consistency.

4. **Transfer learning works for GANs too.** The same face→drawing mapping learned on one dataset transfers surprisingly well to another drawing style.

5. **Diffusion is NOT always the answer.** For small paired datasets with a clear translation task, GANs still win on training speed and data efficiency. Diffusion shines with large-scale diverse generation.

6. **Build one canonical inference function.** Whether GAN (single forward pass) or diffusion (iterative denoising), the user-facing API should be: `generate(photo) → drawing`.
