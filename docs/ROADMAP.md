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

## 🗺️ Four Phases

```
Phase 1 ──────► Phase 2 ──────► Phase 3 ──────► Phase 4
pix2pix GAN    Finetune to    DDPM from       ControlNet
Photo→Sketch   Caricature     Scratch         + LoRA
✅ DONE        ❌ GAN limit   Educational     Production
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

## 📋 Phase 3: Conditional DDPM from Scratch — Educational (2-3 weeks)

### Why DDPM First?

Before using black-box production systems (ControlNet), you need to understand how diffusion actually works: forward noising process, reverse denoising, noise prediction, DDIM sampling. Building a DDPM from scratch teaches you the internals that every diffusion-based system is built on.

### Approach

```
  Photo + Noisy Sketch ──► Conditional U-Net ──► Predicted Noise
  (concatenated)
```

- Build DDPM/DDIM scheduler from scratch (no diffusers, no HuggingFace)
- Reuse Phase 1 U-Net architecture, modify to 6 input channels (photo + noisy target)
- Train on CUHK + SKSF-A (322 pairs) for proof-of-concept
- Then try on TwitterPicasso (184 pairs) with the same architecture

### Why This Might Work for Caricatures

Unlike GANs, DDPM doesn't need a discriminator. The loss is simple: predict the noise that was added to the image. No adversarial balance to maintain, no D dying on small data. The model learns a distribution over possible outputs — which means it can generate different caricatures from the same photo (diversity that GANs can't provide).

### What You Build

| Component | File | What You Learn |
|-----------|------|----------------|
| **DDPM Scheduler** | `src/diffusion.py` | Forward diffusion (add noise), reverse process, β schedule |
| **DDIM Sampler** | `src/diffusion.py` | Fast deterministic sampling, skip-step inference |
| **Noise Predictor** | `src/unet.py` (modified) | Conditional U-Net that predicts noise given photo + noisy sketch |

### Architecture

```
Noise Predictor (modified Phase 1 U-Net):
  Input: concatenated(photo, noisy_sketch, timestep_embedding) → 6+ channels
  Encoder: same 5-level structure as Phase 1
  Bottleneck: same
  Decoder: same 5-level structure + skip connections
  Output: predicted noise (3 channels)
```

### Loss

```
L = MSE(ε, ε_θ(photo, noisy_sketch, t))
```

Where ε is the actual noise added, ε_θ is the model's prediction. Simple, stable, no adversarial dynamics.

### Schedule

| Week | Task |
|------|------|
| 1 | Implement DDPM scheduler (forward + reverse) |
| 1 | Implement noise predictor U-Net (timestep embedding, conditioning) |
| 2 | Train on synthetic data: add known noise, verify recovery |
| 2-3 | Train on CUHK (322 pairs), then TwitterPicasso (184 pairs) |
| 3 | Implement DDIM sampling for fast inference |

### Success Criteria

- [ ] Forward diffusion: clean sketch → pure noise over T steps
- [ ] Reverse diffusion: pure noise → recognizable sketch
- [ ] Conditioning works: photo guides the denoising to the correct face
- [ ] DDIM sampling: generates in 20-50 steps instead of 1000
- [ ] Diversity: same photo produces different sketches on different runs

---

## 📋 Phase 4: ControlNet + LoRA on Stable Diffusion — Production (2-3 weeks)

### Why ControlNet?

Once you understand diffusion fundamentals from Phase 3, ControlNet is the production-grade way to do conditional generation. Instead of training from scratch, you leverage Stable Diffusion 1.5 — a model trained on billions of images that already knows how to draw faces, textures, and lighting.

### Approach

```
  Face Photo ──► ControlNet (trainable) ──► Guides SD denoising
                                               │
  Random Noise ──► SD 1.5 U-Net + LoRA ────────┤
                                               ▼
                                        Caricature Output
```

- Use pretrained SD 1.5 (~860M params, frozen)
- Train ControlNet to extract spatial structure from face photos
- Train LoRA (~20M params) on SD U-Net for TwitterPicasso style
- Combined: photo drives structure, LoRA drives style

### Why This Works with 184 Pairs

- **ControlNet:** Pre-trained face ControlNet already exists (MediaPipe face, Canny edges). Only fine-tune, not train from scratch.
- **LoRA:** Designed for few-shot style adaptation. 184 pairs is a reasonable LoRA dataset. Adds small trainable adapters to frozen SD layers.
- **No discriminator needed.** Diffusion's denoising objective is inherently stable.

### What You Build

| Component | File | What You Learn |
|-----------|------|----------------|
| **ControlNet** | `src/controlnet.py` | Zero-convolution, locked encoder copy, spatial conditioning |
| **LoRA Adapter** | `src/lora.py` | Low-rank decomposition, trainable adapters on frozen weights |
| **SD Pipeline** | `src/pipeline.py` | Text-to-image with ControlNet + LoRA, CFG guidance |

### Budget

| | Phase 3 (DDPM) | Phase 4 (ControlNet) |
|---|---|---|
| GPU | T4/P100 | T4/P100 |
| Resolution | 256×256 | 512×512 |
| Epochs | ~500-1000 | ~50 |
| Time | ~10-20h | ~4-6h |
| VRAM | ~8GB | ~12GB |

### Phase Gates

#### Phase 3 → Phase 4: GO when
- [ ] DDPM generates recognizable face sketches from photos
- [ ] Conditioning works: photo controls output structure
- [ ] DDIM sampling produces results in <50 steps
- [ ] Understanding of diffusion internals is solid

#### Phase 4 → Ship: GO when
- [ ] Caricature style transfers to new photos
- [ ] Face identity is preserved (looks like the input person)
- [ ] Multiple different caricatures possible from same photo
- [ ] Fun factor: images make you smile 😄

---

## 📚 Reading List

### Phase 1 — GANs
- [pix2pix (Isola et al. 2017)](https://arxiv.org/abs/1611.07004)
- [U-Net (Ronneberger et al. 2015)](https://arxiv.org/abs/1505.04597)
- [PatchGAN explained](https://machinelearningmastery.com/a-gentle-introduction-to-pix2pix-generative-adversarial-network/)

### Phase 2 — GAN Transfer
- [Transfer Learning in GANs](https://arxiv.org/abs/1812.04948)

### Phase 3 — DDPM from Scratch
- [DDPM (Ho et al. 2020)](https://arxiv.org/abs/2006.11239) — the diffusion bible
- [DDIM (Song et al. 2021)](https://arxiv.org/abs/2010.02502) — fast sampling
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) — line-by-line code
- [What are Diffusion Models? (Lil'Log)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

### Phase 4 — ControlNet + LoRA
- [ControlNet (Zhang et al. 2023)](https://arxiv.org/abs/2302.05543)
- [LoRA (Hu et al. 2021)](https://arxiv.org/abs/2106.09685)
- [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)

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
