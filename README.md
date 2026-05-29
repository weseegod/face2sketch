# 🎨 face2sketch

> Build image-to-image generative models from scratch. Zero wrappers, zero `diffusers`.
> From an empty U-Net to generating funny caricature drawings from face photos.

**Goal:** User uploads a face photo → model outputs a funny cartoon/caricature drawing.

**Approach:** Start with GAN (pix2pix) for paired image translation, then upgrade to conditional diffusion. Learn both paradigms — the industry-standard GAN baseline AND the modern diffusion SOTA.

## 📋 Status

| Phase | Approach | Status |
|-------|----------|--------|
| **Phase 1: pix2pix GAN** | Photo → Realistic Sketch | 🔜 Planning |
| **Phase 2: Finetune** | Photo → Funny Caricature | 🔮 After Phase 1 |
| **Phase 3: Diffusion Upgrade** | Photo → Caricature (SOTA) | 🔮 After Phase 2 |

## 🚀 Quick Start

```bash
git clone https://github.com/weseegod/face2sketch.git
cd face2sketch
pip install -r requirements.txt

# Phase 1: Train pix2pix on CUHK + SKSF-A (photo → sketch)
python src/train.py --config configs/pix2pix_phase1.yaml

# Generate from a photo
python src/sample.py --checkpoint checkpoints/pix2pix_best.pt --input my_face.jpg
```

## 📖 Documentation

| Document | What's in it |
|----------|-------------|
| **[docs/ROADMAP.md](docs/ROADMAP.md)** | Full 3-phase plan with timeline, success criteria, industry context |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | Deep dive: U-Net, PatchGAN, adversarial loss, perceptual loss |
| **[docs/DATA.md](docs/DATA.md)** | Dataset guide: CUHK, SKSF-A, TwitterPicasso |

## 🏗️ Architecture (Phase 1: pix2pix)

```
                    ┌──────────────────────┐
  Face Photo ─────► │    GENERATOR         │
                    │    (U-Net)            │──► Generated Sketch
                    │    ~50M params        │
                    └──────────────────────┘
                              │
                              │ (photo, generated_sketch)
                              ▼
                    ┌──────────────────────┐
                    │  DISCRIMINATOR        │
                    │  (PatchGAN)           │──► REAL or FAKE?
                    │  ~3M params           │     per 70×70 patch
                    └──────────────────────┘

  Loss = L1(real_sketch, generated_sketch)  ← reconstruction
       + adversarial_loss                    ← realism
```

### Generator (U-Net)

- **Encoder-decoder** with skip connections — same architecture as the diffusion version
- **Input:** face photo (3 channels)
- **Output:** sketch drawing (3 channels)
- **Skip connections** pass low-level details (edges, textures) directly to decoder

### Discriminator (PatchGAN)

- **Patch-level** classification: judges 70×70 patches as real/fake, not the whole image
- **Much fewer params** than the generator (~3M vs ~50M)
- **Only used during training** — at inference, you only need the Generator

## 🧠 Learning Path

| Step | Skill |
|------|-------|
| 1. Build U-Net Generator | Conv2d, skip connections, encoder-decoder |
| 2. Build PatchGAN Discriminator | Strided conv, patch-level classification |
| 3. Implement GAN training loop | Adversarial loss, L1 reconstruction, balance |
| 4. Train on CUHK + SKSF-A | GAN dynamics, mode collapse prevention, loss curves |
| 5. Finetune on TwitterPicasso | Transfer learning, domain adaptation |
| 6. Upgrade to Diffusion (Phase 3) | Noise schedules, DDPM/DDIM, ControlNet |

## 🔗 Key References

| Paper | What it teaches | Phase |
|-------|-----------------|-------|
| [pix2pix (Isola et al. 2017)](https://arxiv.org/abs/1611.07004) | Paired image translation, U-Net + PatchGAN | 1, 2 |
| [PatchGAN (Isola et al. 2017)](https://arxiv.org/abs/1611.07004) | Patch-level discriminator, L1+GAN loss | 1 |
| [pix2pixHD (Wang et al. 2018)](https://arxiv.org/abs/1711.11585) | Multi-scale, perceptual loss, instance maps | 2 |
| [DDPM (Ho et al. 2020)](https://arxiv.org/abs/2006.11239) | Diffusion probabilistic models | 3 |
| [DDIM (Song et al. 2021)](https://arxiv.org/abs/2010.02502) | Fast deterministic sampling | 3 |
| [ControlNet (Zhang et al. 2023)](https://arxiv.org/abs/2302.05543) | Conditional control for diffusion | 3 |
| [LoRA (Hu et al. 2021)](https://arxiv.org/abs/2106.09685) | Low-rank adaptation for efficient fine-tuning | 3 |

## 📂 Project Structure

```
face2sketch/
├── src/
│   ├── unet.py              # U-Net generator (Phase 1, reused in Phase 3)
│   ├── discriminator.py     # PatchGAN discriminator (Phase 1)
│   ├── gan_trainer.py       # pix2pix training loop (Phase 1-2)
│   ├── diffusion.py         # DDPM/DDIM scheduler (Phase 3)
│   ├── data_loader.py       # Paired data loading (all phases)
│   ├── train.py             # Main training entry point
│   └── sample.py            # Generation & visualization
├── configs/
│   └── pix2pix_phase1.yaml  # Phase 1 training config
├── docs/
│   ├── ROADMAP.md           # Complete 3-phase plan
│   ├── ARCHITECTURE.md      # Architectural deep dive
│   └── DATA.md              # Dataset guide
├── checkpoints/             # Trained weights (gitignored)
├── data/
│   ├── dataset/             # 322 pairs: CUHK + SKSF-A (Phase 1)
│   └── finetune/            # 184 pairs: TwitterPicasso (Phase 2)
├── samples/                 # Generated images during training
├── notebooks/               # Exploration & demos
├── requirements.txt
└── README.md
```

## 🎯 End Goal

```
Input                          Output
┌──────────┐                   ┌──────────────┐
│  selfie  │                   │  🤪          │
│  photo   │  ──── model ──►  │  funny       │
│  😊      │                   │  caricature  │
│          │                   │  drawing     │
└──────────┘                   └──────────────┘

Phase 1: photo → realistic sketch
Phase 2: photo → funny caricature
Phase 3: photo → high-quality funny caricature (diffusion)
```
# face2sketch
