# 🏗️ Architecture Deep Dive: pix2pix GAN

> Understanding every component — U-Net Generator, PatchGAN Discriminator, and adversarial training dynamics.

---

## 📐 The Big Picture

```
pix2pix = Conditional GAN for paired image translation

                    ┌──────────────────────┐
  Input Photo  ───► │    GENERATOR (G)      │
  x (real face)     │    U-Net              │──► G(x)  (generated sketch)
                    │    ~50M params         │
                    └──────────────────────┘
                              │
                              │
                    ┌─────────▼────────────┐
                    │                      │
              (x, real_y)           (x, G(x))
              real pair             fake pair
                    │                      │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │  DISCRIMINATOR (D)    │
                    │  PatchGAN             │
                    │  ~3M params           │
                    └──────┬───────────────┘
                           ▼
                    30×30 patch predictions
                    Each: P(real) for 70×70 region

  Training: G and D play a minimax game
  Inference: Only G is used — single forward pass photo → sketch
```

### Why Two Networks?

- **Generator alone** (L1 loss only): produces blurry averages — safe but unrealistic
- **Generator + Discriminator** (L1 + adversarial): L1 ensures structural correctness, adversarial loss pushes toward photorealism
- The Discriminator is thrown away after training. It's a teaching tool, not part of the product.

---

## 🧱 1. U-Net Generator — The Artist

### Why U-Net?

U-Net was invented for biomedical image segmentation (Ronneberger 2015). It's ideal for image translation because:

1. **Same-size input/output** — perfect for pixel-level mapping
2. **Skip connections** — low-level features (edges, positions) flow directly from encoder to decoder, preserving fine detail
3. **Bottleneck** — forces information through compressed representation → learns global structure
4. **No fully-connected layers** — purely convolutional → works at any resolution

### Architecture Details

```
Input: photo (B, 3, 256, 256)
    │
    ▼
InputConv (3 → 64, 3×3 conv)    ← No activation yet
    │
    ▼
┌───────────────────────────────────────────┐
│ ENCODER (contracting path)                │
│                                           │
│ Encoder 0 (256×256, 64ch):               │
│   Conv(3→64) → BN → ReLU → Conv(3→64)    │ ──skip──┐
│   └─ MaxPool(2) → (128×128, 128ch)       │          │
│                                           │          │
│ Encoder 1 (128×128, 128ch):               │          │
│   Conv → BN → ReLU → Conv                 │ ──skip──┤
│   └─ MaxPool(2) → (64×64, 256ch)         │          │
│                                           │          │
│ Encoder 2 (64×64, 256ch):                │          │
│   Conv → BN → ReLU → Conv                 │ ──skip──┤
│   └─ MaxPool(2) → (32×32, 512ch)         │          │
│                                           │          │
│ Encoder 3 (32×32, 512ch):                │          │
│   Conv → BN → ReLU → Conv                 │ ──skip──┤
│   └─ MaxPool(2) → (16×16, 512ch)         │          │
│                                           │          │
│ Encoder 4 (16×16, 512ch):                │          │
│   Conv → BN → ReLU → Conv                 │ ──skip──┤
│   └─ MaxPool(2) → (8×8, 512ch)           │          │
└───────────────────────────────────────────┘          │
    │                                                   │
    ▼                                                   │
┌───────────────────────────────────────────┐          │
│ BOTTLENECK (8×8, 512ch)                  │          │
│   Conv → BN → ReLU → Conv → BN → ReLU    │          │
│   (no pooling here, this is the bottom)   │          │
└───────────────────────────────────────────┘          │
    │                                                   │
    ▼                                                   │
┌───────────────────────────────────────────┐          │
│ DECODER (expanding path)                  │          │
│                                           │          │
│ Decoder 4 (8→16×16):                      │          │
│   UpConv(2×) → concat(skip4) →            │ ◄────────┤
│   Conv → BN → ReLU → Conv                 │          │
│                                           │          │
│ Decoder 3 (16→32×32):                     │          │
│   UpConv(2×) → concat(skip3) →            │ ◄────────┤
│   Conv → BN → ReLU → Conv                 │          │
│                                           │          │
│ Decoder 2 (32→64×64):                     │          │
│   UpConv(2×) → concat(skip2) →            │ ◄────────┤
│   Conv → BN → ReLU → Conv                 │          │
│                                           │          │
│ Decoder 1 (64→128×128):                   │          │
│   UpConv(2×) → concat(skip1) →            │ ◄────────┤
│   Conv → BN → ReLU → Conv                 │          │
│                                           │          │
│ Decoder 0 (128→256×256):                  │          │
│   UpConv(2×) → concat(skip0) →            │ ◄────────┘
│   Conv → BN → ReLU → Conv                 │
└───────────────────────────────────────────┘
    │
    ▼
OutputConv (64 → 3, 1×1 conv) → tanh
    │
    ▼
Generated Sketch (B, 3, 256, 256) in [-1, 1]
```

### Key Design Decisions

**Conv → BatchNorm → ReLU pattern:** The standard U-Net block. Two 3×3 convolutions (no padding reduction — use padding=1 to preserve size). After each conv: BatchNorm → ReLU. This pattern repeats at every encoder/decoder level.

> **Note:** In the diffusion version of U-Net (Phase 3), we use GroupNorm instead of BatchNorm (batch-size independence) and SiLU instead of ReLU (smoother gradients). But for GANs, BN + ReLU is standard and well-tested.

**Skip connections:** After each upconvolution in the decoder, concatenate the feature map from the corresponding encoder level along the channel dimension. This DOUBLES the channel count at the decoder input, which the first conv in that decoder level compresses back. This is the U-Net's signature — without it, the decoder would only see bottleneck features (blurry results).

**Upsampling:** Use `nn.ConvTranspose2d` OR `nn.Upsample(scale_factor=2) + Conv2d`. The latter avoids checkerboard artifacts and is preferred in modern implementations.

---

## 🕵️ 2. PatchGAN Discriminator — The Critic

### Why Patch-Level?

A global discriminator outputs a single number: "this entire image is real/fake." This is too easy — the discriminator learns to reject based on broad blurriness and misses local artifacts.

PatchGAN outputs an N×N grid of predictions. Each cell judges whether a 70×70 patch of the input is real or fake. This forces the discriminator to look at local texture consistency:

```
Global Discriminator:         PatchGAN Discriminator:
  Input → Conv → ... → Dense   Input → Conv → ... → 30×30 map
  Output: [0.87]               Output: [0.9 0.2 0.8 ...
  "one judgment"                       0.1 0.9 0.3 ...
                                         0.8 0.4 0.9]
                                       "900 judgments"
```

Each cell in the output corresponds to a 70×70 region in the input. The discriminator effectively runs 900 times on overlapping patches — all in one forward pass!

### Architecture

```
Input: concatenate(photo, sketch_or_fake) → (B, 6, 256, 256)

Layer 0: Conv2d(6 → 64, kernel=4, stride=2, padding=1)
         → LeakyReLU(0.2)                     # (B, 64, 128, 128)

Layer 1: Conv2d(64 → 128, kernel=4, stride=2, padding=1)
         → BatchNorm → LeakyReLU(0.2)          # (B, 128, 64, 64)

Layer 2: Conv2d(128 → 256, kernel=4, stride=2, padding=1)
         → BatchNorm → LeakyReLU(0.2)          # (B, 256, 32, 32)

Layer 3: Conv2d(256 → 512, kernel=4, stride=1, padding=1)
         → BatchNorm → LeakyReLU(0.2)          # (B, 512, 31, 31)

Output: Conv2d(512 → 1, kernel=4, stride=1, padding=1)
        → Sigmoid                              # (B, 1, 30, 30)
```

### Receptive Field Math

With 4×4 kernels and stride-2 downsampling (except layer 3 which is stride-1), each output neuron sees a **70×70 patch** of the input. This was found empirically to work best — large enough to see local structure, small enough to enforce detail.

### Why LeakyReLU(0.2)?

GAN discriminators use LeakyReLU instead of ReLU. Why? Standard ReLU kills negative gradients (gradient = 0 when input < 0). When the discriminator gets too strong, many neurons die. LeakyReLU(0.2) lets a small gradient through for negative inputs, keeping the discriminator "alive" even when it's winning. This prevents the generator from getting zero gradient.

---

## ⚔️ 3. Loss Functions — The Learning Signal

### Generator Loss

```
L_G = L_L1 + λ * L_BCE_G

where:
  L_L1     = mean(|real_y - G(x)|)            ← per-pixel reconstruction
  L_BCE_G  = BCE(D(x, G(x)), target=1)        ← "fool the discriminator"
  λ        = 100 (L1 weight)

  real_y = ground truth sketch
  G(x)   = generated sketch from photo x
  D(x,y) = discriminator output for pair (x,y)
```

**L1 loss** ensures the generator output is structurally close to the ground truth. L1 is preferred over L2 (MSE) because it produces sharper edges — L2 penalizes large errors quadratically and encourages blurry "safe" averages.

**Adversarial loss** pushes the generator to produce crisp, realistic outputs that fool the discriminator. Without it (λ=∞), outputs are blurry averages. Without L1 (λ=0), outputs can be sharp but structurally wrong — the generator might produce a realistic sketch of a DIFFERENT person!

**λ=100** is the standard starting point from the pix2pix paper. It heavily weights structural correctness while still getting the sharpness benefit of adversarial training.

### Discriminator Loss

```
L_D = 0.5 * BCE(D(x, real_y), target=1)           ← real pairs as "real"
    + 0.5 * BCE(D(x, G(x).detach()), target=0)    ← fake pairs as "fake"

.detach() is CRITICAL: prevents generator gradients from flowing
through the discriminator loss. The discriminator always gets clean
"is this real?" labels, never convoluted G+D gradients.
```

The discriminator is a simple binary classifier: real pairs = 1, fake pairs = 0.

### One-Sided Label Smoothing (Stability Trick)

Instead of using hard labels (0.0 and 1.0), smooth the REAL labels slightly:

```
target_real = 0.9  (instead of 1.0)
target_fake = 0.0  (keep at 0.0 — don't smooth fake labels)

Why only smooth real labels?
Smoothing fake labels would encourage the discriminator
to be less certain about rejecting fakes — bad for the generator.
Smoothing real labels prevents the discriminator from becoming
overconfident, which helps balance the game.
```

---

## 🏋️ 4. Training Dynamics

### The Arms Race

```
Epoch 1:    G produces random noise, D easily distinguishes
            D loss: 0.1 (easy), G loss: 10+ (high L1)
            
Epoch 10:   G produces blurry face shapes, D has to work harder
            D loss: 0.5, G loss: 2-3
            
Epoch 50:   G produces recognizable sketches, D is confused
            D loss: 0.7 (struggling), G loss: 1-2
            
Epoch 200:  G produces sharp, clean sketches
            D loss: 0.5-0.7 (balanced), G loss: < 1

Ideal equilibrium: D accuracy ~50-70% (slightly better than random)
```

### The Alternating Update

```python
for batch in dataloader:
    photo, real_sketch = batch

    # ── Update Discriminator ──
    fake_sketch = generator(photo).detach()
    d_loss = discriminator_loss(photo, real_sketch, fake_sketch)
    d_loss.backward()
    d_optimizer.step()
    d_optimizer.zero_grad()

    # ── Update Generator ──
    fake_sketch = generator(photo)
    g_loss = generator_loss(photo, real_sketch, fake_sketch)
    g_loss.backward()
    g_optimizer.step()
    g_optimizer.zero_grad()
```

**Critical:** The discriminator must be updated FIRST in each iteration, and `fake_sketch` must be detached for the discriminator step. Getting this wrong silently breaks training.

### Common Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| D loss → 0, G loss stagnant | D is too strong | Lower D learning rate, add D noise |
| G produces identical outputs for different inputs | Mode collapse | Increase D capacity, add L1 weight |
| D loss oscillates wildly | Unstable training | Lower both LRs, add gradient clipping |
| Outputs are sharp but wrong person | λ_L1 too low | Increase λ_L1 (100 → 200) |
| Outputs are blurry, no improvement | λ_L1 too high / D too weak | Decrease λ_L1, increase D capacity |

### Learning Rates

```
Generator: 2e-4 (Adam, β1=0.5, β2=0.999)
Discriminator: 2e-4 (Adam, β1=0.5, β2=0.999)

β1=0.5 instead of default 0.9 is standard for GANs.
The lower momentum helps both networks adapt to each other
more quickly — they're in a dynamic equilibrium, not converging
to a fixed optimum.
```

---

## 🎨 5. Inference — Single Forward Pass

```python
def generate(photo, generator):
    """
    Unlike diffusion (1000 denoising steps), GAN inference is O(1):
    one forward pass through the generator.
    """
    generator.eval()
    with torch.no_grad():
        sketch = generator(photo)       # (B, 3, 256, 256) in [-1, 1]
        sketch = (sketch + 1) / 2       # [-1, 1] → [0, 1]
        sketch = sketch.clamp(0, 1)
    return sketch

# That's it. No loop. No noise schedule. No iterative refinement.
# This is the key advantage of GANs: speed.
```

---

## 📊 Comparison: GAN vs Diffusion (pix2pix style)

| | pix2pix GAN | Conditional DDPM |
|---|---|---|
| **Inference speed** | 1 forward pass (~10ms) | 50-1000 denoising steps (~1-40s) |
| **Training data needed** | 100-1000 pairs | 5000-50K+ pairs |
| **Output diversity** | Deterministic (1 input → 1 output) | Stochastic (1 input → many outputs) |
| **Training stability** | Unstable (adversarial game) | Very stable (MSE regression) |
| **Sharpness** | Excellent (adversarial loss) | Good (but can be slightly soft) |
| **Mode collapse** | Common problem | Does not occur |
| **Architecture complexity** | 2 networks, tricky loss | 1 network, simple MSE loss |
| **When to use** | Small paired datasets, fast inference needed | Large datasets, diversity important |

---

## 🔗 6. When You Upgrade to Diffusion (Phase 3)

The Phase 3 conditional DDPM will reuse:

1. **The same U-Net architecture** — with modifications: GroupNorm instead of BatchNorm, SiLU instead of ReLU, plus time embedding injection at every level
2. **The same input format** — concatenate(photo, noisy_sketch) → 6 channels
3. **The same data** — your 322 + 184 pairs

The differences:

| Component | GAN (Phase 1-2) | Diffusion (Phase 3) |
|---|---|---|
| U-Net input channels | 3 (photo only) | 3 + 3 = 6 (photo + noisy sketch) |
| U-Net extra inputs | None | Timestep t → sinusoidal embedding → injected at each level |
| Output | Sketch image (3 ch) | Noise prediction ε (3 ch) |
| Normalization | BatchNorm | GroupNorm (32 groups) |
| Activation | ReLU | SiLU (smoother for diffusion) |
| Training | Adversarial (D+G loss) | MSE (predict noise) |
| Inference | 1 forward pass | T-step iterative denoising |
