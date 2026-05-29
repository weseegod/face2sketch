# 📦 Datasets Guide

> What data you have, where it came from, and how it's structured.

---

## Data Inventory

| Directory | Pairs | Source | Task | Resolution | Style |
|-----------|-------|--------|------|------------|-------|
| `data/dataset/` | **322** | CUHK + SKSF-A | Photo → Realistic Sketch | 200–1024 px | Clean pencil sketch, academic |
| `data/finetune/` | **184** | TwitterPicasso | Photo → Funny Caricature | 169–2048 px | Exaggerated caricature, humorous |

---

## Phase 1: `data/dataset/` — CUHK + SKSF-A

### Source Breakdown

```
322 total pairs ──┬── 134 from SKSF-A  (Korean face-sketch dataset)
                  │      - 1024×1024 PNG, centered faces
                  │      - Professional artist, clean pencil lines
                  │
                  └── 188 from CUHK    (Chinese University of Hong Kong)
                         - ~200×250 JPG, centered faces
                         - Student sketches, varied quality
                         - Includes male (93), female (39), and subsets (56)
```

### What They Look Like

- **Photos:** Frontal face portraits, neutral expression, plain background
- **Sketches:** Hand-drawn pencil art, clean lines, good shading
- **Paired:** Each photo has exactly one matching sketch with the same filename

### Directory Structure

```
data/dataset/
├── photos/
│   ├── cuhk_f-005-01.jpg      # CUHK female student photo
│   ├── cuhk_m-001-01.jpg      # CUHK male student photo
│   ├── ...
│   ├── sksf_1.png             # SKSF-A photo
│   └── sksf_134.png
│
└── sketches/
    ├── cuhk_f-005-01.jpg      # Matching sketch (same filename!)
    ├── cuhk_m-001-01.jpg
    ├── ...
    ├── sksf_1.png
    └── sksf_134.png

Total: 322 files in each directory
```

### Why CUHK + SKSF-A?

- **CUHK Student Sketch Dataset:** One of the most cited face-sketch datasets. Widely used in academic benchmarks. Simple, clean, well-studied.
- **SKSF-A (Sungkyunkwan University Sketch Face Dataset-A):** Higher resolution (1024×1024), professional quality sketches. Complements CUHK's lower-res student-drawn sketches.
- Together they provide both quantity and quality diversity — the model learns to handle different resolutions, drawing styles, and face types.

### Known Issues

- **CUHK resolution is low** (200×250). Resize to 256×256 will upsample these — expect some blur.
- **Sketch quality varies** in CUHK (student-drawn, some are rough).
- **All faces are frontal** — model will struggle with profile/side-angle inputs.
- **All faces are Asian** (Chinese + Korean) — bias in training data, model may not generalize to other ethnicities well.

---

## Phase 2: `data/finetune/` — TwitterPicasso

### Source

TwitterPicasso was a Twitter bot that posted side-by-side photo + caricature drawings. The drawings are in a distinctive exaggerated style — big heads, emphasized features, humorous expression.

### Directory Structure

```
data/finetune/
├── photos/
│   ├── Tw1tterPicasso_20160903_011822_771879977591517184.jpg
│   ├── Tw1tterPicasso_20160903_023219_771898590453047296.jpg
│   └── ... (184 files)
│
└── sketches/
    ├── Tw1tterPicasso_20160903_011822_771879977591517184.jpg
    ├── Tw1tterPicasso_20160903_023219_771898590453047296.jpg
    └── ... (184 files)

Total: 184 pairs
```

### Resolution

Highly variable — photos range from 169×201 to 2048×2048, sketches from 480×818 to 1823×2048. All will be resized to 256×256 during training.

### Style Characteristics

- **Exaggerated features:** Big heads, large eyes, emphasized noses/mouths
- **Bold lines:** Thick outlines, strong contrast, not subtle shading
- **Colorful:** Often includes color, not just B&W
- **Humorous:** The goal — these are funny drawings, not serious portraits

### Why TwitterPicasso?

- **The style matches your goal:** "funny drawing" is literally what these are
- **Paired data is rare:** Finding photo↔caricature pairs is hard. Most caricature datasets are unpaired (just drawings, no source photo)
- **184 pairs is enough for fine-tuning** a pretrained model (transfer learning)

### Known Issues

- **Inconsistent quality:** Some caricatures are more detailed than others
- **Background variation:** Unlike CUHK/SKSF-A (plain backgrounds), TwitterPicasso has varied backgrounds
- **Not all are faces:** Some images might be full-body or group shots that got included — you may want to filter manually
- **Resolution inconsistency:** Very wide range requires careful resizing

---

## 🧹 Data Preprocessing Pipeline

### Current (in `data_loader.py`)

Your `FaceDataset` already handles:
- ✅ Paired loading by filename
- ✅ Format validation (PNG, JPG, etc.)
- ✅ Deterministic ordering (sorted)
- ✅ Shared random seed for paired augmentations (flip, rotation applied identically to both images)
- ✅ Mean/std normalization computed from actual data

### What You Might Add

```python
# 1. Face detection & alignment (for production use)
#    Use MTCNN or dlib to detect face, crop, and align
#    Ensures consistent face position across all inputs

# 2. Background removal (optional)
#    For TwitterPicasso — removes varied backgrounds
#    Use rembg or similar

# 3. Resolution standardization
#    Currently handled by transforms.Resize((256, 256))
#    Consider keeping aspect ratio with padding for better quality

# 4. Data cleaning script
#    Remove corrupted images, non-face images, duplicates
#    Use perceptual hash (phash) for duplicate detection
```

### Normalization Constants

```python
# Phase 1: CUHK + SKSF-A (322 pairs)
DATASET_MEAN = [0.7108, 0.7137, 0.6957]
DATASET_STD  = [0.2983, 0.2965, 0.3196]

# Phase 2: TwitterPicasso (184 pairs)
FINETUNE_MEAN = [0.6433, 0.5730, 0.5383]
FINETUNE_STD  = [0.2706, 0.2684, 0.2594]
```

Note: TwitterPicasso has lower mean values (darker images overall) and lower std (less variation) — consistent with cartoon drawings having more uniform colors vs photographs with varied lighting.

### Augmentation Strategy

Current augmentations (from `data_loader.py`):
```python
transforms.RandomHorizontalFlip()    # Faces are roughly symmetric
transforms.RandomRotation(15°)       # Small rotations for robustness
```

Good starting point. Consider adding for Phase 2:
```python
transforms.RandomAffine(0, translate=(0.1, 0.1))  # Small translations
transforms.ColorJitter(brightness=0.1, contrast=0.1)  # Photo only? Be careful
```

**Warning:** Color jitter on sketches would create unrealistic training pairs (sketches don't naturally vary in color). Consider applying color augmentations only to photos, or skipping them entirely.

---

## 📊 Data Budget Summary

| Phase | Dataset | Pairs | Resolution (resized) | Disk | RAM/batch (BS=16) |
|-------|---------|-------|---------------------|------|---------------------|
| 1 | CUHK + SKSF-A | 322 | 256×256 | ~50 MB | ~12 MB |
| 2 | TwitterPicasso | 184 | 256×256 | ~30 MB | ~12 MB |

Both fit easily in L4 24GB VRAM with room to spare.

---

## 🔮 If You Need More Data (Phase 3 Diffusion)

For Phase 3 Path B (DDPM from scratch), you'd need 5K-20K pairs. Options:

### Option 1: Synthetic Pairs (Fastest)
Apply sketch/edge-detection filters to FFHQ/CelebA faces:
```python
# 1. Download FFHQ (70K faces at 1024×1024)
# 2. Apply sketch filter (e.g., PhotoSketch, XDoG, or a pretrained model)
# 3. Pair: (original photo, filtered sketch)
# Result: 70K pairs instantly
```

Trade-off: synthetic sketches are less diverse than human-drawn ones, model may learn the filter's artifacts.

### Option 2: More Caricature Datasets
- **iCartoonFace** — 2K cartoon faces with identity labels
- **CartoonSet** — 10K avatar images
- **Danbooru portrait subset** — filtered for face portraits
- **Web scraping** — Pinterest, DeviantArt, Instagram caricature artists

### Option 3: Bootstrap with Existing Model
Train Phase 1-2 GAN, then use it to generate pseudo-pairs:
```python
# For each photo in CelebA/FFHQ:
#   sketch = gan(photo)
#   Save (photo, sketch) as a training pair
```
Self-supervised data augmentation. Quality depends on your GAN's quality.

---

## 🧪 Quick Data Sanity Checks

```bash
# Count pairs in each directory
echo "Phase 1 photos:" $(ls data/dataset/photos/ | wc -l)
echo "Phase 1 sketches:" $(ls data/dataset/sketches/ | wc -l)
echo "Phase 2 photos:" $(ls data/finetune/photos/ | wc -l)
echo "Phase 2 sketches:" $(ls data/finetune/sketches/ | wc -l)

# Verify every photo has a matching sketch
for f in data/dataset/photos/*; do
    name=$(basename "$f")
    if [ ! -f "data/dataset/sketches/$name" ]; then
        echo "MISSING: $name"
    fi
done
# (should print nothing — all paired)

# Check for corrupted images
python3 -c "
from PIL import Image
import os
for d in ['data/dataset/photos', 'data/dataset/sketches',
          'data/finetune/photos', 'data/finetune/sketches']:
    for f in os.listdir(d):
        try:
            img = Image.open(f'{d}/{f}')
            img.verify()
        except:
            print(f'CORRUPTED: {d}/{f}')
"
```
