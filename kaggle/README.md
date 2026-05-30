# Kaggle Dataset: face2sketch

## One-time setup: Upload dataset to Kaggle

### Step 1: Prepare files locally

```bash
cd /Users/thanhbm/Projects/face2sketch
rm -rf kaggle_dataset && mkdir kaggle_dataset

# Copy raw data folders (no zip needed)
cp -a data/dataset   kaggle_dataset/
cp -a data/finetune  kaggle_dataset/
cp -a data/test      kaggle_dataset/

# Copy Phase 1 checkpoint
cp checkpoints/pix2pix_best.pt kaggle_dataset/
```

You now have:
```
kaggle_dataset/
├── dataset/              (photos/ + sketches/ — 322 pairs)
├── finetune/             (photos/ + sketches/ — 184 pairs)
├── test/                 (photos/ + sketches/ — 100 pairs)
└── pix2pix_best.pt       (454MB — Phase 1 checkpoint)
```

### Step 2: Upload

1. Go to https://www.kaggle.com/datasets → **New Dataset**
2. Title: `face2sketch`
3. Drag the **`kaggle_dataset/` folder** (the whole thing)
4. Visibility: Private
5. Click **Create**

### Step 3: Add to notebook

1. Open `kaggle/kaggle_finetune.ipynb` on Kaggle
2. Right sidebar → **Input** → **Add Input** → search `face2sketch`
3. Enable **GPU** + **Internet**
4. Run cells 1→5
