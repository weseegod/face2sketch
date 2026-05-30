# Kaggle Dataset: face2sketch

## One-time setup: Upload dataset to Kaggle

### Step 1: Create the zip + gather files

```bash
cd /Users/thanhbm/Projects/face2sketch
rm -rf kaggle_dataset && mkdir kaggle_dataset

# Zip data folders FLAT (no data/ prefix — unzips cleanly)
cd data && zip -r -q ../kaggle_dataset/data.zip dataset/ finetune/ test/ && cd ..

# Copy checkpoint
cp checkpoints/pix2pix_best.pt kaggle_dataset/
```

`kaggle_dataset/` now contains:
```
kaggle_dataset/
├── data.zip            (287MB — dataset/ finetune/ test/ at root)
└── pix2pix_best.pt     (454MB — Phase 1 checkpoint)
```

### Step 2: Upload

1. https://www.kaggle.com/datasets → **New Dataset**
2. Title: `face2sketch`
3. Drag `kaggle_dataset/` folder
4. Private → **Create**

### Step 3: Use in notebook

1. Open `kaggle/kaggle_finetune.ipynb` on Kaggle
2. Sidebar → Input → Add Input → `face2sketch`
3. Enable GPU + Internet
4. Run cells 1→5
