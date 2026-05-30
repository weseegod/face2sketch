# Kaggle Dataset: face2sketch

## One-time setup: Upload dataset to Kaggle

### Step 1: Prepare files locally

```bash
cd /Users/thanhbm/Projects/face2sketch
rm -rf kaggle_dataset && mkdir -p kaggle_dataset
cp data.zip kaggle_dataset/
cp checkpoints/pix2pix_best.pt kaggle_dataset/
```

You now have:
```
kaggle_dataset/
├── data.zip            (287MB)
└── pix2pix_best.pt     (454MB)
```

### Step 2: Upload to Kaggle

1. Go to https://www.kaggle.com/datasets → **New Dataset**
2. Title: `face2sketch`
3. **Drag the `kaggle_dataset/` folder** (not the files inside, the whole folder)
4. Visibility: Private
5. Click **Create**

⚠️  **Drag the FOLDER, not the files.** If you drag files individually, Kaggle won't nest them and the notebook's recursive search will still find them — but the folder upload is cleaner.

### Step 3: Add to notebook

1. Open `kaggle/kaggle_finetune.ipynb` on Kaggle
2. Right sidebar → **Input** → **Add Input** → search `face2sketch`
3. Enable **GPU** (Accelerator) + **Internet**
4. Run cells 1→5

---

## How the data flows

```
kaggle_dataset/              ──upload──►  Kaggle Dataset "face2sketch"
  data.zip                                 │
  pix2pix_best.pt                          ▼
                                    /kaggle/input/.../data.zip
                                    /kaggle/input/.../pix2pix_best.pt
                                           │
                                    cell 2: find + unzip + copy
                                           ▼
                                    /kaggle/working/face2sketch/
                                      data/dataset/   (322 pairs)
                                      data/finetune/  (184 pairs)  ← Phase 2
                                      data/test/      (100 pairs)
                                      checkpoints/pix2pix_best.pt
```

Cell 2 recursively searches all of `/kaggle/input/` so it works regardless of how Kaggle nests your dataset folder.
