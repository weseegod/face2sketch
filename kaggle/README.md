# Kaggle Dataset: face2sketch

## How to Upload

### 1. Create the dataset archive

On your local machine:
```bash
cd /Users/thanhbm/Projects/face2sketch
mkdir -p kaggle_dataset
cp data.zip kaggle_dataset/
cp checkpoints/pix2pix_best.pt kaggle_dataset/
```

### 2. Upload to Kaggle

Go to: https://www.kaggle.com/datasets → New Dataset

**Option A — Upload via web UI:**
- Title: face2sketch
- Drag `kaggle_dataset/` folder (contains data.zip + pix2pix_best.pt)
- Set visibility to Private
- Create

**Option B — Upload via Kaggle API:**
```bash
pip install kaggle
# Get API key from https://www.kaggle.com/settings/account
# Place kaggle.json at ~/.kaggle/

cd /Users/thanhbm/Projects/face2sketch
kaggle datasets create -p kaggle_dataset --dir-mode zip
```

### 3. Use in Kaggle Notebook

- Open `kaggle/kaggle_finetune.ipynb` on Kaggle
- Add the face2sketch dataset as input: Notebook → Input → Add Input → face2sketch
- Enable GPU + Internet
- Run cells in order

### Dataset contents

```
kaggle_dataset/
├── data.zip              # 287MB — data/dataset/ + data/finetune/ + data/test/
└── pix2pix_best.pt       # 454MB — Phase 1 pretrained generator
```

### Output

After training, download from Kaggle Output tab:
- `checkpoints/phase2_best.pt`
- `samples/epoch_*.png`
- `outputs/phase1_vs_phase2.png`
