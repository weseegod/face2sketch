"""
Phase 1 Gate Check — evaluate a trained pix2pix checkpoint on unseen test data.

Two modes:
  quick  → 10 random test photos, generate, save grid (visual check)
  full   → all test photos, compare against ground-truth sketches (L1 metric)

The test set (data/test/) is NEVER used in training — L1 here is genuine generalization.

Usage:
  python src/evaluate.py --checkpoint checkpoints/pix2pix_best.pt --mode quick
  python src/evaluate.py --checkpoint checkpoints/pix2pix_best.pt --mode full
"""

import argparse
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from unet import UNetGenerator
from data_loader import DATASET_MEAN, DATASET_STD
from sample import postprocess_tensor


def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("generator", ckpt)

    cfg = ckpt.get("config", {})
    gen = UNetGenerator(
        in_channels=cfg.get("in_channels", 3),
        out_channels=cfg.get("out_channels", 3),
        ngf=cfg.get("ngf", 64),
        num_levels=cfg.get("num_levels", 5),
        use_dropout=cfg.get("use_dropout", True),
        dropout=cfg.get("dropout", 0.5),
    ).to(device)

    # Strip DataParallel 'module.' prefix if checkpoint was saved with it
    if any(k.startswith('module.') for k in state.keys()):
        state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}

    gen.load_state_dict(state)
    gen.eval()

    epoch = ckpt.get("epoch", "?")
    train_l1 = ckpt.get("g_l1_loss", "?")
    return gen, epoch, train_l1


def preprocess(img_path, size=(256, 256)):
    img = Image.open(img_path).convert('RGB')
    tfs = T.Compose([T.Resize(size), T.ToTensor(),
                     T.Normalize(DATASET_MEAN, DATASET_STD)])
    return tfs(img).unsqueeze(0)


def get_test_pairs(test_dir):
    """Pair test photos with test sketches by numeric suffix.
    photo: imageXXXX.jpg  →  sketch: sketchXXXX.jpg
    If no sketch exists, it's still included (for quick mode).
    """
    photo_dir = Path(test_dir) / "photos"
    sketch_dir = Path(test_dir) / "sketches"

    if not photo_dir.exists():
        return []

    pairs = []
    for f in sorted(photo_dir.iterdir()):
        if not f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp', '.bmp'):
            continue
        # Match: imageXXXX → sketchXXXX
        base = f.stem  # e.g. "image0001"
        if base.startswith("image"):
            num = base[len("image"):]  # "0001"
            sketch_path = sketch_dir / f"sketch{num}{f.suffix}"
        elif base.startswith("sketch"):
            continue  # skip sketches in photos dir by accident
        else:
            sketch_path = sketch_dir / f.name

        pairs.append({
            "photo": str(f),
            "sketch": str(sketch_path) if sketch_path.exists() else None,
            "name": f.stem,
        })

    return pairs


# ═══════════════════════════════════════════════════════════════
#  MODE 1: QUICK — 10 random photos, visual check
# ═══════════════════════════════════════════════════════════════

def evaluate_quick(gen, test_pairs, device, num_samples=10):
    print(f"\n  {'='*56}")
    print(f"  👁️  QUICK CHECK — {num_samples} random test photos")
    print(f"  {'='*56}")

    n = min(num_samples, len(test_pairs))
    samples = random.sample(test_pairs, n)

    photos_list, fakes_list = [], []
    for s in tqdm(samples, desc="  Generating", leave=False):
        photo_t = preprocess(s["photo"]).to(device)
        with torch.no_grad():
            fake = gen(photo_t)
        photos_list.append(postprocess_tensor(photo_t.cpu()))
        fakes_list.append(postprocess_tensor(fake.cpu()))

    # Grid: top = photos, bottom = generated
    photos_grid = make_grid(torch.cat(photos_list), nrow=n)
    fakes_grid = make_grid(torch.cat(fakes_list), nrow=n)
    combined = torch.cat([photos_grid, fakes_grid], dim=1)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "phase1_quick_eval.png"
    save_image(combined, path)

    print(f"\n  📸  Saved: {path}")
    print(f"      Top row  = {n} test photos (unseen)")
    print(f"      Bot row  = generated sketches")
    print(f"\n  ❓  Do the generated sketches look like recognizable")
    print(f"     drawings of the faces above?")
    print(f"     YES → good  |  NO → needs more training")


# ═══════════════════════════════════════════════════════════════
#  MODE 2: FULL — all test photos vs ground-truth sketches
# ═══════════════════════════════════════════════════════════════

def evaluate_full(gen, test_pairs, device):
    print(f"\n  {'='*56}")
    print(f"  📏  FULL EVALUATION — all test pairs (photo → sketch)")
    print(f"  {'='*56}")

    # Only pairs that have a ground-truth sketch
    paired = [p for p in test_pairs if p["sketch"] is not None]
    if not paired:
        print(f"  ❌  No paired (photo, sketch) found in data/test/.")
        print(f"      Expected: photos/imageXXXX.jpg + sketches/sketchXXXX.jpg")
        return

    print(f"  Paired test samples: {len(paired)}")
    l1_fn = nn.L1Loss()

    total_l1 = 0.0
    fakes_for_grid = []
    reals_for_grid = []

    with torch.no_grad():
        for p in tqdm(paired, desc="  Evaluating", leave=False):
            photo_t = preprocess(p["photo"]).to(device)
            sketch_t = preprocess(p["sketch"]).to(device)
            fake = gen(photo_t)
            total_l1 += l1_fn(fake, sketch_t).item()

            # Save first 8 for grid
            if len(fakes_for_grid) < 8:
                fakes_for_grid.append(postprocess_tensor(fake.cpu()))
                reals_for_grid.append(postprocess_tensor(sketch_t.cpu()))

    avg_l1 = total_l1 / len(paired)

    # Grid: generated vs ground-truth
    fakes_grid = make_grid(torch.cat(fakes_for_grid), nrow=8)
    reals_grid = make_grid(torch.cat(reals_for_grid), nrow=8)
    combined = torch.cat([fakes_grid, reals_grid], dim=1)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "phase1_full_eval.png"
    save_image(combined, path)

    # Threshold (from ROADMAP: L1 < 0.1 on validation)
    l1_pass = avg_l1 < 0.15
    status = "✅ GOOD" if l1_pass else "❌ NEEDS WORK"

    print(f"\n  📊  RESULTS (on unseen test set)")
    print(f"      Test pairs:      {len(paired)}")
    print(f"      Avg L1 loss:     {avg_l1:.4f}  [{status}]")
    print(f"      Threshold:       < 0.15 (from pix2pix paper)")
    print(f"      Comparison grid: {path}")
    print(f"         Top row = generated  |  Bot row = ground-truth")
    print(f"")

    if not l1_pass:
        print(f"  💡  L1 > 0.15 — model hasn't converged yet. Tips:")
        print(f"      - Train more epochs (200 → 300)")
        print(f"      - Increase λ_l1 (100 → 200)")
        print(f"      - Check if G_L1 was still decreasing at final epoch")
    else:
        print(f"  🎉  L1 < 0.15 on unseen data — ready for Phase 2!")

    return avg_l1


# ═══════════════════════════════════════════════════════════════
#  MODE 3: COMPARE — up to 4 checkpoints side-by-side
# ═══════════════════════════════════════════════════════════════

def evaluate_compare(checkpoints, test_pairs, device, num_samples=8):
    """Generate side-by-side: photo | ckpt1 output | ckpt2 output | ..."""
    models = []
    names = []
    for path in checkpoints:
        gen, ep, l1 = load_checkpoint(path, device)
        models.append(gen)
        names.append(f"{Path(path).stem}")
        print(f"  [{len(models)}] {Path(path).name} (epoch {ep}, G_L1={l1:.4f})")

    print(f"\n  {'='*56}")
    print(f"  ⚖️  COMPARISON — {len(models)} checkpoints")
    print(f"  {'='*56}")

    n = min(num_samples, len(test_pairs))
    samples = random.sample(test_pairs, n)

    photos_list = []
    fake_lists = [[] for _ in models]

    with torch.no_grad():
        for s in tqdm(samples, desc="  Generating", leave=False):
            photo_t = preprocess(s["photo"]).to(device)
            photos_list.append(postprocess_tensor(photo_t.cpu()))
            for i, model in enumerate(models):
                fake = model(photo_t)
                fake_lists[i].append(postprocess_tensor(fake.cpu()))

    grid_photos = make_grid(torch.cat(photos_list), nrow=n)
    grids = [grid_photos] + [make_grid(torch.cat(fl), nrow=n) for fl in fake_lists]
    combined = torch.cat(grids, dim=1)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "phase1_vs_phase2.png"
    save_image(combined, path)

    print(f"\n  📸  Saved: {path}")
    print(f"      Row 1 = {n} test photos")
    for i, name in enumerate(names):
        print(f"      Row {i+2} = {name}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1/2 Evaluation")
    parser.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint")
    parser.add_argument("--checkpoint2", default=None,
                        help="Second checkpoint (for compare mode)")
    parser.add_argument("--checkpoints", nargs="+", default=None,
                        help="Multiple checkpoints for compare mode (v3 v4 v5)")
    parser.add_argument("--mode", choices=["quick", "full", "compare"], default="quick",
                        help="quick=10 random | full=all+L1 | compare=Phase 1 vs Phase 2")
    parser.add_argument("--device", default="cpu", help="Device: cuda or cpu")
    parser.add_argument("--test-dir", default="data/test",
                        help="Test data directory")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Compare mode: needs at least 2 checkpoints (--checkpoints or --checkpoint + --checkpoint2)
    if args.mode == "compare":
        ckpts = args.checkpoints or []
        if args.checkpoint:
            ckpts.insert(0, args.checkpoint)
        if args.checkpoint2:
            ckpts.append(args.checkpoint2)
        if len(ckpts) < 2:
            print("❌  Compare mode needs at least 2 checkpoints")
            print("    Use: --checkpoints ckpt1.pt ckpt2.pt ckpt3.pt")
            sys.exit(1)
        print(f"\n{'='*60}")
        print(f"📋  Multi-Checkpoint Comparison ({len(ckpts)} models)")
        print(f"{'='*60}")
        print(f"  Device: {device}")
        test_pairs = get_test_pairs(args.test_dir)
        if not test_pairs:
            print(f"\n  ❌  No test photos found in {args.test_dir}/photos/")
            sys.exit(1)
        evaluate_compare(ckpts, test_pairs, device)
        sys.exit(0)

    if not args.checkpoint:
        print("❌  --checkpoint is required for quick/full mode")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"📋  Phase 1 Evaluation")
    print(f"{'='*60}")
    print(f"  Device: {device}")

    # Load model
    gen, epoch, train_l1 = load_checkpoint(args.checkpoint, device)
    n_params = sum(p.numel() for p in gen.parameters())
    print(f"  Checkpoint:  {Path(args.checkpoint).name}")
    print(f"  Epoch:       {epoch}")
    print(f"  Params:      {n_params:,}")
    print(f"  Train G_L1:  {train_l1}")

    # Load test pairs
    test_pairs = get_test_pairs(args.test_dir)
    paired = sum(1 for p in test_pairs if p["sketch"] is not None)
    print(f"  Test data:   {len(test_pairs)} photos, {paired} have sketches")

    if not test_pairs:
        print(f"\n  ❌  No test photos found in {args.test_dir}/photos/")
        sys.exit(1)

    if args.mode == "quick":
        evaluate_quick(gen, test_pairs, device)
    else:
        evaluate_full(gen, test_pairs, device)

    print(f"{'='*60}\n")
