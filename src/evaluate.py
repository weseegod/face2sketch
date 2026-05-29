"""
Phase 1 Gate Check — qualitative evaluation of a trained pix2pix checkpoint.

Success criteria (from ROADMAP):
  1. No mode collapse — different inputs produce different outputs
  2. Outputs are sharp (not blurry averages)
  3. 👁️  VISUAL: generated sketches look like recognizable face drawings

Since we use 100% data for training (no val split), the primary metric is
visual quality. Look at outputs/phase1_eval_grid.png — top row is input
photos, bottom row is generated sketches.

Usage:
  python src/evaluate.py --checkpoint checkpoints/pix2pix_best.pt
  python src/evaluate.py --checkpoint checkpoints/pix2pix_best.pt --device cuda
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from unet import UNetGenerator
from data_loader import FaceDataset, get_transformations, DATASET_MEAN, DATASET_STD
from sample import postprocess_tensor


def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
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
    gen.load_state_dict(state)
    gen.eval()

    epoch = ckpt.get("epoch", "?")
    train_l1 = ckpt.get("g_l1_loss", "?")
    return gen, epoch, train_l1


def evaluate(generator, checkpoint_path, device='cpu', data_dir="data/dataset"):
    """Run Phase 1 gate checks. Qualitative — look at the grid."""

    print(f"\n{'='*60}")
    print(f"📋  Phase 1 Gate Evaluation")
    print(f"{'='*60}")

    gen, epoch, train_l1 = load_checkpoint(checkpoint_path, device)
    n_params = sum(p.numel() for p in gen.parameters())
    print(f"\n  Checkpoint:  {Path(checkpoint_path).name}")
    print(f"  Epoch:       {epoch}")
    print(f"  Params:      {n_params:,}")
    print(f"  Train G_L1:  {train_l1}")

    # ── Load dataset ──
    dataset = FaceDataset(root_dir=data_dir)
    n_pairs = len(dataset)
    print(f"  Dataset:     {n_pairs} pairs (100% train)")

    main_tf, _ = get_transformations(
        DATASET_MEAN, DATASET_STD, size=(256, 256),
    )

    all_ok = True

    # ═══════════════════════════════════════════════════════════
    # CHECK 1: Mode Collapse
    # ═══════════════════════════════════════════════════════════
    print(f"\n  ── Check 1: Mode Collapse ──")

    indices = [0, 10, 20, 30, 40, 50, 60, 70]
    images = []
    with torch.no_grad():
        for idx in indices:
            photo, _ = dataset[idx]
            photo_t = main_tf(photo).unsqueeze(0).to(device)
            fake = gen(photo_t)
            images.append(fake.cpu())

    diffs = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            diff = (images[i] - images[j]).pow(2).mean().sqrt().item()
            diffs.append(diff)

    avg_diff = sum(diffs) / len(diffs)
    min_diff = min(diffs)
    collapse_ok = min_diff > 0.01

    status = "✅" if collapse_ok else "❌ MODE COLLAPSE"
    print(f"  Pairwise L2 diff: avg={avg_diff:.4f}  min={min_diff:.4f}  [{status}]")
    if not collapse_ok:
        print(f"  ⚠️  All outputs look the same — generator collapsed.")
        all_ok = False

    # ═══════════════════════════════════════════════════════════
    # CHECK 2: Output Sharpness (Laplacian variance)
    # ═══════════════════════════════════════════════════════════
    print(f"\n  ── Check 2: Sharpness ──")

    laplacian = torch.tensor([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ], dtype=torch.float32).view(1, 1, 3, 3).to(device)

    variances = []
    with torch.no_grad():
        for idx in tqdm(range(min(n_pairs, 30)), desc="  Computing sharpness",
                         leave=False):
            photo, _ = dataset[idx]
            photo_t = main_tf(photo).unsqueeze(0).to(device)
            fake = gen(photo_t)
            for c in range(3):
                channel = fake[:, c:c+1, :, :]
                lap = F.conv2d(channel, laplacian, padding=1)
                variances.append(lap.var().item())

    avg_var = sum(variances) / len(variances)
    sharp_ok = avg_var > 0.001

    status = "✅ sharp" if sharp_ok else "⚠️  blurry"
    print(f"  Laplacian variance: {avg_var:.6f}  [{status}]")
    if not sharp_ok:
        print(f"  💡  Outputs are blurry — need more adversarial training.")

    # ═══════════════════════════════════════════════════════════
    # CHECK 3: Output Range
    # ═══════════════════════════════════════════════════════════
    print(f"\n  ── Check 3: Output Range ──")
    with torch.no_grad():
        photo, _ = dataset[0]
        photo_t = main_tf(photo).unsqueeze(0).to(device)
        fake = gen(photo_t)
    vmin, vmax = fake.min().item(), fake.max().item()
    range_ok = vmin > -1.01 and vmax < 1.01

    status = "✅" if range_ok else "❌"
    print(f"  Range: [{vmin:.3f}, {vmax:.3f}]  [{status}]")

    # ═══════════════════════════════════════════════════════════
    # 👁️  PRIMARY: Sample Grid
    # ═══════════════════════════════════════════════════════════
    print(f"\n  ── 👁️  PRIMARY: Sample Grid ──")

    num_show = 8
    photos_list, fakes_list = [], []
    with torch.no_grad():
        for idx in range(min(n_pairs, num_show)):
            photo, _ = dataset[idx]
            photo_t = main_tf(photo).unsqueeze(0).to(device)
            fake = gen(photo_t)
            photos_list.append(postprocess_tensor(photo_t.cpu()))
            fakes_list.append(postprocess_tensor(fake.cpu()))

    photos_grid = make_grid(torch.cat(photos_list), nrow=num_show)
    fakes_grid = make_grid(torch.cat(fakes_list), nrow=num_show)
    combined = torch.cat([photos_grid, fakes_grid], dim=1)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    save_path = out_dir / "phase1_eval_grid.png"
    save_image(combined, save_path)

    # ═══════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"📊  VERDICT")
    print(f"{'='*60}")
    print(f"  Checkpoint:      {Path(checkpoint_path).name} (epoch {epoch})")
    print(f"  Mode collapse:   {'✅ none' if collapse_ok else '❌'}")
    print(f"  Sharpness:       {'✅ good' if sharp_ok else '⚠️  blurry'}")
    print(f"  Output range:    {'✅' if range_ok else '❌'}")
    print(f"")
    print(f"  👁️  Sample grid:  {save_path}")
    print(f"     Top row = input photos, bottom row = generated sketches")
    print(f"")
    print(f"  ❓  Do the generated sketches look like recognizable")
    print(f"     pencil drawings of the faces above?")
    print(f"")
    print(f"     YES → 🎉  PHASE 1 PASS — proceed to Phase 2")
    print(f"     NO  → ⚠️   Continue training, tune hyperparams")
    print(f"{'='*60}\n")

    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 Gate Check")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--device", default="cpu", help="Device: cuda or cpu")
    parser.add_argument("--data-dir", default="data/dataset", help="Data directory")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ok = evaluate(None, args.checkpoint, device=device, data_dir=args.data_dir)
    sys.exit(0 if ok else 1)
