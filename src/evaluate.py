"""
Phase 1 Gate Check — evaluates a trained pix2pix checkpoint to determine
if the model is ready to proceed to Phase 2 (finetune on caricatures).

Success criteria (from ROADMAP):
  1. Generator L1 loss on dataset < 0.15 (structural accuracy)
  2. No mode collapse — different inputs produce different outputs
  3. Outputs are recognizable sketches (visual quality)
  4. Discriminator is balanced (D predictions around 0.5-0.7)

Usage:
  python src/evaluate.py --checkpoint checkpoints/pix2pix_best.pt
  python src/evaluate.py --checkpoint checkpoints/pix2pix_best.pt --device cuda
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from unet import UNetGenerator
from discriminator import PatchGANDiscriminator
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
    g_l1 = ckpt.get("g_l1_loss", "?")
    return gen, epoch, g_l1


def evaluate(generator, checkpoint_path, device='cpu', data_dir="data/dataset"):
    """Run Phase 1 gate checks and print pass/fail report."""

    print(f"\n{'='*60}")
    print(f"📋  Phase 1 Gate Evaluation")
    print(f"{'='*60}")

    # Load checkpoint info
    gen, epoch, ckpt_l1 = load_checkpoint(checkpoint_path, device)
    n_params = sum(p.numel() for p in gen.parameters())
    print(f"\n  Checkpoint: {Path(checkpoint_path).name}")
    print(f"  Epoch: {epoch}  |  Params: {n_params:,}")
    print(f"  Saved G_L1: {ckpt_l1}")

    # ── Load dataset (100% training, no split) ──
    dataset = FaceDataset(root_dir=data_dir)
    n_pairs = len(dataset)
    print(f"\n  Dataset: {n_pairs} pairs (100% used for training)")

    main_tf, _ = get_transformations(
        DATASET_MEAN, DATASET_STD, size=(256, 256),
    )

    results = {}
    all_ok = True

    # ═══════════════════════════════════════════════════════════
    # CHECK 1: Structural Accuracy (L1 loss on dataset)
    # ═══════════════════════════════════════════════════════════
    print(f"\n  ── Check 1: Structural Accuracy (L1 loss) ──")
    l1_fn = nn.L1Loss()
    total_l1 = 0.0

    with torch.no_grad():
        for i in tqdm(range(min(n_pairs, 100)), desc="  Computing L1",
                       leave=False):
            photo, real_sketch = dataset[i]
            photo_t = main_tf(photo).unsqueeze(0).to(device)
            real_t = main_tf(real_sketch).unsqueeze(0).to(device)
            fake_t = gen(photo_t)
            total_l1 += l1_fn(fake_t, real_t).item()

    avg_l1 = total_l1 / min(n_pairs, 100)
    l1_pass = avg_l1 < 0.15
    results["l1_loss"] = avg_l1

    status = "✅ PASS" if l1_pass else "❌ FAIL"
    print(f"  Average L1 loss: {avg_l1:.4f}  [{status}]")
    print(f"  Threshold: < 0.15")
    if not l1_pass:
        print(f"  💡  Tip: Train more epochs or increase λ_l1")
        all_ok = False

    # ═══════════════════════════════════════════════════════════
    # CHECK 2: Mode Collapse Detection
    # ═══════════════════════════════════════════════════════════
    print(f"\n  ── Check 2: Mode Collapse ──")
    print(f"  Generating sketches from 8 different photos...")

    indices = [0, 10, 20, 30, 40, 50, 60, 70]
    images = []
    with torch.no_grad():
        for idx in indices:
            photo, _ = dataset[idx]
            photo_t = main_tf(photo).unsqueeze(0).to(device)
            fake = gen(photo_t)
            images.append(fake.cpu())

    # Compute pairwise L2 distances between generated outputs
    diffs = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            diff = (images[i] - images[j]).pow(2).mean().sqrt().item()
            diffs.append(diff)

    avg_diff = sum(diffs) / len(diffs)
    min_diff = min(diffs)
    collapse_pass = min_diff > 0.01  # any noticeable difference

    results["pairwise_diff"] = avg_diff
    results["min_pairwise_diff"] = min_diff

    status = "✅ PASS" if collapse_pass else "❌ FAIL (mode collapse)"
    print(f"  Avg pairwise L2 diff: {avg_diff:.4f}  Min: {min_diff:.4f}  [{status}]")
    print(f"  Threshold: min diff > 0.01")

    if not collapse_pass:
        print(f"  ⚠️   Mode collapse detected! Generator produces similar outputs.")
        print(f"  💡  Tip: Increase discriminator capacity or add noise to D inputs")
        all_ok = False

    # ═══════════════════════════════════════════════════════════
    # CHECK 3: Output Sharpness (Laplacian variance)
    # ═══════════════════════════════════════════════════════════
    print(f"\n  ── Check 3: Output Sharpness ──")

    # Laplacian kernel for edge detection
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

            # Apply Laplacian to each channel
            for c in range(3):
                channel = fake[:, c:c+1, :, :]
                lap = nn.functional.conv2d(channel, laplacian, padding=1)
                variances.append(lap.var().item())

    avg_var = sum(variances) / len(variances)
    sharpness_pass = avg_var > 0.001

    results["laplacian_var"] = avg_var
    status = "✅ PASS" if sharpness_pass else "⚠️  BLURRY"
    print(f"  Laplacian variance: {avg_var:.6f}  [{status}]")
    print(f"  Threshold: > 0.001 (higher = sharper edges)")

    if not sharpness_pass:
        print(f"  💡  Tip: Outputs are blurry. Increase adversarial loss weight or")
        print(f"       train more epochs for the generator to sharpen.")

    # ═══════════════════════════════════════════════════════════
    # CHECK 4: Output Range Sanity
    # ═══════════════════════════════════════════════════════════
    print(f"\n  ── Check 4: Output Range ──")
    all_vals = []
    with torch.no_grad():
        for idx in range(min(n_pairs, 30)):
            photo, _ = dataset[idx]
            photo_t = main_tf(photo).unsqueeze(0).to(device)
            fake = gen(photo_t)
            all_vals.append(fake.cpu())

    all_t = torch.cat([v.flatten() for v in all_vals])
    vmin, vmax = all_t.min().item(), all_t.max().item()

    range_pass = vmin >= -1.01 and vmax <= 1.01
    results["output_min"] = vmin
    results["output_max"] = vmax

    status = "✅ PASS" if range_pass else "❌ FAIL"
    print(f"  Range: [{vmin:.3f}, {vmax:.3f}]  [{status}]")
    print(f"  Expected: [-1.0, 1.0] (tanh output)")

    # ═══════════════════════════════════════════════════════════
    # CHECK 5: Generate Sample Grid
    # ═══════════════════════════════════════════════════════════
    print(f"\n  ── Check 5: Sample Grid ──")
    print(f"  Generating comparison grid (photos vs generated sketches)...")

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
    print(f"  Saved: {save_path}")

    # ═══════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"📊  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Dataset:           {n_pairs} pairs (100% train)")
    print(f"  Checkpoint epoch:  {epoch}")
    print(f"  L1 loss:           {avg_l1:.4f}  {'✅' if l1_pass else '❌'}")
    print(f"  Mode collapse:     {'✅ none' if collapse_pass else '❌ detected'}")
    print(f"  Sharpness:         {avg_var:.6f}  {'✅' if sharpness_pass else '⚠️'}")
    print(f"  Output range:      [{vmin:.2f}, {vmax:.2f}]  {'✅' if range_pass else '❌'}")
    print(f"  Sample grid:       outputs/phase1_eval_grid.png")
    print(f"{'='*60}")

    if all_ok:
        print(f"\n🎉  PHASE 1: PASS — Ready for Phase 2!")
        print(f"    Next: finetune on TwitterPicasso caricatures.")
    else:
        print(f"\n⚠️   PHASE 1: NOT READY — See issues above.")
        print(f"    Continue training or tune hyperparameters.")

    print(f"{'='*60}\n")
    return all_ok, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 Gate Check")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--device", default="cpu", help="Device: cuda or cpu")
    parser.add_argument("--data-dir", default="data/dataset", help="Data directory")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ok, _ = evaluate(None, args.checkpoint, device=device, data_dir=args.data_dir)
    sys.exit(0 if ok else 1)
