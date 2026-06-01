"""
Phase 3: DDPM Training Loop — Conditional Diffusion from Scratch.

Trains a noise-predicting U-Net on (photo, sketch) pairs.
No discriminator needed — just MSE between predicted and actual noise.

Usage:
    # Phase 3: train from scratch on CUHK + SKSF-A
    python src/train_diffusion.py --device cuda --name phase3_v1_

    # With config overrides
    python src/train_diffusion.py --device cuda --epochs 500 --batch-size 4 --lr 2e-4
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure src/ is on path when running from repo root
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.optim as optim

from data_loader import get_dataloaders, MEAN, MEAN_STD, FINETUNE_MEAN, FINETUNE_STD
from diffusion import DiffusionModel

# ── Default config ──
CONFIG = {
    "T": 1000,
    "schedule": "cosine",
    "base_ch": 64,
    "time_dim": 256,
    "lr": 2e-4,
    "epochs": 500,
    "batch_size": 8,
    "grad_accum_steps": 1,
    "val_fraction": 0.10,
    "test_fraction": 0.05,
    "train_fraction": 1.0,
    "val_every": 25,
    "save_every": 100,
    "image_size": 256,
    "amp": False,   # disabled — float16 underflows small β values in diffusion
    "clip_grad": 1.0,
    "ema_decay": 0.9999,
    "num_workers": 2,
    "patience": 100,
    # DDIM sampling for validation visualization
    "sample_steps": 50,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3: DDPM training")
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    parser.add_argument("--grad-accum-steps", type=int, default=CONFIG["grad_accum_steps"])
    parser.add_argument("--val-every", type=int, default=CONFIG["val_every"])
    parser.add_argument("--patience", type=int, default=CONFIG["patience"],
                        help="Early stopping patience (0 = no early stop)")
    parser.add_argument("--name", default="phase3_",
                        help="Prefix for checkpoint filenames (e.g. 'phase3_v1_')")
    parser.add_argument("--resume", default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--finetune", default=None,
                        help="Load pretrained model for finetuning")
    return parser.parse_args()


def train_epoch(model, loader, optimizer, device, grad_accum):
    """One training epoch. Returns average loss."""
    model.model.train()
    total_loss = 0.0
    n_batches = 0

    optimizer.zero_grad()

    for batch_idx, (photos, sketches) in enumerate(loader):
        photos = photos.to(device)
        sketches = sketches.to(device)

        loss = model.training_loss(photos, sketches) / grad_accum

        # NaN guard — skip batch BEFORE backward
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n  ⚠️  NaN/Inf loss at batch {batch_idx} — skipping")
            continue

        loss.backward()

        total_loss += loss.item() * grad_accum
        n_batches += 1

        if (batch_idx + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), CONFIG["clip_grad"])
            optimizer.step()
            optimizer.zero_grad()

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, device):
    """Compute validation loss (MSE on noise prediction)."""
    model.model.eval()
    total_loss = 0.0
    n_batches = 0

    for photos, sketches in loader:
        photos = photos.to(device)
        sketches = sketches.to(device)
        loss = model.training_loss(photos, sketches)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def sample_validation(model, val_loader, device, sample_steps, output_dir):
    """Generate sketches for a few validation photos using DDIM."""
    model.model.eval()
    photos, _ = next(iter(val_loader))
    photos = photos[:8].to(device)

    generated = model.sample(photos, num_steps=sample_steps)

    # Save comparison grid: photos | generated sketches
    os.makedirs(output_dir, exist_ok=True)
    from torchvision.utils import save_image
    import torchvision.transforms.functional as TF

    # Un-normalize
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(MEAN_STD, device=device).view(1, 3, 1, 1)
    photos_vis = (photos * std + mean).clamp(0, 1)
    gen_vis = (generated * std + mean).clamp(0, 1)

    # Interleave: photo | generated
    grid = []
    for i in range(len(photos_vis)):
        grid.append(photos_vis[i])
        grid.append(gen_vis[i])

    from torchvision.utils import make_grid
    grid_img = make_grid(grid, nrow=4)
    save_image(grid_img, os.path.join(output_dir, "progress.png"))


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    """Load checkpoint with optional optimizer state."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.model.load_state_dict(ckpt["model"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_loss = ckpt.get("val_loss", float("inf"))
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return start_epoch, best_loss


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    """Save checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "config": CONFIG,
    }, path)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda"
                          else "cpu")

    print(f"╔══════════════════════════════════════╗")
    print(f"║  Phase 3: Conditional DDPM Training ║")
    print(f"╚══════════════════════════════════════╝")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"  T={CONFIG['T']}  |  schedule={CONFIG['schedule']}  |  base_ch={CONFIG['base_ch']}")
    print(f"  val_every={args.val_every}  |  patience={args.patience}")
    # ⚠️ AMP disabled for diffusion — float16 underflows small β values
    print(f"  AMP: False  |  Grad accum: {args.grad_accum_steps}")
    print()

    # ── Data ──
    root_dir = None
    mean, std = MEAN, MEAN_STD
    if args.finetune:
        root_dir = "data/finetune"
        mean, std = FINETUNE_MEAN, FINETUNE_STD
        print(f"🎯 Finetune mode — data: {root_dir}")

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=args.batch_size,
        val_fraction=CONFIG["val_fraction"],
        test_fraction=CONFIG["test_fraction"],
        train_fraction=CONFIG["train_fraction"],
        num_workers=CONFIG["num_workers"],
        root_dir=root_dir,
        mean=mean,
        std=std,
    )
    print(f"  Train: {len(train_loader.dataset)} pairs, {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader.dataset)} pairs, {len(val_loader)} batches")
    print()

    # ── Model ──
    model = DiffusionModel(T=CONFIG["T"], schedule=CONFIG["schedule"],
                           base_ch=CONFIG["base_ch"], time_dim=CONFIG["time_dim"],
                           device=device)

    # DataParallel for multi-GPU
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model.model = torch.nn.DataParallel(model.model)
        print(f"  🚀 DataParallel: {torch.cuda.device_count()} GPUs")
    print(f"  Params: {sum(p.numel() for p in model.model.parameters()) / 1e6:.1f}M")
    print()

    # ── Optimizer ──
    optimizer = optim.AdamW(model.model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # ── Resume or fresh start ──
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer, device)
        print(f"  🔄 Resumed from {args.resume} (epoch {start_epoch})")
    elif args.finetune:
        # Load only model weights (no optimizer)
        ckpt = torch.load(args.finetune, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt)
        # Handle DataParallel 'module.' prefix
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        # Allow partial match for finetuning
        model_state = model.model.state_dict()
        matched = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
        model_state.update(matched)
        # If DataParallel, need to load into .module
        if isinstance(model.model, torch.nn.DataParallel):
            model.model.module.load_state_dict(model_state)
        else:
            model.model.load_state_dict(model_state)
        print(f"  🎯 Loaded {len(matched)}/{len(model_state)} params from {args.finetune}")
        print()

    # ── Training ──
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    best_ckpt_path = f"checkpoints/{args.name}best.pt"
    final_ckpt_path = f"checkpoints/{args.name}final.pt"

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device,
                                 args.grad_accum_steps)

        epoch_time = time.time() - epoch_start

        # Log
        log_str = f"  Epoch {epoch+1:4d}/{args.epochs}  |  Train Loss: {train_loss:.6f}  |  Time: {epoch_time:.0f}s"

        # Validate
        if (epoch + 1) % args.val_every == 0 or epoch == 0:
            val_loss = validate(model, val_loader, device)
            log_str += f"  |  Val Loss: {val_loss:.6f}"

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(model, optimizer, epoch, val_loss, best_ckpt_path)
                log_str += "  ✅ BEST"
            else:
                patience_counter += 1
                log_str += f"  |  Patience: {patience_counter}/{args.patience}"

            # Generate samples for visual inspection
            sample_validation(model, val_loader, device,
                            CONFIG["sample_steps"], "outputs")

            # Early stopping
            if args.patience > 0 and patience_counter >= args.patience:
                print(log_str)
                print(f"  ⏹️  Early stop at epoch {epoch+1}")
                break

        print(log_str)

    # ── Final checkpoint ──
    final_val = validate(model, val_loader, device)
    save_checkpoint(model, optimizer, args.epochs - 1, final_val, final_ckpt_path)
    print(f"\n  ✅  Training complete  |  Best val loss: {best_val_loss:.6f}")
    print(f"  📦  Best:  {best_ckpt_path}")
    print(f"  📦  Final: {final_ckpt_path}")

    # ── Test evaluation ──
    print("\n  🧪  Evaluating on test set...")
    from evaluate import evaluate_diffusion
    evaluate_diffusion(model, "data/test", device, CONFIG["sample_steps"])


if __name__ == "__main__":
    main()
