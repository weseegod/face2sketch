"""
Phase 1: pix2pix GAN Training Script.

Modes:
  test  → small image, few epochs  (local smoke test)
  train → full config               (Colab / GPU training)

Usage:
  python src/train.py --mode test
  python src/train.py --mode train --config configs/pix2pix_phase1.yaml
  python src/train.py --resume checkpoints/pix2pix_epoch_040.pt
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from unet import UNetGenerator
from discriminator import PatchGANDiscriminator
from data_loader import FaceDataset, get_dataloaders, get_transformations
from data_loader import DATASET_MEAN, DATASET_STD, FINETUNE_MEAN, FINETUNE_STD
from sample import save_sample_grid
from torchvision import transforms as T


ROOT = Path(__file__).parent.parent


# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    "mode": "train",

    # Model
    "image_size": 256,
    "in_channels": 3,
    "out_channels": 3,
    "ngf": 64, "num_levels": 5, "use_dropout": True, "dropout": 0.5,
    "ndf": 32, "n_layers": 3,

    # Data
    "data_dir": "data/dataset",
    "val_fraction": 0.0,
    "test_fraction": 0.0,
    "num_workers": 2,
    "mean": DATASET_MEAN,
    "std": DATASET_STD,

    # Modes
    "test": {
        "max_epochs": 5, "batch_size": 8, "image_size": 128,
        "ngf": 32, "num_levels": 4, "ndf": 32,
        "sample_interval": 1, "save_interval": 5,
        "num_val_samples": 4, "log_interval": 10,
        "patience": 0,
    },
    "train": {
        "max_epochs": 200, "batch_size": 16, "image_size": 256,
        "ngf": 64, "num_levels": 5, "ndf": 32,
        "sample_interval": 10, "save_interval": 20,
        "num_val_samples": 8, "log_interval": 50,
        "patience": 20,
    },
    "finetune": {
        "max_epochs": 100, "batch_size": 16, "image_size": 256,
        "ngf": 64, "num_levels": 5, "ndf": 32,
        "sample_interval": 10, "save_interval": 20,
        "num_val_samples": 8, "log_interval": 50,
        "patience": 20,
    },

    # Optimizer (D gets lower LR to prevent overpowering G)
    "g_lr": 2e-4, "d_lr": 1e-4,

    # Finetune overrides (active when mode=finetune or --finetune flag)
    "finetune_lr": 5e-5,

    "adam_beta1": 0.5, "adam_beta2": 0.999,

    # Loss
    "lambda_l1": 100, "lambda_adv": 1.0,
    "label_smoothing_real": 0.9, "label_smoothing_fake": 0.0,
    "grad_clip": 1.0,

    # Discriminator stabilization (prevents D from dying on small datasets)
    "d_spectral_norm": True,   # bound D's Lipschitz constant
    "d_noise_std": 0.1,        # Gaussian noise injected to D inputs

    # Gradient accumulation (1 = no accumulation, 2+ = accumulate N steps)
    "grad_accum": 1,

    # Paths
    "checkpoint_dir": "checkpoints",
    "sample_dir": "samples",

    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# ═══════════════════════════════════════════════════════════════
#  MODEL CREATION
# ═══════════════════════════════════════════════════════════════

def create_models(cfg, device):
    gen = UNetGenerator(
        in_channels=cfg["in_channels"], out_channels=cfg["out_channels"],
        ngf=cfg["ngf"], num_levels=cfg["num_levels"],
        use_dropout=cfg["use_dropout"], dropout=cfg["dropout"],
    ).to(device)

    disc = PatchGANDiscriminator(
        in_channels=cfg["in_channels"] + cfg["out_channels"],
        ndf=cfg["ndf"], n_layers=cfg["n_layers"],
        use_spectral_norm=cfg.get("d_spectral_norm", True),
        noise_std=cfg.get("d_noise_std", 0.05),
    ).to(device)

    return gen, disc


def wrap_models(gen, disc, device):
    """Wrap in DataParallel if multiple GPUs available."""
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        n = torch.cuda.device_count()
        gen = nn.DataParallel(gen)
        disc = nn.DataParallel(disc)
        print(f"    🖥️   DataParallel: {n} GPUs")
    return gen, disc


# ═══════════════════════════════════════════════════════════════
#  CHECKPOINT
# ═══════════════════════════════════════════════════════════════

def save_ckpt(gen, disc, g_opt, d_opt, epoch, loss, cfg, fname):
    d = ROOT / cfg["checkpoint_dir"]; d.mkdir(exist_ok=True)
    prefix = cfg.get("ckpt_prefix", "")
    full_name = f"{prefix}{fname}" if prefix else fname
    torch.save({
        "epoch": epoch,
        "generator": gen.state_dict(),
        "discriminator": disc.state_dict(),
        "g_optimizer": g_opt.state_dict(),
        "d_optimizer": d_opt.state_dict(),
        "g_l1_loss": loss,
        "config": {
            "image_size": cfg["image_size"], "in_channels": cfg["in_channels"],
            "out_channels": cfg["out_channels"], "ngf": cfg["ngf"],
            "num_levels": cfg["num_levels"], "dropout": cfg["dropout"],
            "ndf": cfg["ndf"], "n_layers": cfg["n_layers"],
        },
        "timestamp": datetime.now().isoformat(),
    }, d / full_name)
    return d / full_name


def load_pretrained_gen(path, gen, device):
    """Load only generator weights from a checkpoint (for finetuning)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["generator"]
    # Handle DataParallel: strip/add 'module.' prefix as needed
    model_is_dp = isinstance(gen, nn.DataParallel)
    state_is_dp = any(k.startswith('module.') for k in state.keys())
    if model_is_dp and not state_is_dp:
        state = {'module.' + k: v for k, v in state.items()}
    elif not model_is_dp and state_is_dp:
        state = {k[7:]: v for k, v in state.items()}
    gen.load_state_dict(state)
    print(f"📂  Loaded pretrained G from {path} (epoch {ckpt['epoch']})")
    print(f"    D and optimizers will start fresh (new style, new dataset)")
    return ckpt["epoch"]


def load_ckpt(path, gen, disc, g_opt, d_opt, device):
    """Resume training — load G, D, and both optimizers."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    for model, key in [(gen, "generator"), (disc, "discriminator")]:
        state = ckpt[key]
        model_is_dp = isinstance(model, nn.DataParallel)
        state_is_dp = any(k.startswith('module.') for k in state.keys())
        if model_is_dp and not state_is_dp:
            state = {'module.' + k: v for k, v in state.items()}
        elif not model_is_dp and state_is_dp:
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state)
    g_opt.load_state_dict(ckpt["g_optimizer"])
    d_opt.load_state_dict(ckpt["d_optimizer"])
    print(f"📂  Resumed from {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"]


def get_finetune_augmentation(mean, std, size=(256, 256)):
    """Phase 2: more aggressive augmentation for style transfer."""
    return T.Compose([
        T.Resize(size),
        T.RandomHorizontalFlip(),
        T.RandomRotation(degrees=30),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05),
                       scale=(0.95, 1.05)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


# ═══════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def train_one_epoch(gen, disc, dataloader, g_opt, d_opt, cfg, epoch, device,
                     use_adversarial=True, grad_accum=1):
    """One training epoch with gradient accumulation.
    grad_accum: number of batches to accumulate before optimizer step.
      effective_batch = batch_size * grad_accum * num_gpus"""
    gen.train(); disc.train()

    bce = nn.BCELoss()
    l1 = nn.L1Loss()
    real_label = torch.tensor([cfg["label_smoothing_real"]], device=device)
    fake_label = torch.tensor([cfg["label_smoothing_fake"]], device=device)
    lambda_l1 = cfg["lambda_l1"]
    lambda_adv = cfg["lambda_adv"]
    grad_clip = cfg["grad_clip"]

    m = {"d_loss": 0.0, "g_adv": 0.0, "g_l1": 0.0, "d_real": 0.0, "d_fake": 0.0}
    n = len(dataloader)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:3d}", leave=False, unit="batch")
    for batch_idx, (photo, real_sketch) in enumerate(pbar):
        photo = photo.to(device)
        real_sketch = real_sketch.to(device)

        d_real = d_fake = d_loss = torch.tensor(0.0, device=device)

        # ── Step 1: Update Discriminator (skip during L1 warmup) ──
        if use_adversarial:
            with torch.no_grad():
                fake_sketch = gen(photo)
            d_real = disc(photo, real_sketch)
            d_fake = disc(photo, fake_sketch)
            d_loss = (bce(d_real, real_label.expand_as(d_real)) +
                      bce(d_fake, fake_label.expand_as(d_fake))) * 0.5 / grad_accum
            d_loss.backward()
            if (batch_idx + 1) % grad_accum == 0:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(disc.parameters(), grad_clip)
                d_opt.step()
                d_opt.zero_grad()

        # ── Step 2: Update Generator ──
        fake_sketch = gen(photo)
        g_l1 = l1(fake_sketch, real_sketch)
        if use_adversarial:
            d_fake_g = disc(photo, fake_sketch)
            g_adv = bce(d_fake_g, real_label.expand_as(d_fake_g))
            g_loss = (g_adv * lambda_adv + g_l1 * lambda_l1) / grad_accum
        else:
            g_adv = torch.tensor(0.0, device=device)
            g_loss = (g_l1 * lambda_l1) / grad_accum
        g_loss.backward()
        if (batch_idx + 1) % grad_accum == 0:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(gen.parameters(), grad_clip)
            g_opt.step()
            g_opt.zero_grad()

        # Track (un-scale for clean logging)
        m["d_loss"] += d_loss.item() * grad_accum if use_adversarial else 0.0
        m["g_adv"] += g_adv.item() if use_adversarial else 0.0
        m["g_l1"] += g_l1.item()
        m["d_real"] += d_real.mean().item() if use_adversarial else 0.5
        m["d_fake"] += d_fake.mean().item() if use_adversarial else 0.5

        if batch_idx % cfg["log_interval"] == 0:
            pbar.set_postfix({
                "D": f"{d_loss.item() * grad_accum:.3f}",
                "G_adv": f"{g_adv.item():.3f}",
                "G_L1": f"{g_l1.item():.3f}",
                "Dr": f"{d_real.mean():.2f}",
                "Df": f"{d_fake.mean():.2f}",
            })

    for k in m: m[k] /= n
    return m


@torch.no_grad()
def evaluate(gen, loader, device, n_batches=5):
    gen.eval()
    total = 0.0; cnt = 0
    l1 = nn.L1Loss()
    for photo, real_sketch in loader:
        if cnt >= n_batches: break
        photo = photo.to(device); real_sketch = real_sketch.to(device)
        fake = gen(photo)
        total += l1(fake, real_sketch).item()
        cnt += 1
    gen.train()
    return total / cnt if cnt > 0 else float("inf")


# ═══════════════════════════════════════════════════════════════
#  MAIN TRAINING
# ═══════════════════════════════════════════════════════════════

def train(config_path=None, resume_from=None, finetune_from=None,
          overfit_batch=False):
    cfg = CONFIG.copy()
    mode = cfg["mode"]
    S = cfg[mode]

    # Override with mode-specific settings
    for k in ["max_epochs", "batch_size", "image_size", "ngf", "num_levels",
              "ndf", "sample_interval", "save_interval", "num_val_samples",
              "log_interval", "patience"]:
        if k in S:
            cfg[k] = S[k]

    # Load YAML config (overrides defaults)
    if config_path:
        with open(config_path, 'r') as f:
            yaml_cfg = yaml.safe_load(f)
        if yaml_cfg.get("training"):
            t = yaml_cfg["training"]
            for k in ["num_epochs", "batch_size", "learning_rate",
                       "lambda_l1", "lambda_adv", "grad_clip",
                       "label_smoothing_real", "label_smoothing_fake"]:
                if k in t: cfg[k.replace("num_epochs", "max_epochs")] = t[k]
            # Separate G/D learning rates (D lower = prevents overpowering)
            if "g_lr" in t: cfg["g_lr"] = t["g_lr"]
            if "d_lr" in t: cfg["d_lr"] = t["d_lr"]
            elif "learning_rate" in t:
                # Backward compat: if only learning_rate, D gets half
                cfg["d_lr"] = t["learning_rate"] / 2
                cfg["g_lr"] = t["learning_rate"]
            if "l1_warmup_epochs" in t: cfg["l1_warmup_epochs"] = t["l1_warmup_epochs"]
            if "adam_beta1" in t: cfg["adam_beta1"] = t["adam_beta1"]
            if "adam_beta2" in t: cfg["adam_beta2"] = t["adam_beta2"]
            for k in ["save_interval", "sample_interval", "num_val_samples",
                       "log_interval", "patience"]:
                if k in t: cfg[k] = t[k]
        if yaml_cfg.get("model"):
            m = yaml_cfg["model"]
            for k in ["image_size", "in_channels", "out_channels"]:
                if k in m: cfg[k] = m[k]
            if m.get("generator"):
                for k in ["ngf", "num_levels", "use_dropout", "dropout"]:
                    if k in m["generator"]: cfg[k] = m["generator"][k]
            if m.get("discriminator"):
                for k in ["ndf", "n_layers"]:
                    if k in m["discriminator"]: cfg[k] = m["discriminator"][k]
        if yaml_cfg.get("data"):
            d = yaml_cfg["data"]
            for k in ["data_dir", "val_fraction", "test_fraction", "num_workers",
                       "mean", "std"]:
                if k in d: cfg[k] = d[k]

    # ── Finetune mode: override config for Phase 2 ──
    is_finetune = finetune_from is not None or mode == "finetune"
    if is_finetune:
        cfg["data_dir"] = "data/finetune"
        cfg["mean"] = FINETUNE_MEAN
        cfg["std"] = FINETUNE_STD
        cfg.setdefault("g_lr", 5e-5)
        cfg.setdefault("d_lr", 2.5e-5)
        cfg.setdefault("lambda_l1", 50)
        cfg.setdefault("val_fraction", 0.1)
        cfg.setdefault("l1_warmup_epochs", 0)
        print("    🎯  FINETUNE mode: Phase 1 G → Phase 2 caricatures")

    device = torch.device(cfg["device"])
    max_epochs = cfg["max_epochs"]

    # ── Banner ──
    is_resume = resume_from is not None
    tag = "RESUME" if is_resume else mode.upper()
    print(f"\n{'='*60}")
    print(f"🎨  face2sketch — pix2pix GAN — {tag}")
    print(f"{'='*60}")
    print(f"   Device: {device}  |  Image: {cfg['image_size']}×{cfg['image_size']}")
    print(f"   Epochs: {max_epochs}  |  Batch: {cfg['batch_size']}  |  "
          f"G_lr: {cfg['g_lr']}  D_lr: {cfg['d_lr']}")
    print(f"   Gen: ngf={cfg['ngf']} levels={cfg['num_levels']}  |  "
          f"Disc: ndf={cfg['ndf']} layers={cfg['n_layers']}")
    print(f"   λ_L1={cfg['lambda_l1']}  λ_adv={cfg['lambda_adv']}  |  "
          f"β1={cfg['adam_beta1']}")

    # ── CLI overrides (batch-size, lr) ──
    if cfg.get("batch_size_override"):
        cfg["batch_size"] = cfg["batch_size_override"]
        print(f"    📦  Batch size override: {cfg['batch_size']}")
    if cfg.get("lr_override"):
        cfg["g_lr"] = cfg["lr_override"]
        cfg["d_lr"] = cfg["lr_override"] / 2
        print(f"    ⚡  LR override: G={cfg['g_lr']}, D={cfg['d_lr']}")
    if cfg.get("patience_override") is not None:
        cfg["patience"] = cfg["patience_override"]
        print(f"    ⏱️   Patience override: {cfg['patience']}")

    # ── Data ──
    print(f"\n📦  Data: {cfg['data_dir']}")
    dataset = FaceDataset(root_dir=cfg['data_dir'])
    print(f"    Pairs: {len(dataset):,}")

    main_tf, aug_tf = get_transformations(
        cfg["mean"], cfg["std"],
        size=(cfg["image_size"], cfg["image_size"]),
    )
    if is_finetune:
        aug_tf = get_finetune_augmentation(
            cfg["mean"], cfg["std"],
            size=(cfg["image_size"], cfg["image_size"]),
        )
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=cfg["batch_size"],
        val_fraction=cfg["val_fraction"],
        test_fraction=cfg["test_fraction"],
        dataset=dataset,
        main_transform=main_tf,
        augmentation_transform=aug_tf,
        num_workers=cfg["num_workers"],
        mean=cfg["mean"], std=cfg["std"],
    )
    # Use val_loader for samples if available, otherwise use train_loader
    sample_loader = val_loader if len(val_loader) > 0 else train_loader
    print(f"    Train: {len(train_loader)} batches  "
          f"Val: {len(val_loader)}  Test: {len(test_loader)}")
    if len(val_loader) == 0:
        print(f"    ⚠️   No val split — using 100% data for training")

    # ── Models ──
    gen, disc = create_models(cfg, device)
    gen, disc = wrap_models(gen, disc, device)
    n_g = sum(p.numel() for p in gen.parameters())
    n_d = sum(p.numel() for p in disc.parameters())
    n_gpu = torch.cuda.device_count() if device.type == 'cuda' else 1
    grad_accum = cfg.get("grad_accum", 1)
    eff_batch = cfg["batch_size"] * grad_accum * n_gpu
    print(f"\n🧱  Generator: {n_g:,} params")
    print(f"    Discriminator: {n_d:,} params")
    print(f"    GPUs: {n_gpu}  |  Batch/GPU: {cfg['batch_size']}"
          f"  |  Grad accum: {grad_accum}  |  Effective: {eff_batch}")

    # ── Optimizers ──
    g_opt = optim.Adam(gen.parameters(), lr=cfg["g_lr"],
                       betas=(cfg["adam_beta1"], cfg["adam_beta2"]))
    d_opt = optim.Adam(disc.parameters(), lr=cfg["d_lr"],
                       betas=(cfg["adam_beta1"], cfg["adam_beta2"]))

    # ── Resume/Load pretrained ──
    start_epoch = 0
    if finetune_from:
        finetune_path = Path(finetune_from)
        if not finetune_path.exists():
            finetune_path = ROOT / finetune_from
        pretrained_epoch = load_pretrained_gen(str(finetune_path), gen, device)
        # D starts fresh for new style, optimizers get new lr settings
        print(f"    🧹  D re-initialized (fresh start for caricature style)")
    elif resume_from:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            resume_path = ROOT / resume_from
        start_epoch = load_ckpt(str(resume_path), gen, disc, g_opt, d_opt, device)

    # ── Overfit batch test ──
    if overfit_batch:
        print("\n🧪  OVERFIT-BATCH SANITY CHECK (200 steps)")
        _overfit_batch(gen, disc, train_loader, g_opt, d_opt, cfg, device)
        save_ckpt(gen, disc, g_opt, d_opt, 0, 0.0, cfg, "overfit_test.pt")
        print("   ✅  Overfit test saved: checkpoints/overfit_test.pt")
        return

    # ── Training ──
    patience = cfg.get("patience", 0)
    l1_warmup = cfg.get("l1_warmup_epochs", 0)
    print(f"\n{'='*60}\n🚀  TRAINING START  (patience={patience}" +
          (f", L1 warmup={l1_warmup} epochs)" if l1_warmup > 0 else ")"))
    print(f"{'='*60}\n")
    best_val_l1 = float("inf")
    plateau_count = 0
    t0 = time.time()

    l1_warmup = cfg.get("l1_warmup_epochs", 0)
    for epoch in range(start_epoch + 1, max_epochs + 1):
        use_adv = epoch > l1_warmup
        if epoch == 1 and l1_warmup > 0:
            print(f"    🔥  L1-only warmup: {l1_warmup} epochs (no adversarial)")
        if epoch == l1_warmup + 1:
            print(f"    ⚔️   Warmup done — adversarial training ON")

        metrics = train_one_epoch(gen, disc, train_loader, g_opt, d_opt, cfg,
                                   epoch, device, use_adversarial=use_adv,
                                   grad_accum=grad_accum)

        # ── Progress ──
        elapsed = time.time() - t0
        eta = elapsed / (epoch - start_epoch) * (max_epochs - epoch) if epoch > start_epoch else 0
        warmup_tag = "🔥" if not use_adv else ""
        print(f"Epoch {epoch:3d}/{max_epochs} {warmup_tag} |  "
              f"D={metrics['d_loss']:.4f}  G_adv={metrics['g_adv']:.4f}  "
              f"G_L1={metrics['g_l1']:.4f}  "
              f"Dr={metrics['d_real']:.3f}  Df={metrics['d_fake']:.3f}  "
              f"⏱ {elapsed:.0f}s  ETA:{eta:.0f}s")

        # ── Validation (skip if no val set) ──
        if len(val_loader) > 0:
            val_l1 = evaluate(gen, val_loader, device)
            is_best = val_l1 < best_val_l1
            trend = "📉" if is_best else "➡️"
            best_val_l1 = min(best_val_l1, val_l1)
            print(f"        Val L1: {val_l1:.4f} {trend}  Best: {best_val_l1:.4f}")
        else:
            val_l1 = metrics["g_l1"]
            is_best = val_l1 < best_val_l1
            best_val_l1 = min(best_val_l1, val_l1)

        # ── Plateau tracking (early stopping) ──
        if is_best:
            plateau_count = 0
        else:
            plateau_count += 1

        # ── Samples ──
        if epoch % cfg["sample_interval"] == 0 or epoch == 1:
            gen.eval()
            save_sample_grid(gen, sample_loader, epoch, cfg["sample_dir"],
                             device=device, num_samples=cfg["num_val_samples"])

        # ── Checkpoints (best + final only) ──
        if is_best:
            save_ckpt(gen, disc, g_opt, d_opt, epoch, val_l1, cfg, "best.pt")
            print(f"        🏆  Best! → best.pt")

        if epoch == max_epochs:
            p = save_ckpt(gen, disc, g_opt, d_opt, epoch, val_l1, cfg, "final.pt")
            print(f"        💾  final.pt")

        # ── Early stop ──
        if patience > 0 and plateau_count >= patience:
            print(f"        ⏹️  No improvement for {patience} epochs — stopping early")
            break

    total_t = time.time() - t0
    print(f"\n{'='*60}")
    label = "Val L1" if len(val_loader) > 0 else "Train L1"
    print(f"✅  Training complete!  {total_t:.0f}s  |  Best {label}: {best_val_l1:.4f}")
    print(f"{'='*60}")


def _overfit_batch(gen, disc, loader, g_opt, d_opt, cfg, device):
    """Overfit on one batch to verify pipeline."""
    bce = nn.BCELoss()
    l1 = nn.L1Loss()
    real_lbl = torch.tensor([cfg["label_smoothing_real"]], device=device)
    fake_lbl = torch.tensor([cfg["label_smoothing_fake"]], device=device)
    la, ll, gc = cfg["lambda_adv"], cfg["lambda_l1"], cfg["grad_clip"]

    photo, real_sketch = next(iter(loader))
    photo = photo.to(device); real_sketch = real_sketch.to(device)
    print(f"    Batch: {photo.shape}, {real_sketch.shape}")

    for step in range(200):
        # D
        with torch.no_grad():
            fake = gen(photo)
        d_loss = (bce(disc(photo, real_sketch), real_lbl.expand_as(disc(photo, real_sketch))) +
                  bce(disc(photo, fake), fake_lbl.expand_as(disc(photo, fake)))) * 0.5
        d_opt.zero_grad(); d_loss.backward()
        if gc > 0: torch.nn.utils.clip_grad_norm_(disc.parameters(), gc)
        d_opt.step()

        # G
        fake = gen(photo)
        g_adv = bce(disc(photo, fake), real_lbl.expand_as(disc(photo, fake)))
        g_l1 = l1(fake, real_sketch)
        g_loss = g_adv * la + g_l1 * ll
        g_opt.zero_grad(); g_loss.backward()
        if gc > 0: torch.nn.utils.clip_grad_norm_(gen.parameters(), gc)
        g_opt.step()

        if step % 40 == 0:
            print(f"    Step {step:3d}: D={d_loss:.4f} G_adv={g_adv:.4f} "
                  f"G_L1={g_l1:.4f} Dr={disc(photo, real_sketch).mean():.3f} "
                  f"Df={disc(photo, fake.detach()).mean():.3f}")

    print("    ✅  Overfit test passed!")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train pix2pix GAN")
    p.add_argument("--mode", choices=["test", "train", "finetune"], default="train",
                   help="test=small local | train=full train | finetune=Phase 2 style")
    p.add_argument("--config", type=str, default=None,
                   help="YAML config file (overrides defaults)")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from checkpoint (loads G+D+optimizers)")
    p.add_argument("--finetune", type=str, default=None,
                   help="Finetune from Phase 1 checkpoint (loads G only)")
    p.add_argument("--overfit-batch", action="store_true",
                   help="Overfit a single batch (sanity check)")
    p.add_argument("--device", type=str, default=None,
                   help="Device: cuda, mps, or cpu")
    p.add_argument("--name", type=str, default="",
                   help="Prefix for checkpoint files")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override batch size (Kaggle T4x2: use 32-48)")
    p.add_argument("--lr", type=float, default=None,
                   help="Override G learning rate")
    p.add_argument("--patience", type=int, default=None,
                   help="Override early stopping patience (0=disable)")
    args = p.parse_args()

    if args.mode:
        CONFIG["mode"] = args.mode
    if args.device:
        CONFIG["device"] = args.device
    if args.name:
        CONFIG["ckpt_prefix"] = args.name
    if args.batch_size:
        CONFIG["batch_size_override"] = args.batch_size
    if args.lr:
        CONFIG["lr_override"] = args.lr

    if args.patience is not None:
        CONFIG["patience_override"] = args.patience

    train(config_path=args.config, resume_from=args.resume,
          finetune_from=args.finetune,
          overfit_batch=args.overfit_batch)
