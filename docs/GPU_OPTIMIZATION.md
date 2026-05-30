# GPU Optimization Patterns in face2sketch

> How we squeeze maximum performance from free GPU hardware without OOM crashes.

---

## The Problem

You're training a 37M-param UNet on Kaggle/Colab. Free tiers give you random hardware — sometimes a single P100, sometimes dual T4s. How do you write one training script that uses whatever it gets?

## Solution 1: DataParallel — Use All GPUs

### What it does

PyTorch, by default, only uses GPU 0. GPU 1 sits idle. `DataParallel` splits each batch across all available GPUs, runs forward+backward in parallel, then gathers gradients on GPU 0.

```
Without DataParallel:           With DataParallel:
┌──────┐  ┌──────┐             ┌──────────────┐  ┌──────────────┐
│ GPU0 │  │ GPU1 │             │    GPU0      │  │    GPU1      │
│ 100% │  │  0%  │             │ batch[:N/2]  │  │ batch[N/2:]  │
│      │  │ idle │             │ forward      │  │ forward      │
└──────┘  └──────┘             │ backward     │  │ backward     │
                               │ gradients ───┼──► gathered     │
                               └──────────────┘  └──────────────┘
```

### How it's implemented

```python
def wrap_models(gen, disc, device):
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        gen = nn.DataParallel(gen)      # splits batch, gathers gradients
        disc = nn.DataParallel(disc)
    return gen, disc
```

One line. No `DistributedDataParallel` boilerplate, no `torch.multiprocessing.spawn`. Just works.

### Tradeoffs

| | DataParallel | DistributedDataParallel |
|---|---|---|
| Code changes | 1 line | ~50 lines |
| Speed overhead | ~5% (gather on GPU0) | Near-linear scaling |
| Works in notebooks | ✅ Yes | ❌ Needs multiprocess |
| Best for | 2-4 GPUs, prototyping | 4+ GPUs, production |

For free Kaggle/Colab (2 GPUs max), DataParallel is the right call.

### The `module.` prefix problem

DataParallel wraps your model: `gen → DataParallel(gen)`. All parameter names get a `module.` prefix:

```
Before DataParallel:           After DataParallel:
initial.weight                 module.initial.weight
encoders.0.conv.weight         module.encoders.0.conv.weight
```

So when you load a checkpoint saved WITHOUT DataParallel into a DataParallel model:

```python
# ❌ Breaks — keys don't match
gen.load_state_dict(ckpt["generator"])
# Error: Missing key(s): "module.initial.weight"...

# ✅ Fix — auto-detect the mismatch
def load_pretrained_gen(path, gen, device):
    state = torch.load(path)["generator"]
    if isinstance(gen, nn.DataParallel) and not any(k.startswith('module.') for k in state):
        state = {'module.' + k: v for k, v in state.items()}  # add prefix
    gen.load_state_dict(state)
```

This is why both `load_pretrained_gen()` and `load_ckpt()` have the prefix auto-fix.

---

## Solution 2: Gradient Accumulation — VRAM Flexibility

### The problem

Your batch size is limited by VRAM. At 256×256, a UNet forward+backward pass with batch=16 uses ~9GB. Batch=32 needs ~18GB — OOM on a 15GB T4.

But you WANT the training stability of a larger effective batch. What do you do?

### How it works

Instead of updating weights after every batch, you accumulate gradients over N smaller batches, then update once:

```
Standard training:                    Gradient accumulation (N=2):
                                      
batch 1 → backward → step            batch 1 → backward (accumulate)
batch 2 → backward → step            batch 2 → backward (accumulate)
batch 3 → backward → step            ──── step (apply accumulated grads)
                                      batch 3 → backward (accumulate)
Effective batch = 16                  batch 4 → backward (accumulate)
                                      ──── step
                                      Effective batch = 32
```

### Implementation

```python
def train_one_epoch(gen, disc, dataloader, ..., grad_accum=1):
    for batch_idx, (photo, sketch) in enumerate(dataloader):
        # Compute loss normally, but divide by accum steps
        g_loss = (g_adv * lambda_adv + g_l1 * lambda_l1) / grad_accum
        g_loss.backward()  # accumulate gradients
        
        # Only update every N steps
        if (batch_idx + 1) % grad_accum == 0:
            g_opt.step()
            g_opt.zero_grad()
```

The `/ grad_accum` is critical — without it, accumulating 2 steps gives 2× the gradient magnitude, which is equivalent to increasing the learning rate, not the batch size.

### Why this matters

| Scenario | batch_size | grad_accum | Effective | VRAM |
|---|---|---|---|---|
| T4 single GPU | 16 | 1 | 16 | ~9GB |
| T4 single GPU, low VRAM | 8 | 2 | 16 | ~5GB |
| T4 ×2 (DataParallel) | 16 | 1 | 32 | ~9GB per GPU |
| T4 ×2, aggressive | 16 | 2 | 64 | ~9GB per GPU |

---

## Combining Both

The banner shows exactly what you're running:

```
🧱  Generator: 36,916,611 params
    Discriminator: 2,768,705 params
    GPUs: 2  |  Batch/GPU: 16  |  Grad accum: 1  |  Effective: 32
```

Effective batch = `batch_size × grad_accum × num_gpus`

- **GPUs=1, Batch=16, Accum=1** → Effective 16 (standard single-GPU)
- **GPUs=2, Batch=16, Accum=1** → Effective 32 (DataParallel splits batch)
- **GPUs=1, Batch=8, Accum=2** → Effective 16 (same as single-GPU, half VRAM)
- **GPUs=2, Batch=8, Accum=2** → Effective 32 (both techniques)

---

## Practical Configurations

### Colab (single T4, 15GB)

```bash
python src/train.py --device cuda
# Default: batch=16, accum=1, 1 GPU → effective 16
```

### Kaggle (dual T4, 15GB each)

```bash
python src/train.py --device cuda
# Auto-detect: batch=16, accum=1, 2 GPUs → effective 32
```

### If you hit OOM

```bash
# Halve per-GPU batch, double accumulation → same effective, less VRAM
python src/train.py --batch-size 8 --device cuda
# Manually set accum=2 in config or inline
```

### If you have spare VRAM

```bash
python src/train.py --batch-size 24 --device cuda
# 2 GPUs × 24 = effective 48
```

---

## What We Didn't Do

- **DistributedDataParallel** — Overkill for 2 GPUs, needs multiprocessing, breaks notebook flow.
- **Mixed precision (AMP)** — P100 supports it, T4 doesn't well. Added complexity for marginal gain on 184-pair dataset.
- **VRAM probing** — We had this, removed it. Industry doesn't guess batch size at runtime — you configure it once for your hardware.
- **Gradient checkpointing** — Trades compute for VRAM. Useful for very large models or high resolutions, not needed at 256×256.

---

## The Key Principle

> **Don't guess. Configure.**

The script auto-detects GPU count (for DataParallel), but batch size is always explicit. You set it once for your hardware, or override with `--batch-size`. Gradient accumulation is there as a safety net if you're VRAM-constrained — same effective batch, less memory.
