"""
Phase 1: pix2pix GAN Training Script — ties together Generator,
Discriminator, data loading, checkpointing, and visualization.

Usage:
  python src/train.py                                    # train with defaults
  python src/train.py --config configs/pix2pix_phase1.yaml
  python src/train.py --resume checkpoints/pix2pix_epoch_40.pt
  python src/train.py --overfit-batch                    # sanity check on 1 batch

═══════════════════════════════════════════════════════════════
TRAINING LOOP STRUCTURE
═══════════════════════════════════════════════════════════════

def train_one_epoch(generator, discriminator, dataloader,
                    g_optimizer, d_optimizer, config, epoch):

    for batch_idx, (photo, real_sketch) in enumerate(dataloader):
        photo = photo.to(device)
        real_sketch = real_sketch.to(device)

        # ═══════════════════════════════════════════════════
        # STEP 1: UPDATE DISCRIMINATOR
        # ═══════════════════════════════════════════════════
        #
        # The discriminator learns to tell real pairs from fake pairs.

        # Generate fake sketch from current generator
        fake_sketch = generator(photo)

        # Discriminator predictions on REAL pair
        d_real = discriminator(photo, real_sketch)
        d_real_loss = BCE(d_real, real_label_smoothed)

        # Discriminator predictions on FAKE pair
        # CRITICAL: .detach() the fake_sketch so D gradients don't flow to G
        d_fake = discriminator(photo, fake_sketch.detach())
        d_fake_loss = BCE(d_fake, fake_label)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Backprop on discriminator ONLY
        d_optimizer.zero_grad()
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
        d_optimizer.step()

        # ═══════════════════════════════════════════════════
        # STEP 2: UPDATE GENERATOR
        # ═══════════════════════════════════════════════════
        #
        # The generator learns to produce sketches that:
        #   a) LOOK like real sketches (fool the discriminator)
        #   b) ARE close to the ground truth (L1 reconstruction)

        # Generate again (without .detach() — we WANT gradients this time)
        fake_sketch = generator(photo)

        # Adversarial loss: "did I fool the discriminator?"
        d_fake_for_g = discriminator(photo, fake_sketch)
        g_adv_loss = BCE(d_fake_for_g, real_label)  # target=1.0: "be real!"

        # L1 reconstruction loss: "am I structurally correct?"
        g_l1_loss = L1Loss(fake_sketch, real_sketch)

        # Total generator loss
        g_loss = g_adv_loss * lambda_adv + g_l1_loss * lambda_l1

        # Backprop on generator ONLY
        g_optimizer.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        g_optimizer.step()

        # ═══════════════════════════════════════════════════
        # LOGGING
        # ═══════════════════════════════════════════════════
        if batch_idx % log_interval == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"D_loss: {d_loss:.4f}  "
                  f"G_adv: {g_adv_loss:.4f}  G_L1: {g_l1_loss:.4f}  "
                  f"D_real: {d_real.mean():.3f}  D_fake: {d_fake.mean():.3f}")


def main():
    1. Parse config (YAML or argparse)
    2. Create dataloaders (train/val/test) from data_loader.py
    3. Create Generator (UNetGenerator)
    4. Create Discriminator (PatchGANDiscriminator)
    5. Create optimizers (Adam, β1=0.5, β2=0.999, lr=2e-4)
    6. Create loss functions (BCEWithLogitsLoss or MSELoss + L1Loss)
    7. Training loop:
       for epoch in range(num_epochs):
           train_one_epoch(...)
           if epoch % sample_interval == 0:
               generate_val_samples(generator, val_loader)
               save_sample_grid(epoch)
           if epoch % save_interval == 0:
               save_checkpoint(generator, discriminator, optimizers, epoch)
    8. Save final checkpoint


# ═══════════════════════════════════════════════════════════════
# KEY DIFFERENCES FROM CLASSIFIER TRAINING
# ═══════════════════════════════════════════════════════════════
#
# 1. TWO optimizers, updated SEPARATELY
#    G and D are adversaries — they should NEVER share gradients.
#    Always: zero_grad → compute_D_loss → backward → step(D)
#            zero_grad → compute_G_loss → backward → step(G)
#
# 2. .detach() is critical
#    When computing D_loss: fake_sketch = generator(photo).detach()
#    When computing G_loss: fake_sketch = generator(photo)  ← no detach!
#    Getting this wrong silently corrupts training.
#
# 3. Discriminator accuracy tells you if training is balanced
#    D_real → 1.0: D correctly identifies real pairs
#    D_fake → 0.0: D correctly identifies fake pairs
#    Ideal: D_real ≈ 0.7-0.9, D_fake ≈ 0.3-0.5 (D slightly better than random)
#    If D_real → 1.0 AND D_fake → 0.0: D is too strong, G can't learn
#    If D_real ≈ D_fake ≈ 0.5: D is too weak, G isn't being challenged
#
# 4. No val loss in the traditional sense
#    GANs don't have a clean "validation loss" that always correlates
#    with quality. Monitor:
#      - D accuracy balance (real vs fake predictions)
#      - G_L1 on val set (should decrease)
#      - VISUAL QUALITY of generated samples (MOST IMPORTANT)
#
# 5. β1=0.5 instead of 0.9
#    Standard Adam uses β1=0.9 (heavy momentum). GANs train in a dynamic
#    equilibrium — lower momentum (0.5) helps both networks adapt faster
#    to each other's changes.


# ═══════════════════════════════════════════════════════════════
# COMMON PITFALLS (read before implementing)
# ═══════════════════════════════════════════════════════════════
#
# 1. FORGETTING .detach()
#    Symptom: G_loss goes down but D_loss never changes
#    Cause: Fake sketch gradients flowing to G through D_loss path
#    Fix: Always .detach() fake_sketch when computing D_loss
#
# 2. MODE COLLAPSE
#    Symptom: Generator produces same/similar output for any input
#    Cause: G found a "safe" output that fools D consistently
#    Fix: Increase λ_l1 (more ground truth signal), add noise to D inputs,
#         use a buffer of generated images for D training
#
# 3. D OVERPOWERING G
#    Symptom: D_loss → 0, G_loss stagnant or increasing
#    Cause: D is too strong, G gets zero gradient through adversarial loss
#    Fix: Decrease D learning rate, add label smoothing, reduce D capacity
#
# 4. CHECKERBOARD ARTIFACTS
#    Symptom: Grid-like patterns in generated images
#    Cause: ConvTranspose2d overlapping kernels
#    Fix: Use Upsample + Conv2d instead of ConvTranspose2d
#
# 5. NAN LOSSES
#    Symptom: Loss becomes NaN
#    Cause: Gradients exploding, or log(0) in BCE with logits
#    Fix: Gradient clipping (1.0), use BCEWithLogitsLoss (numerically stable),
#         lower learning rate
#
# 6. G_L1 DOMINATES G_ADV
#    Symptom: Outputs are blurry, G_adv loss doesn't change
#    Cause: λ_l1 too high relative to λ_adv
#    Fix: Decrease λ_l1 from 100 to 50 or 20


# ═══════════════════════════════════════════════════════════════
# OVERFIT-BATCH SANITY CHECK
# ═══════════════════════════════════════════════════════════════
#
# Before full training, ALWAYS overfit on a single batch:
#   python src/train.py --overfit-batch
#
# This catches 90% of bugs:
#   - Model should memorize the batch (loss → 0, outputs match targets)
#   - D accuracy should converge to ~50% (can't distinguish on 1 batch)
#   - If it can't overfit, something is broken (architecture, loss, data)
#
# Same pattern as overfitting on a single text sequence
# when debugging a language model.
