# data_loader.py

import os
from typing import List, Tuple

from torch.utils.data import Dataset, random_split, Subset, DataLoader
from pathlib import Path
from torchvision import transforms
import torch
from tqdm.auto import tqdm
from PIL import Image

path_dataset = Path.cwd() / 'data/dataset'

class SubsetWithTransform(Dataset):
    """Wrap a SubSet applying the same transform to both photo and sketch."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        photo, sketch = self.subset[index]
        if self.transform:
            # Apply transforms with shared random seed so flip/rotation
            # are identical for both images in the pair.
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            photo = self.transform(photo)
            torch.manual_seed(seed)
            sketch = self.transform(sketch)
        return photo, sketch


class FaceDataset(Dataset):
    def __init__(self, root_dir=path_dataset, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[str, str]] = self._make_dataset()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        photo_path, sketch_path = self.samples[index]
        photo = Image.open(photo_path).convert('RGB')
        sketch = Image.open(sketch_path).convert('RGB')
        return photo, sketch

    def _make_dataset(self) -> List[Tuple[str, str]]:
        """Pair each photo with its matching sketch by filename.
        Expects flat photos/ and sketches/ dirs under root_dir."""
        photos_dir = os.path.join(self.root_dir, 'photos')
        sketches_dir = os.path.join(self.root_dir, 'sketches')

        assert os.path.isdir(photos_dir), f"Missing photos dir: {photos_dir}"
        assert os.path.isdir(sketches_dir), f"Missing sketches dir: {sketches_dir}"

        samples = []
        IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')

        for entry in os.scandir(photos_dir):
            if not entry.is_file():
                continue
            ext = os.path.splitext(entry.name)[1].lower()
            if ext not in IMAGE_EXT:
                continue

            sketch_path = os.path.join(sketches_dir, entry.name)
            if not os.path.isfile(sketch_path):
                continue

            samples.append((entry.path, sketch_path))

        samples.sort()  # deterministic order
        assert len(samples) > 0, "No paired samples found!"
        return samples


def get_transformations(mean, std, size=(128, 128)):
    main_tfs = [
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]

    # Only geometric augments — no ColorJitter.
    # Sketches are black-on-white; color/brightness jitter would turn
    # the background gray/pink and create unrealistic training pairs.
    augmentation_tfs = [
        transforms.Resize(size),                                                                                                                                                                
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),                                                                                                                                                                        
        transforms.ToTensor(),                                                                                                                                                                        
        transforms.Normalize(mean, std),    
    ]

    main_transform = transforms.Compose(main_tfs)
    augmentation_transform = transforms.Compose(augmentation_tfs)

    return main_transform, augmentation_transform

def get_mean_std(dataset: Dataset):
    """Single-pass mean & std over BOTH photos and sketches."""
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    sum_pixels = torch.zeros(3)
    sum_sq_pixels = torch.zeros(3)
    total_pixels = 0

    loader = tqdm(dataset, desc="Computing mean & std")

    for photo, sketch in loader:
        for img in (photo, sketch):
            img_tensor = preprocess(img)
            pixels = img_tensor.view(3, -1)          # (3, H*W)
            sum_pixels += pixels.sum(dim=1)           # sum per channel
            sum_sq_pixels += (pixels ** 2).sum(dim=1) # sum of squares per channel
            total_pixels += pixels.size(1)            # count pixels

    mean = sum_pixels / total_pixels
    variance = sum_sq_pixels / total_pixels - mean ** 2
    std = torch.sqrt(variance.clamp(min=1e-8))

    return mean.tolist(), std.tolist()

# ── Dataset normalization constants ──────────────────────────────

# Phase 1: CUHK + SKSF-A (322 pairs, standard face photos/sketches)
DATASET_MEAN = [0.7107771635055542, 0.7137131094932556, 0.6956883072853088]
DATASET_STD  = [0.2982601225376129, 0.296517014503479, 0.3196362555027008]

# Phase 2: TwitterPicasso finetune (184 pairs, caricature style)
FINETUNE_MEAN = [0.6433422565460205, 0.5730469226837158, 0.5383039712905884]
FINETUNE_STD  = [0.27061960101127625, 0.2684483528137207, 0.2593892216682434]

# Default to dataset (Phase 1)
MEAN = DATASET_MEAN
MEAN_STD = DATASET_STD

face_dataset = FaceDataset()
main_transform, transform_with_augmentation = get_transformations(MEAN, MEAN_STD)

def get_dataloaders(
    batch_size, val_fraction, test_fraction,
    dataset=face_dataset,
    main_transform=main_transform,
    augmentation_transform=transform_with_augmentation,
    train_fraction=1.0,
    num_workers=1,
    root_dir=None,          # override dataset root, e.g. 'data/finetune'
    mean=None, std=None,    # override normalization values
):
    # If root_dir is given, create a new dataset + transforms for that dir
    if root_dir is not None:
        m = mean if mean is not None else MEAN
        s = std if std is not None else MEAN_STD
        dataset = FaceDataset(root_dir=root_dir)
        main_transform, augmentation_transform = get_transformations(m, s)
    
    total_size = len(dataset)
    val_size = int(total_size * val_fraction)
    test_size = int(total_size * test_fraction)
    train_size = total_size - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Use only a fraction of training data (for fast Optuna trials)
    if train_fraction < 1.0:
        subset_size = int(len(train_dataset) * train_fraction)
        train_dataset = Subset(train_dataset, range(subset_size))

    train_dataset = SubsetWithTransform(subset=train_dataset, transform=augmentation_transform)
    val_dataset = SubsetWithTransform(subset=val_dataset, transform=main_transform)
    test_dataset = SubsetWithTransform(subset=test_dataset, transform=main_transform)


    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader



# ═══════════════════════════════════════════════════════════════
#  DRY-RUN TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # --- Already computed, uncomment to recompute ---
    mean, std = get_mean_std(face_dataset)
    print(f"MEAN = {mean}")
    print(f"MEAN_STD  = {std}")

    # from torchvision.utils import save_image, make_grid

    # train_loader, val_loader, test_loader = get_dataloaders(
    #     batch_size=8, val_fraction=0.1, test_fraction=0.05
    # )

    # # --- Validation batch (main transform — no augmentation) ---
    # photos, sketches = next(iter(val_loader))

    # # Un-normalize: x = x * std + mean
    # mean_t = torch.tensor(MEAN).view(1, 3, 1, 1)
    # std_t = torch.tensor(MEAN_STD).view(1, 3, 1, 1)
    # photos_vis = (photos * std_t + mean_t).clamp(0, 1)
    # sketches_vis = (sketches * std_t + mean_t).clamp(0, 1)

    # # Side-by-side: photo | sketch alternating
    # pairs = []
    # for i in range(photos_vis.size(0)):
    #     pairs.append(photos_vis[i])
    #     pairs.append(sketches_vis[i])
    # grid = make_grid(pairs, nrow=2)
    # save_image(grid, 'samples/pairs_val.png')
    # print("Saved samples/pairs_val.png  (no augment — val set)")

    # # --- Training batch (WITH augmentation) ---
    # photos_aug, sketches_aug = next(iter(train_loader))
    # photos_aug_vis = (photos_aug * std_t + mean_t).clamp(0, 1)
    # sketches_aug_vis = (sketches_aug * std_t + mean_t).clamp(0, 1)

    # pairs_aug = []
    # for i in range(photos_aug_vis.size(0)):
    #     pairs_aug.append(photos_aug_vis[i])
    #     pairs_aug.append(sketches_aug_vis[i])
    # grid_aug = make_grid(pairs_aug, nrow=2)
    # save_image(grid_aug, 'samples/pairs_train_aug.png')
    # print("Saved samples/pairs_train_aug.png  (with flip + rotation)")

    # print(f"\nDataset: {len(FaceDataset())} pairs")
    # print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)}, Test: {len(test_loader)}")
    # print(f"\nOpen samples/pairs_val.png to verify paired transforms — "
    #       f"each photo-sketch pair shares the same flip/rotation.")