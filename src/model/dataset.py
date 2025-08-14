# Copyright (c) 2025 András Kalapos
# Licensed under the MIT License. See LICENSE file in the project root for details.

import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from torchvision import tv_tensors
from torchvision.io import read_image, ImageReadMode
from pathlib import Path


class SegmentationDataset(Dataset):
    """Custom dataset for segmentation tasks"""

    def __init__(self, image_paths, mask_paths, transforms=None, num_classes=1):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms
        assert num_classes in [1, 3], "out_classes must be 1 (binary) or 3 (multiclass)"
        self.num_classes = num_classes
        print(self.image_paths)
        print(self.mask_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask using torchvision
        image_path = str(self.image_paths[idx])
        mask_path = str(self.mask_paths[idx])

        # Load image as RGB
        image = read_image(image_path, mode=ImageReadMode.RGB).float() / 255.0

        # Load mask as grayscale
        mask = read_image(mask_path, mode=ImageReadMode.GRAY)
        # import numpy as np
        # print(np.unique(mask))
        # replace 128 with 1 and 255 with 2 for multiclass
        if self.num_classes == 3:
            mask = torch.where(mask == 119, torch.tensor(1), mask)
            mask = torch.where(mask == 255, torch.tensor(2), mask)
        mask = mask.float()

        # Apply transforms
        if self.transforms:
            image_tv = tv_tensors.Image(image)
            mask_tv = tv_tensors.Mask(mask)
            image, mask = self.transforms(image_tv, mask_tv)
        return image, mask.long()


def get_transforms():
    """Define augmentation transforms using torchvision v2"""

    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=(512, 512), scale=(0.01, 0.5)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180, expand=False),
            transforms.RandomAffine(
                degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=None
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3
            ),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0
            ),
            # Images are grayscae, 0-255, masks are 0-1
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),  # This is the resolution of val images
        ]
    )

    return train_transforms, val_transforms


def prepare_data(data_dir, image_folder, mask_folder, repeats=1):
    """Prepare training and validation datasets"""

    # Assuming data structure: data_dir/images/ and data_dir/masks/
    image_dir = Path(data_dir) / image_folder
    mask_dir = Path(data_dir) / mask_folder
    # Resolve paths
    image_dir = image_dir.resolve()
    mask_dir = mask_dir.resolve()

    image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    mask_paths = sorted(list(mask_dir.glob("*.jpg")) + list(mask_dir.glob("*.png")))

    # Ensure matching number of images and masks
    assert len(image_paths) == len(mask_paths), (
        f"Mismatch: {len(image_paths)} images vs {len(mask_paths)} masks"
    )

    # Repeat data for augmentation
    image_paths = image_paths * repeats
    mask_paths = mask_paths * repeats

    return image_paths, mask_paths
