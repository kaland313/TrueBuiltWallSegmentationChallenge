import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
import torchvision.transforms.v2 as transforms
from torchvision import tv_tensors 
from torchvision.io import read_image, ImageReadMode
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

from model import SegmentationModel

class SegmentationDataset(Dataset):
    """Custom dataset for segmentation tasks"""
    
    def __init__(self, image_paths, mask_paths, transforms=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms
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
        mask = read_image(mask_path, mode=ImageReadMode.GRAY).float() / 255.0
        # mask = mask.squeeze(0)  # Remove channel dimension for mask
        
        # Apply transforms
        if self.transforms:
            image_tv = tv_tensors.Image(image)
            mask_tv = tv_tensors.Mask(mask)
            image, mask = self.transforms(image_tv, mask_tv)
        return image, mask.long()

def get_transforms():
    """Define augmentation transforms using torchvision v2"""
    
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=(512, 512), scale=(0.08, 0.5)), 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90, expand=False),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=None
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0),    
        # Images are grayscae, 0-255, masks are 0-1
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((1024, 1024)), # This is the resolution of val images
    ])
    
    return train_transforms, val_transforms


def prepare_data(data_dir, repeats=1):
    """Prepare training and validation datasets"""
    
    # Assuming data structure: data_dir/images/ and data_dir/masks/
    image_dir = Path(data_dir) / "images"
    mask_dir = Path(data_dir) / "masks"
    # Resolve paths
    image_dir = image_dir.resolve()
    mask_dir = mask_dir.resolve()

    image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    mask_paths = sorted(list(mask_dir.glob("*.jpg")) + list(mask_dir.glob("*.png")))
    
    # Ensure matching number of images and masks
    assert len(image_paths) == len(mask_paths), f"Mismatch: {len(image_paths)} images vs {len(mask_paths)} masks"

    # Repeat data for augmentation
    image_paths = image_paths * repeats
    mask_paths = mask_paths * repeats

    return image_paths, mask_paths

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    """Main training function using Hydra configuration"""
    
    # Print configuration
    print(OmegaConf.to_yaml(cfg))

    L.seed_everything(cfg.seed, workers=True)  # Set random seed for reproducibility
    
    # Prepare data
    train_images, train_masks = prepare_data(cfg.data.train_data_dir, repeats=cfg.data.repeats)
    val_images, val_masks = prepare_data(cfg.data.val_data_dir)
    train_transforms, val_transforms = get_transforms()
    
    # Create datasets
    train_dataset = SegmentationDataset(train_images, train_masks, train_transforms)
    val_dataset = SegmentationDataset(val_images, val_masks, val_transforms)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=True, 
        num_workers=cfg.data.num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=False, 
        num_workers=cfg.data.num_workers, 
        pin_memory=True
    )
    
    # Initialize model
    model = SegmentationModel(
        arch=cfg.model.arch,
        encoder_name=cfg.model.encoder_name,
        out_classes=1, # Wall / not wall binary segmentation
        in_channels=3, # Grayscale images
        learning_rate=cfg.training.learning_rate,
        encoder_weights=cfg.model.encoder_weights,
        loss_fn=cfg.training.loss_fn
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        filename='{epoch}-{val_loss:.2f}-{val_iou:.2f}',
        monitor=cfg.training.monitor_metric,
        mode='min',
        save_top_k=cfg.training.save_top_k,
        save_last=True
    )

    
    # Logger
    logger = WandbLogger(
        project="TrueBuildchallenge",
        name=cfg.experiment_name,
        save_dir="artifacts",
            group=cfg.logging.get("wandb_group", None),
        )

    
    # Trainer
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() and cfg.training.use_gpu else 'cpu',
        devices=cfg.training.gpus if torch.cuda.is_available() and cfg.training.use_gpu else 'auto',
        callbacks=[checkpoint_callback],
        logger=logger,
        default_root_dir=Path(cfg.training.save_dir) / cfg.experiment_name,
        precision=cfg.training.precision,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    print(f"Training completed! Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()