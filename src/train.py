import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
import torchvision.transforms.v2 as transforms
from torchvision import tv_tensors 
from torchvision.io import read_image, ImageReadMode
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

from model import SegmentationModel, SegmentationDataset, prepare_data, get_transforms

@hydra.main(version_base=None, config_path="./model", config_name="config")
def main(cfg: DictConfig):
    """Main training function using Hydra configuration"""
    
    # Print configuration
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed, workers=True)  # Set random seed for reproducibility
    
    # Prepare data
    train_images, train_masks = prepare_data(cfg.data.train_data_dir, 
                                             image_folder=cfg.data.image_folder,
                                             mask_folder=cfg.data.mask_folder,
                                             repeats=cfg.data.repeats)
    val_images, val_masks = prepare_data(cfg.data.val_data_dir, 
                                             image_folder=cfg.data.image_folder,
                                             mask_folder=cfg.data.mask_folder,
                                             repeats=cfg.data.repeats)
    train_transforms, val_transforms = get_transforms()
    
    # Create datasets
    train_dataset = SegmentationDataset(train_images, train_masks, train_transforms, num_classes=cfg.model.out_classes)
    val_dataset = SegmentationDataset(val_images, val_masks, val_transforms, num_classes=cfg.model.out_classes)
    
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
        out_classes=cfg.model.out_classes, # Wall / not wall binary segmentation
        in_channels=3,
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
    trainer = pl.Trainer(
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