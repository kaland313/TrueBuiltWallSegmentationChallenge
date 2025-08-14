import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import wandb


class SegmentationModel(pl.LightningModule):
    """PyTorch Lightning module for segmentation"""

    def __init__(
        self,
        arch="Unet",
        encoder_name="resnet34",
        in_channels=3,
        out_classes=3,
        learning_rate=1e-4,
        encoder_weights="imagenet",
        loss_fn="DiceLoss",
    ):
        super().__init__()
        self.learning_rate = learning_rate
        assert out_classes in [1, 3], "out_classes must be 1 (binary) or 3 (multiclass)"
        self.out_classes = out_classes
        if out_classes == 1:
            self.class_labels = {0: "Not Wall", 1: "Wall"}
        else:
            self.class_labels = {0: "Not Wall", 1: "Wall", 2: "DoorWindow"}
        self.save_hyperparameters()

        # Initialize model
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            encoder_weights=encoder_weights,
        )

        # Loss function
        if self.out_classes == 1:
            if loss_fn == "DiceLoss":
                self.criterion = smp.losses.DiceLoss(
                    smp.losses.BINARY_MODE, from_logits=True
                )
            else:
                raise ValueError(f"Unsupported loss function: {loss_fn}")
        else:
            if loss_fn == "DiceLoss":
                self.criterion = smp.losses.DiceLoss(
                    smp.losses.MULTICLASS_MODE, from_logits=True
                )
            else:
                raise ValueError(f"Unsupported loss function: {loss_fn}")

    def forward(self, x):
        return self.model(x)

    def common_step(self, batch, mode="train"):
        images, masks = batch
        outputs = self(images)

        loss = self.criterion(outputs, masks)

        if self.out_classes > 1:
            # For multiclass, we need to apply softmax to outputs
            preds = torch.argmax(outputs, dim=1, keepdim=True).detach()
        else:
            preds = (torch.sigmoid(outputs) > 0.5).float().detach()

        # Calculate IoU
        tp, fp, fn, tn = smp.metrics.get_stats(
            preds,
            masks,
            mode="binary" if self.out_classes == 1 else "multiclass",
            threshold=0.5 if self.out_classes == 1 else None,
            num_classes=self.out_classes if self.out_classes > 1 else None,
        )
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        self.log(f"{mode}_iou", iou, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{mode}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        # log images and masks
        if mode == "val":
            preds = preds.cpu().numpy().astype("uint8")
            masks = masks.cpu().numpy().astype("uint8")
            images = images.cpu().numpy().transpose(0, 2, 3, 1).astype("uint8")

            for i in range(min(4, len(images))):
                wandb.log(
                    {
                        f"val_image_{i}": wandb.Image(
                            images[i],
                            masks={
                                "predictions": {
                                    "mask_data": preds[i].squeeze(),
                                    "class_labels": self.class_labels,
                                },
                                "ground_truth": {
                                    "mask_data": masks[i].squeeze(),
                                    "class_labels": self.class_labels,
                                },
                            },
                        )
                    }
                )

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, mode="val")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
