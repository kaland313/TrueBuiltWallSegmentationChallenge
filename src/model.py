import torch
import lightning as L
import segmentation_models_pytorch as smp
import wandb

class SegmentationModel(L.LightningModule):
    """PyTorch Lightning module for segmentation"""
    
    def __init__(
        self, 
        arch="Unet", 
        encoder_name="resnet34", 
        in_channels=3, 
        out_classes=1, 
        learning_rate=1e-4,
        encoder_weights="imagenet", 
        loss_fn="DiceLoss"
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        
        # Initialize model
        self.model = smp.create_model(
            arch, 
            encoder_name=encoder_name, 
            in_channels=in_channels, 
            classes=out_classes,
            encoder_weights=encoder_weights
        )
        
        # Loss function
        if loss_fn == "DiceLoss":
            self.criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        elif loss_fn == "FocalLoss":
            self.criterion = smp.losses.FocalLoss(smp.losses.BINARY_MODE)
        elif loss_fn == "MCCLoss":
            self.criterion = smp.losses.MCCLoss()    

    def forward(self, x):
        return self.model(x)
    
    def common_step(self, batch, mode="train"):
        images, masks = batch
        outputs = self(images)
        
        loss = self.criterion(outputs, masks)
        
        # Calculate IoU
        tp, fp, fn, tn = smp.metrics.get_stats(outputs, masks, mode='binary', threshold=0.5)
        # then compute metrics with required reduction (see metric docs)
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        self.log(f'{mode}_iou', iou, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{mode}_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        # log images and masks
        if mode == "val":
            outputs = torch.sigmoid(outputs.detach())
            outputs = (outputs > 0.5).cpu().numpy().astype('uint8')
            masks = masks.cpu().numpy().astype('uint8')
            images = images.cpu().numpy().transpose(0, 2, 3, 1).astype('uint8')
            class_labels = {0: "Not Wall", 1: "Wall"}

            for i in range(len(images)):
                wandb.log(
                    {f"val_image_{i}" : wandb.Image(images[i], masks={
                        "predictions" : {
                            "mask_data" : outputs[i].squeeze(),
                            "class_labels" : class_labels
                        },
                        "ground_truth" : {
                            "mask_data" : masks[i].squeeze(),
                            "class_labels" : class_labels
                        }
                    })})

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
