# Segmentation Training with PyTorch Lightning and Hydra

This repository contains a boilerplate template for training segmentation models using PyTorch Lightning and segmentation-models-pytorch (SMP) library, with Hydra for configuration management.

## Features

- **Modular Design**: Clean separation of data loading, model definition, and training logic
- **Configuration Management**: Hydra-based configuration with YAML files
- **Multiple Architectures**: Support for various SMP architectures (U-Net, U-Net++, DeepLabV3, etc.)
- **Data Augmentation**: Comprehensive augmentation pipeline using torchvision v2 transforms
- **Monitoring**: TensorBoard logging and model checkpointing
- **Best Practices**: Early stopping, learning rate scheduling, mixed precision training

## Installation

```bash
pip install -r requirements.txt
```

## Data Structure

Organize your data as follows:

```
data/
├── images/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── masks/
    ├── image1.png
    ├── image2.png
    └── ...
```

## Configuration

The training script uses Hydra for configuration management. The main configuration file is `config.yaml`. You can modify this file or create new configuration files to experiment with different settings.

### Key Configuration Sections:

- **data**: Dataset paths and data loading parameters
- **model**: Model architecture and encoder settings
- **training**: Training hyperparameters and optimization settings
- **logging**: TensorBoard and experiment tracking settings

## Usage

### Basic Training

```bash
python train.py
```

This will use the default configuration from `config.yaml`.

### Custom Configuration

You can override any configuration parameter from the command line:

```bash
# Change data directory and batch size
python train.py data.data_dir=/path/to/your/data training.batch_size=32

# Use different model architecture
python train.py model.arch=UnetPlusPlus model.encoder_name=efficientnet-b0

# Adjust training parameters
python train.py training.max_epochs=50 training.learning_rate=0.001
```

### Using Custom Config Files

Create a new config file (e.g., `experiment.yaml`) and use it:

```bash
python train.py --config-name=experiment
```

### Multi-GPU Training

```bash
python train.py training.gpus=2
```

## Configuration Examples

### High-Resolution Training
```yaml
# config_highres.yaml
data:
  data_dir: "./data"
  batch_size: 8

augmentation:
  resize_height: 512
  resize_width: 512

training:
  max_epochs: 200
  learning_rate: 0.0005
```

### Quick Experiment
```yaml
# config_quick.yaml
training:
  max_epochs: 10
  batch_size: 32
  
data:
  num_workers: 8
```

## Model Architectures

Supported architectures include:
- Unet
- UnetPlusPlus
- MAnet
- Linknet
- FPN
- PSPNet
- DeepLabV3
- DeepLabV3Plus
- PAN

## Encoder Backbones

Popular encoder options:
- resnet34, resnet50, resnet101
- efficientnet-b0 through efficientnet-b7
- densenet121, densenet169, densenet201
- mobilenet_v2
- timm-* (any timm model)

## Monitoring

Training metrics are logged to TensorBoard. View them with:

```bash
tensorboard --logdir=tb_logs
```

## Output

- **Checkpoints**: Saved in the directory specified by `training.save_dir`
- **Logs**: TensorBoard logs in `logging.log_dir`
- **Best Model**: Automatically saved based on validation loss

## Tips

1. **Mixed Precision**: Enable with `training.precision=16` for faster training and lower memory usage
2. **Data Parallel**: Use multiple GPUs with `training.gpus=N`
3. **Gradient Accumulation**: Increase effective batch size with `training.accumulate_grad_batches=N`
4. **Early Stopping**: Adjust patience with `training.patience=N`

## Extending the Template

### Adding New Loss Functions
Modify the `SegmentationModel` class to include additional loss functions:

```python
# In SegmentationModel.__init__()
self.criterion = smp.losses.DiceLoss() + smp.losses.FocalLoss()
```

### Custom Augmentations
Update the `get_transforms()` function to add new augmentations:

```python
def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ElasticTransform(alpha=250.0, sigma=5.0),  # Add new augmentation
        # ... existing transforms
    ])
```

### Multi-class Segmentation
For multi-class segmentation, update the configuration:

```yaml
model:
  out_classes: 3  # Number of classes

# And modify the loss function in the model
self.criterion = smp.losses.DiceLoss(mode=smp.losses.MULTICLASS_MODE)
```
