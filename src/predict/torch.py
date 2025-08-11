import cv2
import numpy as np
import torch

from model import SegmentationModel

def predict(image, model):
    """Run prediction on a single image using the trained model.
    Args:
        image: Input image as a numpy array
        model: Trained segmentation model
    Returns:
        Predicted mask as a numpy array
    """
    if len(image.shape) == 2:  # If grayscale, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Ensure image is in RGB format
    image = image.transpose(2, 0, 1)  # Change to CxHxW format
    image = image / 255.0  # Normalize to [0, 1]

    # Pad to next multiple of 32
    h, w = image.shape[1:3]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    with torch.no_grad():
        # Preprocess the image
        input_tensor = torch.from_numpy(image).unsqueeze(0).float().to(model.device)  
        pred = model(input_tensor)
        pred_mask = torch.sigmoid(pred).squeeze().cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype('uint8') * 255
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            pred_mask = pred_mask[:h, :w]
        return pred_mask

    
def load_model(model_ckpt="model.ckpt", gpu_id=0):
    """Load the trained segmentation model from checkpoint.
    Args:
        model_ckpt: Path to the model checkpoint
        gpu_id: GPU ID to use for inference
    Returns:
        Loaded model
    """
    model = SegmentationModel.load_from_checkpoint(model_ckpt)
    model.eval()
    if torch.cuda.is_available() and gpu_id >= 0:
        device = torch.device(f'cuda:{gpu_id}')
        model.to(device)
        print(f"Using GPU: {gpu_id}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return model