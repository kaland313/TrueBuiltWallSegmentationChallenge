
import cv2
import numpy as np
from tqdm import tqdm


def predict_in_patches(image, model, pred_fn, patch_size=512):
    """
    Run prediction on an image in patches to avoid memory issues and see progress.
    Args:
        image: Input image
        model: Trained segmentation model
        pred_fn: Function running prediction on a single patch
        patch_size: Size of each patch (default: 512)
    Returns:
        Predicted mask
    """
    
    h, w = image.shape[0:2]
    predicted_mask = np.zeros((h, w), dtype=np.uint8)

    for y in tqdm(range(0, h, patch_size), leave=False):
        for x in tqdm(range(0, w, patch_size), leave=False):
            patch = image[y:y + patch_size, x:x + patch_size]
            pred_patch = pred_fn(patch, model)
            predicted_mask[y:y + patch_size, x:x + patch_size] = pred_patch

    return predicted_mask

def clean_mask_via_morph_ops(mask, kernel_size=10):
    """
    Apply morphological closing followed by opening to clean the mask.
    
    Args:
        mask: Binary mask image
        kernel_size: Size of the structuring element
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)
    return cleaned_mask

def overlay_mask(image, mask, alpha=0.5):
    """
    Overlay the mask on the original image.
    
    Args:
        image: Original image
        mask: Binary mask
    Returns:
        Image with mask overlay
    """
    # a nice blue color 
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Ensure image is in BGR format
    mask_color = (2, 118, 50) # Green color for the mask
    mask_colored = np.zeros_like(image)
    mask_colored[mask > 0] = mask_color
    overlay = cv2.addWeighted(image.astype(np.float32), 
                              1 - alpha,
                              mask_colored.astype(np.float32),
                              alpha,
                              0)
    overlay[mask == 0] = image[mask == 0]  # Keep original pixels where mask is not present
 
    return overlay.astype(np.uint8)