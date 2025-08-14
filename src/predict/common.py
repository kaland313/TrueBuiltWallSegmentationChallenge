
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

def clean_mask_via_morph_ops(mask, close_kernel_size=10, open_kernel_size=5):
    """
    Apply morphological closing followed by opening to clean the mask.
    
    Args:
        mask: Binary mask image
        close_kernel_size: Size of the kernel for closing operation
        open_kernel_size: Size of the kernel for opening operation
    """
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))

    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=3)
    cleaned_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, open_kernel)
    return cleaned_mask
