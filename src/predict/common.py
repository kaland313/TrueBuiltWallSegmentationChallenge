
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

def overlay_mask(image:np.ndarray, mask:np.ndarray, alpha=0.5):
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
    mask_colors = ((237, 111, 100), # Blue
                   (2, 118, 50),)    # Green
    mask_values = np.unique(mask)
    mask_colored = np.zeros_like(image)
    for i, value in enumerate(mask_values):
        if value == 0:
            continue
        mask_colored[mask == value] = mask_colors[i % len(mask_colors)]
    overlay = cv2.addWeighted(image.astype(np.float32), 
                              1 - alpha,
                              mask_colored.astype(np.float32),
                              alpha,
                              0)
    overlay[mask == 0] = image[mask == 0]  # Keep original pixels where mask is not present
 
    return overlay.astype(np.uint8)


def overlay_rooms(image: np.ndarray, room_image: np.ndarray, alpha=0.5):
    """
    Overlay the room segmentation on the original image.
    
    Args:
        image: Original image
        room_image: Room segmentation mask
    Returns:
        Image with room segmentation overlay
    """
    # Convert to BGR if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    
    # Blend the original image and the room overlay
    blended = cv2.addWeighted(image.astype(np.float32), 
                              1 - alpha,
                              room_image.astype(np.float32),
                              alpha,
                              0)
    
    return blended.astype(np.uint8)