import numpy as np
import cv2
from tqdm import trange


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha=0.5):
    """
    Overlay the mask on the original image.

    Args:
        image: Original image
        mask: Binary mask
    Returns:
        Image with mask overlay
    """
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Ensure image is in BGR format
    mask_colors = (
        (237, 111, 100),  # Blue
        (2, 118, 50), # Green
    )  
    mask_values = np.unique(mask)
    mask_colored = np.zeros_like(image)
    for i, value in enumerate(mask_values):
        if value == 0:
            continue
        mask_colored[mask == value] = mask_colors[i % len(mask_colors)]
    overlay = cv2.addWeighted(
        image.astype(np.float32), 1 - alpha, mask_colored.astype(np.float32), alpha, 0
    )
    overlay[mask == 0] = image[
        mask == 0
    ]  # Keep original pixels where mask is not present

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
    blended = cv2.addWeighted(
        image.astype(np.float32), 1 - alpha, room_image.astype(np.float32), alpha, 0
    )
    # Keep background pixels from the original image
    blended[room_image == 0] = image[room_image == 0]

    return blended.astype(np.uint8)


def colorize_regions(mask):
    """Colorize indexed regions in a mask.
    0 is not colored, its considered background and colored black.
    Args:
        mask: Indexed mask image
    Returns:
        Colorized image
    """

    # Create a color map
    num_labels = np.max(mask) + 1
    colors = np.random.randint(0, 256, size=(num_labels, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background stays black

    # Vectorized colorization - this is the key speedup
    colorized_image = colors[mask]
    return colorized_image
