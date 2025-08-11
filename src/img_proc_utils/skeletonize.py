import numpy as np
from skimage.morphology import skeletonize
from tqdm import tqdm


def thin_image(image):
    """
    Thin a binary image using morphological thinning.
    
    Args:
        image: Input binary image
    
    Returns:
        Thinned binary image
    """
    # Ensure the image is binary (0 or 255)
    binary_image = (image > 0).astype(np.uint8) * 255
    thinned_image = skeletonize(binary_image // 255) * 255
    return thinned_image.astype(np.uint8)

def skeletonize_in_patches(image, patch_size=515):
    """
    Skeletonize a large image in patches.
    Args:
        image: Input binary image
        patch_size: Size of each patch (default: 515)
    Returns:
        Thinned binary image
    """
    h, w = image.shape

    thinned_image = np.zeros_like(image)
    
    for y in tqdm(range(0, h, patch_size)):
        for x in tqdm(range(0, w, patch_size), leave=False):
            patch = image[y:y + patch_size, x:x + patch_size]
            thinned_patch = thin_image(patch)
            thinned_image[y:y + patch_size, x:x + patch_size] = thinned_patch
            
    return thinned_image