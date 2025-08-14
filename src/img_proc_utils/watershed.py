import cv2
import numpy as np
import random
from tqdm import trange


def watershed_segmentation(mask):
    """Apply watershed segmentation to a binary mask.
    Based on: https://docs.opencv2.org/3.4/d2/dbd/tutorial_distance_transform.html    
    Args:
        mask: Binary mask image
    Returns:
        Segmented image
    """
    # Binarize the mask if not already binary
    mask = (mask > 0).astype(np.uint8) * 255
    # Invert the mask to treat walls as background that separates watershed regions
    mask = 255 - mask

    # Perform the distance transform algorithm
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    # Normalize the distance image by subtracting the minimum value 
    dist = dist - dist.min()
    # Threshold the image to obtain "peaks"
    # This will be the markers for the foreground objects
    # The arbitrary threshold value of 32 is chosen empirically
    # Adaptive thresholding doesn't work well, because there are very large and small regions
    dist_tresh_large = dist > 32

    # Identify connected components in the thresholded distance image
    dist_uint8 = (dist_tresh_large * 255).astype(np.uint8) 
    num_labels, markers = cv2.connectedComponents(dist_uint8, connectivity=4, ltype=cv2.CV_32S)

    # Threshold using a smaller value to readd some of the smaller regions
    dist_tresh_small = dist > 32
    dist_uint8_small = (dist_tresh_small * 255).astype(np.uint8)
    num_labels_small, markers_small = cv2.connectedComponents(dist_uint8_small, connectivity=4, ltype=cv2.CV_32S)
    # Add small regions to the markers if they don't overlap with any pixels from dist_tresh_large
    for label in trange(1, num_labels_small):
        small_region_mask = markers_small == label
        if not np.any(small_region_mask & dist_tresh_large):
            markers[small_region_mask] = num_labels + 1
            num_labels += 1

    dist = markers > 0
    
    # Watershed expects 8-bit 3-channel image
    mask_8UC3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # Perform the watershed algorithm
    cv2.watershed(mask_8UC3, markers)
    #mark = np.zeros(markers.shape, dtype=np.uint8)
    # Set walls to background
    markers[mask==0] = 0

    # Compare the four corners of the image, if all belong to the same region, set them to background. Look at offset = 3 pixels from the exact corners.
    offset = 3
    h, w = markers.shape
    corners = [
        markers[offset, offset],
        markers[offset, w - offset - 1],
        markers[h - offset - 1, offset],
        markers[h - offset - 1, w - offset - 1]
    ]
    if len(set(corners)) == 1:
        # If all corners belong to the same region, set them to background
        markers[markers == corners[0]] = 0

    return markers, dist