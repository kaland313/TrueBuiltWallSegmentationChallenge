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
    # Normalize the distance image for range = {0.0, 1.0}
    # so we can visualize and threshold it
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

    # Threshold to obtain the peaks
    # This will be the markers for the foreground objects
    _, dist = cv2.threshold(dist, 0.01, 1.0, cv2.THRESH_BINARY)
    # Dilate a bit the dist image
    # kernel1 = np.ones((3,3), dtype=np.uint8)
    # dist = cv2.dilate(dist, kernel1)

    # Find total markers
    # contours, _ = cv2.findContours(dist.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # Create the marker image for the watershed algorithm
    # markers = np.zeros(dist.shape, dtype=np.int32)
    # # Draw the foreground markers
    # for i in range(len(contours)):
    #     cv2.drawContours(markers, contours, i, (i+2), -1)
    # # Draw the background marker
    # # cv2.circle(markers, (5,5), 3, (1,1,1), -1)

    dist_uint8 = (dist * 255).astype(np.uint8) 
    num_labels, markers = cv2.connectedComponents(dist_uint8, connectivity=8)
    
    # Watershed expects 8-bit 3-channel image
    mask_8UC3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # Perform the watershed algorithm
    cv2.watershed(mask_8UC3, markers)
    #mark = np.zeros(markers.shape, dtype=np.uint8)
    mark = markers.astype('uint8')
    # Set walls to background
    mark[mask==0] = 0

    return markers, dist


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
    colors = [tuple(np.random.randint(0, 256, size=3)) for _ in range(num_labels)]
    
    # Create an output image
    colorized_image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    # Colorize each region
    for label in trange(1, num_labels, desc="   Coloring labels"):
        colorized_image[mask == label] = colors[label]
    
    return colorized_image