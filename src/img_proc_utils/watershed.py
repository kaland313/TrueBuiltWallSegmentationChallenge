# Copyright (c) 2025 András Kalapos
# Licensed under the MIT License. See LICENSE file in the project root for details.

import cv2
import numpy as np


def watershed_segmentation(mask):
    """Apply watershed segmentation to a binary mask.
    Based on: https://docs.opencv2.org/3.4/d2/dbd/tutorial_distance_transform.html
    Args:
        mask: Binary mask image
    Returns:
        markers: Segmented image as watershed markers
        dist: Distance transform of the mask (for development purposes)
    """
    # Binarize the mask if not already binary
    mask = (mask > 0).astype(np.uint8) * 255
    # Invert the mask to treat walls as background that separates watershed regions
    mask = 255 - mask

    # Perform the distance transform algorithm
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    # Normalize the distance image by subtracting the minimum value
    dist = dist - dist.min()
    # Threshold the image to obtain "peaks", these will be the markers for the foreground objects
    # The arbitrary threshold value of 32 is chosen empirically
    # Adaptive thresholding doesn't work well, because there are very large and small regions
    dist_threshold = dist > 32

    # Identify connected components in the thresholded distance image
    dist_uint8 = (dist_threshold * 255).astype(np.uint8)
    num_labels, markers, stats, _ = cv2.connectedComponentsWithStats(
        dist_uint8, connectivity=4, ltype=cv2.CV_32S, stats=cv2.CC_STAT_AREA
    )

    # Apply a higher distance threshold to the largest component(s)
    # This is to help differentiate between the large outside areas and rooms adjecent to them. The sample plans have 1-2 such large areas.
    # Hacky, but works well for the current dataset.
    areas = stats[1:, cv2.CC_STAT_AREA]
    top_areas = np.argsort(areas)[-2:]  # Get indices of the top 10 largest components
    # Drop if the area is less than 1000000 pixels
    top_areas = top_areas[areas[top_areas] > 1_000_000]
    if len(top_areas) > 0:
        # add 1 to the indices, because the markers array starts from 1
        top_areas += 1
        top_areas_mask = np.isin(markers, top_areas)
        dist[~top_areas_mask] = 0
        dist_threshold_large = dist > 100
        # Merge the two distance maps
        dist_threshold[top_areas_mask] = dist_threshold_large[top_areas_mask]

        dist_uint8 = (dist_threshold * 255).astype(np.uint8)
        num_labels, markers = cv2.connectedComponents(
            dist_uint8, connectivity=4, ltype=cv2.CV_32S
        )

    dist = dist_threshold

    # Watershed expects 8-bit 3-channel image
    mask_8UC3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # Perform the watershed algorithm
    cv2.watershed(mask_8UC3, markers)
    # Set watershed boundaries to background
    markers[markers == -1] = 0

    # Set walls to background
    markers[mask == 0] = 0

    # Compare the four corners of the image, if all belong to the same region, set them to background. Look at offset = 3 pixels from the exact corners.
    offset = 3
    h, w = markers.shape
    corners = [
        markers[offset, offset],
        markers[offset, w - offset - 1],
        markers[h - offset - 1, offset],
        markers[h - offset - 1, w - offset - 1],
    ]
    if len(set(corners)) == 1:
        # If all corners belong to the same region, set them to background
        markers[markers == corners[0]] = 0

    return markers, dist
