import cv2
import numpy as np

def close_gaps(image, kernel_size=15, iterations=1):
    """
    Close door and window gaps using elongated morphological operations of certain directions.
    Args:
        image: Input binary image
        kernel_size: Size of the structuring element (default: 5)
        iterations: Number of iterations for morphological operations (default: 1)
    Returns: 
        Image with closed gaps
    """
    # Create horizontal, vertical, and diagonal kernels
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
    diagonal_kernel = np.eye(kernel_size, dtype=np.uint8)
    inverted_diagonal_kernel = np.fliplr(diagonal_kernel)

    # Apply morphological closing with vertical kernel
    for kernel in [vertical_kernel, horizontal_kernel]:
        closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        image = cv2.bitwise_or(image, closed_image)

    return image