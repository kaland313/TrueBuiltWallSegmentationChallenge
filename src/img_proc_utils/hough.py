import cv2
import numpy as np

def hough_lines(image, threshold=150, min_line_length=100, max_line_gap=100):
    """
    Detect lines in a binary image using Hough Transform.
    
    Args:
        image: Input binary image
        threshold: Minimum number of votes to consider a line (default: 50)
        min_line_length: Minimum length of a line to be detected (default: 100)
        max_line_gap: Maximum gap between segments to link them into a single line (default: 10)
    Returns:
        List of detected lines
    """
    lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    if lines is not None:
        print(f"  Detected {len(lines)} lines")
        # Plot the lines on a copy of the image
        line_image = np.zeros_like(image)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 1)
        return lines, line_image
    else:
        print("  No lines detected")
        return [], np.zeros_like(image)

def line_segment_detector(image):
    """
    Detect line segments in a grayscale image.
    
    Args:
        image: Input grayscale image
    
    Returns:
        List of detected line segments
    """
    # lsd = cv2.LineSegmentDetector('Refine','Standard')
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(image)[0]
    drawnLines1 = lsd.drawSegments(image, lines)
    return lines, drawnLines1