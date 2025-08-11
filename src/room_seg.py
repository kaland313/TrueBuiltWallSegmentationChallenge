import argparse
from pathlib import Path
import cv2
import torch
from tqdm import tqdm
import numpy as np

from img_proc_utils import (
    skeletonize_in_patches,
    hough_lines,
    line_segment_detector,
    close_gaps
)

def process_architectural_drawing(input_path, output_dir, model_ckpt="model.ckpt", gpu_id=0):
    """
    Process an architectural drawing: remove non-black colors and thin lines.
    
    Args:
        input_path: Path to input PNG file
        output_dir: Directory to save processed images
    """
    try:
        # Read the image in grayscale
        image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Error: Could not read image {input_path}")
            return False
        
        print(f"Processing {input_path.name}...")
        
        # Get the base filename without extension
        base_name = input_path.stem

        # # Skeletonize the image
        # print("   Skeletonizing the image...")
        # thinned_image = skeletonize_in_patches(image)
        # thinned_img_path = output_dir / f"{base_name}_seg_thinned.png"
        # cv2.imwrite(str(thinned_img_path), thinned_image)
        # print(f"    Saved: {thinned_img_path.name}")
        
        # Run hough transform to detect lines
        print("   Detecting lines using Hough Transform...")
        lines, line_image = line_segment_detector(image)
        hough_img_path = output_dir / f"{base_name}_seg_lines.png"
        cv2.imwrite(str(hough_img_path), line_image)
        print(f"    Saved: {hough_img_path.name}")

        # Close gaps in the lines
        print("   Closing gaps in the lines...")
        closed_image = close_gaps(image)
        closed_img_path = output_dir / f"{base_name}_seg_closed.png"
        cv2.imwrite(str(closed_img_path), closed_image)
        print(f"    Saved: {closed_img_path.name}")

        return True
        
    except Exception as e:
        print(f"   Error processing {input_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Process architectural drawings: remove non-black colors and thin lines")
    parser.add_argument("input_path", help="Input PNG file or folder containing PNG files")
    parser.add_argument("--file_suffix", default="_cleaned.png", help="File suffix to process (default: _cleaned.png)")
    parser.add_argument("-o", "--output", help="Output folder (default: data/intermediaries)")
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist.")
        return
    
    # Set output folder
    if args.output:
        output_folder = Path(args.output)
    else:
        output_folder = Path("data/intermediaries")
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Process files
    if input_path.is_file():
        # Single file
        if str(input_path).lower().endswith(args.file_suffix):
            png_files = [input_path]
        else:
            print(f"Error: '{input_path}' is not a supported image file.")
            return
    else:
        # Folder
        png_files = list(input_path.glob("*" + args.file_suffix))

    if not png_files:
        print(f"No files with suffix {args.file_suffix} found in '{input_path}'")
        return
    
    print(f"Found {len(png_files)} image file(s)")
    print(f"Output folder: {output_folder}")
    print("-" * 60)
    
    # Process each image
    successful = 0
    for image_file in png_files:
        if process_architectural_drawing(
            image_file, 
            output_folder, 
        ):
            successful += 1
        print()
    
    print("-" * 60)
    print(f"Processing complete: {successful}/{len(png_files)} files processed successfully")

if __name__ == "__main__":
    main()