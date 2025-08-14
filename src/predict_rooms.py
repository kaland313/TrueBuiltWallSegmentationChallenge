import argparse
from pathlib import Path
import cv2
import torch
from tqdm import tqdm
import numpy as np

from img_proc_utils import (
    watershed_segmentation,
    colorize_regions,
    overlay_rooms,
    overlay_mask,
)
from predict.common import clean_mask_via_morph_ops    

def process_architectural_drawing(input_path, output_dir, image_path=None):
    """
    Process an architectural drawing: remove non-black colors and thin lines.
    
    Args:
        input_path: Path to input PNG file
        output_dir: Directory to save processed images
    """
    try:
        # Read the mask in grayscale
        mask = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"Error: Could not read image {input_path}")
            return False

        if image_path:
            # Read the original image if provided
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Error: Could not read image {image_path}")    
        else:
            image = None
        print(f"Processing {input_path.name}...")
        
        # Get the base filename without extension
        base_name = input_path.stem

        # Clean the mask using morphological operations
        print("   Cleaning mask with morphological operations...")
        cleaned_mask = clean_mask_via_morph_ops(mask)

        # Run watershed segmentation
        print("   Running watershed segmentation...")
        markers, dist = watershed_segmentation(cleaned_mask)
        cv2.imwrite(str(output_dir / f"{base_name}_room_markers.png"), markers*(255/markers.max()))
        if image is not None:
            overlayed_dist = overlay_mask(image, dist*255)
            cv2.imwrite(str(output_dir / f"{base_name}_room_dist.png"), overlayed_dist)
        print("   Colorizing regions...")
        segmented_image = colorize_regions(markers)
        cv2.imwrite(str(output_dir / f"{base_name}_room_colorized.png"), segmented_image)

        # Overlay the segmented image on the original
        print("   Overlaying segmented image on the original...")
        if image is not None:
            overlayed_image = overlay_rooms(image, segmented_image)
            cv2.imwrite(str(output_dir / f"{base_name}_room_overlay.png"), overlayed_image)

        return True
        
    except Exception as e:
        print(f"   Error processing {input_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Process architectural drawings: remove non-black colors and thin lines")
    parser.add_argument("-i", "--input_path", default="./results_wall/",
                        help="Input PNG file or folder containing PNG files")
    parser.add_argument("--file_suffix", default="_segmented.png", help="File suffix to process (default: _segmented.png)")
    parser.add_argument("-o", "--output_folder", default="./results_room/",
                        help="Output folder")
    parser.add_argument("--img", default=None, help="Directory containing original images, only used for visualization (default: None)")
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist.")
        return

    output_folder = Path(args.output_folder)
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    image_folder = None
    if args.img:
        image_folder = Path(args.img)
        if not image_folder.exists():
            print(f"Error: Image folder '{image_folder}' does not exist.")
            image_folder = None
        
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
    for mask_file in png_files:
        image_file = image_folder / mask_file.name.replace(args.file_suffix, ".png") if image_folder else None
        if process_architectural_drawing(
            mask_file, 
            output_folder,
            image_file
        ):
            successful += 1
        print()
    
    print("-" * 60)
    print(f"Processing complete: {successful}/{len(png_files)} files processed successfully")

if __name__ == "__main__":
    main()

# python src/predict_rooms.py -i data/train/masks_wd -o results_room_gt/train --img data/train/images/ --file_suffix=".png"
# python src/predict_rooms.py -i data/val/masks_wd -o results_room_gt/val --img data/val/images/ --file_suffix=".png"


# python src/predict_rooms.py -i results_wall/walls/ -o results_room/walls/ --img data/walls_png/
# python src/predict_rooms.py -i results_wall/rooms/ -o results_room/rooms/ --img data/rooms_png/

# python src/predict_rooms.py -i results_wall/train/ -o results_room/train/ --img data/train/images/
# python src/predict_rooms.py -i results_wall/val/ -o results_room/val/ --img data/val/images/