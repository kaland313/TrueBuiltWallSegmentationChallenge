import argparse
from pathlib import Path
import cv2
import torch
from tqdm import tqdm
import numpy as np

from predict.torch import load_model, predict
from predict.common import (
    predict_in_patches,
    clean_mask_via_morph_ops
)
from img_proc_utils import overlay_mask

def process_architectural_drawing(input_path, output_dir, model_ckpt="model.ckpt", gpu_id=0):
    """
    Process an architectural drawing.
    
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

        # # Run edge detection on the image and add edges to the image
        # print("   Running edge detection...")
        # edges = cv2.Canny(image, 100, 200)
        # # Thicken edges for better visibility
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # edges = cv2.dilate(edges, kernel, iterations=1)
        # # Draw edges on the original image
        # image = 255-edges
        # # image[edges > 0] = 0
        # cv2.imwrite(str(output_dir / f"{base_name}_edges.png"), image)

        # Segment the image using the cnn model
        print("   Running CNN model for segmentation...")
        model = load_model(model_ckpt, gpu_id)
        predicted_mask = predict_in_patches(image, model, pred_fn=predict)
        cv2.imwrite(str(output_dir / f"{base_name}_segmented.png"), predicted_mask*100)

        # Clean the mask using morphological operations
        print("   Cleaning mask with morphological operations...")
        cleaned_mask = clean_mask_via_morph_ops(predicted_mask)
        cleaned_mask = (cleaned_mask * (255//cleaned_mask.max())).astype(np.uint8)
        cv2.imwrite(str(output_dir / f"{base_name}_cleaned.png"), cleaned_mask)

        # Overlay the initial and cleaned masks on the original image
        print("   Overlaying masks on the original image...")
        overlayed_initial = overlay_mask(image, predicted_mask)
        cv2.imwrite(str(output_dir / f"{base_name}_segmented_overlay.png"), overlayed_initial)
        overlayed_cleaned = overlay_mask(image, cleaned_mask)
        cv2.imwrite(str(output_dir / f"{base_name}_cleaned_overlay.png"), overlayed_cleaned)

        return True
        
    except Exception as e:
        print(f"   Error processing {input_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Process architectural drawings: remove non-black colors and thin lines")
    parser.add_argument("input_path", 
                        help="Input PNG file or folder containing PNG files")
    parser.add_argument("-o", "--output", default="./results_wall/",
                        help="Output folder (default: ./resuts_wall/)")
    parser.add_argument("-g", "--gpu_id", type=int, default=0,
                        help="GPU ID to use for inference (default: 0)")
    parser.add_argument("-m", "--model_ckpt", default="model_wd.ckpt",
                        help="Path to the model checkpoint (default: model.ckpt)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist.")
        return
    
    # Set output folder
    output_folder = Path(args.output)
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Process files
    if input_path.is_file():
        # Single file
        if input_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            png_files = [input_path]
        else:
            print(f"Error: '{input_path}' is not a supported image file.")
            return
    else:
        # Folder
        png_files = list(input_path.glob("*.png")) + list(input_path.glob("*.PNG"))
        png_files += list(input_path.glob("*.jpg")) + list(input_path.glob("*.JPG"))
        png_files += list(input_path.glob("*.jpeg")) + list(input_path.glob("*.JPEG"))
    
    if not png_files:
        print(f"No image files found in '{input_path}'")
        return
    
    print(f"Found {len(png_files)} image file(s)")
    print(f"Output folder: {output_folder}")
    print(f"GPU ID: {args.gpu_id}")
    print("-" * 60)
    
    # Process each image
    successful = 0
    for image_file in png_files:
        if process_architectural_drawing(
            image_file, 
            output_folder, 
            model_ckpt=args.model_ckpt, 
            gpu_id=args.gpu_id
        ):
            successful += 1
        print()
    
    print("-" * 60)
    print(f"Processing complete: {successful}/{len(png_files)} files processed successfully")

if __name__ == "__main__":
    main()

# python src/predict.py data/walls_png/ -o results_wall/walls -m model_wd_aug_fill.ckpt 
# python src/predict.py data/rooms_png/ -o results_wall/rooms
# python src/predict.py data/train/images/ -o results_wall/train -m model_wd_aug_fill.ckpt 
# python src/predict.py data/val/images/ -o results_wall/val -m model_wd_aug_fill.ckpt 