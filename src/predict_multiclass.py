import argparse
from pathlib import Path
import cv2
import torch
from tqdm import tqdm
import numpy as np

from model_multiclass import SegmentationModel

def predict(image, model):
    """Run prediction on a single image using the trained model.
    Args:
        image: Input image as a numpy array
        model: Trained segmentation model
    Returns:
        Predicted mask as a numpy array
    """
    if len(image.shape) == 2:  # If grayscale, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Ensure image is in RGB format
    image = image.transpose(2, 0, 1)  # Change to CxHxW format
    image = image / 255.0  # Normalize to [0, 1]

    # Pad to next multiple of 32
    h, w = image.shape[1:3]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    with torch.no_grad():
        # Preprocess the image
        input_tensor = torch.from_numpy(image).unsqueeze(0).float().to(model.device)  
        pred = model(input_tensor)
        # pred_mask = torch.sigmoid(pred).squeeze().cpu().numpy()
        print(f"Pred shape: {pred.shape}")
        pred_mask = torch.argmax(pred.squeeze(), dim=0).cpu().numpy().astype('uint8')*100
        # pred_mask = (pred_mask > 0.5).astype('uint8') * 255
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            pred_mask = pred_mask[:h, :w]
        return pred_mask

def load_model(model_ckpt="model.ckpt", gpu_id=0):
    """Load the trained segmentation model from checkpoint.
    Args:
        model_ckpt: Path to the model checkpoint
        gpu_id: GPU ID to use for inference
    Returns:
        Loaded model
    """
    model = SegmentationModel.load_from_checkpoint(model_ckpt)
    model.eval()
    if torch.cuda.is_available() and gpu_id >= 0:
        device = torch.device(f'cuda:{gpu_id}')
        model.to(device)
        print(f"Using GPU: {gpu_id}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return model

    

def predict_in_patches(image, model, patch_size=512):
    """
    Run prediction on an image in patches to avoid memory issues and see progress.
    Args:
        image: Input image
        model: Trained segmentation model
        patch_size: Size of each patch (default: 512)
    Returns:
        Predicted mask
    """
    
    h, w = image.shape[0:2]
    predicted_mask = np.zeros((h, w), dtype=np.uint8)

    for y in tqdm(range(0, h, patch_size), leave=False):
        for x in tqdm(range(0, w, patch_size), leave=False):
            patch = image[y:y + patch_size, x:x + patch_size]
            pred_patch = predict(patch, model)
            predicted_mask[y:y + patch_size, x:x + patch_size] = pred_patch

    return predicted_mask

def clean_mask_via_morph_ops(mask, kernel_size=10):
    """
    Apply morphological closing followed by opening to clean the mask.
    
    Args:
        mask: Binary mask image
        kernel_size: Size of the structuring element
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)
    return cleaned_mask

def overlay_mask(image, mask, alpha=0.5):
    """
    Overlay the mask on the original image.
    
    Args:
        image: Original image
        mask: Binary mask
    Returns:
        Image with mask overlay
    """
    # a nice blue color 
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Ensure image is in BGR format
    mask_color = (2, 118, 50) # Green color for the mask
    mask_colored = np.zeros_like(image)
    mask_colored[mask > 0] = mask_color
    overlay = cv2.addWeighted(image.astype(np.float32), 
                              1 - alpha,
                              mask_colored.astype(np.float32),
                              alpha,
                              0)
    overlay[mask == 0] = image[mask == 0]  # Keep original pixels where mask is not present
 
    return overlay.astype(np.uint8)

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

        # Segment the image using the cnn model
        print("   Running CNN model for segmentation...")
        model = load_model(model_ckpt, gpu_id)
        predicted_mask = predict_in_patches(image, model)
        segmented_img_path = output_dir / f"{base_name}_step1_segmented.png"
        cv2.imwrite(str(segmented_img_path), predicted_mask)
        print(f"    Saved: {segmented_img_path.name}")

        # Clean the mask using morphological operations
        print("   Cleaning mask with morphological operations...")
        cleaned_mask = clean_mask_via_morph_ops(predicted_mask)
        cleaned_img_path = output_dir / f"{base_name}_step2_cleaned.png"
        cv2.imwrite(str(cleaned_img_path), cleaned_mask)
        print(f"    Saved: {cleaned_img_path.name}")

        # Overlay the initial and cleaned masks on the original image
        print("   Overlaying masks on the original image...")
        overlayed_initial = overlay_mask(image, predicted_mask)
        overlayed_initial_path = output_dir / f"{base_name}_overlay_initial.png"
        cv2.imwrite(str(overlayed_initial_path), overlayed_initial)
        print(f"    Saved: {overlayed_initial_path.name}")
        overlayed_cleaned = overlay_mask(image, cleaned_mask)
        overlayed_cleaned_path = output_dir / f"{base_name}_overlay_cleaned.png"
        cv2.imwrite(str(overlayed_cleaned_path), overlayed_cleaned)
        print(f"    Saved: {overlayed_cleaned_path.name}")

        return True
        
    except Exception as e:
        print(f"   Error processing {input_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Process architectural drawings: remove non-black colors and thin lines")
    parser.add_argument("input_path", help="Input PNG file or folder containing PNG files")
    parser.add_argument("-o", "--output", help="Output folder (default: data/intermediaries)")
    parser.add_argument("-g", "--gpu_id", type=int, default=0,
                        help="GPU ID to use for inference (default: 0)")
    parser.add_argument("-m", "--model_ckpt", default="model.ckpt",
                        help="Path to the model checkpoint (default: model.ckpt)")
    
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