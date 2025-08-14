import os
import sys
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import argparse
import io


def convert_pdf_to_png(pdf_path, output_dir, dpi=300):
    """
    Convert a PDF file to PNG images with high DPI for architectural drawings.

    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str): Directory to save PNG files
        dpi (int): DPI for output images (default 300)
    """
    try:
        # Open the PDF
        pdf_document = fitz.open(pdf_path)
        pdf_name = Path(pdf_path).stem

        print(f"Converting {pdf_path}...")

        # Process each page
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]

            # Create a transformation matrix for high DPI
            zoom = dpi / 72.0  # 72 is the default DPI
            matrix = fitz.Matrix(zoom, zoom)

            # Render page to pixmap
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)

            # Convert to PIL Image for better quality control
            img_data = pixmap.tobytes("ppm")
            pil_image = Image.open(io.BytesIO(img_data))

            # Save as PNG with high quality
            if pdf_document.page_count == 1:
                output_filename = f"{pdf_name}.png"
            else:
                output_filename = f"{pdf_name}_page_{page_num + 1:03d}.png"

            output_path = os.path.join(output_dir, output_filename)
            pil_image.save(output_path, "PNG", optimize=True, compress_level=1)

            print(f"  Saved: {output_filename}")

        pdf_document.close()
        return True

    except Exception as e:
        print(f"Error converting {pdf_path}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF files to high-quality PNG images"
    )
    parser.add_argument("input_folder", help="Folder containing PDF files")
    parser.add_argument("output_folder", help="Output folder")
    parser.add_argument(
        "-d",
        "--dpi",
        type=int,
        default=300,
        help="DPI for output images (default: 300)",
    )

    args = parser.parse_args()

    input_folder = Path(args.input_folder)

    if not input_folder.exists():
        print(f"Error: Input folder '{input_folder}' does not exist.")
        sys.exit(1)

    # Set output folder
    output_folder = Path(args.output_folder)

    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    # Find all PDF files
    pdf_files = list(input_folder.glob("*.pdf")) + list(input_folder.glob("*.PDF"))

    if not pdf_files:
        print(f"No PDF files found in '{input_folder}'")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF file(s)")
    print(f"Output folder: {output_folder}")
    print(f"DPI: {args.dpi}")
    print("-" * 50)

    # Convert each PDF
    successful = 0
    for pdf_file in pdf_files:
        if convert_pdf_to_png(str(pdf_file), str(output_folder), args.dpi):
            successful += 1

    print("-" * 50)
    print(
        f"Conversion complete: {successful}/{len(pdf_files)} files converted successfully"
    )


if __name__ == "__main__":
    main()
