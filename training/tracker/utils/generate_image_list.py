#!/usr/bin/env python3
import os
import sys
import argparse

def generate_image_list(images_dir, output_file):
    """
    Crawls the specified directory recursively using os.walk, filters image files
    whose paths contain 'DCIM', and writes their absolute paths to a text file.
    
    Args:
        images_dir (str): Root directory to start crawling.
        output_file (str): Output text file path to write the list to.
    """
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff")
    matching_paths = []
    
    print(f"Crawling directory: {images_dir}")
    print("Filtering image paths containing 'DCIM'...")
    
    # Traverse the directory recursively
    for root, _, files in os.walk(images_dir):
        for file in files:
            # Check if file has a valid image extension
            if file.lower().endswith(image_extensions):
                full_path = os.path.join(root, file)
                # Filter: Path must contain 'DCIM' (case-sensitive)
                if "DCIM" in full_path:
                    matching_paths.append(os.path.abspath(full_path))
                    
    # Write paths to output file
    if not matching_paths:
        print("Warning: No images containing 'DCIM' in their paths were found.")
    else:
        print(f"Found {len(matching_paths)} matching images. Writing to {output_file}...")
        
    with open(output_file, "w", encoding="utf-8") as f:
        for path in matching_paths:
            f.write(path + "\n")
            
    print("Image list generation completed successfully.")

def main():
    parser = argparse.ArgumentParser(
        description="Recursively crawls a directory using os.walk and creates a text file of image paths containing 'DCIM'."
    )
    parser.add_argument(
        "images_dir", 
        help="Path to the root directory containing the images to crawl."
    )
    parser.add_argument(
        "-o", "--output", 
        default="background_images.txt", 
        help="Path to the output text file where the list of images will be saved (default: background_images.txt)."
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.images_dir):
        print(f"Error: The directory '{args.images_dir}' does not exist or is not a directory.")
        sys.exit(1)
        
    generate_image_list(args.images_dir, args.output)

if __name__ == "__main__":
    main()
