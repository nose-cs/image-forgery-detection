import os
from PIL import Image
import shutil

PATH = 'data/CASIA2'

old_tampered_dir = os.path.join(PATH, 'Tp')
new_tampered_dir = os.path.join(PATH, 'Tp2')


def process_images(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        # Check if it's a file
        if os.path.isfile(input_path):
            # If it's a .tif file, convert it to .jpeg
            if filename.lower().endswith('.tif') or filename.lower().endswith('.tiff'):
                # Change the extension to .jpeg
                output_filename = os.path.splitext(filename)[0] + '.jpeg'
                output_path = os.path.join(output_folder, output_filename)
                try:
                    with Image.open(input_path) as img:
                        # Convert and save the image as .jpeg
                        img.convert("RGB").save(output_path, "JPEG")
                    print(f"Converted: {filename} -> {output_filename}")
                except Exception as e:
                    print(f"Error converting {filename}: {str(e)}")
            # If it's not a .tif file, simply copy it
            else:
                output_path = os.path.join(output_folder, filename)
                try:
                    shutil.copy2(input_path, output_path)
                    print(f"Copied: {filename}")
                except Exception as e:
                    print(f"Error copying {filename}: {str(e)}")


def main():
    process_images(old_tampered_dir, new_tampered_dir)


if __name__ == "__main__":
    main()
