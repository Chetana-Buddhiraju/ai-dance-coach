"""
Resize images in a folder recursively.
Usage:
    python src/preprocessing.py --input_dir /path/to/frames --output_dir /path/to/resized --size 224
"""
import os
from PIL import Image
import argparse

def resize_images(input_dir, output_dir, size=(224,224)):
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for root, _, files in os.walk(input_dir):
        rel = os.path.relpath(root, input_dir)
        out_root = os.path.join(output_dir, rel)
        os.makedirs(out_root, exist_ok=True)
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                in_path = os.path.join(root, f)
                out_path = os.path.join(out_root, f)
                try:
                    img = Image.open(in_path).convert("RGB")
                    img = img.resize(size)
                    img.save(out_path)
                    count += 1
                except Exception as e:
                    print("Error:", in_path, e)
    print(f"Resized {count} images to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()
    resize_images(args.input_dir, args.output_dir, size=(args.size, args.size))
