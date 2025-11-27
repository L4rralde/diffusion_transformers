import os
import re

import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image


def find_one_file_by_regex(directory, pattern):
    # Compile the regex pattern for better performance
    regex = re.compile(pattern)
    
    # Traverse the directory and subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            # If the file matches the regex pattern
            if regex.match(file):
                # Return the full path of the first matching file
                return os.path.join(root, file)
    
    # If no match is found
    return None


def create_npz_from_sample_folder(sample_dir: os.PathLike, num: int=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        pattern = f"sample_{i:05d}_.*.png"
        img_path = find_one_file_by_regex(sample_dir, pattern)
        sample_pil = Image.open(img_path)
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}/samples.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-samples",
        type=int,
        default=64, #5_000
        help="Número total de imágenes a generar.",
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        required=True,
        help="Path del folder de imágenes"
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    create_npz_from_sample_folder(
        sample_dir=args.image_folder,
        num=args.num_samples
    )


if __name__ == '__main__':
    main()
