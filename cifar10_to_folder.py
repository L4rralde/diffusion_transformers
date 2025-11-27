import os
import argparse

import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm


classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Resolución de muestreo (64 o 128, múltiplo de 8).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directorio para CIFAR-10.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="cifar10_samples",
        help="Directorio donde guardar las imágenes generadas.",
    )
    args = parser.parse_args()
    return args


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    transform = transforms.Lambda(
        lambda pil_image: center_crop_arr(pil_image, args.image_size)
    )

    dataset = torchvision.datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=transform,
    )

    for i, (image, y) in tqdm(enumerate(dataset)):
        save_path = os.path.join(
            args.out_dir,
            f"sample_{i:05d}_{classes[y]}.png"
        )
        image.save(save_path)


if __name__ == '__main__':
    main()
