
import torch
from diffusers.models import AutoencoderKL
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5], inplace=True)
    ])

    dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=128,
        pin_memory=True,
        drop_last=True
    )

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)

    for x, y in tqdm(dataloader):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            x = vae.encode(x).latent_dist.sample()

    #takes 3 minutes on mac m1 (same for cpu and mps)


if __name__ == '__main__':
    main()
