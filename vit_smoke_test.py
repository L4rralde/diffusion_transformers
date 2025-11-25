
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.vit import VisionTransformer


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    vit = VisionTransformer(
        n_blocks=12,
        patch_size=8,
        n_channels=3,
        dim=384,
        num_heads=6,
        mlp_ratio=2
    ) #ViT-S
    vit = vit.to(device)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=128,
        pin_memory=True,
        drop_last=True
    )

    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x = x.to(device)
            y = y.to(device)
            tokens = vit(x)

if __name__ == '__main__':
    main()
