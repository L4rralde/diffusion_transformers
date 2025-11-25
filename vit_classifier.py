
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.vit import VisionTransformerClassToken


class ClassifierHead(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(token)
        return logits


class ViTClassifier(VisionTransformerClassToken):
    def __init__(self, n_blocks: int, patch_size: int, n_channels: int, dim: int, num_heads: int, mlp_ratio: int) -> None:
        super().__init__(n_blocks, patch_size, n_channels, dim, num_heads, mlp_ratio)
        self.classifier_head = ClassifierHead(dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        preds = super().forward(image)
        cls_token = preds['cls_token']

        logits = self.classifier_head(cls_token)
        return logits


class MlpClassifier(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        assert len(image.shape) == 4, "Expected an image of shape B x C x H x W"
        x = image.view(image.size(0), -1)
        logits = self.mlp(x)

        return logits


def to_categorical(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    eye = torch.eye(num_classes).to(y.device)
    return eye[y]


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    #elif torch.backends.mps.is_available():
    #    device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = ViTClassifier(
        n_blocks=12,
        patch_size=8,
        n_channels=3,
        dim=384,
        num_heads=6,
        mlp_ratio=2
    ) #ViT-S
    #model = MlpClassifier(3 * 32 * 32)
    model = model.to(device)

    torch.autograd.set_detect_anomaly(True)
    optim = torch.optim.Adam(model.parameters())
    loss_fn = nn.modules.loss.BCEWithLogitsLoss()

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

    for epoch in tqdm(range(1, 10)):
        total_loss = 0
        optim.zero_grad()
        for x, y in tqdm(dataloader):
            x = x.to(device)
            y = y.to(device)
            y = to_categorical(y, 10)
            logits = model(x)

            loss = loss_fn(logits, y)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        
        tqdm.write(f"Epoch {epoch}. Training loss: {total_loss}")


if __name__ == '__main__':
    main()
