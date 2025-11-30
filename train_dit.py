
import os
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
import numpy as np

from src.dit import create_dit
from src.diffusion import DiffusionSchedule


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.collected = {}
        self.register(model)

    def register(self, model: nn.Module):
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        decay = self.decay
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow
            new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
            self.shadow[name] = new_average.clone()

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow
            param.data.copy_(self.shadow[name])


def get_cifar10_dataloader(data_dir: str,
                           image_size: int,
                           batch_size: int,
                           num_workers: int = 4) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        #transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5, inplace=True),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader


def train(args) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    # 1) Data
    train_loader = get_cifar10_dataloader(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 2) VAE pre-entrenada de Stable Diffusion
    print(f"Cargando VAE pre-entrenada ({args.vae_name})...")
    vae = AutoencoderKL.from_pretrained(
        f"stabilityai/sd-vae-ft-{args.vae_name}"
    ).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # factor de escala usado en Stable Diffusion para el espacio latente
    latent_scaling = 0.18215

    # 3) Modelo DiT
    print(f"Creando modelo DiT ({args.model_type})...")
    model = create_dit(
        model_type=args.model_type,
        image_size=args.image_size,
        num_classes=10,  # CIFAR-10
        patch_size=args.patch_size,
        device=device,
    )

    # 4) EMA
    ema = EMA(model, decay=args.ema_decay)

    # 5) Diffusion schedule
    schedule = DiffusionSchedule(
        num_steps=args.num_diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )

    # 6) Optimizador
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0,
    )

    # 7) Loop de entrenamiento
    global_step = 0
    total_steps = args.epochs * len(train_loader)
    print(f"Comenzando entrenamiento por {args.epochs} epochs "
          f"({total_steps} iteraciones)...")

    train_stats = defaultdict(list)

    model.train()
    for epoch in range(args.epochs):
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 1) Codificar imágenes a latente con VAE (sin gradiente)
            with torch.no_grad():
                posterior = vae.encode(images)
                latents = posterior.latent_dist.sample() * latent_scaling
                # latents: (B, 4, H/8, W/8)

            # 2) Muestrear timesteps y ruido
            noise = torch.randn_like(latents)
            t = schedule.sample_timesteps(
                batch_size=latents.shape[0],
                device=device,
            )  # (B,). Beta numbers from 0 to num_time_steps=1000

            #Diffusion step. x_t = \sqrt{\bar \alpha_t x_0} + noise * \sqrt{1 - \bar \alpha_t }
            x_t = schedule.q_sample(latents, t, noise)

            # 3) Forward del modelo: predice ruido
            eps_pred = model(x_t, t, labels) #eps_\theta (x_t)

            # 4) Loss MSE entre ruido real y predicho
            loss = F.mse_loss(eps_pred, noise) # L = mean[ (e_t - eps_\theta (x_t))^2 ]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            ema.update(model)

            global_step += 1

            if global_step % args.log_every == 0:
                print(
                    f"[Epoch {epoch+1}/{args.epochs}] "
                    f"Step {global_step}/{total_steps} "
                    f"Loss: {loss.item():.4f}"
                )
                train_stats['epoch'].append(epoch)
                train_stats['step'].append(global_step)
                train_stats['loss'].append(loss.item())

        if epoch % 10 != 9:
            continue

        # Guardar checkpoint al final de cada 10 epochs
        ckpt_path = os.path.join(
            args.results_dir,
            f"dit_{args.model_type}_epoch{epoch+1}.pt",
        )
        ema_ckpt_path = os.path.join(
            args.results_dir,
            f"dit_{args.model_type}_epoch{epoch+1}_ema.pt",
        )

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step,
            },
            ckpt_path,
        )

        last_ckpt_path = os.path.join(args.results_dir, 'dit_last.pt')
        try:
            os.remove(last_ckpt_path)
        except OSError:
            pass
        os.symlink(os.path.realpath(ckpt_path), last_ckpt_path)

        # Guardar versión EMA de los pesos
        ema_model = create_dit(
            model_type=args.model_type,
            image_size=args.image_size,
            num_classes=10,
            patch_size=args.patch_size,
            device=device,
        )
        ema.copy_to(ema_model)
        torch.save(
            {
                "model": ema_model.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step,
            },
            ema_ckpt_path,
        )

        last_ema_path = os.path.join(args.results_dir, 'dit_last_ema.pt')
        try:
            os.remove(last_ema_path)
        except OSError:
            pass
        os.symlink(os.path.realpath(ema_ckpt_path), last_ema_path)

        del ema_model
        torch.cuda.empty_cache()
        print(f"Guardados checkpoints en {ckpt_path} y {ema_ckpt_path}")

    print("Entrenamiento finalizado.")
    stats_path = os.path.join(
        args.results_dir,
        f"dit_{args.model_type}_training_stats.npz"
    )
    print(f"Guardando estadísticas de entrenamiento en: {stats_path}")
    np.savez(stats_path, **train_stats)
    return train_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento de DiT en CIFAR-10 con VAE pre-entrenada."
    )

    parser.add_argument(
        "--model-type",
        type=str,
        choices=["adaln", "cross", "incontext"],
        default="adaln",
        help="Variante de DiT a usar.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Resolución de entrenamiento (64 o 128, múltiplo de 8).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Tamaño de batch.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Número de epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4, #Constante en el paper
        help="Learning rate del AdamW.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directorio para CIFAR-10.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results_dit",
        help="Directorio donde guardar checkpoints.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Dispositivo ('cuda' o 'cpu').",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Workers del DataLoader.",
    )
    parser.add_argument(
        "--num-diffusion-steps",
        type=int,
        default=1000,
        help="Número de pasos de difusión (T).",
    )
    parser.add_argument(
        "--beta-start",
        type=float,
        default=1e-4,
        help="Valor inicial de beta.",
    )
    parser.add_argument(
        "--beta-end",
        type=float,
        default=0.02,
        help="Valor final de beta.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.9999,
        help="Factor de decaimiento de EMA.",
    )
    parser.add_argument(
        "--vae-name",
        type=str,
        choices=['ema', 'mse'],
        default="ema",  # 'mse' o 'ema'
        help="Nombre del VAE SD: 'mse' o 'ema'.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Frecuencia de logging en pasos.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Máximo L2 de gradiente (clipping).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=2,
        help="Tamano de parche. Por default 2 (2x2)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
