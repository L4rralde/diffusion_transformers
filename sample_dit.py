import os
from pathlib import Path
import argparse

import torch
import torch.nn as nn
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL

from src.diffusion import DiffusionSchedule, p_sample_step
from src.dit import create_dit


classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


@torch.no_grad()
def sample_images(model: nn.Module,
                  vae: AutoencoderKL,
                  schedule: DiffusionSchedule,
                  num_samples: int,
                  batch_size: int,
                  image_size: int,
                  cfg_scale: float,
                  num_classes: int,
                  out_dir: str):
    device = next(model.parameters()).device
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    latent_scaling = 0.18215
    T = schedule.num_steps

    samples_saved = 0
    sample_id = 0

    model.eval()
    vae.eval()

    while samples_saved < num_samples:
        current_batch = min(batch_size, num_samples - samples_saved)

        # latentes iniciales N(0, I)
        latent_H = image_size // 8
        latent_W = image_size // 8
        x_t = torch.randn(
            current_batch,
            4,
            latent_H,
            latent_W,
            device=device,
        )

        # muestreamos etiquetas de clase uniformemente de CIFAR-10 (0..9)
        y = torch.randint(
            low=0,
            high=num_classes,
            size=(current_batch,),
            device=device,
            dtype=torch.long,
        )

        # cadena de muestreo
        for t_step in reversed(range(T)):
            t = torch.full(
                (current_batch,),
                t_step,
                device=device,
                dtype=torch.long,
            )
            x_t = p_sample_step(
                model=model,
                x_t=x_t,
                t=t,
                y=y,
                schedule=schedule,
                cfg_scale=cfg_scale,
                num_classes=num_classes,
            )

        # decodificar con VAE
        latents = x_t / latent_scaling
        imgs = vae.decode(latents).sample  # (B, 3, H, W)
        imgs = imgs.clamp(-1.0, 1.0)
        imgs = (imgs + 1.0) / 2.0  # [0,1]

        # guardar imágenes individuales
        for i in range(current_batch):
            img = imgs[i]
            save_path = os.path.join(out_dir, f"sample_{sample_id:05d}_{classes[y[i]]}.png")
            save_image(img, save_path)
            sample_id += 1

        samples_saved += current_batch
        print(f"Generadas {samples_saved}/{num_samples} imágenes")


# -------------------------------------------------------------------------
# main / argumentos
# -------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sampling de DiT entrenado en CIFAR-10 con VAE pre-entrenada."
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Ruta al checkpoint (idealmente *_ema.pt).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["adaln", "cross", "incontext"],
        default="adaln",
        help="Variante de DiT (adaln, cross, incontext).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Resolución de muestreo (64 o 128, múltiplo de 8).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="Número total de imágenes a generar.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Tamaño de batch para la cadena de muestreo.",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=3.0,
        help="Escala de classifier-free guidance (1.0 = sin CFG).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./samples_dit_cifar",
        help="Directorio donde guardar las imágenes generadas.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Dispositivo ('cuda' o 'cpu').",
    )
    parser.add_argument(
        "--num-diffusion-steps",
        type=int,
        default=1000,
        help="Número de pasos de difusión (T). Debe coincidir con entrenamiento.",
    )
    parser.add_argument(
        "--beta-start",
        type=float,
        default=1e-4,
        help="Beta inicial (igual que en entrenamiento).",
    )
    parser.add_argument(
        "--beta-end",
        type=float,
        default=0.02,
        help="Beta final (igual que en entrenamiento).",
    )
    parser.add_argument(
        "--vae-name",
        type=str,
        default="ema",  # 'mse' o 'ema'
        help="Nombre del VAE SD: 'mse' o 'ema'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla manual de torch. Si no se usa, es 'aleatorio'"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_classes = 10  # CIFAR-10
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # 1) Modelo
    print("Creando modelo DiT...")
    model = create_dit(
        model_type=args.model_type,
        image_size=args.image_size,
        num_classes=num_classes,
        device=device,
    )

    print(f"Cargando checkpoint desde {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt["model"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 2) VAE
    print(f"Cargando VAE pre-entrenada (sd-vae-ft-{args.vae_name})...")
    vae = AutoencoderKL.from_pretrained(
        f"stabilityai/sd-vae-ft-{args.vae_name}"
    ).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # 3) Schedule
    schedule = DiffusionSchedule(
        num_steps=args.num_diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )

    # 4) Sampling
    print("Iniciando muestreo...")
    sample_images(
        model=model,
        vae=vae,
        schedule=schedule,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        image_size=args.image_size,
        cfg_scale=args.cfg_scale,
        num_classes=num_classes,
        out_dir=args.out_dir,
    )
    print("Muestreo completado.")


if __name__ == "__main__":
    main()
