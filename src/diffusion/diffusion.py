import torch
import torch.nn as nn


class DiffusionSchedule:
    def __init__(self,
                 num_steps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02,
                 device: torch.device = torch.device("cpu")):
        self.num_steps = num_steps
        betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(
            low=0,
            high=self.num_steps,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
        """
        # t: (B,)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)  # (B,1,1,1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise

    def get_step_params(self, t: torch.Tensor):
        """
        Devuelve alpha_t, alpha_bar_t, beta_t para un batch de timesteps t.
        t: (B,) en {0,...,T-1}
        """
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        return alpha_t, alpha_bar_t, beta_t


@torch.no_grad()
def p_sample_step(model: nn.Module,
                  x_t: torch.Tensor,
                  t: torch.Tensor,
                  y: torch.Tensor,
                  schedule: DiffusionSchedule,
                  cfg_scale: float,
                  num_classes: int) -> torch.Tensor:
    """
    Un paso de muestreo p(x_{t-1} | x_t) con DDPM y CFG.

    x_t: (B, 4, H/8, W/8)
    t: (B,)
    y: (B,) labels condicionales en [0, num_classes-1]
    """

    device = x_t.device
    B = x_t.shape[0]

    # Classifier-free guidance: cond + uncond en un solo forward.
    if cfg_scale is not None and cfg_scale != 1.0:
        # etiquetas condicionales
        y_cond = y
        # etiquetas "null" (el último índice es el null token)
        y_null = torch.full_like(y, num_classes, device=device)
        x_in = torch.cat([x_t, x_t], dim=0)
        t_in = torch.cat([t, t], dim=0)
        y_in = torch.cat([y_cond, y_null], dim=0)

        eps = model(x_in, t_in, y_in)  # (2B, 4, H/8, W/8)
        eps_cond, eps_uncond = eps.chunk(2, dim=0)
        eps_theta = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
    else:
        eps_theta = model(x_t, t, y)

    alpha_t, alpha_bar_t, beta_t = schedule.get_step_params(t)

    # Ecuación DDPM para la media de p_theta(x_{t-1} | x_t)
    # mu_t = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps_theta)
    mu_t = (1.0 / torch.sqrt(alpha_t)) * (
        x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_theta
    )

    # Variancia simple: beta_t
    if t.min() > 0:
        noise = torch.randn_like(x_t)
        x_prev = mu_t + torch.sqrt(beta_t) * noise
    else:
        # en t=0 no se añade ruido
        x_prev = mu_t

    return x_prev
