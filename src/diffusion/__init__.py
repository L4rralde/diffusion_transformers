import numpy as np

from .diffusion import DiffusionSchedule, p_sample_step
from .respace import SpacedDiffusion, space_timesteps
from .gaussian_diffusion import ModelMeanType, ModelVarType, LossType


def create_diffusion(diffusion_steps: int = 1000):
    timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas = get_beta_schedule(diffusion_steps), #1000 steps from 1e/4 to 0.02
        model_mean_type = ModelMeanType.EPSILON,
        model_var_type = ModelVarType.LEARNED,
        loss_type = LossType.MSE
    )


def get_beta_schedule(num_steps: int) -> np.ndarray:
    scale = 1000/num_steps
    return np.linspace(
        1e-4 * scale,
        0.02 * scale,
        num_steps,
        dtype=np.float64
    )
