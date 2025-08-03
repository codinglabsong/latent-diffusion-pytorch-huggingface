import torch
import os
from diffusers import DDPMScheduler

from ldm.utils import generate, create_path_if_not_exists
from ldm.config import (
    cfg,
    unet_model_path,
    vae_model_path,
    project_root,
    vae_plots_path,
    unet_plots_path,
)
from ldm.models import vae, unet
from ldm.train import train


def sample():
    if os.path.exists(unet_model_path) and os.path.exists(vae_model_path):
        vae.load_state_dict(torch.load(vae_model_path))
        unet.load_state_dict(torch.load(unet_model_path))
        noise_scheduler = DDPMScheduler(num_train_timesteps=cfg.denoising_timesteps)
        generate(vae, unet, noise_scheduler, 101)
    else:
        print("Models not trained, training...")
        train()


if __name__ == "__main__":
    create_path_if_not_exists(str(project_root / "models"))
    create_path_if_not_exists(str(project_root / "plots"))
    create_path_if_not_exists(vae_plots_path)
    create_path_if_not_exists(unet_plots_path)

    sample()
