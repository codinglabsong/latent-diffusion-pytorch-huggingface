import os
import gradio as gr
import numpy as np
import torch
from diffusers import DDPMScheduler
from PIL import Image

from ldm.config import (
    cfg,
    device,
    ema_unet_model_path,
    vae_model_path,
)
from ldm.models import unet, vae
from ldm.utils import revert_images


def load_models() -> None:
    """Load pre-trained model weights."""
    if not (os.path.exists(ema_unet_model_path) and os.path.exists(vae_model_path)):
        msg = "Model weights not found. Train the models before running the app."
        raise RuntimeError(msg)

    vae.load_state_dict(torch.load(vae_model_path, map_location=device))
    unet.load_state_dict(torch.load(ema_unet_model_path, map_location=device))
    vae.eval()
    unet.eval()


@torch.no_grad()
def generate_digit(digit: int) -> Image.Image:
    """Generate a single digit conditioned on user's requested digit class."""
    noise_scheduler = DDPMScheduler(num_train_timesteps=cfg.denoising_timesteps)
    latents = torch.randn((1, cfg.latent_channels, 8, 8)).to(device)
    label = torch.tensor([digit]).to(device)

    for t in noise_scheduler.timesteps:
        noise_pred = unet(
            latents, t, class_labels=label, encoder_hidden_states=None
        ).sample
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    recon = vae.decode(latents).sample
    img_np = revert_images(recon)[0].astype(np.uint8)
    img = Image.fromarray(img_np, mode="L")
    return img.resize((512, 512), resample=Image.NEAREST)


def infer(digit: str) -> Image.Image:
    return generate_digit(int(digit))


def build_demo() -> gr.Interface:
    dropdown = gr.Dropdown(
        choices=[str(i) for i in range(10)], value="0", label="Choose Digit to Generate"
    )
    output = gr.Image(type="pil", height=512, width=512)
    return gr.Interface(
        fn=infer,
        inputs=dropdown,
        outputs=output,
        title="Digit Generator with Latent Diffuser Model",
    )


if __name__ == "__main__":
    load_models()
    demo = build_demo()
    demo.launch()
