import torch

from ldm.models import vae, unet
from ldm.config import cfg, device


def test_vae_encode_decode_roundtrip():
    x = torch.randn(2, 1, cfg.img_size, cfg.img_size).to(device)
    posterior = vae.encode(x)
    z = posterior["latent_dist"].sample()
    recon = vae.decode(z).sample
    assert recon.shape == x.shape


def test_unet_forward_shape():
    latents = torch.randn(2, cfg.latent_channels, 8, 8).to(device)
    timesteps = torch.randint(0, 10, (2,), device=device)
    labels = torch.randint(0, 10, (2,), device=device)
    out = unet(
        latents, timesteps, class_labels=labels, encoder_hidden_states=None
    ).sample
    assert out.shape == latents.shape
