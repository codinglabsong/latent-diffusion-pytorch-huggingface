from diffusers import UNet2DConditionModel, AutoencoderKL

from ldm.config import *

vae = AutoencoderKL(
    in_channels=1,
    out_channels=1,
    latent_channels=cfg.latent_channels,
    sample_size=32,
    block_out_channels=(16, 32, 64),
    norm_num_groups=4,
    down_block_types=("DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D",),
    up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D")
).to(device)


unet = UNet2DConditionModel(
    in_channels=cfg.latent_channels,
    out_channels=cfg.latent_channels,
    sample_size=8,
    
    layers_per_block=2,
    down_block_types=('AttnDownBlock2D', 'AttnDownBlock2D'),
    up_block_types=('AttnUpBlock2D', 'UpBlock2D'),
    block_out_channels=(128, 256),
    norm_num_groups=1,

    num_class_embeds=10,
    time_embedding_act_fn='silu',
    cross_attention_dim=256,
    class_embeddings_concat=True,
).to(device)