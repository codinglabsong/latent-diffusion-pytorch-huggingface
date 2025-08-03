import torch
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
DATA_DIR = project_root / "data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
img_size = 32

latent_channels = 4

lr_vae = 1e-4
vae_epochs = 10 # 50

lr_unet = 1e-4
denoising_timesteps = 10 # 1000
num_warmup_steps = 5 # 500
unet_epochs = 3 # 100

vae_model_path = str(project_root / 'models' / 'vae.pth')
unet_model_path = str(project_root / 'models' / 'unet.pth')

vae_plots_path = str(project_root / 'plots' / 'vae')
unet_plots_path = str(project_root / 'plots' / 'unet')

# vae_model_path = '../../models/vae.pth'
# unet_model_path = '../../models/unet.pth'

# vae_plots_path = '../../plots/vae/'
# unet_plots_path = '../../plots/unet/'