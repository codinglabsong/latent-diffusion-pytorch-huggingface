import torch
import yaml
import os
from pathlib import Path
from types import SimpleNamespace

project_root = Path(__file__).resolve().parent.parent.parent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae_model_path = str(project_root / 'models' / 'vae.pth')
unet_model_path = str(project_root / 'models' / 'unet.pth')

vae_plots_path = str(project_root / 'plots' / 'vae')
unet_plots_path = str(project_root / 'plots' / 'unet')

# Get the config in project root dir
config_path = os.path.join(project_root, "config.yaml")
with open(config_path, "r") as f:
    cfg_dict = yaml.safe_load(f)

cfg = SimpleNamespace(**cfg_dict)