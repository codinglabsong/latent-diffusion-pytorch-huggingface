import importlib
from unittest.mock import MagicMock

import torchvision.datasets as datasets
from torchvision.datasets import FakeData

from ldm.config import cfg


def test_train_invokes_subroutines(monkeypatch):
    def fake_mnist(root, train, download, transform):
        return FakeData(
            size=1,
            image_size=(1, cfg.img_size, cfg.img_size),
            num_classes=10,
            transform=transform,
        )

    monkeypatch.setattr(datasets, "MNIST", fake_mnist)
    train = importlib.reload(importlib.import_module("ldm.train"))

    monkeypatch.setattr(train.os.path, "exists", lambda p: False)
    mock_vae = MagicMock()
    mock_unet = MagicMock()
    monkeypatch.setattr(train, "train_vae", mock_vae)
    monkeypatch.setattr(train, "train_unet", mock_unet)
    monkeypatch.setattr(train, "create_path_if_not_exists", lambda p: None)

    train.train()

    assert mock_vae.called
    assert mock_unet.called
