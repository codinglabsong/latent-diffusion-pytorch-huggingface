import importlib
import torchvision.datasets as datasets
from torchvision.datasets import FakeData

from ldm.config import cfg


def test_data_loaders_return_expected_shapes(monkeypatch):
    def fake_mnist(root, train, download, transform):
        return FakeData(
            size=20,
            image_size=(1, cfg.img_size, cfg.img_size),
            num_classes=10,
            transform=transform,
        )

    monkeypatch.setattr(datasets, "MNIST", fake_mnist)
    data = importlib.reload(importlib.import_module("ldm.data"))

    images, labels = next(iter(data.train_loader))
    assert images.shape[1:] == (1, cfg.img_size, cfg.img_size)
    assert labels.shape[0] == images.shape[0]

    images, labels = next(iter(data.test_loader))
    assert images.shape[1:] == (1, cfg.img_size, cfg.img_size)
    assert labels.shape[0] == images.shape[0]
