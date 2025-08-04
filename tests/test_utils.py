import torch

from ldm.utils import create_path_if_not_exists, revert_images


def test_create_path_if_not_exists(tmp_path):
    p = tmp_path / "example"
    create_path_if_not_exists(str(p))
    assert p.exists()


def test_revert_images_returns_uint8_range():
    imgs = torch.randn(2, 1, 8, 8)
    reverted = revert_images(imgs)
    assert reverted.shape == (2, 8, 8)
    assert reverted.min() >= 0 and reverted.max() <= 255
