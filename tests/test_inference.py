import importlib
from unittest.mock import MagicMock

import torch


def test_sample_calls_generate_when_models_present(monkeypatch):
    # Import module after patching
    inference = importlib.import_module("ldm.inference")

    monkeypatch.setattr(inference.os.path, "exists", lambda p: True)
    monkeypatch.setattr(torch, "load", lambda p: {})
    inference.vae.load_state_dict = MagicMock()
    inference.unet.load_state_dict = MagicMock()
    monkeypatch.setattr(inference, "generate", MagicMock())

    inference.sample()

    inference.generate.assert_called_once()
