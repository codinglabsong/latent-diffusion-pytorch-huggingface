import torch
from torchvision import datasets, transforms
from ldm.config import project_root, cfg

DATA_DIR = project_root / "data"

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Resize((cfg.img_size, cfg.img_size)),
    ]
)

mnist_train = datasets.MNIST(
    root=str(DATA_DIR), train=True, download=True, transform=transform
)
mnist_test = datasets.MNIST(
    root=str(DATA_DIR), train=False, download=True, transform=transform
)

# Uncomment For Smoke Tests on Small Sub Dataset
# from torch.utils.data import Subset
# mnist_train = Subset(mnist_train, list(range(100)))
# mnist_test = Subset(mnist_test, list(range(100)))

train_loader = torch.utils.data.DataLoader(
    mnist_train, batch_size=cfg.batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    mnist_train, batch_size=cfg.batch_size, shuffle=False
)
