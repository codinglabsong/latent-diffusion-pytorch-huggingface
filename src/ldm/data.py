import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
from ldm.config import batch_size, img_size
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
DATA_DIR = project_root / "data"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Resize((img_size, img_size))
])

mnist_train = datasets.MNIST(root=str(DATA_DIR), train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root=str(DATA_DIR), train=False, download=True, transform=transform)

mnist_train = Subset(mnist_train, list(range(100)))
mnist_test = Subset(mnist_test, list(range(100)))

train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=False)