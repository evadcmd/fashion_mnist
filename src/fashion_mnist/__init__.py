import logging.config

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

logging.config.fileConfig("./log.conf", disable_existing_loggers=False)
logger = logging.getLogger("root")

TRAINING_DATA = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
TEST_DATA = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
logger.info(f"using device: {DEVICE}")
