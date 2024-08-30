import logging

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fashion_mnist import DEVICE

logger = logging.getLogger("archive")


def train(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn,
    optimizer: Optimizer,
):
    size = len(dataloader.dataset)
    model.train()
    for i_batch, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # compute prediction error
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # back propagation
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        if i_batch % 100 == 0:
            loss, current = loss.item(), (i_batch + 1) * len(inputs)
            logger.info(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn,
):
    size = len(dataloader.dataset)
    num_batch = len(dataloader)
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            pred = model(inputs)
            total_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
    total_loss /= num_batch
    correct /= size
    logger.info(
        f"test error: \n accuracy: {100 * correct:>0.1f}% avg loss: {total_loss:>8f}\n"
    )
