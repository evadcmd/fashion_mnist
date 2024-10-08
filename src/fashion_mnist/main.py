import logging

import torch
import typer
from torch import nn

from fashion_mnist import DEVICE, helper
from fashion_mnist.dataloader import test_loader, train_loader
from fashion_mnist.model import NeuralNetwork

cli = typer.Typer()

logger = logging.getLogger("archive")

model = NeuralNetwork().to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


@cli.command()
def train(epoch: int = 5):
    for i in range(epoch):
        try:
            logger.info(f"epoch {i+1}\n---------------------------")
            helper.train(
                dataloader=train_loader,
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
            )
            helper.test(dataloader=test_loader, model=model, loss_fn=loss_fn)
            logger.info("done!")
        except Exception as e:
            logger.error(e)

    torch.save(model.state_dict(), "fashion_mnist.pth")
    logging.info("fashion mnist model has been saved as fashion_mnist.pth")


if __name__ == "__main__":
    cli()
