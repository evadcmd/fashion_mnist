from torch.utils.data import DataLoader

from fashion_mnist import TEST_DATA, TRAINING_DATA

BATCH_SIZE = 64

train_loader = DataLoader(TRAINING_DATA, batch_size=BATCH_SIZE)
test_loader = DataLoader(TEST_DATA, batch_size=BATCH_SIZE)
