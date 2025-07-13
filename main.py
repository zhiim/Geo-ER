import torch.nn as nn

import config
from dataset.dataset import POIPairDataset, ShuffleDataset
from functions import get_device
from model import GeoERX
from train import train

dataset_train = ShuffleDataset(POIPairDataset("data/train.csv"))
dataset_val = ShuffleDataset(POIPairDataset("data/validation.csv"))

device = get_device()
model = GeoERX(device=device, dropout=config.dropout)

criterion = nn.NLLLoss()

train(
    model,
    dataset_train,
    dataset_val,
    criterion,
    device,
    save_path="./saved_models",
    batch_size=32,
    epochs=10,
    lr=3e-5,
)
