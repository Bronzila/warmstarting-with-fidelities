import torch
import torch.nn as nn
import torch.optim as optim

modules = {
    "Adam": optim.Adam,
    "SGD": optim.SGD,
    "CrossEntropyLoss": nn.CrossEntropyLoss,
}