import torch
from torch.utils.tensorboard import SummaryWriter
from warmstarting.testbench.DummyBench import DummyBench
from warmstarting.data_loader import DataHandler
from warmstarting.optimizers.random_search import random_search

def HPOLoop():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = torch.seed()
    writer = SummaryWriter()

    handler = DataHandler()
    handler.set_dataset()
    train_dataloader, valid_dataloader = handler.get_train_and_val_set(batch_size=10, device=torch.device(device))

    problem = DummyBench(train_dataloader, valid_dataloader, writer, seed)

    random_search(problem, n_models=20, epochs=100, writer=writer)


if __name__ == "__main__":
    HPOLoop()
