import torch
from ConfigSpace import Configuration
from torch.utils.data import DataLoader
from optimizers.hyperband import hyperband
from warmstarting.testbench.DummyBench import DummyBench
from warmstarting.DataHandler import DataHandler

def HPOLoop():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = torch.seed()

    handler = DataHandler()
    handler.set_dataset()
    train_dataloader, valid_dataloader = handler.get_train_and_val_set(device=torch.device(device))

    problem = DummyBench(train_dataloader, valid_dataloader, seed)

    cs = problem.get_configuration_space(seed)
    problem.objective_function(cs)



if __name__ == "__main__":
    HPOLoop()
