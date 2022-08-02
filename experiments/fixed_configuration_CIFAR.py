import ConfigSpace as CS
import torch.optim as optim

def load_configurations(cs):
    config0 = CS.Configuration(cs, values={
        "lr": 0.001,
        "momentum": 0.9,
        "optimizer": optim.SGD
    })
    config1 = CS.Configuration(cs, values={
        "lr": 0.002,
        "momentum": 0.6,
        "optimizer": optim.SGD
    })
    config2 = CS.Configuration(cs, values={
        "lr": 0.003,
        "momentum": 0.3,
        "optimizer": optim.SGD
    })
    config3 = CS.Configuration(cs, values={
        "lr": 0.004,
        "momentum": 0.3,
        "optimizer": optim.SGD
    })
    config4 = CS.Configuration(cs, values={
        "lr": 0.005,
        "momentum": 0,
        "optimizer": optim.SGD
    })

    return config0, config1, config2, config3, config4
