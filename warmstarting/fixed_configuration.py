import ConfigSpace as CS
import torch.optim as optim

def load_configurations(cs):
    config0 = CS.Configuration(cs, values={
        "lr": 0.01,
        "momentum": 0.6,
        "optimizer": optim.SGD
    })
    config1 = CS.Configuration(cs, values={
        "lr": 0.01,
        "momentum": 0.3,
        "optimizer": optim.SGD
    })
    config2 = CS.Configuration(cs, values={
        "lr": 0.008,
        "momentum": 0.1,
        "optimizer": optim.Adam
    })
    config3 = CS.Configuration(cs, values={
        "lr": 0.01,
        "momentum": 0.1,
        "optimizer": optim.Adam
    })
    config4 = CS.Configuration(cs, values={
        "lr": 0.012,
        "momentum": 0.1,
        "optimizer": optim.Adam
    })

    return config0, config1, config2, config3, config4
