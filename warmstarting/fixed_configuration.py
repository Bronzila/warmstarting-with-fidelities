import ConfigSpace as CS
import torch.optim as optim

def load_configurations(cs):
    config0 = CS.Configuration(cs, values={
        "lr": 0.00025,
        "momentum": 0.6,
        "optimizer": optim.Adam
    })
    config1 = CS.Configuration(cs, values={
        "lr": 0.0005,
        "momentum": 0.3,
        "optimizer": optim.Adam
    })
    config2 = CS.Configuration(cs, values={
        "lr": 0.00075,
        "momentum": 0.1,
        "optimizer": optim.Adam
    })
    config3 = CS.Configuration(cs, values={
        "lr": 0.001,
        "momentum": 0.1,
        "optimizer": optim.Adam
    })
    config4 = CS.Configuration(cs, values={
        "lr": 0.00125,
        "momentum": 0.1,
        "optimizer": optim.Adam
    })

    return config0, config1, config2, config3, config4
