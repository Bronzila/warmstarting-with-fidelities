import yaml
import argparse
from warmstarting.HPOLoop import HPOLoop
from warmstarting.utils.datasets import select_dataset_id
from warmstarting.utils.torch import modules


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='configuration parser calling the main HPO loop')
    parser.add_argument('--config', help='Config file to run experiments')

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    params = dict()
    fidelity_space = []
    config_space = []

    for topic, data in cfg.items():
        for k, v in data.items():
            if topic == "fidelity":
                fidelity_space.append(k)
            if topic == "configuration":
                config_space.append(k)
            if k == "dataset":
                k = "dataset_id"
                v = select_dataset_id(v)
            params[k] = v

    criterion = modules[params["criterion"]]
    optimizer = [modules[optim] for optim in params["optimizer"]]
    lr = [float(x) for x in params["lr"]]

    HPOLoop(params["model"], lr, params["momentum"], optimizer,
            params["lr_sched"], criterion, params["epoch"],
            config_space, fidelity_space, params["batch_size"], params["n_models"])