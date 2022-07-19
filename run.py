import numpy as np
import yaml
import argparse
from warmstarting.HPOLoop import HPOLoop
from warmstarting.utils.datasets import select_dataset_id
from warmstarting.utils.serialization import load_results
from warmstarting.utils.torch import modules
from warmstarting.visualization.plot import visualize_performance_time, visualize_data_epoch_grid, visualize_performance_subset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='configuration parser calling the main HPO loop')
    parser.add_argument('-c', '--config', nargs='+', help='Config file to run experiments')

    args = parser.parse_args()
    for config in args.config:
        with open(config, "r") as f:
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

        subset_lower_bound = params["subset_bounds"][0] if params["subset_bounds"] else 0.01
        subset_upper_bound = params["subset_bounds"][1] if params["subset_bounds"] else 1

        if params["data_subset_ratio"] is not None:
            data_subset_ratios = params["data_subset_ratio"]
        else:
            if params["step_scaling"] == "linear":
                data_subset_ratios = np.linspace(subset_lower_bound, subset_upper_bound, params["d_sub"])
            elif params["step_scaling"] == "exponential":
                data_subset_ratios = np.geomspace(subset_lower_bound, subset_upper_bound, num=params["d_sub"]).round(2)

        HPOLoop(params["model"], lr, params["momentum"], optimizer,
                params["lr_sched"], criterion, params["epoch_bounds"], [subset_lower_bound, subset_upper_bound],
                config_space, fidelity_space, params["epoch"], data_subset_ratios,
                params["use_checkpoints"], params["shuffle"], params["only_train_on_new"], params["dataset_id"],
                params["results_file_name"])

        score = load_results(file_name=params["results_file_name"])

        for m in params["vis_method"]:
            if m == "time_val":
                title = "checkpoints: {}, only_new: {}, shuffle: {}"\
                    .format(params["use_checkpoints"],  params["only_train_on_new"], params["shuffle"])
                performance = np.array(score["performance"])
                time = np.array(score["full_train_time"])
                configs = np.array(score["configs"])
                visualize_performance_time(performance, time, configs, title)
            elif m == "subset_val":
                title = "checkpoints: {}, only_new: {}, shuffle: {}"\
                    .format(params["use_checkpoints"],  params["only_train_on_new"], params["shuffle"])
                performance = np.array(score["performance"])
                data_subsets = np.array(score["subsets"])
                configs = np.array(score["configs"])
                visualize_performance_subset(performance, data_subsets, configs, title)
            elif m == "grid":
                performance = np.array(score["performance"])
                epochs = np.array(score["epochs"])
                data_subsets = np.array(score["subsets"])
                configs = np.array(score["configs"])
                visualize_data_epoch_grid(performance, epochs, data_subsets, configs)
            else:
                raise ValueError('Visualization method "{}" not found!'.format(m))
