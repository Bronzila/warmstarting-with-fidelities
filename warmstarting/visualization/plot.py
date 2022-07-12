import json

import matplotlib.pylab as plt
import numpy as np
from matplotlib.colors import Normalize
from warmstarting.utils.serialization import load_results


def visualize_data_epoch_grid(performance: np.ndarray, epochs: np.ndarray, data_subsets: np.ndarray, configs):
    """
    performance visualization for every epoch/data_subset lattice point
    every configuration (model) gets a plot

    Parameters
    ----------
    performance
        validation performance for each lattice point
    epochs
        epoch values where the performance was evaluated
    data_subsets
        subset size where the performance was evaluated
    configs
        model configurations
    """
    norm = Normalize(np.min(performance), np.max(performance))
    for model in range(performance.shape[0]):
        plt.scatter(epochs[model], data_subsets[model], c=performance[model], cmap="inferno", norm=norm)
        plt.title("Model with lr={}".format(configs[model]["lr"]))
        plt.xlabel("Number of Epochs"), plt.ylabel("Data Subset Ratio")
        plt.colorbar()
        plt.show()


def visualize_performance_time(performance: np.ndarray, time: np.ndarray, configs, title: str):
    """
    trade-off: validation performance - training time

    Parameters
    ----------
    performance
        validation performance
    time
        training time
    configs
        model configurations
    title
        title of the graph
    """
    for model in range(performance.shape[0]):
        x = time[model].squeeze()
        for time_idx in range(1, len(x)):
            x[time_idx] += x[time_idx - 1]
        label = json.dumps(configs[model])
        plt.plot(x, performance[model][0], label=label)
    plt.xlabel("Training Time"),  plt.ylabel("Validation Performance")
    plt.title(title)
    plt.show()


def visualize_performance_subset(performance: np.ndarray, subset: np.ndarray, configs, title: str):
    """
    validation performance - subset size

    Parameters
    ----------
    performance
        validation performance
    subset
        percentage of data used
    configs
        model configurations
    title
        title of the graph
    """
    for model in range(performance.shape[0]):
        label = json.dumps(configs[model])
        plt.plot(subset[model][0], performance[model][0], label=label)
    plt.xlabel("Data Subset Ratio"),  plt.ylabel("Validation Performance")
    plt.title(title)
    plt.legend(fontsize=6)
    plt.show()

def visualize_fidelity_time(train_times: np.ndarray, subsets: np.ndarray, configs):
    """
    validation fidelity - cost/time

    Parameters
    ----------
    train_times
        training times of both checkpointing and non-checkpointing method.
        train_times[0] is checkpointing, train_times[1] is without
    subsets
        percentages of data used
    configs
        model configurations
    title
        title of the graph
    """

    for model in range(train_times.shape[1]):
        for checkpointing in range(train_times.shape[0]):
            label = "With checkpointing" if checkpointing == 0 else "Without checkpointing"
            x = list(map(str, subsets[model].squeeze()))
            x.append("Accumulated")
            y = train_times[checkpointing, model].squeeze().tolist()
            y.append(sum(y))
            plt.bar(x, y, label=label, width=0.4)
        plt.title("Model with lr={}".format(configs[model]["lr"]))
        plt.xlabel("Data Subset Ratio"), plt.ylabel("Train time in seconds")
        plt.legend()
        plt.show()

def run_vis_fidelity_time():
    score = load_results(file_name="checkpoint", base_path="../../results")
    score_no_checkpoint = load_results(file_name="no_checkpoint", base_path="../../results")

    subsets = np.array(score["subsets"])
    time = np.array([score["full_train_time"], score_no_checkpoint["full_train_time"]])
    configs = np.array(score["configs"])

    visualize_fidelity_time(time, subsets, configs)


if __name__ == "__main__":
    run_vis_fidelity_time()
    # score = load_results(file_name="20220708-120332")
    #
    # performance = np.array(score["performance"])
    # time = np.array(score["time"])
    # configs = np.array(score["configs"])
    #
    # visualize_performance_time(performance, time, configs, "use_checkpoints=True")
