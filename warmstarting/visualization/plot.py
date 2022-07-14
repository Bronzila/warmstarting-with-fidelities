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
        label = json.dumps(configs[model])
        plt.plot(time[model][0], performance[model][0], label=label)
    plt.xlabel("Training Time"),  plt.ylabel("Validation Performance")
    plt.title(title)
    plt.legend(fontsize=6)
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


def visualize_discretization():
    step_scale = ["linear", "exponential"]
    d_sub = [2, 3, 5, 10, 20]

    for model in range(5):
        for s in step_scale:
            for d in d_sub:
                filename = "iris" + "_" + s + "_" + str(d)
                score = load_results(file_name=filename, base_path="../../results")
                flattened_performance = [x for xs in score["performance"][model][0] for x in xs]
                flattened_time_steps = [y - score["start_time_step"] for ys in score["time_step"][model][0] for y in ys]
                color = "red" if s == "linear" else "green"
                plt.plot(flattened_time_steps, flattened_performance, color=color, alpha=0.4)
        baseline = load_results(file_name="baseline", base_path="../../results")
        baseline_flattened_performance = [x for xs in baseline["performance"][model][0] for x in xs]
        baseline_flattened_time_steps = [y - baseline["start_time_step"] for ys in baseline["time_step"][model][0] for y in ys]
        plt.plot(baseline_flattened_time_steps, baseline_flattened_performance, color="blue")
        title = "Model {} with lr={}, optimizer={}".format(
            model + 1,
            baseline['configs'][model]['lr'],
            baseline['configs'][model]['optimizer']
        )
        if baseline['configs'][model]['optimizer'] == "SGD":
            title = title + " and momentum={}".format(
                baseline['configs'][model]['momentum']
            )
        plt.title(title)
        plt.xlabel("Training Time in s"), plt.ylabel("Validation Performance")
        plt.show()

if __name__ == "__main__":
    # score = load_results(file_name="20220708-120332")
    #
    # performance = np.array(score["performance"])
    # time = np.array(score["time"])
    # configs = np.array(score["configs"])
    #
    # visualize_performance_time(performance, time, configs, "use_checkpoints=True")
    visualize_discretization()