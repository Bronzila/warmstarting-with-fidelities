import json

import matplotlib.pylab as plt
import matplotlib.colors as clr
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


def visualize_performance_time(performance: np.ndarray, time: np.ndarray, subsets, configs):
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
    for model in range(time.shape[1]):
        for checkpointing in range(time.shape[0]):
            prev_end = 0
            for subset in range(subsets.shape[-1]):
                curr_subset = np.around(subsets[model, :, subset].squeeze(), 2)
                label = f"CP, sub = {curr_subset}" if checkpointing == 0 else f"BL, sub = {curr_subset}"
                x = time[checkpointing, model, :, subset].squeeze()
                x += prev_end
                y = performance[checkpointing, model, :, subset].squeeze()
                prev_end = x[-1]
                plt.plot(x, y, label=label)
        plt.title("Model with lr={}".format(configs[model]["lr"]))
        plt.xlabel("Time in seconds"), plt.ylabel("Validation loss")
        plt.legend()
        plt.grid(True)
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
    time = np.array([score["time_step"], score_no_checkpoint["time_step"]])
    initial_times = np.tile(time[..., 0, np.newaxis], time.shape[-1])
    time -= initial_times
    time = np.sum(time, axis=-1)
    configs = np.array(score["configs"])

    visualize_fidelity_time(time, subsets, configs)

def run_vis_perf_time():
    score = load_results(file_name="checkpoint", base_path="../../results")
    score_no_checkpoint = load_results(file_name="no_checkpoint", base_path="../../results")

    subsets = np.array(score["subsets"])
    performance = np.array([score["performance"], score_no_checkpoint["performance"]])
    time = np.array([score["time_step"], score_no_checkpoint["time_step"]])
    initial_times = np.tile(time[..., 0, np.newaxis], time.shape[-1])
    time -= initial_times
    configs = np.array(score["configs"])

    visualize_performance_time(performance, time, subsets, configs)

def visualize_discretization():
    step_scale = ["linear", "exponential"]
    d_sub = [2, 3, 5, 10, 20]

    for model in range(5):
        for s in step_scale:
            for d in d_sub:
                filename = "iris" + "_" + s + "_" + str(d)
                score = load_results(file_name=filename, base_path="../../results")
                flattened_performance = [x for xs in score["performance"][model][0] for x in xs]
                flattened_time_steps = [y for ys in score["time"][model][0] for y in ys]
                start = 0
                for i, y in enumerate(flattened_time_steps):
                    flattened_time_steps[i] = start + y
                    start = flattened_time_steps[i]

                color = "red" if s == "linear" else "green"
                plt.plot(flattened_time_steps, flattened_performance, color=color, alpha=0.4)
        baseline = load_results(file_name="baseline", base_path="../../results")
        baseline_flattened_performance = [x for xs in baseline["performance"][model][0] for x in xs]
        baseline_flattened_time_steps = [y for ys in baseline["time"][model][0] for y in ys]
        start = 0
        for i, y in enumerate(baseline_flattened_time_steps):
            baseline_flattened_time_steps[i] = start + y
            start = baseline_flattened_time_steps[i]
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

    # run_vis_fidelity_time()

    run_vis_perf_time()