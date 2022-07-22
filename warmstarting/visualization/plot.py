import json

import matplotlib.pylab as plt
import matplotlib
import numpy as np
from matplotlib.colors import Normalize
from warmstarting.utils.serialization import load_results
import seaborn as sns
import pandas as pd


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
    cmap_bl = matplotlib.cm.get_cmap("winter")
    cmap_cp = matplotlib.cm.get_cmap("autumn")
    color_steps = np.linspace(0, 1, subsets.shape[-1])
    for model in range(time.shape[1]):
        for checkpointing in range(time.shape[0]):
            prev_end = 0
            for subset in range(subsets.shape[-1]):
                curr_subset = np.around(subsets[model, :, subset].squeeze(), 2)
                if checkpointing == 0:
                    color = cmap_cp(color_steps[subset])
                    label = f"CP, sub = {curr_subset}"
                else:
                    color = cmap_bl(color_steps[subset])
                    label = f"BL, sub = {curr_subset}"
                x = time[checkpointing, model, :, subset].squeeze()
                x += prev_end
                y = performance[checkpointing, model, :, subset].squeeze()
                prev_end = x[-1]
                plt.plot(x, y, label=label, c=color)
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

def visualize_fidelity_time(train_times: np.ndarray, subsets: np.ndarray, performance: np.ndarray, configs):
    """
    validation fidelity - cost/time

    Parameters
    ----------
    train_times
        training times of both checkpointing and non-checkpointing method.
        train_times[0] is checkpointing, train_times[1] is without
    subsets
        percentages of data used
    performance

    configs
        model configurations
    title
        title of the graph
    """

    sns.set_theme(palette="tab10", style="whitegrid")

    for model in range(train_times.shape[1]):
        optimum = np.min(performance[:, model])
        d = []
        for checkpointing in range(train_times.shape[0]):
            label = "With checkpointing" if checkpointing == 0 else "Without checkpointing"
            x = list(map(str, np.around(subsets[model].squeeze(), 3)))
            x.append("Accumulated")
            y = train_times[checkpointing, model].squeeze().tolist()
            y.append(sum(y))
            # Take the performances after the last training epoch
            p = performance[checkpointing, model, :, :, -1].squeeze().tolist()
            p.append(sum(p))
            for i, _ in enumerate(x):
                d.append({
                    "Subset Ratio": x[i],
                    "Train Time": y[i],
                    "Checkpoint": label,
                    "Performance": p[i],
                    "Optimum": optimum
                })
        df = pd.DataFrame(d)

        ax = sns.barplot(x="Subset Ratio", y="Train Time", hue="Checkpoint", data=df)
        for i, patch in enumerate(ax.patches):
            if (i + 1) % (len(ax.patches) / 2) == 0:
                continue
            optimum = df["Optimum"][model]
            current_val = df["Performance"][i]

            percentage = ((current_val - optimum) / ((current_val + optimum) / 2)).round(2)

            _x = patch.get_x() + patch.get_width() / 2
            _y = patch.get_y() + patch.get_height() + 0.1

            patch.set_alpha(max(1 - percentage, 0.01))
            ax.text(_x, _y, current_val.round(3), ha="center", rotation='vertical')

        plt.title("Model with lr={}".format(configs[model]["lr"]))
        plt.xlabel("Data Subset Ratio"), plt.ylabel("Train time in seconds")
        plt.legend()
        plt.show()

def get_relative_timestamps(times):
    initial_times = np.tile(times[..., 0, np.newaxis], times.shape[-1])
    times -= initial_times
    return times

def run_vis_fidelity_time():
    score = load_results(file_name="checkpoint", base_path="../../results")
    score_no_checkpoint = load_results(file_name="no_checkpoint", base_path="../../results")

    subsets = np.array(score["subsets"])
    time = np.array([score["time_step"], score_no_checkpoint["time_step"]])
    time = get_relative_timestamps(time)
    time = time[..., -1]
    performance = np.array([score["performance"], score_no_checkpoint["performance"]])
    configs = np.array(score["configs"])

    visualize_fidelity_time(time, subsets, performance, configs)


def run_vis_perf_time():
    score = load_results(file_name="checkpoint", base_path="../../results")
    score_no_checkpoint = load_results(file_name="no_checkpoint", base_path="../../results")

    subsets = np.array(score["subsets"])
    performance = np.array([score["performance"], score_no_checkpoint["performance"]])
    time = np.array([score["time_step"], score_no_checkpoint["time_step"]])
    time = get_relative_timestamps(time)
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