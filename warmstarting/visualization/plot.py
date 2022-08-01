import json

import matplotlib.pylab as plt
import matplotlib
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from warmstarting.utils.serialization import load_results
from warmstarting.utils.plotting import HandlerColormap, cm_to_inch
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D

title_fontsize = 30
label_fontsize = 28
legend_fontsize = 22
tick_fontsize = 20
tick_length = 12
tick_width = 3


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


def visualize_performance_time_bl(performance: np.ndarray, time: np.ndarray, subsets, configs):
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
                curr_subset = np.around(subsets[model, subset], 2)
                if checkpointing == 0:
                    color = cmap_cp(color_steps[subset])
                    label = f"CP, sub = {curr_subset}"
                else:
                    color = cmap_bl(color_steps[subset])
                    label = f"BL, sub = {curr_subset}"
                x = np.array(time[checkpointing, model, subset])
                x += prev_end
                y = performance[checkpointing, model, subset]
                prev_end = x[-1]
                plt.plot(x, y, label=label, c=color)
        plt.title("Model with lr={}".format(configs[model]["lr"]))
        plt.xlabel("Time in seconds"), plt.ylabel("Validation loss")
        plt.legend()
        # plt.yscale('log')
        plt.grid(True)
        plt.show()

def visualize_performance_time_multiple(performance: np.ndarray, time: np.ndarray, subsets, configs, epochs):
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
    cmaps = [matplotlib.cm.get_cmap("copper"), matplotlib.cm.get_cmap("summer"),
             matplotlib.cm.get_cmap("cool"), matplotlib.cm.get_cmap("winter")]
    for model in range(time[0].shape[1]):
        color_lists = []
        plt.figure(figsize=(cm_to_inch(23), cm_to_inch(18)))
        for bucket in range(len(time)):
            color_lists.append([])
            curr_perf = performance[bucket]
            curr_time = time[bucket]
            curr_subsets = subsets[bucket]
            curr_epochs = epochs[bucket]
            prev_end = 0
            for subset in range(curr_subsets[model].shape[-1]):
                color_steps = np.linspace(0.3, 0.9, curr_subsets.shape[-1])
                curr_subset = np.around(curr_subsets[model, subset], 2)
                label = f"sub = {curr_subset}, ep = {curr_epochs[model, subset]}"
                color = cmaps[bucket](color_steps[subset])
                color_lists[bucket].append(color)
                x = np.array(curr_time[model, subset])
                x += prev_end
                y = curr_perf[model, subset]
                prev_end = x[-1]
                plt.plot(x, y, label=label, c=color, linewidth=2)

        plt.title("MNIST - Hyperband-Bucket like scaling", fontsize=title_fontsize)
        plt.xlabel("Time in seconds", fontsize=label_fontsize), plt.ylabel("Validation loss", fontsize=label_fontsize)

        cmap_handles = [Rectangle((0, 0), 1, 1) for _ in cmaps]
        cmap_labels = ["Bucket 0", "Bucket 1", "Bucket 2", "Bucket 3"]

        handler_map = dict(
            zip(cmap_handles, [HandlerColormap(cmaps[i], num_stripes=len(cl), fc=color_lists[i]) for i, cl in enumerate(color_lists)]))
        plt.legend(handles=cmap_handles, labels=cmap_labels, handler_map=handler_map, fontsize=legend_fontsize)
        plt.tick_params(direction='out', length=tick_length, width=tick_width, grid_alpha=0.5, labelsize=tick_fontsize)

        plt.yscale('log'), plt.xscale('log')
        plt.grid(True)
        # plt.rcParams.update({'figure.autolayout': True})

        # plt.show()
        plt.savefig(f'{configs[model]["lr"]}.png', bbox_inches='tight')
        plt.cla()


def visualize_performance_time_diagonal(performance: np.ndarray, time: np.ndarray, subsets, configs, epochs):
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

    color_list_cp = []
    color_list_bl = []

    for model in range(time.shape[1]):
        plt.figure(figsize=(cm_to_inch(23), cm_to_inch(18)))
        for is_linear in range(time.shape[0]):
            prev_end = 0
            for subset in range(subsets.shape[-1]):
                curr_subset = np.around(subsets[is_linear, model, subset], 2)
                if is_linear == 0:
                    color = cmap_cp(color_steps[subset])
                    label = f"Lin, sub = {curr_subset}, ep = {epochs[is_linear, model, subset]}"
                    color_list_cp.append(color)
                else:
                    color = cmap_bl(color_steps[subset])
                    label = f"Exp, sub = {curr_subset}, ep = {epochs[is_linear, model, subset]}"
                    color_list_bl.append(color)
                x = np.array(time[is_linear, model, subset])
                x += prev_end
                y = performance[is_linear, model, subset]
                prev_end = x[-1]
                plt.plot(x, y, label=label, c=color, linewidth=3)
        plt.title("MNIST - Moving along the fidelity diagonal", fontsize=title_fontsize)
        plt.xlabel("Time in seconds", fontsize=label_fontsize), plt.ylabel("Validation loss", fontsize=label_fontsize)

        cmaps = [cmap_cp, cmap_bl]
        cmap_handles = [Rectangle((0, 0), 1, 1) for _ in cmaps]
        color_lists = [color_list_cp, color_list_bl]
        cmap_labels = ["Linear", "Exponential"]

        handler_map = dict(zip(cmap_handles, [HandlerColormap(cm, num_stripes=5, fc=color_lists[i]) for i, cm in enumerate(cmaps)]))
        plt.legend(handles=cmap_handles, labels=cmap_labels, handler_map=handler_map, fontsize=legend_fontsize)
        plt.tick_params(direction='out', length=tick_length, width=tick_width, grid_alpha=0.5, labelsize=tick_fontsize)

        plt.yscale('log')
        plt.grid(True)
        # plt.rcParams.update({'figure.autolayout': True})

        # plt.show()
        plt.savefig(f'diagonal_{configs[model]["lr"]}.png',bbox_inches='tight')
        plt.cla()


def visualize_seeded_performance(performance: np.ndarray, time: np.ndarray, subsets, configs, scaling, dataset_name):
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

    color_list_cp = []
    color_list_bl = []

    for model in range(time.shape[2]):
        plt.figure(figsize=(cm_to_inch(23), cm_to_inch(18)))
        for checkpointing in range(time.shape[0]):
            prev_end = 0
            for subset in range(subsets.shape[-1]):
                curr_subset = np.around(subsets[model, subset], 2)
                if checkpointing == 0:
                    color = cmap_cp(color_steps[subset])
                    label = f"CP, sub = {curr_subset}"
                    color_list_cp.append(color)
                else:
                    color = cmap_bl(color_steps[subset])
                    label = f"BL, sub = {curr_subset}"
                    color_list_bl.append(color)
                x = np.mean(time[checkpointing, :, model, subset, :], axis=0)
                x += prev_end
                y = performance[checkpointing, :, model, subset, :]
                y_std = np.std(y, axis=0)
                y_mean = np.mean(y, axis=0)
                prev_end = x[-1]
                plt.plot(x, y_mean, label=label, c=color)
                plt.fill_between(x, y_mean - y_std, y_mean + y_std, facecolor=color, alpha=0.2)
                plt.yscale('log')
        title = f"{scaling} Scaling on {dataset_name}"
        plt.title(title, fontsize=title_fontsize)
        plt.xlabel("Time in seconds", fontsize=label_fontsize), plt.ylabel("Validation loss", fontsize=label_fontsize)

        plt.tick_params(direction='out', length=tick_length, width=tick_width, grid_alpha=0.5, labelsize=tick_fontsize)

        cmaps = [matplotlib.cm.get_cmap("autumn"), matplotlib.cm.get_cmap("winter")]
        cmap_handles = [Rectangle((0, 0), 1, 1) for _ in cmaps]
        cmap_labels = ["Checkpoint", "Baseline"]
        color_lists = [color_list_cp, color_list_bl]
        handler_map = dict(
            zip(cmap_handles, [HandlerColormap(cmaps[i], num_stripes=len(cl), fc=color_lists[i]) for i, cl in enumerate(color_lists)]))

        plt.legend(handles=cmap_handles, labels=cmap_labels, handler_map=handler_map, fontsize=legend_fontsize, loc='upper right')
        # plt.rcParams.update({'figure.autolayout': True})

        color_list_bl.clear()
        color_list_cp.clear()
        plt.grid(True)

        # plt.show()
        plt.savefig(f'{configs[model]["lr"]}.png', bbox_inches='tight')
        plt.cla()


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
        if type(optimum) is list:
            optimum = min(optimum)
        d = []
        for checkpointing in range(train_times.shape[0]):
            label = "With checkpointing" if checkpointing == 0 else "Without checkpointing"
            x = list(map(str, np.around(subsets[model], 2)))
            y = train_times[checkpointing, model]
            # Take the performances after the last training epoch
            p = []
            for sbs_ep in range(performance.shape[2]):
                p.append(performance[checkpointing, model, sbs_ep][-1])
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

        plt.title(f"Model with lr={configs[model]['lr']}")
        plt.xlabel("Data Subset Ratio"), plt.ylabel("Train time in seconds")
        plt.legend()

        for i, patch in enumerate(ax.patches):
            optimum = df["Optimum"][model]
            current_val = df["Performance"][i]

            percentage = ((current_val - optimum) / ((current_val + optimum) / 2)).round(2)

            _x = patch.get_x() + patch.get_width() / 2
            _y = patch.get_y() + patch.get_height() + 0.1

            patch.set_alpha(max(1 - percentage, 0.1))
        plt.show()

def get_relative_timestamps(times):
    for cp in range(times.shape[0]):
        for mdl in range(times.shape[1]):
            for sbs_ep in range(times.shape[2]):
                time = np.array(times[cp, mdl, sbs_ep])
                time -= time[0]
                times[cp, mdl, sbs_ep] = time.tolist()
    return times

def run_vis_fidelity_time():
    score = load_results(file_name="checkpoint", base_path="../../results")
    score_no_checkpoint = load_results(file_name="no_checkpoint", base_path="../../results")

    subsets = np.array(score["subsets"])
    time = np.array([score["time_step"], score_no_checkpoint["time_step"]])
    time = get_relative_timestamps(time)

    # only save final time of training
    final_times = np.zeros((time.shape[0], time.shape[1], time.shape[2]))
    for cp in range(time.shape[0]):
        for mdl in range(time.shape[1]):
            for sbs_ep in range(time.shape[2]):
                final_times[cp, mdl, sbs_ep] = time[cp, mdl, sbs_ep][-1]

    performance = np.array([score["performance"], score_no_checkpoint["performance"]])
    configs = np.array(score["configs"])

    visualize_fidelity_time(final_times, subsets, performance, configs)


def run_vis_perf_time_with_bl():
    score = load_results(file_name="checkpoint_mnist_proper", base_path="../../results/MNIST_usual")
    score_no_checkpoint = load_results(file_name="no_checkpoint_mnist_proper", base_path="../../results/MNIST_usual")

    subsets = np.array(score["subsets"])
    performance = np.array([score["performance"], score_no_checkpoint["performance"]])
    time = np.array([score["time_step"], score_no_checkpoint["time_step"]])
    time = get_relative_timestamps(time)
    configs = np.array(score["configs"])

    visualize_performance_time_bl(performance, time, subsets, configs)

def run_vis_perf_time_diagonal():
    score_lin = load_results(file_name="diagonal_linear", base_path="../../results/Moving_along_diag")
    score_exp = load_results(file_name="diagonal_exponential", base_path="../../results/Moving_along_diag")

    subsets = np.array([score_lin["subsets"], score_exp["subsets"]])
    epochs = np.array([score_lin["epochs"], score_exp["epochs"]])
    performance = np.array([score_lin["performance"], score_exp["performance"]])
    time = np.array([score_lin["time_step"], score_exp["time_step"]])
    time = get_relative_timestamps(time)
    configs = np.array(score_lin["configs"])

    visualize_performance_time_diagonal(performance, time, subsets, configs, epochs)

def run_vis_perf_time():
    performances = []
    times = []
    subsets = []
    epochs = []
    for filename in ["hb_1", "hb_2", "hb_3", "hb_4"]:
        score = load_results(file_name=filename, base_path="../../results/Hyperband")

        subsets.append(np.array(score["subsets"]))
        epochs.append(np.array(score["epochs"]))
        performances.append(np.array(score["performance"]))
        time = np.expand_dims(np.array(score["time_step"]), axis=0)
        time = get_relative_timestamps(time)
        times.append(time.squeeze())
        configs = np.array(score["configs"])

    visualize_performance_time_multiple(performances, times, subsets, configs, epochs)

def run_seeded_perf():
    base_path = "../../results/MNIST-Seeds-linear-first"
    seeds = [0, 50, 100, 150, 200]
    # seeds = [100, 200, 300, 400, 500]

    cp_performance = []
    bl_performance = []
    cp_time = []
    bl_time = []
    subsets, configs = None, None
    for i, seed in enumerate(seeds):
        cp_score = load_results(file_name=f"checkpoint_seed{seed}", base_path=base_path)
        bl_score = load_results(file_name=f"no_checkpoint_seed{seed}", base_path=base_path)
        # cp_score = load_results(file_name=f"cp_seed_{seed}", base_path=base_path)
        # bl_score = load_results(file_name=f"bl_seed_{seed}", base_path=base_path)

        if i == 0:
            subsets = np.array(cp_score["subsets"])
            configs = np.array(cp_score["configs"])

        cp_performance.append(cp_score["performance"])
        bl_performance.append(bl_score["performance"])

        current_cp_time = get_relative_timestamps(np.expand_dims(np.array(cp_score["time_step"]), axis=0))
        current_bl_time = get_relative_timestamps(np.expand_dims(np.array(bl_score["time_step"]), axis=0))
        cp_time.append(current_cp_time[0])
        bl_time.append(current_bl_time[0])

    performance = np.array([cp_performance, bl_performance])
    time = np.array([cp_time, bl_time])

    visualize_seeded_performance(performance, time, subsets, configs, "Linear", "MNIST")


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
    # run_vis_perf_time_with_bl()
    # run_vis_perf_time_diagonal()
    run_vis_perf_time()
    # run_seeded_perf()
