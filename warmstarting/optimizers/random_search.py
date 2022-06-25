from torch.utils.tensorboard import SummaryWriter
from warmstarting.testbench import WarmstartingBenchTemplate


def random_search(
        problem: WarmstartingBenchTemplate,
        n_models=10,
        epochs=50,
        writer: SummaryWriter = None
        ) -> None:

    problem.get_configuration_space()
    cs = problem.get_configuration_space()
    fs = problem.get_fidelity_space()

    for model in range(n_models):
        sampled_config = cs.sample_configuration()
        print("Model: {}".format(model))
        for _ in range(epochs):
            results = problem.objective_function(sampled_config)
            # results.append((model, sampled_config, val_errors))
    # save_result('random_result', results)
