from torch.utils.tensorboard import SummaryWriter
from warmstarting.testbench import WarmstartingBenchTemplate


def random_search(
        problem: WarmstartingBenchTemplate,
        n_models=10,
        epochs=None
        ) -> None:
    """ Implementation of a random search algorithm

    Parameters
    ----------
    problem: WarmstartingBenchTemplate
        Testbench containing an objective function and 
    n_models: int
        Number of configurations to train
    epochs: int or None
        Number of epochs to train
        If not specified, the sampled fidelity is used
    """

    problem.get_configuration_space()
    cs = problem.get_configuration_space()
    fs = problem.get_fidelity_space()

    for model in range(n_models):
        sampled_config = cs.sample_configuration()
        sampled_fidelity = fs.sample_configuration()

        if epochs == None:
            epochs = sampled_fidelity["epoch"]
        print("Model {} trained with {} epochs".format(model, epochs))

        for _ in range(epochs):
            results = problem.objective_function(sampled_config, sampled_fidelity)
            # results.append((model, sampled_config, val_errors))
    # save_result('random_result', results)
