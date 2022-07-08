import json
import os
from datetime import datetime
from typing import Dict, Union, Tuple, List
from ConfigSpace import Configuration


def serialize_results(score: Dict[str, Union[list, object]],
                      configs: Tuple[Configuration],
                      critical_objects: List[str] = None,
                      base_path: str = "./results/"):
    """
    serializes the results as well as the corresponding configuration for plotting
    if the configuration contains objects (e.g. torch optimizer) set the parameter respectively
    Parameters
    ----------
    score
        object containing result scores and empty "configs" array
    configs
        tuple of all configurations used for the experiment
    critical_objects
        list of keys of all objects in the configuration space
    base_path
        path for saving the file
    """
    if critical_objects is None:
        critical_objects = ["optimizer"]

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    file_name = os.path.join(base_path, "%s" % datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Make configuration object serializable for json
    for c in configs:
        configuration = c.get_dictionary()

        # Change torch object to its respective name to be serializable via json
        for o in critical_objects:
            configuration[o] = configuration[o].__name__
        score["configs"].append(configuration)

    with open(file_name, 'w') as f:
        json.dump(score, f)
