# WarmstartingWithFidelitites
This framework allows continuation for the fidelities data subsets and epochs in the field of multi-fidelity hyperparameter optimization.
It allows for various experimentation and thorough result logging as well as their visualization.

It is a project created during the [Deep Learning Lab course 2022](https://rl.uni-freiburg.de/teaching/ss22/laboratory-deep-learning-lab) of the Albert-Ludwigs-Universität Freiburg.


## Installation
```
git clone https://github.com/Bronzila/warmstarting-with-fidelities.git
cd warmstarting-with-fidelities
conda create -n warmstarting python=3.7
conda activate warmstarting

# Install for usage
pip install -r requirements.txt

# Run an experiment defined in config.yml
python run.py --config=config.yml
```

## Usage
Define your own experimentation configs or use the predefined in the [`experiments`](experiments) folder.  
Use the [`run.py`](run.py) file to execute your experiments.
To visualize your results use the functions defined in [`warmstarting/visualization/plots.py`](warmstarting/visualization/plot.py).  

## License
[MIT](https://choosealicense.com/licenses/mit/)
