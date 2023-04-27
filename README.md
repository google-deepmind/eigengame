# EigenGame: Top-k Eigendecompositions for Large, Streaming Data

## Background and Description

EigenGame formulates top-k eigendecomposition as a k-player game, enabling
a distributed approach that iteratively learns eigenvectors of large
matrices defined as expectations over large datasets. This setting is
common to many settings in machine learning, statistics, and science more
generally.

This repository contains an implementation of the some of the algorithms
and experiments described in a series of papers:
- “EigenGame: PCA as a Nash Equilibrium”, Ian Gemp, Brian McWilliams, Claire Vernade, Thore Graepel, ICLR (2021)
- “EigenGame Unloaded: When playing games is better than optimizing”, Ian Gemp, Brian McWilliams, Claire Vernade, Thore Graepel, ICLR (2022)
- “The Generalized Eigenvalue Problem as a Nash Equilibrium”, Ian Gemp, Charlie Chen, Brian McWilliams, ICLR (2023).

Charlie Chen was the primary author of the source code with guidance and support from Ian Gemp and Zhe Wang. Brian McWilliams and Ian Gemp wrote initial versions of the implementation which was used for inspiration. Sukhdeep Singh was the program manager for this project.

WARNING: This is a research-level release of a JAX implementation and is under active development.


## Installation

`pip install -e .` will install all required dependencies. This is best done
inside a [virtual environment](https://docs.python-guide.org/dev/virtualenvs/) (pip install virtualenv).

```shell
cd eigengame
virtualenv ~/venv/eigengame
source ~/venv/eigengame/bin/activate
pip install -e .
```

Note that the jaxlib version (which may be specified in `setup.py`) must correspond to the existing CUDA installation you wish to use. Please see the [JAX documentation](https://github.com/google/jax#installation) for more details.

## Usage

eigengame uses the `ConfigDict` from [ml_collections](https://github.com/google/ml_collections) to configure the system. A few example scripts are included under `eigengame/configs/`. These are mostly for testing so may need additional settings for a production-level calculation.

Taking the `synthetic_dataset_pca` as an example.

```shell
cd eigengame/examples/synthetic_dataset_pca
python experiment.py --config ./config.py --jaxline_mode train_eval_multithreaded
```

This will train EigenGame to find the top-256 eigenvectors of a dataset drawn
from a 1000 dimensional multivariate normal distribution. The system and
hyperparameters can be controlled by modifying the config file. Details of all
available config settings are in `eigengame/eg_base_config.py`.


Other systems can easily be set up, by creating a new config, data_pipeline,
and experiment file.

Note: to train on larger datasets with large batch sizes, multi-GPU
parallelisation is essential. This is supported via JAX's [pmap](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap). Multiple GPUs will be automatically detected and used if available.

## Output

Evaluation metrics such as cosine similarity are saved in
`[config.checkpoint_dir]/eval` as tensorboard event logs. Note that EigenGame
must be run with jaxline_mode=train_eval_multithreaded as indicated in the
example for these metrics to be saved.

The eigenvectors are saved to `[config.checkpoint_dir]` as `.npy` files
containing numpy arrays of shape `(num_devices, k, dimensionality)`.

## Giving Credit

If you use this code in your work, we would appreciate it if you please cite the associated papers. The initial paper details the architecture and results on a range of systems:

```
@inproceedings{gemp2021eigengame,
  author    = {Gemp, Ian and
               McWilliams, Brian and
               Vernade, Claire and
               Graepel, Thore},
  title     = {{EigenGame}: {PCA} as a {N}ash Equilibrium},
  booktitle = {International Conference on Learning Representations},
  year      = {2021}
}
```

```
@inproceedings{gemp2022eigengame,
  author    = {Gemp, Ian and
               McWilliams, Brian and
               Vernade, Claire and
               Graepel, Thore},
  title     = {{EigenGame} Unloaded: When playing games is better than optimizing},
  booktitle = {International Conference on Learning Representations},
  year      = {2022}
}
```

and an arXiv paper describes the most current implementation:

```
@article{gemp2023generalized,
  author    = {Gemp, Ian and
               Chen, Charlie and
               McWilliams, Brian},
  title     = {The Generalized Eigenvalue Problem as a {N}ash Equilibrium},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}
```

This repository can be cited using:

```
@software{eigengame_github,
  author  = {Chen, Charlie and
             Wang, Zhe and
             Gemp, Ian and
             Singh, Sukhdeep and
             {EigenGame} Contributors},
  title   = {{EigenGame}},
  url     = {http://github.com/deepmind/eigengame},
  year    = {2023}
}
```

## Disclaimer

This is not an official Google product.
