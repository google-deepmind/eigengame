# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Boilerplate needed to do a jaxline experiment with Eigengame."""
import abc
import functools
from typing import Callable, Dict, Iterator, Optional, Tuple

import chex
from eigengame import eg_gradients
from eigengame import eg_objectives
from eigengame import eg_utils
import jax
import jax.numpy as jnp
from jaxline import experiment
from jaxline import utils
import ml_collections
import optax


def get_experiment_type(
    experiment_type: eg_utils.ExperimentType
) -> Tuple[eg_utils.EigenGameGradientFunction,
           eg_utils.EigenGameQuotientFunction,
           int]:
  """Selects the a gradient function and default evaluation functions.

  More experiment types may be added here as their respective matrix products
  are implemented.

  Args:
    experiment_type: string value which selects the model type

  Returns:
    Returns two functions and an integer. The first function evaluates the
    gradients and is used in AbstractEigenGameExperiment to update the
    eigenvectors for the given type of experiment. The latter is used to
    evaluate the numerator and denominator of the rayleigh quotient, which
    serves as estimates for the generalized eigenvalues. The last value, the
    integer, indicates how many independent minibatches the functions expect as
    input.
  """
  if experiment_type == eg_utils.ExperimentType.BIASED_CCA:
    return (
        eg_gradients.biased_cca_gradients,
        eg_objectives.biased_cca_rayleigh_quotient_components,
        1
    )
  elif experiment_type == eg_utils.ExperimentType.CCA:
    return (
        eg_gradients.unbiased_cca_gradients,
        eg_objectives.unbiased_cca_rayleigh_quotient_components,
        2
    )
  elif experiment_type == eg_utils.ExperimentType.PLS:
    return (
        eg_gradients.unbiased_pls_gradients,
        eg_objectives.unbiased_pls_rayleigh_quotient_components,
        2
    )
  elif experiment_type == eg_utils.ExperimentType.PCA:
    return (
        eg_gradients.pca_generalized_eigengame_gradients,
        eg_objectives.pca_generalised_eigengame_rayleigh_quotient_components,
        1
    )
  elif experiment_type == eg_utils.ExperimentType.ICA:
    # Not yet tested
    return (
        eg_gradients.unbiased_ica_gradients,
        eg_objectives.unbiased_ica_rayleigh_quotient_components,
        3
    )
  elif experiment_type == eg_utils.ExperimentType.MATRIX_INVERSE:
    return (
        eg_gradients.matrix_inverse_gradients,
        eg_objectives.matrix_inverse_rayleigh_quotient_components,
        1
    )
  else:
    raise ValueError('Please specify a valid experiment Type e.g. "cca"')


def decaying_schedule_with_warmup(
    step: int,
    warm_up_step: int,
    end_step: int,
    base_lr: float,
    end_lr: float,
) -> float:
  """Learning rate schedule with warmup and harmonic decay.

  We have a warmup period for the eigenvalues as the auxiliary values and the
  mean estimates are learned. During this period, the learning rate increases
  linearly until it reaches the base learning rate after the period ends.

  This is followed by an harmonically decaying learning rate which end_lr at
  the end_state.

  Args:
    step: global step of the schedule
    warm_up_step: number of warmup steps
    end_step: step at which end_lr is reached
    base_lr: maximum learning rate. Reached when warmup finishes.
    end_lr: learning rate at end_step

  Returns:
    The learning rate at the current step:
  """
  warmup_lr = step * base_lr / warm_up_step
  # calculate shift and scale such that scale/(step+shift) satisfies
  # schedule(warm_up_step) = base_lr and
  # schedule(end_step) = end_lr
  decay_shift = (warm_up_step * base_lr - end_step * end_lr) / (
      end_lr - base_lr)
  decay_scale = base_lr * (warm_up_step + decay_shift)
  decay_lr = decay_scale / (step + decay_shift)
  return jnp.where(step < warm_up_step, warmup_lr, decay_lr)


def create_checkpointer(
    config: ml_collections.ConfigDict,
    mode: str,
) -> utils.Checkpointer:
  """Creates an object to be used as a checkpointer."""
  return eg_utils.InMemoryCheckpointerPlusSaveEigVecs(config, mode)


class AbstractEigenGameExperiment(experiment.AbstractExperiment):
  """Jaxline object for running Eigengame Experiments."""

  NON_BROADCAST_CHECKPOINT_ATTRS = {
      '_eigenvectors': 'eigenvectors',
      '_auxiliary_variables': 'auxiliary_variables',
      '_opt_state': 'opt_state',
      '_aux_opt_state': 'aux_opt_state',
      '_mean_estimate': 'mean_estimate',
      '_mean_opt_state': 'mean_opt_state',
  }

  def __init__(self, mode: str, init_rng: chex.Array,
               config: ml_collections.ConfigDict):
    super().__init__(mode=mode, init_rng=init_rng)
    # Get a different seed for each host
    if jax.process_count() > 1:
      init_rngs = jax.random.split(init_rng, jax.process_count())
      init_rng = init_rngs[jax.process_index()]
    self._eigenvector_count = config.eigenvector_count
    self._epsilon = config.epsilon
    self._maximize = config.maximize
    self._track_mean = config.track_mean

    self._net_activations = self.build_preprocess_function(
        config.preprocess_config,)
    self.data_config = config.dataset_config
    self._data_generator = self.build_dataset(self.data_config)

    (
        self._gradient_function,
        self._rayleigh_quotient_function,
        self._num_samples,
    ) = get_experiment_type(config.experiment_type)

    # Initialize the eigenvalues and mean estimate from a batch of data
    (
        self._eigenvectors,
        self._mean_estimate,
        self._auxiliary_variables,
    ) = self._initialize_eigenvector_params(init_rng)

    if self._track_mean:
      self._mean_opt = optax.sgd(lambda step: 1 / (step + 1))
      self._mean_opt_state = jax.pmap(self._mean_opt.init)(self._mean_estimate,)
    else:
      self._mean_opt = None
      self._mean_opt_state = None

    if mode == 'train':
      # Setup the update function
      self._update = self._build_update_function()

      # Initialize the data generators and optimizers
      self._optimizer = optax.adam(
          functools.partial(decaying_schedule_with_warmup,
                            **config.optimizer_schedule_config),
          **config.optimizer_config)
      self._opt_state = jax.pmap(self._optimizer.init)(self._eigenvectors)

      #  Create optimizer for the auxiliary variables. Don't use ADAM for this
      #  at the same time as ADAM for the main optimiser! It may cause the
      #  experiment to become unstable.
      self._aux_optimizer = optax.sgd(**config.aux_optimizer_config)
      self._aux_opt_state = jax.pmap(self._aux_optimizer.init)(
          self._auxiliary_variables,)

      #  Create optimizer for the mean estimate
      #  This effectively calculates the mean up til the latest step.
      self._eval_batches = None
    else:
      self._optimizer = None
      self._opt_state = None
      self._aux_optimizer = None
      self._aux_opt_state = None
      self._update = None
      self._eval_batches = config.eval_batches

  @abc.abstractmethod
  def build_preprocess_function(
      self,
      preprocess_config: ml_collections.ConfigDict,
  ) -> Callable[[chex.ArrayTree, chex.PRNGKey], chex.ArrayTree]:
    """Build a pmappable function which is called on all machines in parallel.

    This function will be pmapped inside the updatefunction, and called
    immediately on the batch taken from built_dataset.

    Args:
      preprocess_config: config dict specified in the experiment configs.
        Contains parameters for this function.

    Returns:
      Pmappable function which takes in a local batch of data from build_dataset
      of shape [per_device_batch_size, ...] and a rng key. Returns preprocessed
      batch of shape [per_device_batch_size, ...]
    """

  @abc.abstractmethod
  def build_dataset(
      self,
      dataset_config: ml_collections.ConfigDict,
  ) -> Iterator[chex.ArrayTree]:
    """Iterator which continuously returns the batches of the dataset.

    Args:
      dataset_config: config dict specified in the experiment configs. Contains
        parameters for this function.

    Returns:
      Iterator which will return batches of data which will be sharded across
      across machines. This means we need pytrees of shape:
      [num_local_devices, per_device_batch_size, ...]
    """

  def _initialize_eigenvector_params(
      self,
      init_rng: chex.PRNGKey,
  ) -> chex.ArrayTree:
    """Initializes the eigenvalues, mean estimates and auxiliary variables."""
    init_batch = next(self._data_generator)
    local_init_data = eg_utils.get_first(init_batch)
    model_rng, eigenvector_rng = jax.random.split(init_rng, 2)
    local_activiation_batch = self._net_activations(local_init_data, model_rng)
    if self._num_samples > 1:
      local_activiation_batch = local_activiation_batch[0]
    initial_eigenvectors = eg_utils.initialize_eigenvectors(
        self._eigenvector_count,
        local_activiation_batch,
        eigenvector_rng,
    )

    initial_mean_estimate = jax.device_put_replicated(
        jax.tree_map(
            lambda x: jnp.zeros(x.shape[1:]),
            local_activiation_batch,
        ),
        jax.local_devices(),
    )
    auxiliary_variables = eg_utils.init_aux_variables(
        jax.device_count(),
        initial_eigenvectors,
    )
    return initial_eigenvectors, initial_mean_estimate, auxiliary_variables

  def _build_update_function(self):
    """pmaps and applies masks to the update functions."""
    sliced_identity = eg_gradients.create_sharded_identity(
        self._eigenvector_count)
    mask = eg_gradients.create_sharded_mask(self._eigenvector_count)
    return functools.partial(
        jax.pmap(
            self._update_eigenvectors,
            axis_name='devices',
            in_axes=0,
            out_axes=0,
        ),
        mask=mask,
        sliced_identity=sliced_identity)

  def _update_eigenvectors(
      self,
      local_eigenvectors: chex.ArrayTree,
      opt_state: chex.ArrayTree,
      auxiliary_variables: Optional[eg_utils.AuxiliaryParams],
      aux_opt_state: Optional[eg_utils.AuxiliaryParams],
      batch: chex.Array,
      mean_estimate: Optional[chex.ArrayTree],
      mean_opt_state: Optional[chex.ArrayTree],
      rng: chex.PRNGKey,
      mask: chex.Array,
      sliced_identity: chex.Array,
  ) -> Tuple[chex.ArrayTree, chex.ArrayTree, eg_utils.AuxiliaryParams,
             eg_utils.AuxiliaryParams, Optional[chex.ArrayTree],
             Optional[chex.ArrayTree],]:
    """Calculates the new vectors, applies update and then renormalize."""
    # Generate activations from the batch of data.
    data = self._net_activations(batch, rng)

    # Calculate the gradient and the new auxiliary variable values.
    gradient, new_aux = self._gradient_function(
        local_eigenvectors=local_eigenvectors,
        sharded_data=data,
        auxiliary_variables=auxiliary_variables,
        mask=mask,
        sliced_identity=sliced_identity,
        mean_estimate=mean_estimate,
        epsilon=self._epsilon,
        maximize=self._maximize
    )
    # Update and normalize the eigenvectors variables.
    update, new_opt_state = self._optimizer.update(
        # (TODO ccharlie) implement __neg__ for this object when overhauling it
        jax.tree_map(lambda x: -x, gradient),
        opt_state,
    )
    new_eigenvectors = optax.apply_updates(local_eigenvectors, update)
    new_eigenvectors = eg_utils.normalize_eigenvectors(new_eigenvectors)

    # Update the auxiliary as well. In this case we're minimising the
    # squared error between the new target and the old.
    auxiliary_error = jax.tree_map(
        lambda x, y: x - y,
        auxiliary_variables,
        new_aux,
    )
    aux_update, new_aux_opt_state = self._aux_optimizer.update(
        auxiliary_error,
        aux_opt_state,
    )
    new_aux_value = optax.apply_updates(auxiliary_variables, aux_update)

    if self._track_mean:
      # The mean also needs to be estimated -- since we're looking at the
      # covariances we need the data to be centered.
      if self._num_samples == 1:
        data_tuple = (data,)
      else:
        data_tuple = data
      minibatch_mean = lambda x: jnp.mean(x, axis=0)
      ind_batch_mean = lambda *x: sum(x) / len(x)  # independent batches
      # average over independent sample dim, minibatch dim, device dim
      batch_mean_estimate = jax.lax.pmean(
          jax.tree_util.tree_map(minibatch_mean,
                                 jax.tree_util.tree_map(ind_batch_mean,
                                                        *data_tuple)),
          axis_name='devices',
      )
      mean_error = jax.tree_map(
          lambda x, y: x - y,
          mean_estimate,
          batch_mean_estimate,
      )
      mean_update, new_mean_opt_state = self._mean_opt.update(
          mean_error,
          mean_opt_state,
      )
      new_mean_estimate = optax.apply_updates(mean_estimate, mean_update)
    else:
      new_mean_opt_state = None
      new_mean_estimate = None

    return (  # pytype: disable=signature-mismatch  # jax-ndarray
        new_eigenvectors,
        new_opt_state,
        new_aux_value,
        new_aux_opt_state,
        new_mean_estimate,
        new_mean_opt_state,
    )

  def get_eigenvectors(self) -> chex.ArrayTree:
    """Returns the current eigenvectors as jax array."""
    return self._eigenvectors

  def step(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      global_step: int,
      rng: chex.Array,
      **unused_kwargs,
  ) -> Dict[str, chex.Array]:
    """Calls the update function on the eigen vectors and aux variables."""
    batch = next(self._data_generator)

    (
        self._eigenvectors,
        self._opt_state,
        self._auxiliary_variables,
        self._aux_opt_state,
        self._mean_estimate,
        self._mean_opt_state,
    ) = self._update(
        self._eigenvectors,
        self._opt_state,
        self._auxiliary_variables,
        self._aux_opt_state,
        batch,
        self._mean_estimate,
        self._mean_opt_state,
        rng,
    )
    return {}

  @functools.partial(
      jax.pmap,
      in_axes=0,
      out_axes=0,
      axis_name='devices',
      static_broadcasted_argnums=0,
  )
  def _eval_eigenvalues(
      self,
      local_eigenvectors: chex.ArrayTree,
      batch: chex.Array,
      mean_estimate: Optional[chex.ArrayTree],
      rng: chex.PRNGKey,  # pytype: disable=signature-mismatch  # jax-ndarray
  ) -> Tuple[chex.Array, chex.Array]:
    """pmaps the cosine similarity function."""

    data = self._net_activations(batch, rng)
    return self._rayleigh_quotient_function(
        local_eigenvectors,
        data,
        mean_estimate,
        self._epsilon,
        self._maximize,
    )

  def evaluate(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      global_step: int,
      rng: chex.Array,
      **unused_kwargs,
  ) -> Dict[str, chex.Array]:
    """Calculate the eigenvalues for each eigenvector."""
    numerator, denominator = 0, 0
    self._data_generator = self.build_dataset(self.data_config)
    for _ in range(self._eval_batches):
      batch = next(self._data_generator)
      new_numerator, new_denominator = self._eval_eigenvalues(
          self._eigenvectors, batch, self._mean_estimate, rng)
      numerator += eg_utils.get_first(new_numerator)
      denominator += eg_utils.get_first(new_denominator)
    eigenvalues = numerator / denominator
    return eg_utils.per_vector_metric_log('eigenvalue', eigenvalues)  # pytype: disable=bad-return-type  # numpy-scalars
