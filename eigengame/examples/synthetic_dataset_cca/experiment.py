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

"""Implementation of eigengame CCA on a trivial dataset."""
import functools
from typing import Callable, Dict, Iterator, Tuple

from absl import app
from absl import flags
import chex
from eigengame import eg_experiment
from eigengame import eg_objectives
from eigengame import eg_utils
from eigengame.examples.synthetic_dataset_cca import data_pipeline
import jax
import jax.numpy as jnp
from jaxline import platform
import ml_collections

FLAGS = flags.FLAGS


class Experiment(eg_experiment.AbstractEigenGameExperiment):
  """Run CCA on low dimensional synthetic data."""
  NON_BROADCAST_CHECKPOINT_ATTRS = {
      '_covariance': 'covariance',
      '_target_eigenvalues': 'target_eigenvalues',
      '_target_eigenvectors': 'target_eigenvectors',
      **eg_experiment.AbstractEigenGameExperiment.NON_BROADCAST_CHECKPOINT_ATTRS
  }

  def build_dataset(
      self,
      dataset_config: ml_collections.ConfigDict,
  ) -> Iterator[chex.ArrayTree]:
    """Initialize ground truths and returns iterator of samples."""
    # Initialize the ground truths
    key = jax.random.PRNGKey(dataset_config.seed)
    if jax.host_count() > 1:
      # In the case of multihost training, we want deach host to get a different
      # sample.
      key = jax.random.split(key, jax.host_count())[jax.host_id()]
    (
        self._covariance,
        self._target_eigenvalues,
        self._target_eigenvectors,
    ) = data_pipeline.get_sharded_ground_truths(
        key,
        dataset_config.eigenvector_count,
        dataset_config.x_size,
        dataset_config.y_size,
    )
    global_batch_size = dataset_config.global_batch_size
    per_device_batch_size = global_batch_size // jax.device_count()
    def data_iterator(key: chex.PRNGKey):
      """Function to create the iterator which samples from the distribution."""
      sample_from_key = jax.pmap(
          functools.partial(
              data_pipeline.generate_correlated_data,
              x_size=dataset_config.x_size,
              y_size=dataset_config.y_size,
              covariance=self._covariance,
              batch_size=per_device_batch_size,
          ),)
      while True:
        num_keys = jax.local_device_count() + 1
        key, *sharded_keys = jax.random.split(key, num_keys)
        batch = sample_from_key(jnp.asarray(sharded_keys))
        # some experiment types (e.g., CCA) require multiple i.i.d. samples
        # to construct unbiased gradient estimates. see third_party/py/eigegame/
        # eg_experiment.py/get_experiment_type for self._num_samples info
        if self._num_samples <= 1:  # num_samples determined by experiment_type
          yield batch
        else:
          batches = [batch]
          for _ in range(self._num_samples -1):
            key, *sharded_keys = jax.random.split(key, num_keys)
            batch = sample_from_key(jnp.asarray(sharded_keys))
            batches.append(batch)
          yield tuple(batches)
    # We need a separate function call here, since the otherwise, the
    # initialization of the ground truths would be executed the first time
    # next() is called instead of when when build_dataset is called.
    return data_iterator(key)

  def build_preprocess_function(
      self,
      preprocess_config: ml_collections.ConfigDict,
  ) -> Callable[[chex.ArrayTree, chex.PRNGKey], chex.ArrayTree]:
    """No need to do any preprocessing."""
    return lambda batch, _: batch

  @functools.partial(
      jax.pmap,
      in_axes=0,
      out_axes=0,
      axis_name='devices',
      static_broadcasted_argnums=0,
  )
  def _eval_similarity(
      self,
      eigenvectors: chex.Array,
      target_vectors: chex.Array,
  ) -> Tuple[chex.Array, chex.Array]:
    """pmaps the cosine similarity function."""
    cosine_similarity = eg_objectives.cosine_similarity(
        eigenvectors,
        target_vectors,
    )
    return cosine_similarity  # pytype: disable=bad-return-type  # numpy-scalars

  def evaluate(
      self,
      global_step: int,
      rng: chex.Array,
      **unused_kwargs,
  ) -> Dict[str, chex.Array]:
    """Override the evaluate function to return cosine similarity."""
    replicated_cosine_similarity = self._eval_similarity(
        self._eigenvectors, self._target_eigenvectors)
    cosine_similarities = eg_utils.get_first(replicated_cosine_similarity)
    return eg_utils.per_vector_metric_log(  # pytype: disable=bad-return-type  # numpy-scalars
        'cosine_similarity',
        cosine_similarities,
    )

if __name__ == '__main__':
  flags.mark_flag_as_required('config')
  app.run(functools.partial(
      platform.main, Experiment,
      checkpointer_factory=eg_experiment.create_checkpointer))
