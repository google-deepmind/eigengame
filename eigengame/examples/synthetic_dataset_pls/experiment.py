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

"""Implementation of eigengame PLS on a trivial dataset."""

import functools
from typing import Callable, Dict, Iterator, Tuple

from absl import app
from absl import flags
import chex
from eigengame import eg_experiment
from eigengame import eg_objectives
from eigengame import eg_utils
from eigengame.examples.synthetic_dataset_pls import data_pipeline
import jax
from jaxline import platform
import ml_collections


FLAGS = flags.FLAGS


class Experiment(eg_experiment.AbstractEigenGameExperiment):
  """Run PLS on low dimensional synthetic data."""
  NON_BROADCAST_CHECKPOINT_ATTRS = {
      '_a_matrix': 'a_matrix',
      '_b_matrix': 'b_matrix',
      '_target_eigenvalues': 'target_eigenvalues',
      '_target_eigenvectors': 'target_eigenvectors',
      **eg_experiment.AbstractEigenGameExperiment.NON_BROADCAST_CHECKPOINT_ATTRS
  }

  def build_dataset(
      self,
      dataset_config: ml_collections.ConfigDict,
  ) -> Iterator[Tuple[chex.ArrayTree, chex.ArrayTree]]:
    """Initialize ground truths and returns iterator of samples."""
    # Initialize the ground truths
    (
        self._a_matrix,
        self._b_matrix,
        self._target_eigenvalues,
        self._target_eigenvectors,
    ) = data_pipeline.get_sharded_ground_truths(
        dataset_config.eigenvector_count,
        dataset_config.n_samples,
        dataset_config.n_latents,
        dataset_config.seed
    )
    global_batch_size = dataset_config.global_batch_size
    per_device_batch_size = global_batch_size // jax.local_device_count()
    def data_iterator(
    ) -> Iterator[Tuple[Dict[chex.Array, chex.Array],
                        Dict[chex.Array, chex.Array]]]:
      """Function to create the iterator which samples from the dataset."""
      sample_device_batch = data_pipeline.get_preprocessed_generator(
          batch_size=per_device_batch_size,
          repeat=dataset_config.repeat,
          seed=dataset_config.seed,
          n_samples=dataset_config.n_samples,
          n_latents=dataset_config.n_latents,
      )
      while True:
        batches = next(sample_device_batch)
        # shard batches and repackage
        sharded_batches = []
        for batch in batches:
          x = batch['x']
          y = batch['y']
          x_sharded = jax.device_put_sharded(list(x), jax.local_devices())
          y_sharded = jax.device_put_sharded(list(y), jax.local_devices())
          sharded_batch = {'x': x_sharded, 'y': y_sharded}
          sharded_batches.append(sharded_batch)
        yield tuple(batches)
    # We need a separate function call here, since the otherwise, the
    # initialization of the ground truths would be executed the first time
    # next() is called instead of when when build_dataset is called.
    return data_iterator()

  def build_preprocess_function(
      self,
      preprocess_config: ml_collections.ConfigDict,
  ) -> Callable[[chex.ArrayTree, chex.PRNGKey], chex.ArrayTree]:
    """Simple preprocessing."""
    def trivial_preprocess(
        batch: chex.ArrayTree,
        _: chex.PRNGKey,
    ) -> Tuple[eg_utils.SplitVector, eg_utils.SplitVector]:
      split_batch = [eg_utils.SplitVector(x=b['x'], y=b['y']) for b in batch]
      return tuple(split_batch)

    return trivial_preprocess

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
