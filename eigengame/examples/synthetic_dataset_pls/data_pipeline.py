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

"""Creates a correlated dataset of (x, y) pairs for PLS."""

from typing import Dict, Iterator, Tuple

import chex
from eigengame import eg_utils
import jax
import numpy as np
import scipy.linalg
import tensorflow as tf


SplitVector = eg_utils.SplitVector


def get_data_generator(
    batch_size: int,
    repeat: bool,
    shuffle_buffer: int = 1024,
    data_seed: int = 0,
    shuffle_seed: int = 12345,
    n_samples: int = 500,
    n_latents: int = 8,
    **kwargs
) -> Iterator[tf.Tensor]:
  """Pulls data and creates a generator using TensorFlow Datasets."""
  del kwargs
  x, y = generate_dataset(n_samples=n_samples,
                          n_latents=n_latents,
                          seed=data_seed)

  data_set = tf.data.Dataset.from_tensor_slices((x, y))

  data_set = data_set.shuffle(shuffle_buffer, shuffle_seed)
  data_set = data_set.prefetch(tf.data.AUTOTUNE)
  data_set = data_set.batch(batch_size, drop_remainder=True)
  data_set = data_set.batch(jax.local_device_count(), drop_remainder=True)
  if repeat:
    data_set = data_set.repeat()
  return iter(data_set)


def preprocess_sample(
    sample: Tuple[tf.Tensor, tf.Tensor]) -> Dict[chex.Array, chex.Array]:
  """Convert samples to chex Arrays."""
  x, y = sample
  output = {}
  output['x'] = x.numpy()
  output['y'] = y.numpy()
  return output  # pytype: disable=bad-return-type  # numpy-scalars


def get_preprocessed_generator(
    batch_size: int,
    repeat: bool,
    shuffle_buffer: int = 1024,
    seed: int = 0,
    n_samples: int = 500,
    n_latents: int = 8,
    **kwargs
) -> Iterator[Tuple[Dict[chex.Array, chex.Array],
                    Dict[chex.Array, chex.Array]]]:
  """Returns a generator which has been preprocessed."""
  del kwargs
  num_minibatches = 2  # need 2 independent minibatches for PLS
  rnd = np.random.RandomState(seed=seed)
  shuffle_seeds = [rnd.randint(12345, 999999) for _ in range(num_minibatches)]
  data_generators = []
  for shuffle_seed in shuffle_seeds:
    dg = get_data_generator(batch_size,
                            repeat,
                            shuffle_buffer,
                            seed,
                            shuffle_seed,
                            n_samples,
                            n_latents)
    data_generators.append(dg)
  for batches in zip(*data_generators):
    yield tuple([preprocess_sample(batch) for batch in batches])


def generate_dataset(n_samples: int = 500,
                     n_latents: int = 8,
                     seed: int = 0) -> Tuple[chex.Array, chex.Array]:
  """Generates the dataset."""
  # Synthetic example modified from sklearn example:
  # https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_compare_cross_decomposition.html

  rnd = np.random.RandomState(seed=seed)

  latents = [rnd.normal(size=n_samples) for _ in range(n_latents)]

  latents = np.array(latents).T
  latents = np.repeat(latents, repeats=2, axis=1)

  n_noise = 2 * n_latents * n_samples
  shape_noise = (n_samples, 2 * n_latents)
  x = latents + rnd.normal(size=n_noise).reshape(shape_noise)
  y = latents + rnd.normal(size=n_noise).reshape(shape_noise)

  return x, y


def generate_ground_truths(
    n_samples: int = 500,
    n_latents: int = 8,
    seed: int = 0) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
  """Generates the covariance matrix along with the true eigenvalue/vectors."""
  x, y = generate_dataset(n_samples=n_samples, n_latents=n_latents, seed=seed)
  n_samples, x_dim = x.shape
  y_dim = y.shape[1]
  dim = x_dim + y_dim

  covariance_xx = np.dot(x.T, x) / n_samples
  covariance_xy = np.dot(x.T, y) / n_samples

  a_matrix = np.zeros((dim, dim))
  a_matrix[:x_dim, x_dim:] = covariance_xy
  a_matrix[x_dim:, :x_dim] = covariance_xy.T

  b_matrix = np.eye(dim)
  b_matrix[:x_dim, :x_dim] = covariance_xx

  # Solve for Av = lambda v to get the ground_truths
  true_eigenvalues, true_eigenvectors = scipy.linalg.eigh(
      a_matrix,
      b_matrix
  )

  # Order the eigenvalues and vectors from biggest to smallest
  idxs = np.argsort(true_eigenvalues)[::-1]

  # You need to transpose this, since eigh returns eigenvectors on columns!
  true_eigenvectors = true_eigenvectors[:, idxs].T
  true_eigenvalues = true_eigenvalues[idxs]
  return (
      a_matrix,
      b_matrix,
      true_eigenvalues,
      true_eigenvectors,
  )


def get_sharded_ground_truths(
    total_eigenvector_count: int,
    n_samples: int = 500,
    n_latents: int = 8,
    seed: int = 0
) -> Tuple[chex.ArraySharded, chex.ArraySharded, chex.ArraySharded,
           SplitVector]:
  """Shards the ground truths to different machines."""
  (
      a_matrix,
      b_matrix,
      true_eigenvalues,
      true_eigenvectors,
  ) = generate_ground_truths(n_samples=n_samples,
                             n_latents=n_latents,
                             seed=seed)
  shard_shape = (
      jax.local_device_count(),
      total_eigenvector_count // jax.local_device_count(),
  )
  dim = a_matrix.shape[0]
  # We need to shard the eigenvalues and eigenvectors to the corresponding
  # machines responsible for them.
  true_eigenvalues = true_eigenvalues[:total_eigenvector_count].reshape(
      shard_shape,)
  true_eigenvectors = true_eigenvectors[:total_eigenvector_count].reshape(
      shard_shape + (dim,),)
  x_dim = n_latents * 2
  x_eigenvector, y_eigenvector = np.split(
      true_eigenvectors,
      (x_dim,),
      axis=-1,
  )

  # The true eigenvectors also need to be converted to SplitVector.
  true_generalized_eigenvectors = SplitVector(
      x=jax.device_put_sharded(
          list(x_eigenvector),
          jax.local_devices(),
      ),
      y=jax.device_put_sharded(
          list(y_eigenvector),
          jax.local_devices(),
      ))
  true_eigenvalues = jax.device_put_sharded(
      list(true_eigenvalues),
      jax.local_devices(),
  )
  return (a_matrix, b_matrix, true_eigenvalues, true_generalized_eigenvectors)
