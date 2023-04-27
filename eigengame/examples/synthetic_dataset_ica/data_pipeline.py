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

"""Creates a dataset of mixed periodic signals for ICA."""

from typing import Iterator, Tuple

import chex
import jax
import numpy as np
import scipy
from scipy import signal
import tensorflow as tf


def get_data_generator(
    batch_size: int,
    repeat: bool,
    shuffle_buffer: int = 1024,
    data_seed: int = 0,
    shuffle_seed: int = 12345,
    n_samples: int = 2000,
    **kwargs
) -> Iterator[tf.Tensor]:
  """Pulls data and creates a generator using TensorFlow Datasets."""
  del kwargs
  mixed_sources = generate_dataset(n_samples=n_samples, seed=data_seed)

  data_set = tf.data.Dataset.from_tensor_slices(mixed_sources)

  data_set = data_set.shuffle(shuffle_buffer, shuffle_seed)
  data_set = data_set.prefetch(tf.data.AUTOTUNE)
  data_set = data_set.batch(batch_size, drop_remainder=True)
  data_set = data_set.batch(jax.local_device_count(), drop_remainder=True)
  if repeat:
    data_set = data_set.repeat()
  return iter(data_set)


def preprocess_sample(
    sample: tf.Tensor) -> chex.Array:
  """Convert samples to chex Arrays."""
  return sample.numpy()


def get_preprocessed_generator(
    batch_size: int,
    repeat: bool,
    shuffle_buffer: int = 1024,
    seed: int = 0,
    n_samples: int = 2000,
    **kwargs
) -> Iterator[Tuple[chex.Array, chex.Array, chex.Array]]:
  """Returns a generator which has been preprocessed."""
  del kwargs
  num_minibatches = 3  # need 3 independent minibatches for ICA
  rnd = np.random.RandomState(seed=seed)
  shuffle_seeds = [rnd.randint(12345, 999999) for _ in range(num_minibatches)]
  data_generators = []
  for shuffle_seed in shuffle_seeds:
    dg = get_data_generator(batch_size,
                            repeat,
                            shuffle_buffer,
                            seed,
                            shuffle_seed,
                            n_samples)
    data_generators.append(dg)
  for batches in zip(*data_generators):
    yield tuple([preprocess_sample(batch) for batch in batches])


def generate_dataset(n_samples: int = 2000, seed: int = 0) -> chex.Array:
  """Generates the dataset."""
  # Synthetic example modified from sklearn example:
  # https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html

  # Generate three (+1) signals, linearly mix them, and then add Guassian noise
  time = np.linspace(0, 8, n_samples)

  s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
  s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
  s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal
  # Dummy signal so k % num_device = 0 for tpu test
  s4 = np.cos(time)  # Signal 4: sinusoidal signal
  s5 = np.cos(4 * time)  # Signal 5: sinusoidal signal
  s6 = np.cos(8 * time)  # Signal 6: sinusoidal signal
  s7 = np.cos(16 * time)  # Signal 7: sinusoidal signal
  s8 = np.cos(32 * time)  # Signal 8: sinusoidal signal

  # Stack signals on columns
  sources = np.c_[s1, s2, s3, s4, s5, s6, s7, s8]

  # Add noise
  rnd = np.random.RandomState(seed=seed)
  sources += 0.2 * rnd.normal(size=sources.shape)

  sources /= sources.std(axis=0)  # Standardize data

  # Mix data
  mix = np.zeros((8, 8))
  mix[3:, 3:] = np.eye(5)
  mix_main = np.array([[1.0, 1.0, 1.0],
                       [0.5, 2.0, 1.0],
                       [1.5, 1.0, 2.0]])  # Mixing matrix
  mix[:3, :3] = mix_main
  mixed_sources = np.dot(sources, mix.T)  # Generate observations

  return mixed_sources


def generate_ground_truths(
    n_samples: int = 2000,
    seed: int = 0) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
  """Generates the covariance matrix along with the true eigenvalue/vectors."""
  mixed_sources = generate_dataset(n_samples=n_samples, seed=seed)
  n_samples = mixed_sources.shape[0]

  covariance = np.dot(mixed_sources.T, mixed_sources) / n_samples
  kurtosis = sum([np.outer(xi, xi) * np.inner(xi, xi) for xi in mixed_sources])
  kurtosis /= n_samples
  kurtosis -= np.trace(covariance) * covariance + 2 * covariance.dot(covariance)

  # Solve for Av = lambda v to get the ground_truths
  true_eigenvalues, true_eigenvectors = scipy.linalg.eigh(
      kurtosis,
      covariance
  )

  # Order the eigenvalues and vectors from smallest to biggest
  idxs = np.argsort(true_eigenvalues)

  # You need to transpose this, since eigh returns eigenvectors on columns!
  true_eigenvectors = true_eigenvectors[:, idxs].T
  true_eigenvalues = true_eigenvalues[idxs]
  return (
      kurtosis,
      covariance,
      true_eigenvalues,
      true_eigenvectors,
  )


def get_sharded_ground_truths(
    total_eigenvector_count: int,
    n_samples: int = 2000,
    seed: int = 0
) -> Tuple[chex.ArraySharded, chex.ArraySharded, chex.ArraySharded,
           chex.ArraySharded]:
  """Shards the ground truths to different machines."""
  (
      kurtosis_matrix,
      covariance_matrix,
      true_eigenvalues,
      true_eigenvectors,
  ) = generate_ground_truths(n_samples=n_samples, seed=seed)
  shard_shape = (
      jax.device_count(),
      total_eigenvector_count // jax.device_count(),
  )
  dim = kurtosis_matrix.shape[0]
  # We need to shard the eigenvalues and eigenvectors to the corresponding
  # machines responsible for them.
  true_eigenvalues = true_eigenvalues[:total_eigenvector_count].reshape(
      shard_shape,)
  true_eigenvectors = true_eigenvectors[:total_eigenvector_count].reshape(
      shard_shape + (dim,),)

  # The true eigenvectors also need to be converted to CCAVector.
  true_generalized_eigenvectors = jax.device_put_sharded(
      list(true_eigenvectors),
      jax.devices(),
  )
  true_eigenvalues = jax.device_put_sharded(
      list(true_eigenvalues),
      jax.devices(),
  )
  return (kurtosis_matrix, covariance_matrix, true_eigenvalues,  # pytype: disable=bad-return-type  # numpy-scalars
          true_generalized_eigenvectors)
