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

"""Creates a trivial dataset of pair of correlated normals for the CCA."""

from typing import Tuple, Union

import chex
from eigengame import eg_utils
import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg

SplitVector = eg_utils.SplitVector


def generate_ground_truths(
    key: chex.PRNGKey,
    x_size: int,
    y_size: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
  """Generates the covariance matrix along with the true eigenvalue/vectors."""
  total_len = x_size + y_size

  # Create a matrix with 1's on the upper and bottom right squares.
  variance_mask = np.zeros(shape=(total_len, total_len))
  # This usually wouldn't work on TPU, but we only run this once before training
  variance_mask[:x_size, :x_size] = 1
  variance_mask[x_size:, x_size:] = 1

  # Create a random gram matrix for our covariance.
  random_matrix = jax.random.normal(key, shape=(total_len, total_len))
  covariance_matrix = np.einsum(
      'dn, dm -> nm',
      random_matrix,
      random_matrix,
  ) / np.sqrt(total_len)

  variance_matrix = variance_mask * covariance_matrix  # B
  interaction_matrix = (1 - variance_mask) * covariance_matrix  # A
  # Solve for Av = lambda Bv to get the ground_truths
  true_eigenvalues, true_eigenvectors = scipy.linalg.eigh(
      interaction_matrix,
      variance_matrix,
  )

  # Order the eigenvalues and vectors from biggest to smallest
  idxs = np.argsort(true_eigenvalues)[::-1]

  # You need to transpose this, since eigh returns eigenvectors on columns!
  true_eigenvectors = true_eigenvectors[:, idxs].T
  true_eigenvalues = true_eigenvalues[idxs]
  return (
      covariance_matrix,
      true_eigenvalues,
      true_eigenvectors,
  )


def generate_correlated_data(
    key: chex.PRNGKey,
    x_size: int,
    y_size: int,
    covariance: chex.Array,
    batch_size: int,
) -> SplitVector:
  """Returns a pair of correlated data given a covariance matrix."""
  dimension = x_size + y_size
  merged_data = jax.random.multivariate_normal(
      key=key,
      mean=jnp.zeros(dimension),
      cov=covariance,
      shape=(batch_size,),
  )
  # In general splits are really slow on TPU and should be avoided. However,
  # since this is a trivial small synthetic example, it doesn't matter too much.
  x_data, y_data = jnp.split(
      merged_data,
      (x_size,),
      axis=-1,
  )
  return SplitVector(x=x_data, y=y_data)


def get_sharded_ground_truths(
    key: chex.PRNGKey,
    total_eigenvector_count: int,
    x_size: int,
    y_size: int,
) -> Tuple[chex.ArraySharded, chex.ArraySharded, SplitVector]:
  """Shards the ground truths to different machines."""
  (
      covariance_matrix,
      true_eigenvalues,
      true_eigenvectors,
  ) = generate_ground_truths(
      key,
      x_size,
      y_size,
  )
  shard_shape = (
      jax.local_device_count(),
      total_eigenvector_count // jax.local_device_count(),
  )
  # We need to shard the eigenvalues and eigenvectors to the corresponding
  # machines responsible for them.
  true_eigenvalues = true_eigenvalues[:total_eigenvector_count].reshape(
      shard_shape,)
  true_eigenvectors = true_eigenvectors[:total_eigenvector_count].reshape(
      shard_shape + (x_size + y_size,),)
  x_eigenvector, y_eigenvector = np.split(
      true_eigenvectors,
      (x_size,),
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
  return covariance_matrix, true_eigenvalues, true_generalized_eigenvectors


def linear_correlation_data(
    rng: chex.PRNGKey,
    data_dimension: int,
    batch_size: int,
    diagonal_variance: Union[float, chex.Array] = 1
) -> SplitVector:
  """Generate high dimensional correlated data with known ground truths.

  We generate two samples independent, X_1 and X_2 with a diagonal variance. Our
  correlated data is thereby:
    X = X_1
    Y = aX_1 + bX_2
  Where a^2 + b^2 = 1. This means that the correlation equal to a. This can be
  sampled, and we can get the ground truths without dealing with a massive
  covariance matrix. a is a linear range, so we get a range of spectrums.

  The ground truth eigenvectors can be generated by the
  get_linearly_correlated_eigenvectors function.
  Args:
    rng: RNG key for jax to generate samples.
    data_dimension: int denoting the data dimension for the x and y data
    batch_size: Number of samples we generate.
    diagonal_variance: Defines the varianes of X and Y. This is applied
      identically to the two data sources.
  Returns:
    CCA vector containing correlated data with a linear spectrum
  """
  keys = jax.random.split(rng, 2)
  correlation = np.linspace(1, 0, num=data_dimension, endpoint=True)
  noise = np.sqrt(1 - correlation**2)
  x = jax.random.normal(
      keys[1], shape=(
          batch_size,
          data_dimension,
      )) * diagonal_variance
  y = jax.random.normal(
      keys[0], shape=(
          batch_size,
          data_dimension,
      )) * diagonal_variance
  return SplitVector(
      x=x,
      y=noise * y + correlation * x,
  )


def get_linearly_correlated_eigenvectors(
    data_dimension: int,
    eigenvector_count: int,
) -> chex.Array:
  """Ground truth Eigenvalues for linearly correlated data."""
  ground_truth = jnp.reshape(
      jnp.eye(eigenvector_count, data_dimension),
      (
          jax.device_count(),
          eigenvector_count // jax.device_count(),
          data_dimension,
      ),
  )
  normalized_vector = eg_utils.normalize_eigenvectors(
      SplitVector(
          x=ground_truth,
          y=ground_truth,
      ))
  return jax.device_put_sharded(
      normalized_vector,
      jax.devices(),
  )
