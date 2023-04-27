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

"""Creates a trivial dataset of pair of correlated normals for the PCA."""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg


def generate_ground_truths(
    key: chex.PRNGKey,
    dim: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
  """Generates the covariance matrix along with the true eigenvalue/vectors."""
  # Create a random gram matrix for our covariance.
  random_matrix = jax.random.normal(key, shape=(dim, dim))
  covariance_matrix = np.einsum(
      'dn, dm -> nm',
      random_matrix,
      random_matrix,
  ) / np.sqrt(dim)

  # Solve for Av = lambda v to get the ground_truths
  true_eigenvalues, true_eigenvectors = scipy.linalg.eigh(
      covariance_matrix,
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


def generate_data(
    key: chex.PRNGKey,
    dim: int,
    covariance: chex.Array,
    batch_size: int,
) -> chex.Array:
  """Returns a vector of data given a covariance matrix."""
  data = jax.random.multivariate_normal(
      key=key,
      mean=jnp.zeros(dim),
      cov=covariance,
      shape=(batch_size,),
  )
  return data


def get_sharded_ground_truths(
    key: chex.PRNGKey,
    total_eigenvector_count: int,
    dim: int,
) -> Tuple[chex.ArraySharded, chex.ArraySharded, chex.ArraySharded]:
  """Shards the ground truths to different machines."""
  (
      covariance_matrix,
      true_eigenvalues,
      true_eigenvectors,
  ) = generate_ground_truths(
      key,
      dim,
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
      shard_shape + (dim,),)

  # The true eigenvectors also need to be converted to CCAVector.
  true_generalized_eigenvectors = jax.device_put_sharded(
      list(true_eigenvectors),
      jax.local_devices(),
  )
  true_eigenvalues = jax.device_put_sharded(
      list(true_eigenvalues),
      jax.local_devices(),
  )
  return covariance_matrix, true_eigenvalues, true_generalized_eigenvectors  # pytype: disable=bad-return-type  # numpy-scalars
