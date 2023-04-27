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

"""Different Eigengame gradients and their associated helpers.

einsum legend:
  ...: the massive data dimension over which we're running CCA/ICA/whatever
  other method based on GEVP we're doing. This data may not be contiguious, and
  may consist of arbitrary shapes in a pytree. Hence we use ellipses to denote
  it here.
  l: local eigenvector index -- the number of eigenvectors per machine.
  k: global eigenvector index -- number of eigenvectors over all machines.
  b: batch dimension, batch size per machine.
"""
from typing import Optional, Tuple

import chex
from eigengame import eg_gradients
from eigengame import eg_utils
import jax
import jax.numpy as jnp
import scipy


def cosine_similarity(
    eigenvectors: chex.ArrayTree,
    target_vectors: chex.ArrayTree,
) -> chex.Array:
  """Calculate the cosine similarity on each machine then replicate to all."""
  normed_targets = eg_utils.normalize_eigenvectors(target_vectors)
  similarity = eg_utils.tree_einsum(
      'l..., l... -> l',
      normed_targets,
      eigenvectors,
      reduce_f=lambda x, y: x + y,
  )
  return jax.lax.all_gather(
      similarity,
      axis_name='devices',
      axis=0,
      tiled=True,
  )


def unbiased_cca_rayleigh_quotient_components(
    local_eigenvectors: eg_utils.SplitVector,
    sharded_data: Tuple[eg_utils.SplitVector, eg_utils.SplitVector],
    mean_estimate: Optional[eg_utils.SplitVector] = None,
    epsilon: Optional[float] = None,
    maximize: bool = True,
) -> Tuple[chex.Array, chex.Array]:
  """Use the Rayleigh quotient to estimate the generalized eigenvalues.

  Calculates the numerator (vTAv) and denominator (vTBv) separately so we can
  loop over the entire dataset then divide. The calculation is distributed
  across machines.

  Args:
    local_eigenvectors: eg_utils.SplitVector holding the generalized
      eigenvectors sharded across machines. Array Tree with leaves of shape
      [l, ...] denoting v_l.
    sharded_data: eg_utils.SplitVector holding data from our two data sources.
      Array Tree with leaves of shape [b, ...].
    mean_estimate: eg_utils.SplitVector containing an estimate of the mean of
      the input data if it is not centered by default. This is used to calculate
      the covariances. Array tree with leaves of shape [...].
    epsilon: Add an optional isotropic term for the variance matrix to make it
      positive definite. In this case, we're solving for: Av =
        lambda*(B+epsilon*I)v.
    maximize: Whether to search for top-k eigenvectors of Av = lambda*Bv (True)
      or the top-k of (-A)v = lambda*Bv (False). Due to the symmetry of the CCA
      problem, setting to True or False should not change the performance.

  Returns:
    Tuple of two arrays of shape [k] denoting the numerator and denominator of
    the rayleigh quotient. These can be summed and then divided outside of this
    function in order to estimate the eigenvalues.
  """
  all_eigenvectors = jax.lax.all_gather(
      eg_utils.normalize_eigenvectors(local_eigenvectors),
      axis_name='devices',
      axis=0,
      tiled=True,
  )
  data_vector_x = []
  data_vector_y = []
  for sample in sharded_data:
    if mean_estimate is not None:
      sample = jax.tree_map(lambda x, y: x - y, sample, mean_estimate)

    data_vector = eg_utils.tree_einsum(
        'k..., b... -> kb',
        all_eigenvectors,
        sample,
    )
    data_vector_x.append(
        jax.tree_util.tree_reduce(
            lambda x, y: x + y,
            data_vector.x,  # pytype: disable=attribute-error  # numpy-scalars
        ))
    data_vector_y.append(
        jax.tree_util.tree_reduce(
            lambda x, y: x + y,
            data_vector.y,  # pytype: disable=attribute-error  # numpy-scalars
        ))

  sharded_numerator = 2 * jnp.einsum(
      'kb, kb -> k',
      data_vector_x[0],
      data_vector_y[0],
  )  #  <v,Av>
  sharded_denominator = jnp.einsum(
      'kb, kb -> k',
      data_vector_x[1],
      data_vector_x[1],
  ) + jnp.einsum(
      'kb, kb -> k',
      data_vector_y[1],
      data_vector_y[1],
  )  #  <v,Bv>
  numerator = jax.lax.pmean(sharded_numerator, axis_name='devices')
  denominator = jax.lax.pmean(sharded_denominator, axis_name='devices')
  per_machine_batch_size = data_vector_x[0].shape[1]
  numerator /= per_machine_batch_size
  denominator /= per_machine_batch_size
  if not maximize:
    numerator = -numerator
  if epsilon is not None:
    # We can just add epsilon since we normalized the eigenvectors.
    denominator += epsilon
  return (
      numerator,
      denominator,
  )  # vAv and v(B + epsilonI)v


def biased_cca_rayleigh_quotient_components(
    local_eigenvectors: eg_utils.SplitVector,
    sharded_data: eg_utils.SplitVector,
    mean_estimate: Optional[eg_utils.SplitVector] = None,
    epsilon: Optional[float] = None,
    maximize: bool = True,
) -> Tuple[chex.Array, chex.Array]:
  """Use the Rayleigh quotient to estimate the cca generalized eigenvalues.

  Calculates the numerator (vTAv) and denominator (vTBv) separately so we can
  loop over the entire dataset then divide. The calculation is distributed
  across machines.

  Args:
    local_eigenvectors: eg_utils.SplitVector holding the generalized
      eigenvectors sharded across machines. Array Tree with leaves of shape
      [l, ...] denoting v_l.
    sharded_data: eg_utils.SplitVector holding data from our two data sources.
      Array Tree with leaves of shape [b, ...].
    mean_estimate: eg_utils.SplitVector containing an estimate of the mean of
      the input data if it is not centered by default. This is used to calculate
      the covariances. Array tree with leaves of shape [...].
    epsilon: Add an optional isotropic term for the variance matrix to make it
      positive definite. In this case, we're solving for: Av =
        lambda*(B+epsilon*I)v.
    maximize: Whether to search for top-k eigenvectors of Av = lambda*Bv (True)
      or the top-k of (-A)v = lambda*Bv (False). Due to the symmetry of the CCA
      problem, setting to True or False should not change the performance.

  Returns:
    Tuple of two arrays of shape [k] denoting the numerator and denominator of
    the rayleigh quotient. These can be summed and then divided outside of this
    function in order to estimate the eigenvalues.
  """
  all_eigenvectors = jax.lax.all_gather(
      eg_utils.normalize_eigenvectors(local_eigenvectors),
      axis_name='devices',
      axis=0,
      tiled=True,
  )
  if mean_estimate is not None:
    sharded_data = jax.tree_map(
        lambda x, y: x - y,
        sharded_data,
        mean_estimate,
    )
  data_vector = eg_utils.tree_einsum(
      'k..., b... -> kb',
      all_eigenvectors,
      sharded_data,
  )
  data_vector_x = jax.tree_util.tree_reduce(lambda x, y: x + y, data_vector.x)  # pytype: disable=attribute-error  # numpy-scalars
  data_vector_y = jax.tree_util.tree_reduce(lambda x, y: x + y, data_vector.y)  # pytype: disable=attribute-error  # numpy-scalars

  sharded_numerator = 2 * jnp.einsum(
      'kb, kb -> k',
      data_vector_x,
      data_vector_y,
  )  #  <v,Av>
  sharded_denominator = jnp.einsum(
      'kb, kb -> k',
      data_vector_x,
      data_vector_x,
  ) + jnp.einsum(
      'kb, kb -> k',
      data_vector_y,
      data_vector_y,
  )  #  <v,Bv>
  numerator = jax.lax.pmean(sharded_numerator, axis_name='devices')
  denominator = jax.lax.pmean(sharded_denominator, axis_name='devices')
  per_machine_batch_size = data_vector_x.shape[1]
  numerator /= per_machine_batch_size
  denominator /= per_machine_batch_size
  if not maximize:
    numerator = -numerator
  if epsilon is not None:
    # We can just add epsilon since we normalized the eigenvectors.
    denominator += epsilon
  return (
      numerator,
      denominator,
  )  # vAv and v(B + epsilonI)v


def unbiased_pls_rayleigh_quotient_components(
    local_eigenvectors: eg_utils.SplitVector,
    sharded_data: Tuple[eg_utils.SplitVector, eg_utils.SplitVector],
    mean_estimate: Optional[eg_utils.SplitVector] = None,
    epsilon: Optional[float] = None,
    maximize: bool = True,
) -> Tuple[chex.Array, chex.Array]:
  """Use the Rayleigh quotient to estimate the generalized eigenvalues.

  Calculates the numerator (vTAv) and denominator (vTBv) separately so we can
  loop over the entire dataset then divide. The calculation is distributed
  across machines.

  Args:
    local_eigenvectors: eg_utils.SplitVector holding the generalized
      eigenvectors sharded across machines. Array Tree with leaves of shape
      [l, ...] denoting v_l.
    sharded_data: eg_utils.SplitVector holding data from our two data sources.
      Array Tree with leaves of shape [b, ...].
    mean_estimate: eg_utils.SplitVector containing an estimate of the mean of
      the input data if it is not centered by default. This is used to calculate
      the covariances. Array tree with leaves of shape [...].
    epsilon: Add an optional isotropic term for the variance matrix to make it
      positive definite. In this case, we're solving for: Av =
        lambda*(B+epsilon*I)v.
    maximize: unused- Solving Av = lambda * Bv is the only sensible approach for
      PLS. We do not foresee a use case for (-Av) = lambda * Bv. Please contact
      authors if you have a need for it.

  Returns:
    Tuple of two arrays of shape [k] denoting the numerator and denominator of
    the rayleigh quotient. These can be summed and then divided outside of this
    function in order to estimate the eigenvalues.
  """
  del maximize

  all_eigenvectors = jax.lax.all_gather(
      eg_utils.normalize_eigenvectors(local_eigenvectors),
      axis_name='devices',
      axis=0,
      tiled=True,
  )
  data_vector_x = []
  data_vector_y = []
  for sample in sharded_data:
    if mean_estimate is not None:
      sample = jax.tree_map(lambda x, y: x - y, sample, mean_estimate)

    data_vector = eg_utils.tree_einsum(
        'k..., b... -> kb',
        all_eigenvectors,
        sample,
    )
    data_vector_x.append(
        jax.tree_util.tree_reduce(
            lambda x, y: x + y,
            data_vector.x,  # pytype: disable=attribute-error  # numpy-scalars
        ))
    data_vector_y.append(
        jax.tree_util.tree_reduce(
            lambda x, y: x + y,
            data_vector.y,  # pytype: disable=attribute-error  # numpy-scalars
        ))

  sharded_numerator = 2 * jnp.einsum(
      'kb, kb -> k',
      data_vector_x[0],
      data_vector_y[0],
  )  #  <v,Av>
  sharded_denominator = jnp.einsum(
      'kb, kb -> k',
      data_vector_x[1],
      data_vector_x[1],
  ) + jnp.einsum(
      'k..., k... -> k',
      all_eigenvectors.y,
      all_eigenvectors.y,
  )  #  <v,Bv>
  numerator = jax.lax.pmean(sharded_numerator, axis_name='devices')
  denominator = jax.lax.pmean(sharded_denominator, axis_name='devices')
  per_machine_batch_size = data_vector_x[0].shape[1]
  numerator /= per_machine_batch_size
  denominator /= per_machine_batch_size

  if epsilon is not None:
    # We can just add epsilon since we normalized the eigenvectors.
    denominator += epsilon
  return (
      numerator,
      denominator,
  )  # vAv and v(B + epsilonI)v


def unbiased_ica_rayleigh_quotient_components(
    local_eigenvectors: chex.ArrayTree,
    sharded_data: Tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree,],
    mean_estimate: Optional[chex.ArrayTree] = None,
    epsilon: Optional[float] = None,
    maximize: bool = True
) -> Tuple[chex.Array, chex.Array]:
  """Use the Rayleigh quotient to estimate the ica generalized eigenvalues.

  Calculates the numerator (vTAv) and denominator (vTBv) separately so we can
  loop over the entire dataset then divide. The calculation is distributed
  across machines.


  Args:
    local_eigenvectors: eg_utils.SplitVector holding the generalized
      eigenvectors sharded across machines. Array Tree with leaves of shape
      [l, ...] denoting v_l.
    sharded_data: Tuple of eg_utils.SplitVectors holding independent batches of
      data from the two data sources. Array Tree with leaves of shape [b, ...].
    mean_estimate: eg_utils.SplitVector containing an estimate of the mean of
      the input data if it is not centered by default. This is used to calculate
      the covariances. Array tree with leaves of shape [...].
    epsilon: Add an optional isotropic term for the variance matrix to make it
      positive definite. In this case, we're solving for: Av =
        lambda*(B+epsilon*I)v.
    maximize: Whether to search for top-k eigenvectors of Av = lambda*Bv (True)
      or the top-k of (-A)v = lambda*Bv (False).

  Returns:
    Tuple of two arrays of shape [k] denoting the numerator and denominator of
    the rayleigh quotient. These can be summed and then divided outside of this
    function in order to estimate the eigenvalues.
  """
  all_eigenvectors = jax.lax.all_gather(
      eg_utils.normalize_eigenvectors(local_eigenvectors),
      axis_name='devices',
      axis=0,
      tiled=True,
  )
  if mean_estimate is not None:
    sharded_data = tuple(
        jax.tree_map(lambda x, y: x - y, sample, mean_estimate)
        for sample in sharded_data)
  (
      kurtosis,
      independent_covariance
  ) = eg_gradients.unbiased_ica_matrix_products(
      all_eigenvectors,
      sharded_data,
  )
  numerator = eg_utils.tree_einsum(
      'k..., k... -> k',
      kurtosis,
      all_eigenvectors,
      reduce_f=lambda x, y: x + y,
  )
  denominator = eg_utils.tree_einsum(
      'k..., k... -> k',
      independent_covariance,
      all_eigenvectors,
      reduce_f=lambda x, y: x + y,
  )
  if not maximize:
    numerator = -numerator  # pytype: disable=unsupported-operands  # numpy-scalars
  if epsilon is not None:
    # We can just add epsilon since we normalized the eigenvectors.
    denominator += epsilon
  return numerator, denominator


def pca_generalised_eigengame_rayleigh_quotient_components(
    local_eigenvectors: chex.ArrayTree,
    sharded_data: chex.ArrayTree,
    mean_estimate: Optional[eg_utils.SplitVector] = None,
    epsilon: Optional[float] = None,
    maximize: bool = True
) -> Tuple[chex.Array, chex.Array]:
  """Use the Rayleigh quotient to estimate the pca generalized eigenvalues.

  Calculates the numerator (vTAv) and returns I separately so we can
  loop over the entire dataset then divide. The calculation is distributed
  across machines.

  Args:
    local_eigenvectors: eg_utils.SplitVector holding the generalized
      eigenvectors sharded across machines. Array Tree with leaves of shape
      [l, ...] denoting v_l.
    sharded_data: Tuple of eg_utils.SplitVectors holding independent batches of
      data from the two data sources. Array Tree with leaves of shape [b, ...].
    mean_estimate: eg_utils.SplitVector containing an estimate of the mean of
      the input data if it is not centered by default. This is used to calculate
      the covariances. Array tree with leaves of shape [...].
    epsilon: Add an optional isotropic term for the variance matrix to make it
      positive definite. In this case, we're solving for: Av =
        lambda*(B+epsilon*I)v.
    maximize: unused- Solving Av = lambda * v is the only sensible approach for
      PCA. We do not foresee a use case for (-Av) = lambda * v. Please contact
      authors if you have a need for it.

  Returns:
    Tuple of two arrays of shape [k] denoting the numerator and denominator of
    the rayleigh quotient. These can be summed and then divided outside of this
    function in order to estimate the eigenvalues.
  """
  del maximize

  all_eigenvectors = jax.lax.all_gather(
      eg_utils.normalize_eigenvectors(local_eigenvectors),
      axis_name='devices',
      axis=0,
      tiled=True,
  )
  if mean_estimate is not None:
    sharded_data = jax.tree_map(lambda x, y: x - y, sharded_data, mean_estimate)

  data_vector_product = eg_utils.tree_einsum(
      'b..., k... -> bk',
      sharded_data,
      all_eigenvectors,
      reduce_f=lambda x, y: x + y,
  )
  local_numerator = jnp.einsum(
      'bk, bk -> k',
      data_vector_product,
      data_vector_product,
  )
  per_machine_batch_size, total_eigenvector_count = data_vector_product.shape  # pytype: disable=attribute-error  # numpy-scalars
  numerator = jax.lax.pmean(local_numerator, axis_name='devices')
  numerator = numerator / per_machine_batch_size
  denominator = jnp.ones(total_eigenvector_count)
  if epsilon is not None:
    # We can just add epsilon since we normalized the eigenvectors.
    denominator += epsilon
  return numerator, denominator


def matrix_inverse_rayleigh_quotient_components(
    local_eigenvectors: chex.ArrayTree,
    sharded_data: chex.ArrayTree,
    mean_estimate: Optional[eg_utils.SplitVector] = None,
    epsilon: Optional[float] = None,
    maximize: bool = True,
) -> Tuple[chex.Array, chex.Array]:
  """Use the Rayleigh quotient to estimate the inv generalized eigenvalues.

  Calculates the numerator (vTAv) and returns I separately so we can
  loop over the entire dataset then divide. The calculation is distributed
  across machines.

  Args:
    local_eigenvectors: eg_utils.SplitVector holding the generalized
      eigenvectors sharded across machines. Array Tree with leaves of shape
      [l, ...] denoting v_l.
    sharded_data: Tuple of eg_utils.SplitVectors holding independent batches of
      data from the two data sources. Array Tree with leaves of shape [b, ...].
    mean_estimate: eg_utils.SplitVector containing an estimate of the mean of
      the input data if it is not centered by default. This is used to calculate
      the covariances. Array tree with leaves of shape [...].
    epsilon: Add an optional isotropic term for the variance matrix to make it
      positive definite. In this case, we're solving for: Av =
        lambda*(B+epsilon*I)v.
    maximize: unused- Solving Iv = lambda * Bv is the only sensible approach for
      inverting B. We do not foresee a use case for (-Iv) = lambda * Bv. Please
      contact authors if you have a need for it.

  Returns:
    Tuple of two arrays of shape [k] denoting the numerator and denominator of
    the rayleigh quotient. These can be summed and then divided outside of this
    function in order to estimate the eigenvalues.
  """
  del maximize

  all_eigenvectors = jax.lax.all_gather(
      eg_utils.normalize_eigenvectors(local_eigenvectors),
      axis_name='devices',
      axis=0,
      tiled=True,
  )
  if mean_estimate is not None:
    sharded_data = jax.tree_map(lambda x, y: x - y, sharded_data, mean_estimate)

  data_vector_product = eg_utils.tree_einsum(
      'b..., k... -> bk',
      sharded_data,
      all_eigenvectors,
      reduce_f=lambda x, y: x + y,
  )
  local_denominator = jnp.einsum(
      'bk, bk -> k',
      data_vector_product,
      data_vector_product,
  )
  per_machine_batch_size, total_eigenvector_count = data_vector_product.shape  # pytype: disable=attribute-error  # numpy-scalars
  denominator = jax.lax.pmean(local_denominator, axis_name='devices')
  denominator = denominator / per_machine_batch_size
  if epsilon is not None:
    # We can just add epsilon since we normalized the eigenvectors.
    denominator += epsilon
  return jnp.ones(total_eigenvector_count), denominator


def calculate_eigenvalues(
    local_eigenvectors: chex.ArrayTree,
    sharded_data: chex.ArrayTree,
) -> chex.Array:
  """Calculates (vX)TXv, which gives the estimate of the eigenvalues.

  We do this in a distributed fashion by first calculating Xv and then
  concatenating it across machines, resulting in a larger effective batch size.

  Args:
    local_eigenvectors: Array Tree holding the generalized eigenvectors sharded
      across machines. Array Tree with leaves of shape [l, ...] denoting v_l.
    sharded_data: Array Tree holding data from our two data sources. Array Tree
      with leaves of shape [b, ...].

  Returns:
    Duplicated copies of all the eigenvalues on all devices, Shape of:
    [total_eigenvector_count]
  """
  all_eigenvectors = jax.lax.all_gather(
      local_eigenvectors,
      axis_name='devices',
      axis=0,
      tiled=True,
  )
  # Calculate Xv for all v. Unlike the gradient, we can generate the eigenvalues
  # with just this.
  sharded_data_vector = eg_utils.tree_einsum(
      'b..., k... -> kb',
      sharded_data,
      all_eigenvectors,
      reduce_f=lambda x, y: x + y,
  )
  full_data_vector = jax.lax.all_gather(
      sharded_data_vector,
      axis_name='devices',
      axis=1,
      tiled=True,
  )
  return jnp.einsum('kb,kb->k', full_data_vector, full_data_vector)


def subspace_error(
    approx_eigenvectors: chex.Array,
    true_eigenvectors: chex.Array,
    matrix_b: Optional[chex.Array] = None,
) -> float:
  """Compute subspace error between approximate solution and ground truth.

  Given the top-k ground truth eigenvectors W* and approximations W to the EVP,
  subspace error can be computed as:

  1 - 1 / k * trace(W* pinv(W*) W pinv(W)).

  Where:
    W* = top-k eigenvectors of B^{-1/2} A B^{-1/2}
    W = B^{1/2} V.

  Let v be a solution to the GEVP, Av = lambda' Bv
  Then w = B^{1/2} v is a solution to the normalized EVP,
    B^{-1/2} A B^{-1/2} w = lambda w, with eigenvalue lambda = lambda'.

  Leveraging this equivalence, we can measure subspace error of the GEVP
  solution by first mapping it to the normalized case and computing subspace
  error there.


  Args:
    approx_eigenvectors: Array of shape (k, d) approximate top-k solution to Av
      = lambda Bv. This function assumes that the eigenvectors are flattened
      into a single dimension.
    true_eigenvectors: Array of shape (k, d) exact top-k solution to Av = lambda
      Bv.
    matrix_b: Array of shape (d, d) of the matrix B in Av = lambda Bv. Default
      assumes B = I (i.e. simple eigenvalue problem instead of general)

  Returns:
    float, subspace error > 0
  """
  k = approx_eigenvectors.shape[0]

  if matrix_b is not None:
    matrix_b_sqrt = scipy.linalg.sqrtm(matrix_b)
    # Transform into a space where the general eigenvectors are orthogonal
    # in the general eigenvalue problem
    transformed_approx_eigenvectors = jnp.einsum(
        'kD,DE->kE',
        approx_eigenvectors,
        matrix_b_sqrt,
    )
    transformed_true_eigenvectors = jnp.einsum(
        'kD,DE->kE',
        true_eigenvectors,
        matrix_b_sqrt,
    )
  else:
    # Keep it as is in simple eigenvector case.
    transformed_approx_eigenvectors = approx_eigenvectors
    transformed_true_eigenvectors = true_eigenvectors

  # Normalize all the vectors
  normalized_approx_eigenvectors = eg_utils.normalize_eigenvectors(
      transformed_approx_eigenvectors,)
  normalized_true_eigenvectors = eg_utils.normalize_eigenvectors(
      transformed_true_eigenvectors,)

  # Calculate the Penrose inverses.
  approx_eigenvector_pinv = jnp.linalg.pinv(normalized_approx_eigenvectors)
  true_eigenvectors_pinv = jnp.linalg.pinv(normalized_true_eigenvectors)

  # Apinv(A) creates a projection into the space spanned by A. Therefore if row
  # vectors of B spans a similar space, it will preserve it.
  approximate_projector = jnp.einsum(
      'kD,Ek->DE',
      normalized_approx_eigenvectors,
      approx_eigenvector_pinv,
  )
  true_projector = jnp.einsum(
      'kD,Ek->DE',
      normalized_true_eigenvectors,
      true_eigenvectors_pinv,
  )

  # The trace will be shrunk if the projection removes components.
  # Alternatively, this can be interpreted as the dot product of the matrices
  # representing the projection maps dual space.
  subspace_similarity = jnp.einsum(
      'KD, KD',
      approximate_projector,
      true_projector,
  )
  # normalize and subtract from 1 so value > 0
  return 1 - (subspace_similarity / k)
