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


Einsum legend:
...: the massive data dimension of the eigenvector which we're trying to do
PCA/CCA/etc. on. The input data may not be contiguious, and may consist of
multiple shapes in a pytree (e.g. images, activations of multiple layers of a
neural net). Hence we use ellipses to denote it.
l: local eigenvector index -- the number of eigenvectors per machine.
k: global eigenvector index -- number of eigenvectors over all machines.
b: batch dimension, batch size per machine.

"""
from typing import Tuple, Optional

import chex
from eigengame import eg_utils
import jax
import jax.numpy as jnp
import numpy as np

SplitVector = eg_utils.SplitVector


def pca_unloaded_gradients(
    local_eigenvectors: chex.ArrayTree,
    sharded_data: chex.ArrayTree,
    mask: chex.Array,
    sliced_identity: chex.Array,
) -> chex.ArrayTree:
  """Calculates the gradients for each of the eigenvectors for EG unloaded.

  Calculates the gradients of eigengame unloaded (see Algorithm 1. of
  https://arxiv.org/pdf/2102.04152.pdf)

  This is done in a distributed manner. Each TPU is responsible for updating
  the whole of a subset of eigenvectors, and get a different batch of the data.
  The training data is applied and then aggregated to increase the effective
  batch size.

  Args:
    local_eigenvectors: eigenvectors sharded across machines in
      a ShardedDeviceArray. ArrayTree with leaves of shape:
      [eigenvectors_per_device, ...]
    sharded_data: Training data sharded across machines. ArrayTree with leaves
      of shape: [batch_size_per_device, ...]
    mask: Mask with 1s below the diagonal and then sharded across devices, used
      to calculate the penalty. Shape of: [eigenvectors_per_device,
        total_eigenvector_count]
    sliced_identity: Sharded copy of the identity matrix used to calculate the
      reward. Shape of: [eigenvectors_per_device, total_eigenvector_count]

  Returns:
    Gradients for the local eigenvectors. ArrayTree with leaves of shape:
   [eigenvectors_per_device, ...] on each device.
  """
  # Collect the matrices from all the other machines.
  all_eigenvectors = jax.lax.all_gather(
      local_eigenvectors,
      axis_name='devices',
      axis=0,
      tiled=True,
  )
  # Calculate X^TXv/b for all v. This and v are all you need to construct the
  # gradient. This is done on each machine with a different batch.
  data_vector_product = eg_utils.tree_einsum(
      'b..., k... -> bk',
      sharded_data,
      all_eigenvectors,
      reduce_f=lambda x, y: x + y,
  )
  # divide by the batch size
  data_vector_product /= data_vector_product.shape[0]  # pytype: disable=attribute-error  # numpy-scalars
  sharded_gram_vector = eg_utils.tree_einsum_broadcast(
      'b..., bk -> k...',
      sharded_data,
      data_vector_product,
  )
  # Aggregate the gram matrix products across machines.
  gram_vector = jax.lax.pmean(
      sharded_gram_vector,
      axis_name='devices',
  )
  # Calculate <Xv_i, Xv_j> for all vectors on each machine.
  scale = eg_utils.tree_einsum(
      'l..., k... -> lk',
      local_eigenvectors,
      gram_vector,
      reduce_f=lambda x, y: x + y,
  )

  # The weights is 1 below diagonals
  penalty = eg_utils.tree_einsum_broadcast(
      'k..., lk, lk -> l...',
      all_eigenvectors,
      scale,
      mask,
  )

  reward = eg_utils.get_local_slice(sliced_identity, gram_vector)  # XtXv_i
  return jax.tree_map(lambda x, y: x - y, reward, penalty)


def create_sharded_mask(eigenvector_count: int) -> chex.ArraySharded:
  """Defines a mask of 1s under the diagonal and shards it."""
  mask = np.ones((eigenvector_count, eigenvector_count))
  r, c = np.triu_indices(eigenvector_count)
  mask[..., r, c] = 0
  start_index = jax.process_index() * eigenvector_count // jax.process_count()
  end_index = (jax.process_index()+1) * eigenvector_count // jax.process_count()
  mask = mask[start_index:end_index]
  mask = mask.reshape((
      jax.local_device_count(),
      eigenvector_count // jax.device_count(),
      eigenvector_count,
  ))
  return jax.device_put_sharded(list(mask), jax.local_devices())


def create_sharded_identity(eigenvector_count: int) -> chex.ArraySharded:
  """Create identity matrix which is then split across devices."""
  identity = np.eye(eigenvector_count)

  start_index = jax.process_index() * eigenvector_count // jax.process_count()
  end_index = (jax.process_index() +
               1) * eigenvector_count // jax.process_count()
  identity = identity[start_index:end_index]
  identity = identity.reshape((
      jax.local_device_count(),
      eigenvector_count // jax.device_count(),
      eigenvector_count,
  ))
  return jax.device_put_sharded(list(identity), jax.local_devices())


def _generalized_eg_matrix_inner_products(
    local_eigenvectors: chex.ArrayTree,
    all_eigenvectors: chex.ArrayTree,
    b_vector_product: chex.ArrayTree,
    a_vector_product: chex.ArrayTree,
    aux_b_vector_product: chex.ArrayTree,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array,]:
  """Compute various inner product quantities used in the gradient.

  In particular, the loss requires the various forms of inner product
  <v_i, Av_j> in order to function. This function calculates all of them to keep
  the gradient function less cluttered.


  Args:
    local_eigenvectors: ArrayTree with local copies of generalised eigen
      vectors with leaves of shape [l, ...] denoting v_l.
    all_eigenvectors: ArrayTree with all generalised eigen vectors with leaves
      of shape [k, ...] denoting v_k
    b_vector_product: ArrayTree with all B matrix eigenvector products with
      leaves of shape [k, ...] denoting Bv_k
    a_vector_product: ArrayTree with all A matrix eigenvector products with
      leaves of shape [k, ...] denoting Av_k
    aux_b_vector_product: ArrayTree with all B matrix eigenvector products from
      the axuiliary variable with leaves of shape [k, ...] denoting Bv_k

  Returns:
    Tuple of arrays containing:
      local_b_inner_product: <v_l, Bv_k>
      local_a_inner_product: <v_l, Av_k>
      b_inner_product_diag: <v_k, Bv_k>
      a_inner_product_diag: : <v_k, Bv_k>
  """

  # Evaluate the various forms of the inner products used in the gradient
  local_aux_b_inner_product = eg_utils.tree_einsum(
      'l... , k... -> lk',
      local_eigenvectors,
      aux_b_vector_product,
      reduce_f=lambda x, y: x + y,
  )  # v_l B v_k, used in the penalty term
  local_a_inner_product = eg_utils.tree_einsum(
      'l..., k... -> lk',
      local_eigenvectors,
      a_vector_product,
      reduce_f=lambda x, y: x + y,
  )  # v_l A v_k, used in the penalty term
  b_inner_product_diag = eg_utils.tree_einsum(
      'k..., k... -> k',
      all_eigenvectors,
      b_vector_product,
      reduce_f=lambda x, y: x + y,
  )  # v_k B v_k, used in the penalty and reward terms
  a_inner_product_diag = eg_utils.tree_einsum(
      'k..., k... -> k',
      all_eigenvectors,
      a_vector_product,
      reduce_f=lambda x, y: x + y,
  )  # v_k A v_k, used in the reward term
  return (
      local_aux_b_inner_product,
      local_a_inner_product,
      b_inner_product_diag,
      a_inner_product_diag,
  )


def _generalized_eg_gradient_reward(
    local_b_inner_product_diag: chex.Array,
    local_a_inner_product_diag: chex.Array,
    local_b_vector_product: chex.ArrayTree,
    local_a_vector_product: chex.ArrayTree,
) -> chex.ArrayTree:
  """Evaluates the reward term for the eigengame gradient for local vectors.

  This attempts to maximise the rayleigh quotient for each eigenvector, and
  by itself would find the eigenvector with the largest generalized eigenvalue.

  The output corresponds to the equation for all l:
    <v_l,Bv_l>Av_l - <v_l,Av_l>Bv_l

  Args:
    local_b_inner_product_diag: Array of shape [l] corresponding to <v_l,Bv_l>.
    local_a_inner_product_diag: Array of shape [l] corresponding to <v_l,Av_l>.
    local_b_vector_product: ArrayTree with local eigen vectors products with
      B with leaves of shape [l, ...] denoting Bv_l.
    local_a_vector_product: ArrayTree with local eigen vectors products with
      A with leaves of shape [l, ...] denoting Av_l.

  Returns:
    The reward gradient for the eigenvectors living on the current machine.
    Array tree with leaves of shape [l, ...]
  """
  # Evaluates <v_l,Bv_l>Av_l
  scaled_a_vectors = eg_utils.tree_einsum_broadcast(
      'l..., l -> l...',
      local_a_vector_product,
      local_b_inner_product_diag,
  )
  # Evaluates <v_l,Av_l>Bv_l
  scaled_b_vectors = eg_utils.tree_einsum_broadcast(
      'l..., l -> l...',
      local_b_vector_product,
      local_a_inner_product_diag,
  )
  return jax.tree_map(
      lambda x, y: x - y,
      scaled_a_vectors,
      scaled_b_vectors,
  )


def _generalized_eg_gradient_penalty(
    local_b_inner_product: chex.Array,
    local_a_inner_product: chex.Array,
    local_b_inner_product_diag: chex.Array,
    b_inner_product_diag: chex.Array,
    local_b_vector_product: chex.ArrayTree,
    b_vector_product: chex.ArrayTree,
    mask: chex.Array,
    b_diag_min: float = 1e-6) -> chex.ArrayTree:
  r"""Evaluates the penalty term for the eigengame gradient for local vectors.

  This attempts to force each eigenvalue to be B orthogonal (i.e. <v,Bw> = 0) to
  all its parents. Combining this with the reward terms means each vector
  learns to maximise the eigenvalue whilst staying orthogonal, giving us the top
  k generalized eigenvectors.


  The output corresponds to the equation for all l on the local machine:
   \\sum_{k<l}(<v_l,Bv_l>Bv_k - <v_l,Bv_k>Bv_l) <v_l,Av_k>/<v_k,Bv_k>

  Note: If the gradient returned must be unbiased, then any estimated quantities
  in the formula below must be independent and unbiased (.e.g, the numerator of
  the first term (<v_l,Av_k>/<v_k,Bv_k>)<v_l,Bv_l>Bv_k must use independent,
  unbiase estimates for each element of this product, otherwise the estimates
  are correlated and will bias the computation). Furthermore, any terms in the
  denominator must be deterministic (i.e., <v_k,Bv_k> must be computed without
  using sample estimates which can be accomplished by introducing an auxiliary
  learning variable).

  Args:
    local_b_inner_product: Array of shape [l,k] denoting <v_l, Bv_k>
    local_a_inner_product: Array of shape [l,k] denoting <v_l, Av_k>
    local_b_inner_product_diag: Array of shape [l] denoting <v_l, Bv_l>
    b_inner_product_diag: Array of shape [k] denoting <v_k, Bv_k>. Insert an
      auxiliary variable here for in order to debias the receprocal.
    local_b_vector_product: ArrayTree with local eigen vectors products with
      B with leaves of shape [l, ...] denoting Bv_l.
    b_vector_product: ArrayTree with all eigen vectors products with
      B with leaves of shape [k, ...] denoting Bv_k.
    mask: Slice of a k x k matrix which is 1's under the diagonals and 0's
      everywhere else. This is used to denote the parents of each vector. Array
      of shape [l, k].
    b_diag_min: Minimum value for the b_inner_product_diag. This value is
      divided, so we use this to ensure we don't get a division by zero.

  Returns:
    The penalty gradient for the eigenvectors living on the current machine.
  """
  # Calculate <v_l,Av_k>/<v_k,Bv_k> with mask
  scale = jnp.einsum(
      'lk, lk, k -> lk',
      mask,
      local_a_inner_product,
      1 / jnp.maximum(b_inner_product_diag, b_diag_min),
  )
  # Calculate scale * <v_l,Bv_l>Bv_k  term
  global_term = eg_utils.tree_einsum_broadcast(
      'k..., lk, l -> l...',
      b_vector_product,
      scale,
      local_b_inner_product_diag,
  )
  # Calculate scale *  <v_l,Bv_k>Bv_l term
  local_term = eg_utils.tree_einsum_broadcast(
      'l..., lk, lk -> l...',
      local_b_vector_product,
      scale,
      local_b_inner_product,
  )
  return jax.tree_map(lambda x, y: x - y, global_term, local_term)


def generalized_eigengame_gradients(
    *,
    local_eigenvectors: chex.ArrayTree,
    all_eigenvectors: chex.ArrayTree,
    a_vector_product: chex.ArrayTree,
    b_vector_product: chex.ArrayTree,
    auxiliary_variables: eg_utils.AuxiliaryParams,
    mask: chex.Array,
    sliced_identity: chex.Array,
    epsilon: Optional[float] = None,
    maximize: bool = True,
) -> Tuple[chex.ArrayTree, eg_utils.AuxiliaryParams,]:
  """Solves for Av = lambda Bv using eigengame in a data parallel manner.

  Algorithm pseudocode can be found in Algorithm 1 of overleaf doc:
  https://ol.deepmind.host/read/kxdfdtfbsdxc

  For moderately sized models this is fine, but for really large models (
  moderate number of eigenvectors of >1m params) the memory overhead might be
  strained in the parallel case.

  Args:
    local_eigenvectors: ArrayTree with local eigen vectors with leaves
      of shape [l, ...] denoting v_l.
    all_eigenvectors: ArrayTree with all eigen vectors with leaves
      of shape [k, ...] denoting v_k.
    a_vector_product: ArrayTree with all eigen vectors products with
      A with leaves of shape [k, ...] denoting Av_k.
    b_vector_product: ArrayTree with all eigen vectors products with
      B with leaves of shape [k, ...] denoting Bv_k.
    auxiliary_variables: AuxiliaryParams object which holds all the variables
      which we want to update separately in order to avoid bias.
    mask: Mask with 1s below the diagonal and then sharded across devices, used
      to calculate the penalty. Shape of: [eigenvectors_per_device,
        total_eigenvector_count]
    sliced_identity: Sharded copy of the identity matrix used to calculate the
      reward. Shape of: [eigenvectors_per_device, total_eigenvector_count]
    epsilon: Add an isotropic term to the B matrix to make it
      positive semi-definite. In this case, we're solving for: Av =
        lambda*(B+epsilon*I)v.
    maximize: Whether to search for top-k eigenvectors of Av = lambda*Bv (True)
      or the top-k of (-A)v = lambda*Bv (False).

  Returns:
    returns the gradients for the eigenvectors and a new entry to update
    auxiliary variable estimates.
  """
  if not maximize:
    a_vector_product = -a_vector_product  # pytype: disable=unsupported-operands  # numpy-scalars

  # Add a small epsilon to the b vector product to make it positive definite.
  # This can help with convergence.
  if epsilon is not None:
    b_vector_product = jax.tree_map(
        lambda x, y: x + epsilon * y,
        b_vector_product,
        all_eigenvectors,
    )
  # Calculate the various matrix inner products.
  (
      # Get <v_l ,Bv_k> and <v_l, Av_k>.
      local_aux_b_inner_product,
      local_a_inner_product,
      # Get <v_k ,Bv_k> and <v_k, Av_k>.
      b_inner_product_diag,
      a_inner_product_diag,
  ) = _generalized_eg_matrix_inner_products(
      local_eigenvectors,
      all_eigenvectors,
      b_vector_product,
      a_vector_product,
      auxiliary_variables.b_vector_product,
  )

  # Get the local slices of these quantities. Going from k rows to l rows.
  (
      local_b_vector_product,
      local_a_vector_product,
      local_b_inner_product_diag,
      local_a_inner_product_diag,
  ) = eg_utils.get_local_slice(
      sliced_identity,
      (
          b_vector_product,
          a_vector_product,
          b_inner_product_diag,
          a_inner_product_diag,
      ),
  )
  # Calculate the reward
  reward = _generalized_eg_gradient_reward(
      local_b_inner_product_diag,
      local_a_inner_product_diag,
      local_b_vector_product,
      local_a_vector_product,
  )

  # Calculate the penalty using the associated auxiliary variables.
  penalty = _generalized_eg_gradient_penalty(
      local_aux_b_inner_product,
      local_a_inner_product,
      local_b_inner_product_diag,
      auxiliary_variables.b_inner_product_diag,
      local_b_vector_product,
      auxiliary_variables.b_vector_product,
      mask,
  )

  gradient = jax.tree_map(lambda x, y: x - y, reward, penalty)

  # Propagate the existing auxiliary variables
  new_auxiliary_variable = eg_utils.AuxiliaryParams(
      b_vector_product=b_vector_product,
      b_inner_product_diag=b_inner_product_diag,
  )
  return gradient, new_auxiliary_variable


def _biased_cca_matrix_products(
    eigenvectors: SplitVector,
    sharded_data: SplitVector,
) -> Tuple[SplitVector, SplitVector]:
  """Calculate and aggregate the equivalent gram matrices for CCA.

  CCA of data sources with dim n, m is equivalent to solving the generalized
  eigenvalue problem Av = lambda Bv, where v is dim n + m, A has the
  covariances between the data sources and B the covariances within the data
  sources. This function unpacks v then calculates Av and Bv from the two data
  sources before aggregating them across machines.

  The majority of the CCA specific logic lives here. The remainder of the
  algorithm should work with any other generalized eigenvalue problem.

  Args:
    eigenvectors: SplitVector holding the generalized eigenvectors estimates for
      the two data sources.
    sharded_data: Tuple containing two independent batches of data from our
      two data sources. Tuple of SplitVectors with leaves of shape [b, ...].
      Array Tree with leaves of shape [k, ...] denoting v_k.
  Returns:
    Tuple of two SplitVectors, each containing array trees with leaves of shape
    [k, ...] containing the gram matrix product of covariances between the data
    sources (Av) and within the data (Bv) respectively.
  """

  data_vector = eg_utils.tree_einsum(
      'k...,b...-> bk',
      eigenvectors,
      sharded_data,
  )
  data_vector_x = jax.tree_util.tree_reduce(lambda x, y: x + y, data_vector.x)  # pytype: disable=attribute-error  # numpy-scalars
  data_vector_y = jax.tree_util.tree_reduce(lambda x, y: x + y, data_vector.y)  # pytype: disable=attribute-error  # numpy-scalars
  # divide by the batch size
  data_vector_x /= data_vector_x.shape[0]
  data_vector_y /= data_vector_y.shape[0]

  def _gram_product(
      data: chex.ArrayTree,
      data_vector: chex.Array,
  ) -> chex.ArrayTree:
    return eg_utils.tree_einsum_broadcast('b..., bk -> k...', data, data_vector)

  sharded_variance_vector = SplitVector(
      x=_gram_product(sharded_data.x, data_vector_x),
      y=_gram_product(sharded_data.y, data_vector_y))
  sharded_interaction_vector = SplitVector(
      x=_gram_product(sharded_data.x, data_vector_y),
      y=_gram_product(sharded_data.y, data_vector_x))
  # Aggregate this across all machines.
  variance_vector_product = jax.lax.pmean(
      sharded_variance_vector, axis_name='devices')
  interaction_vector_product = jax.lax.pmean(
      sharded_interaction_vector, axis_name='devices')

  return (
      variance_vector_product,
      interaction_vector_product,
  )  # Bv and Av


def biased_cca_gradients(
    *,
    local_eigenvectors: SplitVector,
    sharded_data: SplitVector,
    auxiliary_variables: eg_utils.AuxiliaryParams,
    mask: chex.Array,
    sliced_identity: chex.Array,
    mean_estimate: Optional[SplitVector] = None,
    epsilon: Optional[float] = None,
    maximize: bool = True,
) -> Tuple[SplitVector, eg_utils.AuxiliaryParams,]:
  """Evaluates CCA gradients for two data sources with local data parallelism.

  Algorithm pseudocode can be found in Algorithm 1 of overleaf doc:
  https://ol.deepmind.host/read/kxdfdtfbsdxc

  Args:
    local_eigenvectors: SplitVector with local eigen vectors products. Array
      tree with leaves of shape [l, ...] denoting v_l.
    sharded_data: SplitVector containing a batch of data from our two data
      sources. Array tree with leaves of shape [b, ...].
    auxiliary_variables: AuxiliaryParams object which holds all the variables
      which we want to update separately in order to avoid bias.
    mask: Mask with 1s below the diagonal and then sharded across devices, used
      to calculate the penalty. Shape of: [eigenvectors_per_device,
        total_eigenvector_count]
    sliced_identity: Sharded copy of the identity matrix used to calculate the
      reward. Shape of: [eigenvectors_per_device, total_eigenvector_count]
    mean_estimate: SplitVector containing an estimate of the mean of the input
      data if it is not centered by default. This is used to calculate the
      covariances. Array tree with leaves of shape [...].
    epsilon: Add an isotropic term to the variance matrix to make it
      positive semi-definite. In this case, we're solving for: Av =
        lambda*(B+epsilon*I)v.
    maximize: Whether to search for top-k eigenvectors of Av = lambda*Bv (True)
      or the top-k of (-A)v = lambda*Bv (False). Due to the symmetry of the CCA
      problem, setting to True or False should not change the performance.

  Returns:
    returns the gradients for the eigenvectors and a new entry to update
    auxiliary variance estimates.
  """
  # First, collect all the eigenvectors v_k
  all_eigenvectors = jax.lax.all_gather(
      local_eigenvectors,
      axis_name='devices',
      axis=0,
      tiled=True,
  )
  if mean_estimate is not None:
    sharded_data = jax.tree_map(lambda x, y: x - y, sharded_data, mean_estimate)

  # Evaluate the matrix products for all eigenvectors on all machines using
  # different batches and aggregate them to get Bv_k and Av_k.
  variance_vector_product, interaction_vector_product = (
      _biased_cca_matrix_products(all_eigenvectors, sharded_data)
  )

  return generalized_eigengame_gradients(
      local_eigenvectors=local_eigenvectors,
      all_eigenvectors=all_eigenvectors,
      a_vector_product=interaction_vector_product,
      b_vector_product=variance_vector_product,
      auxiliary_variables=auxiliary_variables,
      mask=mask,
      sliced_identity=sliced_identity,
      epsilon=epsilon,
      maximize=maximize,
  )


def _unbiased_cca_matrix_products(
    eigenvectors: SplitVector,
    sharded_data: Tuple[SplitVector, SplitVector],
) -> Tuple[SplitVector, SplitVector]:
  """Calculate and aggregate the equivalent gram matrices for CCA.

  CCA of data sources with dim n, m is equivalent to solving the generalized
  eigenvalue problem Av = lambda Bv, where v is dim n + m, A has the
  covariances between the data sources and B the covariances within the data
  sources. This function unpacks v then calculates Av and Bv from the two data
  sources before aggregating them across machines.

  The majority of the CCA specific logic lives here. The remainder of the
  algorithm should work with any other generalized eigenvalue problem.

  Args:
    eigenvectors: SplitVector holding the generalized eigenvectors estimates for
      the two data sources.
      Array Tree with leaves of shape [k, ...] denoting v_k.
    sharded_data: SplitVector holding data from our two data sources.
      Array Tree with leaves of shape [b, ...].
  Returns:
    Tuple of two SplitVectors, each containing array trees with leaves of shape
    [k, ...] containing the gram matrix product of covariances between the data
    sources (Av) and within the data (Bv) respectively.
  """
  data_vector_x = []
  data_vector_y = []
  for sample in sharded_data:
    data_vector = eg_utils.tree_einsum(
        'k...,b...-> bk',
        eigenvectors,
        sample,
    )
    data_vector_x_sample = jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        data_vector.x,  # pytype: disable=attribute-error  # numpy-scalars
    )
    data_vector_y_sample = jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        data_vector.y,  # pytype: disable=attribute-error  # numpy-scalars
    )
    # divide by the batch size
    data_vector_x.append(data_vector_x_sample / data_vector_x_sample.shape[0])
    data_vector_y.append(data_vector_y_sample / data_vector_y_sample.shape[0])

  def _gram_product(
      data: chex.ArrayTree,
      data_vector: chex.Array,
  ) -> chex.ArrayTree:
    return eg_utils.tree_einsum_broadcast('b..., bk -> k...', data, data_vector)

  sharded_variance_vector = SplitVector(
      x=_gram_product(sharded_data[0].x, data_vector_x[0]),
      y=_gram_product(sharded_data[0].y, data_vector_y[0]))
  sharded_interaction_vector = SplitVector(
      x=_gram_product(sharded_data[1].x, data_vector_y[1]),
      y=_gram_product(sharded_data[1].y, data_vector_x[1]))
  # Aggregate this across all machines.
  variance_vector_product = jax.lax.pmean(
      sharded_variance_vector, axis_name='devices')
  interaction_vector_product = jax.lax.pmean(
      sharded_interaction_vector, axis_name='devices')

  return (
      variance_vector_product,
      interaction_vector_product,
  )  # Bv and Av


def unbiased_cca_gradients(
    *,
    local_eigenvectors: SplitVector,
    sharded_data: Tuple[SplitVector, SplitVector],
    auxiliary_variables: eg_utils.AuxiliaryParams,
    mask: chex.Array,
    sliced_identity: chex.Array,
    mean_estimate: Optional[SplitVector] = None,
    epsilon: Optional[float] = None,
    maximize: bool = True,
) -> Tuple[SplitVector, eg_utils.AuxiliaryParams,]:
  """Evaluates unbiased CCA gradients with data parallelism.

  Algorithm pseudocode can be found in Algorithm 1 of overleaf doc:
  https://ol.deepmind.host/read/kxdfdtfbsdxc

  In this case, we take in two independent batches of data, one to calculate Av
  and one to calculate Bv. Using two different samples results in an unbiased
  gradient, but there is a bias-variance tradeoff in the performance.

  Args:
    local_eigenvectors: SplitVector with local eigen vectors products. Array
      tree with leaves of shape [l, ...] denoting v_l.
    sharded_data: Tuple containing two independent batches of data from our
      two data sources. Tuple of SplitVectors with leaves of shape [b, ...].
    auxiliary_variables: AuxiliaryParams object which holds all the variables
      which we want to update separately in order to avoid bias.
    mask: Mask with 1s below the diagonal and then sharded across devices, used
      to calculate the penalty. Shape of: [eigenvectors_per_device,
        total_eigenvector_count]
    sliced_identity: Sharded copy of the identity matrix used to calculate the
      reward. Shape of: [eigenvectors_per_device, total_eigenvector_count]
    mean_estimate: SplitVector containing an estimate of the mean of the input
      data if it is not centered by default. This is used to calculate the
      covariances. Array tree with leaves of shape [...].
    epsilon: Add an isotropic term to the variance matrix to make it
      positive semi-definite. In this case, we're solving for: Av =
        lambda*(B+epsilon*I)v.
    maximize: Whether to search for top-k eigenvectors of Av = lambda*Bv (True)
      or the top-k of (-A)v = lambda*Bv (False). Due to the symmetry of the CCA
      problem, setting to True or False should not change the performance.

  Returns:
    returns the gradients for the eigenvectors and a new entry to update
    auxiliary variance estimates.
  """
  # First, collect all the eigenvectors v_k
  all_eigenvectors = jax.lax.all_gather(
      local_eigenvectors,
      axis_name='devices',
      axis=0,
      tiled=True,
  )
  if mean_estimate is not None:
    sharded_data = (
        jax.tree_map(lambda x, y: x - y, sharded_data[0], mean_estimate),
        jax.tree_map(lambda x, y: x - y, sharded_data[1], mean_estimate),
    )

  # Evaluate the matrix products for all eigenvectors on all machines using
  # different batches and aggregate them to get Bv_k and Av_k.
  (
      variance_vector_product,
      interaction_vector_product,
  ) = _unbiased_cca_matrix_products(
      all_eigenvectors,
      sharded_data,
  )

  return generalized_eigengame_gradients(
      local_eigenvectors=local_eigenvectors,
      all_eigenvectors=all_eigenvectors,
      a_vector_product=interaction_vector_product,
      b_vector_product=variance_vector_product,
      auxiliary_variables=auxiliary_variables,
      mask=mask,
      sliced_identity=sliced_identity,
      epsilon=epsilon,
      maximize=maximize,
  )


def _unbiased_pls_matrix_products(
    eigenvectors: SplitVector,
    sharded_data: Tuple[SplitVector, SplitVector],
) -> Tuple[SplitVector, SplitVector]:
  """Calculate and aggregate the equivalent gram matrices for PLS.

  PLS of data source with dim n and response (label) of dim m is equivalent to
  solving the generalized eigenvalue problem Av = lambda Bv, where v is
  dim n + m, A has the covariances between the data source and response and B
  the covariance within the data source and identity for the response. This
  function unpacks v then calculates Av and Bv from the data source and response
  before aggregating them across machines.

  The majority of the PLS specific logic lives here. The remainder of the
  algorithm should work with any other generalized eigenvalue problem.

  PLS is performed as a specific case of CCA therefore some eg_utils have been
  reused, e.g., SplitVector.

  Args:
    eigenvectors: SplitVector holding the generalized eigenvectors estimates for
      the two data sources.
      Array Tree with leaves of shape [k, ...] denoting v_k.
    sharded_data: SplitVector holding data from our two data sources.
      Array Tree with leaves of shape [b, ...].
  Returns:
    Tuple of two SplitVectors, each containing array trees with leaves of shape
    [k, ...] containing the gram matrix product of covariances between the data
    sources (Av) and within the data (Bv) respectively.
  """
  data_vector_x = []
  data_vector_y = []
  for sample in sharded_data:
    data_vector = eg_utils.tree_einsum(
        'k...,b...-> bk',
        eigenvectors,
        sample,
    )
    data_vector_x_sample = jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        data_vector.x,  # pytype: disable=attribute-error  # numpy-scalars
    )
    data_vector_y_sample = jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        data_vector.y,  # pytype: disable=attribute-error  # numpy-scalars
    )
    # divide by the batch size
    data_vector_x.append(data_vector_x_sample / data_vector_x_sample.shape[0])
    data_vector_y.append(data_vector_y_sample / data_vector_y_sample.shape[0])

  def _gram_product(
      data: chex.ArrayTree,
      data_vector: chex.Array,
  ) -> chex.ArrayTree:
    return eg_utils.tree_einsum_broadcast('b..., bk -> k...', data, data_vector)

  sharded_variance_vector = SplitVector(
      x=_gram_product(sharded_data[0].x, data_vector_x[0]),
      y=eigenvectors.y)  # diff to CCA is here <--
  sharded_interaction_vector = SplitVector(
      x=_gram_product(sharded_data[1].x, data_vector_y[1]),
      y=_gram_product(sharded_data[1].y, data_vector_x[1]))
  # Aggregate this across all machines.
  variance_vector_product = jax.lax.pmean(
      sharded_variance_vector, axis_name='devices')
  interaction_vector_product = jax.lax.pmean(
      sharded_interaction_vector, axis_name='devices')

  return (
      variance_vector_product,
      interaction_vector_product,
  )  # Bv and Av


def unbiased_pls_gradients(
    *,
    local_eigenvectors: SplitVector,
    sharded_data: Tuple[SplitVector, SplitVector],
    auxiliary_variables: eg_utils.AuxiliaryParams,
    mask: chex.Array,
    sliced_identity: chex.Array,
    mean_estimate: Optional[SplitVector] = None,
    epsilon: Optional[float] = None,
    maximize: float = True,
) -> Tuple[SplitVector, eg_utils.AuxiliaryParams,]:
  """Evaluates unbiased PLS gradients with data parallelism.

  Algorithm pseudocode can be found in Algorithm 1 of overleaf doc:
  https://ol.deepmind.host/read/kxdfdtfbsdxc

  In this case, we take in two independent batches of data, one to calculate Av
  and one to calculate Bv. Using two different samples results in an unbiased
  gradient, but there is a bias-variance tradeoff in the performance.

  Args:
    local_eigenvectors: SplitVector with local eigen vectors products. Array
      tree with leaves of shape [l, ...] denoting v_l.
    sharded_data: Tuple containing two independent batches of data from our
      two data sources. Tuple of SplitVectors with leaves of shape [b, ...].
    auxiliary_variables: AuxiliaryParams object which holds all the variables
      which we want to update separately in order to avoid bias.
    mask: Mask with 1s below the diagonal and then sharded across devices, used
      to calculate the penalty. Shape of: [eigenvectors_per_device,
        total_eigenvector_count]
    sliced_identity: Sharded copy of the identity matrix used to calculate the
      reward. Shape of: [eigenvectors_per_device, total_eigenvector_count]
    mean_estimate: SplitVector containing an estimate of the mean of the input
      data if it is not centered by default. This is used to calculate the
      covariances. Array tree with leaves of shape [...].
    epsilon: Add an isotropic term to the variance matrix to make it
      positive semi-definite. In this case, we're solving for: Av =
        lambda*(B+epsilon*I)v.
    maximize: unused- Solving Av = lambda * Bv is the only sensible approach for
      PLS. We do not foresee a use case for (-Av) = lambda * Bv. Please contact
      authors if you have a need for it.

  Returns:
    returns the gradients for the eigenvectors and a new entry to update
    auxiliary variance estimates.
  """
  del maximize

  # First, collect all the eigenvectors v_k
  all_eigenvectors = jax.lax.all_gather(
      local_eigenvectors,
      axis_name='devices',
      axis=0,
      tiled=True,
  )
  if mean_estimate is not None:
    sharded_data = (
        jax.tree_map(lambda x, y: x - y, sharded_data[0], mean_estimate),
        jax.tree_map(lambda x, y: x - y, sharded_data[1], mean_estimate),
    )

  # Evaluate the matrix products for all eigenvectors on all machines using
  # different batches and aggregate them to get Bv_k and Av_k.
  (
      variance_vector_product,
      interaction_vector_product,
  ) = _unbiased_pls_matrix_products(
      all_eigenvectors,
      sharded_data,
  )

  return generalized_eigengame_gradients(
      local_eigenvectors=local_eigenvectors,
      all_eigenvectors=all_eigenvectors,
      a_vector_product=interaction_vector_product,
      b_vector_product=variance_vector_product,
      auxiliary_variables=auxiliary_variables,
      mask=mask,
      sliced_identity=sliced_identity,
      epsilon=epsilon,
  )


def pca_generalized_eigengame_gradients(
    local_eigenvectors: chex.ArrayTree,
    sharded_data: chex.ArrayTree,
    auxiliary_variables: eg_utils.AuxiliaryParams,
    mask: chex.Array,
    sliced_identity: chex.Array,
    mean_estimate: Optional[chex.ArrayTree] = None,
    epsilon: Optional[float] = None,
    maximize: bool = True,
    )->Tuple[chex.ArrayTree, eg_utils.AuxiliaryParams,]:
  """Evaluates PCA gradients for two data sources with local data parallelism.

  Implements PCA. In this case, we simply set the B matrix as I, which means
  the problem is solving for Av=lambda v and we're back to the classic eigen
  value problem.

  This is not as lightweight as eigengame unloaded due to additional terms in
  the calculation and handling of the auxiliary variables, and is mostly here to
  demonstrate the flexibility of the generalized eigenvalue method. However,
  adaptive optimisers may have an easier time with this since the gradients
  for the generalized eigengame are naturally tangential to the unit sphere.

  einsum legend:
    l: local eigenvector index -- the number of eigenvectors per machine.
    k: global eigenvector index -- number of eigenvectors over all machines.

  Args:
    local_eigenvectors: SplitVector with local eigen vectors products. Array
      tree with leaves of shape [l, ...] denoting v_l.
    sharded_data: SplitVector containing a batch of data from our two data
      sources. Array tree with leaves of shape [b, ...].
    auxiliary_variables: AuxiliaryParams object which holds all the variables
      which we want to update separately in order to avoid bias.
    mask: Mask with 1s below the diagonal and then sharded across devices, used
      to calculate the penalty. Shape of: [eigenvectors_per_device,
        total_eigenvector_count]
    sliced_identity: Sharded copy of the identity matrix used to calculate the
      reward. Shape of: [eigenvectors_per_device, total_eigenvector_count]
    mean_estimate: SplitVector containing an estimate of the mean of the input
      data if it is not centered by default. This is used to calculate the
      covariances. Array tree with leaves of shape [...].
    epsilon: Add an isotropic term to the A matrix to make it
      positive semi-definite. In this case, we're solving for: (A+epsilon*I)v =
        lambda*v.
    maximize: unused- Solving Av = lambda * v is the only sensible approach for
      PCA. We do not foresee a use case for (-Av) = lambda * v. Please contact
      authors if you have a need for it.

  Returns:
    returns the gradients for the eigenvectors and a new entry to update
    auxiliary variance estimates.
  """
  del maximize

  # First, collect all the eigenvectors v_k
  all_eigenvectors = jax.lax.all_gather(
      local_eigenvectors,
      axis_name='devices',
      axis=0,
      tiled=True,
  )
  if mean_estimate is not None:
    sharded_data = jax.tree_map(lambda x, y: x - y, sharded_data, mean_estimate)

  # Calculate X^TXv/b for all v. This and v are all you need to construct the
  # gradient. This is done on each machine with a different batch.
  data_vector_product = eg_utils.tree_einsum(
      'b..., k... -> bk',
      sharded_data,
      all_eigenvectors,
      reduce_f=lambda x, y: x + y,
  )
  # divide by the batch size
  data_vector_product /= data_vector_product.shape[0]  # pytype: disable=attribute-error  # numpy-scalars
  sharded_gram_vector = eg_utils.tree_einsum_broadcast(
      'b..., bk -> k...',
      sharded_data,
      data_vector_product,
  )
  gram_vector = jax.lax.pmean(
      sharded_gram_vector,
      axis_name='devices',
  )

  # Add a small epsilon to the gram vector (Av) to make it positive definite.
  if epsilon is not None:
    gram_vector = jax.tree_map(
        lambda x, y: x + epsilon * y,
        gram_vector,
        all_eigenvectors,
    )
  return generalized_eigengame_gradients(
      local_eigenvectors=local_eigenvectors,
      all_eigenvectors=all_eigenvectors,
      a_vector_product=gram_vector,
      b_vector_product=all_eigenvectors,  # Just return V here.
      auxiliary_variables=auxiliary_variables,
      mask=mask,
      sliced_identity=sliced_identity,
      epsilon=None,
  )


def matrix_inverse_gradients(
    local_eigenvectors: chex.ArrayTree,
    sharded_data: chex.ArrayTree,
    auxiliary_variables: eg_utils.AuxiliaryParams,
    mask: chex.Array,
    sliced_identity: chex.Array,
    mean_estimate: Optional[chex.ArrayTree] = None,
    epsilon: Optional[float] = None,
    maximize: float = True,
) -> Tuple[chex.ArrayTree, eg_utils.AuxiliaryParams,]:
  """Finds the biggest matrix inverse components of B in a data parallel manner.

  In this case, we set A = I, and the problem becomes v = lambda Bv. Assuming
  only positive eigenvalues, The inverse of B may then be approximated as:

    B^{-1} = sum_{i=1}^k lambda_i v_i v_i^T

  with lambda_i>lambda_j for i<j meaning the approximation contains the most
  significant terms.

  einsum legend:
    l: local eigenvector index -- the number of eigenvectors per machine.
    k: global eigenvector index -- number of eigenvectors over all machines.

  Args:
    local_eigenvectors: SplitVector with local eigen vectors products. Array
      tree with leaves of shape [l, ...] denoting v_l.
    sharded_data: SplitVector containing a batch of data from our two data
      sources. Array tree with leaves of shape [b, ...].
    auxiliary_variables: AuxiliaryParams object which holds all the variables
      which we want to update separately in order to avoid bias.
    mask: Mask with 1s below the diagonal and then sharded across devices, used
      to calculate the penalty. Shape of: [eigenvectors_per_device,
        total_eigenvector_count]
    sliced_identity: Sharded copy of the identity matrix used to calculate the
      reward. Shape of: [eigenvectors_per_device, total_eigenvector_count]
    mean_estimate: SplitVector containing an estimate of the mean of the input
      data if it is not centered by default. This is used to calculate the
      covariances. Array tree with leaves of shape [...].
    epsilon: Add an isotropic term to the variance matrix to make it
      positive semi-definite. In this case, we're solving for: v =
        lambda*(B+epsilon*I)v.
    maximize: unused- Solving Iv = lambda * Bv is the only sensible approach for
      inverting B. We do not foresee a use case for (-Iv) = lambda * Bv. Please
      contact authors if you have a need for it.

  Returns:
    returns the gradients for the eigenvectors and a new entry to update
    auxiliary variance estimates.
  """
  del maximize

  # First, collect all the eigenvectors v_k
  all_eigenvectors = jax.lax.all_gather(
      local_eigenvectors,
      axis_name='devices',
      axis=0,
      tiled=True,
  )
  if mean_estimate is not None:
    sharded_data = jax.tree_map(lambda x, y: x - y, sharded_data, mean_estimate)

  # Calculate X^TXv/b for all v. This and v are all you need to construct the
  # gradient. This is done on each machine with a different batch.
  data_vector_product = eg_utils.tree_einsum(
      'b..., k... -> bk',
      sharded_data,
      all_eigenvectors,
      reduce_f=lambda x, y: x + y,
  )
  # divide by the batch size
  data_vector_product /= data_vector_product.shape[0]  # pytype: disable=attribute-error  # numpy-scalars
  sharded_gram_vector = eg_utils.tree_einsum_broadcast(
      'b..., bk -> k...',
      sharded_data,
      data_vector_product,
  )
  gram_vector = jax.lax.pmean(
      sharded_gram_vector,
      axis_name='devices',
  )
  return generalized_eigengame_gradients(
      local_eigenvectors=local_eigenvectors,
      all_eigenvectors=all_eigenvectors,
      a_vector_product=all_eigenvectors,
      b_vector_product=gram_vector,
      auxiliary_variables=auxiliary_variables,
      mask=mask,
      sliced_identity=sliced_identity,
      epsilon=epsilon,
  )


def unbiased_ica_matrix_products(
    eigenvectors: chex.ArrayTree,
    sharded_data: Tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree]
) -> Tuple[chex.ArrayTree, chex.ArrayTree]:
  """Evaluates the matrix vector products required for ICA.

  In this case, we need to product of the following matrices with the
  eigenvectors:
  A = E(<x_i, x_i>x_i x_i^T) - tr(B)B - 2 B^2
  (computed with samples (0), (1,0), (1,0) for each term respectively)
  B = E(x_i x_i^T)
  (computed with sample (2))

  In order to have unbiased gradients when we sample, we need to have
  independent samples for each random variance which is multiplied together.
  Since E(x)E(y)=/=E(xy) in general unless x, y are independent.

  So tr(B)B - 2 B^2 will need two data batches, and B by itself will need a
  third batch.

  This is further complicated by the fact that in order for some of the products
  of averages to be computed (e.g. the tr(B)B term), we will need multiple
  pmeans.

  Args:
    eigenvectors: Array Tree holding all eigen vectors products. Array tree
      with leaves of shape [k, ...] denoting v_k.
    sharded_data: Triple of array trees containing independently sampled
      batches of input data, consisting array tree with leaves of shape [k,
      ...].

  Returns:
    returns a Tuple of two Array Trees. First Denoting Av, second denoting Bv
    with leaves of shape v_k.
  """

  # Compute <x_i, x_i> term for the first term of A
  data_norm = eg_utils.tree_einsum(
      'b..., b...->b',
      sharded_data[0],
      sharded_data[0],
      reduce_f=lambda x, y: x + y,
  )
  per_machine_batch_size = data_norm.shape[0]  # pytype: disable=attribute-error  # numpy-scalars
  data_vector_products = []
  for sample in sharded_data:
    # Calculate Xv for each sample and divide by the per machine batch size
    vector_products = eg_utils.tree_einsum(
        'b..., k... -> bk',
        sample,
        eigenvectors,
        reduce_f=lambda x, y: x + y,
    )
    # device by the per machine batch size
    data_vector_products.append(vector_products / per_machine_batch_size)
  # estimate of  E(<x_i, x_i>x_i x_i^T)v. For this term, we need to use
  # the same data for everything since it's a single expectation.
  fourth_power_term = eg_utils.tree_einsum_broadcast(
      'b..., bk, b -> k...',
      sharded_data[0],
      data_vector_products[0],
      data_norm,
  )

  # estimate of trace of B = XTX/n. We need to psum immediately here since
  # This needs to be multiplied to B, which will also need to be psumed later.
  average_data_norm = eg_utils.tree_einsum(
      'b..., b...->',
      sharded_data[1],
      sharded_data[1],
      reduce_f=lambda x, y: x + y,
  ) / per_machine_batch_size
  average_data_norm = jax.lax.pmean(
      average_data_norm,
      axis_name='devices',
  )
  # An estimate of Bv = XTXv using a sample independent from Tr(B).
  covariance_product = eg_utils.tree_einsum_broadcast(
      'b..., bk -> k...',
      sharded_data[0],
      data_vector_products[0],
  )
  mean_covariance_product = jax.lax.pmean(
      covariance_product,
      axis_name='devices',
  )
  # B^2v. This is computed with batches 0, 1
  covariance_squared_product = eg_utils.tree_einsum_broadcast(
      'b..., bk -> k...',
      sharded_data[1],
      eg_utils.tree_einsum(
          'b..., k... -> bk',
          sharded_data[1],
          mean_covariance_product,
          reduce_f=lambda x, y: x + y,
      ) / per_machine_batch_size,
  )
  # Calculate a portion of the A matrix and then pmean
  # this is (E(<x_i,x_i>x_ix_iT) - 2 B^2)v
  kurtosis = jax.lax.pmean(
      jax.tree_map(
          lambda x, y: x - 2 * y,
          fourth_power_term,
          covariance_squared_product,
      ),
      axis_name='devices',
  )
  # We don't include the Tr(B)B term since that is already pmeaned.
  kurtosis = jax.tree_map(
      lambda x, y: x - average_data_norm * y,
      kurtosis,
      mean_covariance_product,
  )
  # Finally, calculate another covariace matrix using another independent sample
  independent_covariance_product = jax.lax.pmean(
      eg_utils.tree_einsum_broadcast(
          'b..., bk -> k...',
          sharded_data[2],
          data_vector_products[2],
      ),
      axis_name='devices',
  )
  return kurtosis, independent_covariance_product


def unbiased_ica_gradients(
    *,
    local_eigenvectors: chex.ArrayTree,
    sharded_data: Tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree],
    sliced_identity: chex.Array,
    mask: chex.Array,
    auxiliary_variables: Optional[eg_utils.AuxiliaryParams],
    mean_estimate: Optional[chex.ArrayTree] = None,
    epsilon: Optional[float] = None,
    maximize: Optional[bool] = True,
) -> Tuple[chex.ArrayTree, eg_utils.AuxiliaryParams,]:
  """Evaluates the gradients required to compute ICA on the dataset.

    For ICA, we are attempting to separate out multiple components of a data
    source. We can reduce the problem into a generalized eigenvalue problem
    using the following matrices:

    A = E(<x_i, x_i>x_i x_i^T) - tr(B)B - 2 B^2
    B = E(x_i x_i^T)

    However, the twist here is that we need three separate data samples in
    order to avoid bias -- one for the first term of A, one for the remaning
    terms of A and one for B.

  Args:
    local_eigenvectors: Array tree with local eigen vectors products. Array
      tree with leaves of shape [l, ...] denoting v_l.
    sharded_data: Triple of array trees containing independently sampled
      batches of input data, consisting array tree with leaves of shape [k,
      ...].
    sliced_identity: Sharded copy of the identity matrix used to calculate the
      reward. Shape of: [eigenvectors_per_device, total_eigenvector_count]
    mask: Mask with 1s below the diagonal and then sharded across devices, used
      to calculate the penalty. Shape of: [eigenvectors_per_device,
        total_eigenvector_count]
    auxiliary_variables: AuxiliaryParams object which holds all the variables
      which we want to update separately in order to avoid bias.
    mean_estimate: SplitVector containing an estimate of the mean of the input
      data if it is not centered by default. This is used to calculate the
      covariances. Array tree with leaves of shape [...].
    epsilon: Add an isotropic term to the variance matrix to make it
      positive semi-definite. In this case, we're solving for: v =
        lambda*(B+epsilon*I)v.
    maximize: Whether to search for top-k eigenvectors of Av = lambda*Bv (True)
      or the top-k of (-A)v = lambda*Bv (False)

  Returns:
    returns the gradients for the eigenvectors and a new entry to update
    auxiliary variance estimates.
  """
  # First, collect all the eigenvectors v_k
  all_eigenvectors = jax.lax.all_gather(
      local_eigenvectors,
      axis_name='devices',
      axis=0,
      tiled=True,
  )
  if mean_estimate is not None:
    sharded_data = tuple(
        jax.tree_map(lambda x, y: x - y, sample, mean_estimate)
        for sample in sharded_data
        )

  kurtosis, independent_covariance = unbiased_ica_matrix_products(
      all_eigenvectors,
      sharded_data,
  )

  return generalized_eigengame_gradients(
      local_eigenvectors=local_eigenvectors,
      all_eigenvectors=all_eigenvectors,
      a_vector_product=kurtosis,
      b_vector_product=independent_covariance,
      auxiliary_variables=auxiliary_variables,
      mask=mask,
      sliced_identity=sliced_identity,
      epsilon=epsilon,
      maximize=maximize,
  )
