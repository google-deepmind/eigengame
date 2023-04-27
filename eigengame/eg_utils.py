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

"""Helper functions for the distributed TPU implementation of EigenGame."""
import collections
import enum
import functools
import os
from typing import Callable, Dict, NamedTuple, Optional, Tuple, Union
from absl import logging

import chex
import jax
import jax.numpy as jnp
from jaxline import utils

import numpy as np


class ExperimentType(enum.Enum):
  BIASED_CCA = 'biased_cca'
  CCA = 'unbiased_cca'
  PLS = 'unbiased_pls'
  ICA = 'unbiased_ica'
  PCA = 'pca'
  MATRIX_INVERSE = 'matrix_inverse'


class SplitVector(NamedTuple):
  """Generalized Eigenvector object used for CCA, PLS, etc.

  Concatenations and splits are some of the slowest things you can do on a TPU,
  so we want to keep the portions of the eigenvectors responsible for each data
  source separate.
  """
  x: chex.ArrayTree
  y: chex.ArrayTree


class AuxiliaryParams(NamedTuple):
  r"""Container for auxiliary variables for $\gamma$-EigenGame."""
  b_vector_product: chex.ArrayTree
  b_inner_product_diag: chex.Array


EigenGameGradientFunction = Callable[..., Tuple[chex.ArrayTree,
                                                AuxiliaryParams]]

EigenGameQuotientFunction = Callable[..., Tuple[chex.ArrayTree, chex.ArrayTree]]


@functools.partial(
    jax.pmap,
    in_axes=0,
    out_axes=0,
    axis_name='devices',
    static_broadcasted_argnums=0,
)
def init_aux_variables(
    device_count: int,
    init_vectors: chex.ArrayTree,
) -> AuxiliaryParams:
  """Initializes the auxiliary variables from the eigenvalues."""
  leaves, _ = jax.tree_util.tree_flatten(init_vectors)
  per_device_eigenvectors = leaves[0].shape[0]
  total_eigenvectors = per_device_eigenvectors * device_count
  return AuxiliaryParams(
      b_vector_product=jax.tree_map(
          lambda leaf: jnp.zeros((total_eigenvectors, *leaf.shape[1:])),
          init_vectors,
      ),
      b_inner_product_diag=jnp.zeros(total_eigenvectors),
  )


class AuxiliaryMovingAverage:
  """Simple class which computes the moving average of the auxiliary variables.

  This is used in the generalized eigenvalue problem, where the reciprocal of
  an estimate may have a severe biasing effect for small batches.
  """

  def __init__(self, max_len: int):
    self._values = collections.deque(maxlen=max_len)

  def get_moving_average(self) -> Optional[chex.ArrayTree]:
    if not self._values:
      return None
    length = len(self._values)
    values_sum = jax.tree_map(
        lambda *x: sum(x),
        self._values[0],
        *list(self._values)[1:],
    )
    return jax.tree_map(lambda x: x / length, values_sum)

  def add_value(self, new_value: chex.ArrayTree) -> None:
    self._values.append(new_value)


def get_spherical_gradients(
    gradient: chex.ArrayTree,
    eigenvectors: chex.ArrayTree,
) -> chex.ArrayTree:
  """Project gradients to a perpendicular to each of the eigenvectors."""
  tangential_component_scale = tree_einsum(
      'l..., l...-> l',
      gradient,
      eigenvectors,
      reduce_f=lambda x, y: x + y,
  )
  tangential_component = tree_einsum_broadcast(
      'l..., l -> l... ',
      eigenvectors,
      tangential_component_scale,
  )

  return jax.tree_map(
      lambda x, y: x - y,
      gradient,
      tangential_component,
  )


def normalize_eigenvectors(eigenvectors: chex.ArrayTree) -> chex.ArrayTree:
  """Normalize all eigenvectors."""
  squared_norm = jax.tree_util.tree_reduce(
      lambda x, y: x + y,
      jax.tree_map(
          lambda x: jnp.einsum('k..., k... -> k', x, x),
          eigenvectors,
      ))
  return jax.tree_map(
      lambda x: jnp.einsum('k..., k -> k...', x, 1 / jnp.sqrt(squared_norm)),
      eigenvectors,
  )


def initialize_eigenvectors(
    eigenvector_count: int,
    batch: chex.ArrayTree,
    rng_key: chex.PRNGKey,
) -> chex.ArrayTree:
  """Initialize the eigenvectors on a unit sphere and shards it.

  Args:
    eigenvector_count: Total number of eigenvectors (i.e. k)
    batch: A batch of the data we're trying to find the eigenvalues of. The
      initialized vectors will take the shape and tree structure of this data.
      Array tree with leaves of shape [b, ...].
    rng_key: jax rng seed. For multihost, each host should have a different
      seed in order to initialize correctly

  Returns:
    A pytree of initialized, normalized vectors in the same structure as the
    input batch. Array tree with leaves of shape [num_devices, l, ...].
  """
  device_count = jax.device_count()
  local_device_count = jax.local_device_count()
  if eigenvector_count % device_count != 0:
    raise ValueError(f'Number of devices ({device_count}) must divide number of'
                     'eigenvectors ({eigenvector_count}).')
  per_device_count = eigenvector_count // device_count
  leaves, treedef = jax.tree_flatten(batch)
  shapes = [(per_device_count, *leaf.shape[1:]) for leaf in leaves]

  eigenvectors = []
  per_device_keys = jax.random.split(rng_key, local_device_count)
  for per_device_key in per_device_keys:
    # generate a different key for each leaf on each device
    per_leaf_keys = jax.random.split(per_device_key, len(leaves))
    # generate random number for each leaf
    vector_leaves = [
        jax.random.normal(key, shape)
        for key, shape in zip(per_leaf_keys, shapes)
    ]
    eigenvector_tree = jax.tree_unflatten(treedef, vector_leaves)
    normalized_eigenvector = normalize_eigenvectors(eigenvector_tree)
    eigenvectors.append(normalized_eigenvector)
  return jax.device_put_sharded(eigenvectors, jax.local_devices())


def get_local_slice(
    local_identity_slice: chex.Array,
    input_vector_tree: chex.ArrayTree,
) -> chex.ArrayTree:
  """Get the local portion from all the eigenvectors.

  Multiplying by a matrix here to select the vectors that we care about locally.
  This is significantly faster than using jnp.take.

  Args:
    local_identity_slice: A slice of the identity matrix denoting the vectors
      which we care about locally
    input_vector_tree: An array tree of data, with the first index querying all
      the vectors.

  Returns:
      A pytree of the same structure as input_vector_tree, but with only the
      relevant vectors specified in the identity.
  """

  def get_slice(all_vectors):
    return jnp.einsum(
        'k..., lk -> l...',
        all_vectors,
        local_identity_slice,
    )

  return jax.tree_map(get_slice, input_vector_tree)


def per_vector_metric_log(
    metric_name: str,
    metric: chex.Array,
) -> Dict[str, float]:
  """Creates logs for each vector in a sortable way."""
  # get the biggest index length
  max_index_length = len(str(len(metric)))

  def get_key(index: int) -> str:
    """Adds metrix prefix and pads the index so the key is sortable."""
    padded_index = str(index).rjust(max_index_length, '0')
    return metric_name + '_' + padded_index

  return {get_key(i): value for i, value in enumerate(metric)}


def tree_einsum(
    subscripts: str,
    *operands: chex.ArrayTree,
    reduce_f: Optional[Callable[[chex.Array, chex.Array], chex.Array]] = None
) -> Union[chex.ArrayTree, chex.Array]:
  """Applies an leaf wise einsum to a list of trees.

  Args:
    subscripts: subscript string denoting the einsum operation.
    *operands: a list of pytrees with the same structure. The einsum will be
      applied leafwise.
    reduce_f: Function denoting a reduction. If not left empty, this calls a
      tree reduce on the resulting tree after the einsum.

  Returns:
      A pytree with the same structure as the input operands if reduce_f.
      Otherwise an array which is the result of the reduction.
  """
  einsum_function = functools.partial(jnp.einsum, subscripts)
  mapped_tree = jax.tree_map(einsum_function, *operands)
  if reduce_f is None:
    return mapped_tree
  else:
    return jax.tree_util.tree_reduce(reduce_f, mapped_tree)


def tree_einsum_broadcast(
    subscripts: str,
    tree: chex.ArrayTree,
    *array_operands: chex.Array,
    reduce_f: Optional[Callable[[chex.Array, chex.Array], chex.Array]] = None
) -> Union[chex.ArrayTree, chex.Array]:
  """Applies an einsum operation on a list of arrays to all leaves of a tree.

  Args:
    subscripts: subscript string denoting the einsum operation. The first
      argument must denote the tree leaf, followed by the list of arrays.
    tree: A pytree. The einsum will be applied with the leaves of this tree in
      the first argument.
    *array_operands: A list of arrays. The sinsum with these arrays will be
      mapped to each leaf in the tree.
    reduce_f: Function denoting a reduction. If not left empty, this calls a
      tree reduce on the resulting tree after the einsum.

  Returns:
      A pytree with the same structure as the input tree. if reduce_f.
      Otherwise an array which is the result of the reduction.
  """
  einsum_function = lambda leaf: jnp.einsum(subscripts, leaf, *array_operands)
  mapped_tree = jax.tree_map(einsum_function, tree)
  if reduce_f is None:
    return mapped_tree
  else:
    return jax.tree_util.tree_reduce(reduce_f, mapped_tree)


def get_first(xs):
  """Gets values from the first device."""
  return jax.tree_util.tree_map(lambda x: x[0], xs)


class InMemoryCheckpointerPlusSaveEigVecs(utils.InMemoryCheckpointer):
  """A Checkpointer reliant on an in-memory global dictionary."""

  def __init__(self, config, mode: str):
    super().__init__(config, mode)
    self._checkpoint_dir = config.checkpoint_dir

  def save(self, ckpt_series: str) -> None:
    """Saves the checkpoint."""
    super().save(ckpt_series)
    series = utils.GLOBAL_CHECKPOINT_DICT[ckpt_series]
    active_state = self.get_experiment_state(ckpt_series)
    id_ = 0 if not series.history else series.history[-1].id + 1
    filename = ckpt_series + '_' + str(id_)
    filepath = os.path.join(self._checkpoint_dir, filename) + '.npy'
    vecs = np.array(active_state.experiment_module.get_eigenvectors())
    np.save(filepath, vecs)
    logging.info('Saved eigenvectors to %s.', filepath)
