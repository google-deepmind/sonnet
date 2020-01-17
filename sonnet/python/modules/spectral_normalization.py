# Copyright 2017 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Functions and modules for implementing Spectral Normalization.

This implementation follows the use in:
  https://arxiv.org/abs/1802.05957
  https://arxiv.org/abs/1805.08318
  https://arxiv.org/abs/1809.11096
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

# Dependency imports
from sonnet.python.custom_getters import context
from sonnet.python.modules import base
from sonnet.python.modules import util
import tensorflow.compat.v1 as tf


def wrap_with_spectral_norm(module_class,
                            sn_kwargs=None,
                            pow_iter_collection=None):
  """Returns a constructor for the inner class with spectral normalization.

  This function accepts a Sonnet AbstractModule class as argument (the class,
  *not* an instance of that class) alongside an optional dictionary of keyword
  arguments for the spectral_norm function, and returns a constructor which can
  be treated identically to the constructor of the input class, but with
  spectral normalization applied to the weights created by the class.

  Internally, this is just a partially evaluated SpectralNormWrapper module.

  `pow_iter_collection`, if not None, is treated as the name of a TensorFlow
  global collection. Each time the module's weight matrix is accessed ops are
  built for performing one step of power iteration to approximate that weight's
  first singular follow and ops are created for saving this new approximation in
  an internal variable. At build-time the resulting object takes a special
  boolean 'enable_power_iteration' keyword argument. If this is True (the
  default), a control dependency on the operation for updating this internal
  variable is attached to the returned weight. Otherwise, the update is *not*
  attached as a control dependency, but an op is placed into the
  `pow_iter_collection` global collection which causes the internal variable to
  be updated. It is then up to the user to choose whether to run this update.

  Args:
    module_class: A constructor/class reference for a Sonnet module you would
        like to wrap and automatically apply spectral normalization.
    sn_kwargs: Keyword arguments to be passed to the spectral_norm function
        in addition to the weight tensor.
    pow_iter_collection: The name of a global collection for potentially
        storing ops for updating internal variables.
  Returns:
    An snt.AbstractModule class representing the original with spectral norm.
  """
  sn_kwargs = sn_kwargs or {}
  return functools.partial(
      SpectralNormWrapper, module_class, sn_kwargs, pow_iter_collection)


class SpectralNormWrapper(base.AbstractModule):
  """Wraps a Sonnet Module to selectively apply Spectral Normalization."""

  def __init__(self, module, sn_kwargs, pow_iter_collection, *args, **kwargs):
    """Constructs a wrapped Sonnet module with Spectral Normalization.

    The module expects a first argument which should be a Sonnet AbstractModule
    and a second argument which is a dictionary which is passed to the inner
    spectral_norm function as kwargs.

    When connecting this module to the graph,the argument 'pow_iter_collection'
    is treated specially for this wrapper (rather than for the _build
    method of the inner module). If pow_iter_collection is None (the default),
    the approximate first singular value for weights will *not* be updated based
    on the inputs passed at the given _build call. However an op for updating
    the singular value will be placed into the pow_iter_collection global
    collection.

    If pow_iter_collection is None or not passed, a control dependency on the
    update op will be applied to the output of the _build function. Regardless,
    the kwarg is deleted from the list of keywords passed to the inner module.

    Args:
      module: A constructor/class reference for a Sonnet module you would like
          to construct.
      sn_kwargs: Keyword arguments to be passed to the spectral_norm function
          in addition to the weight tensor.
      pow_iter_collection: The name of a global collection for potentially
          storing ops for updating internal variables.
      *args: Construction-time arguments to the module.
      **kwargs: Construction-time  keyword arguments to the module.
    """
    name = kwargs.get('name', 'sn') + '_wrapper'
    # Our getter needs to be able to be disabled.
    getter_immediate_update, getter_deferred_update = self.sn_getter(sn_kwargs)
    w_getter = lambda g: util.custom_getter_router({'.*/w$': g}, lambda s: s)
    getter_immediate_update = w_getter(getter_immediate_update)
    getter_deferred_update = w_getter(getter_deferred_update)
    self._context_getter = context.Context(
        getter_immediate_update, default_getter=getter_deferred_update)
    self.pow_iter_collection = pow_iter_collection
    super(SpectralNormWrapper, self).__init__(
        name=name, custom_getter=self._context_getter)

    # Let's construct our model.
    with self._enter_variable_scope():
      self._module = module(*args, **kwargs)

  def _build(self, *args, **kwargs):
    if kwargs.pop('enable_power_iteration', True):
      with self._context_getter:
        return self._module(*args, **kwargs)
    else:
      return self._module(*args, **kwargs)

  def sn_getter(self, spectral_norm_kwargs):
    """Returns a curried spectral normalization Custom Getter."""
    def getter_immediate_update(getter, *args, **kwargs):
      w = getter(*args, **kwargs)  # This is our variable.
      w_spectral_normalized = spectral_norm(
          w, update_collection=None, **spectral_norm_kwargs)['w_bar']
      return w_spectral_normalized

    def getter_deferred_update(getter, *args, **kwargs):
      w = getter(*args, **kwargs)  # This is our variable.
      w_spectral_normalized = spectral_norm(
          w, update_collection=self.pow_iter_collection,
          **spectral_norm_kwargs)['w_bar']
      return w_spectral_normalized
    return getter_immediate_update, getter_deferred_update


def _l2_normalize(t, axis=None, eps=1e-12):
  """Normalizes along dimension `axis` using an L2 norm.

  We use this over tf.nn.l2_normalize for numerical stability reasons.

  Args:
    t: A `Tensor`.
    axis: Dimension along which to normalize, e.g. `1` to separately normalize
      vectors in a batch. Passing `None` views `t` as a flattened vector when
      calculating the norm (equivalent to Frobenius norm).
    eps: Epsilon to avoid dividing by zero.

  Returns:
    A `Tensor` with the same shape as `t`.
  """
  return t * tf.rsqrt(tf.reduce_sum(tf.square(t), axis, keepdims=True) + eps)


def spectral_norm(weight,
                  num_iters=1,
                  update_collection=None,
                  eps=1e-4):
  """Spectral Weight Normalization.

  Applies first-singular-value spectral normalization to weight and returns a
  tensor equivalent to weight with spectral normalization applies. By default,
  it also updates an inner variable for keeping track of the spectral values of
  this weight matrix. If update_collection is not None, however, this function
  does not update the variable automatically, instead placing an op for this
  update in the 'update_collection' global collection.

  Args:
    weight: The weight tensor which requires spectral normalization
    num_iters: Number of SN iterations.
    update_collection: The update collection for assigning persisted variable u.
      If None, the function will update u0 during the forward pass. Otherwise if
      the update_collection equals 'update_collection', it will put the
      assignment in a collection defined by the user. Then the user will need to
      run the assignment explicitly.
    eps: numerical stability constant > 0.

  Returns:
    A dictionary of:
      w_bar: The normalized weight tensor
      sigma: The estimated singular value for the weight tensor.
      u0: The internal persisted variable.
  """
  if num_iters < 1:
    raise ValueError('num_iters must be a positive integer. {} given.'.format(
        num_iters))

  original_dtype = weight.dtype
  weight = tf.cast(weight, tf.float32)

  w_shape = weight.shape.as_list()
  w_mat = tf.reshape(weight, [-1, w_shape[-1]])
  u0 = tf.get_variable(
      'u0', [1, w_shape[-1]],
      initializer=tf.truncated_normal_initializer(),
      trainable=False)
  u0_ = u0

  # Power iteration for the weight's singular value.
  for _ in range(num_iters):
    v0_ = _l2_normalize(tf.matmul(u0_, w_mat, transpose_b=True), eps=eps)
    u0_ = _l2_normalize(tf.matmul(v0_, w_mat), eps=eps)

  u0_ = tf.stop_gradient(u0_)
  v0_ = tf.stop_gradient(v0_)
  sigma = tf.squeeze(tf.matmul(tf.matmul(v0_, w_mat), u0_, transpose_b=True),
                     axis=[0, 1])

  w_mat /= sigma
  w_bar = tf.reshape(w_mat, w_shape)

  # Potentially add a control dependency on u0s update.
  if update_collection is None:
    u_assign_ops = [u0.assign(u0_)]
    with tf.control_dependencies(u_assign_ops):
      w_bar = tf.identity(w_bar)
  else:
    tf.add_to_collection(update_collection, u0.assign(u0_))
  return {
      'w_bar': tf.cast(w_bar, original_dtype),
      'sigma': tf.cast(sigma, original_dtype),
      'u0': tf.cast(u0, original_dtype)
  }
