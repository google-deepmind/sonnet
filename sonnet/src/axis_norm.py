# Copyright 2019 The Sonnet Authors. All Rights Reserved.
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
"""Generic axis normalization module."""

import collections.abc
from typing import Optional
from sonnet.src import base
from sonnet.src import initializers
from sonnet.src import once
from sonnet.src import types
from sonnet.src import utils
import tensorflow as tf


class LayerNorm(base.Module):
  r"""Normalizes inputs along the given axes.

  This is a generic implementation of normalization along specific axes of the
  input. :class:`InstanceNorm` is a subclass of this module, it normalizes over
  the spatial dimensions.

  It transforms the input ``x`` into:

  .. math::

     \d{outputs} = \d{scale} \dfrac{x - \mu}{\sigma + \epsilon} + \d{offset}

  Where :math:`\mu` and :math:`\sigma` are respectively the mean and standard
  deviation of ``x``.

  There are many different variations for how users want to manage scale and
  offset if they require them at all. These are:

    - No ``scale``/``offset`` in which case ``create_*`` should be set to
      ``False`` and ``scale``/``offset`` aren't passed when the module is
      called.
    - Trainable ``scale``/``offset`` in which case create_* should be set to
      ``True`` and again ``scale``/``offset`` aren't passed when the module is
      called. In this case this module creates and owns the scale/offset
      variables.
    - Externally generated ``scale``/``offset``, such as for conditional
      normalization, in which case ``create_*`` should be set to ``False`` and
      then the values fed in at call time.

  Attributes:
    scale: If ``create_scale=True``, a trainable :tf:`Variable` holding the
      current scale.
    offset: If ``create_offset=True``, a trainable :tf:`Variable` holding the
      current offset.
  """

  def __init__(self,
               axis: types.Axis,
               create_scale: bool,
               create_offset: bool,
               eps: types.FloatLike = 1e-5,
               scale_init: Optional[initializers.Initializer] = None,
               offset_init: Optional[initializers.Initializer] = None,
               data_format: str = "channels_last",
               name: Optional[str] = None):
    r"""Constructs an ``LayerNorm`` module.

    Args:
      axis: An ``int``, ``slice`` or sequence of ``int``\s representing the axes
        which should be normalized across. Typical usages are: ``1`` or ``-1``
        for normalization over just the channels and ``slice(1, None)``,
        ``slice(2, None)`` for normalization over the spatial and channel
        dimensions whilst avoiding the batch and/or time dimensions.
      create_scale: ``bool`` representing whether to create a trainable scale
        per channel applied after the normalization.
      create_offset: ``bool`` representing whether to create a trainable offset
        per channel applied after normalization and scaling.
      eps: Small epsilon to avoid division by zero variance. Defaults to
        ``1e-5``.
      scale_init: Optional initializer for the scale variable. Can only be set
        if ``create_scale=True``. By default scale is initialized to ``1``.
      offset_init: Optional initializer for the offset variable. Can only be set
        if ``create_offset=True``. By default offset is initialized to ``0``.
      data_format: The data format of the input. Can be either
        ``channels_first``, ``channels_last``, ``N...C`` or ``NC...``. By
        default it is ``channels_last``.
      name: Name of the module.
    """
    super().__init__(name=name)

    if isinstance(axis, slice):
      self._axis = axis
    elif isinstance(axis, int):
      self._axis = (axis,)
    elif (isinstance(axis, collections.abc.Iterable) and
          all(isinstance(ax, int) for ax in axis)):
      self._axis = axis
    else:
      raise ValueError("`axis` should be an int, slice or iterable of ints.")

    self._eps = eps

    self._data_format = data_format
    self._channel_index = utils.get_channel_index(data_format)

    self._rank = None

    self._create_scale = create_scale
    self._create_offset = create_offset

    if self._create_scale:
      self._scale_init = (
          scale_init if scale_init is not None else initializers.Ones())
    elif scale_init is not None:
      raise ValueError("Cannot set `scale_init` if `create_scale=False`.")
    if self._create_offset:
      self._offset_init = (
          offset_init if offset_init is not None else initializers.Zeros())
    elif offset_init is not None:
      raise ValueError("Cannot set `offset_init` if `create_offset=False`.")

  def __call__(self,
               inputs: tf.Tensor,
               scale: Optional[tf.Tensor] = None,
               offset: Optional[tf.Tensor] = None) -> tf.Tensor:
    """Returns normalized inputs.

    Args:
      inputs: An n-D tensor of the ``data_format`` specified in the constructor
        on which the transformation is performed.
      scale: A tensor up to n-D. The shape of this tensor must be broadcastable
        to the shape of ``inputs``. This is the scale applied to the normalized
        inputs. This cannot be passed in if the module was constructed with
        ``create_scale=True``.
      offset: A tensor up to n-D. The shape of this tensor must be broadcastable
        to the shape of ``inputs``. This is the offset applied to the normalized
        ``inputs``. This cannot be passed in if the module was constructed with
        ``create_offset=True``.

    Returns:
      An n-d tensor of the same shape as inputs that has been normalized.
    """
    self._initialize(inputs)
    if self._create_scale:
      if scale is not None:
        raise ValueError(
            "Cannot pass `scale` at call time if `create_scale=True`.")
      scale = self.scale

    if self._create_offset:
      if offset is not None:
        raise ValueError(
            "Cannot pass `offset` at call time if `create_offset=True`.")
      offset = self.offset

    if len(inputs.shape) != self._rank:
      raise ValueError(
          "The rank of the inputs cannot change between calls, the"
          " original call was rank={} but this call was rank={}.".format(
              self._rank, len(inputs.shape)))

    mean, var = tf.nn.moments(inputs, self._axis, keepdims=True)

    normalized = tf.nn.batch_normalization(
        inputs,
        mean=mean,
        variance=var,
        scale=scale,
        offset=offset,
        variance_epsilon=self._eps)
    return normalized

  @once.once
  def _initialize(self, inputs: tf.Tensor):
    """Setup of rank specific values."""
    self._rank = len(inputs.shape)

    # Turns slice into list of axis
    if isinstance(self._axis, slice):
      axes = tuple(range(self._rank))
      self._axis = axes[self._axis]

    # Create scale and offset variables
    dtype = inputs.dtype
    if self._channel_index == -1:
      params_shape = [inputs.shape[-1]]
    else:  # self._channel_index == 1
      params_shape = [inputs.shape[1]] + [1] * (self._rank - 2)

    if self._create_scale:
      self.scale = tf.Variable(
          self._scale_init(params_shape, dtype), name="scale")
    else:
      self.scale = None

    if self._create_offset:
      self.offset = tf.Variable(
          self._offset_init(params_shape, dtype), name="offset")
    else:
      self.offset = None


class InstanceNorm(LayerNorm):
  """Normalizes inputs along the spatial dimensions.

  See :class:`LayerNorm` for more details.

  Attributes:
    scale: If ``create_scale=True``, a trainable :tf:`Variable` holding the
      current scale.
    offset: If ``create_offset=True``, a trainable :tf:`Variable` holding the
      current offset.
  """

  def __init__(self,
               create_scale: bool,
               create_offset: bool,
               eps: types.FloatLike = 1e-5,
               scale_init: Optional[initializers.Initializer] = None,
               offset_init: Optional[initializers.Initializer] = None,
               data_format: str = "channels_last",
               name: Optional[str] = None):
    """Constructs an ``InstanceNorm`` module.

    This method creates a module which normalizes over the spatial dimensions.

    Args:
      create_scale: ``bool`` representing whether to create a trainable scale
        per channel applied after the normalization.
      create_offset: ``bool`` representing whether to create a trainable offset
        per channel applied after normalization and scaling.
      eps: Small epsilon to avoid division by zero variance. Defaults to
        ``1e-5``.
      scale_init: Optional initializer for the scale variable. Can only be set
        if ``create_scale=True``. By default scale is initialized to ``1``.
      offset_init: Optional initializer for the offset variable. Can only be set
        if ``create_offset=True``. By default offset is initialized to ``0``.
      data_format: The data format of the input. Can be either
        ``channels_first``, ``channels_last``, ``N...C`` or ``NC...``. By
        default it is ``channels_last``.
      name: Name of the module.
    """
    if utils.get_channel_index(data_format) == 1:
      axis = slice(2, None)
    else:  # channel_index = -1
      axis = slice(1, -1)
    super().__init__(
        axis=axis,
        create_scale=create_scale,
        create_offset=create_offset,
        eps=eps,
        scale_init=scale_init,
        offset_init=offset_init,
        data_format=data_format,
        name=name)
