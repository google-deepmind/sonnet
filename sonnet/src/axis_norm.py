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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import six

from sonnet.src import base
from sonnet.src import initializers
from sonnet.src import once
from sonnet.src import utils
import tensorflow as tf


class AxisNorm(base.Module):
  """Normalizes inputs along the given axes.

  This is a generic implementation of normalization along specific axes of the
  input. LayerNorm and InstanceNorm are subclasses of this module, they
  normalize over the channel and spatial dimensions respectively.
  It transforms the input x into:

    outputs = scale * (x - mu) / (sigma + eps) + offset

  where mu and sigma are respectively the mean and standard deviation of x.

  There are many different variations for how users want to manage scale and
  offset if they require them at all. These are:

    - No scale/offset in which case create_* should be set to False and
      scale/offset aren't passed when the module is called.
    - Trainable scale/offset in which case create_* should be set to True and
      again scale/offset aren't passed when the module is called. In this case
      this module creates and owns the scale/offset variables.
    - Externally generated scale/offset, such as for conditional normalization,
      in which case create_* should be set to False and then the values fed in
      at call time.

  Attributes:
    scale: If `create_scale`, a trainable variable holding the current scale
      after the module is connected for the first time.
    offset: If `create_offset`, a trainable variable holding the current offset
      after the module is connected for the first time.
  """

  def __init__(self, axis, create_scale, create_offset, eps=1e-4,
               scale_init=None, offset_init=None, data_format="channels_last",
               name=None):
    """Constructs a AxisNorm module.

    Args:
      axis: An int, slice or sequence of ints representing the axes which should
        be normalized across.
      create_scale: Boolean representing whether to create a trainable scale per
        channel applied after the normalization.
      create_offset: Boolean representing whether to create a trainable offset
        per channel applied after normalization and scaling.
      eps: Small epsilon to avoid division by zero variance. Defaults to 1e-4.
      scale_init: Optional initializer for the scale variable. Can only be set
        if `create_scale` is True. By default scale is initialized to one.
      offset_init: Optional initializer for the offset variable. Can only be set
        if `create_offset` is True. By default offset is initialized to zero.
      data_format: The data format of the input. Can be either `channels_first`,
        `channels_last`, `N...C` or `NC...`. By default it is `channels_last`.
      name: Name of the module.
    """
    super(AxisNorm, self).__init__(name=name)

    if isinstance(axis, slice):
      self._axis = axis
    elif isinstance(axis, six.integer_types):
      self._axis = (axis,)
    elif (isinstance(axis, collections.Iterable) and
          all(isinstance(ax, six.integer_types) for ax in axis)):
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
      self._scale_init = (scale_init if scale_init is not None
                          else initializers.Ones())
    elif scale_init is not None:
      raise ValueError("Cannot set `scale_init` if `create_scale=False`.")
    if self._create_offset:
      self._offset_init = (offset_init if offset_init is not None
                           else initializers.Zeros())
    elif offset_init is not None:
      raise ValueError("Cannot set `offset_init` if `create_offset=False`.")

  def __call__(self, inputs, scale=None, offset=None):
    """Returns normalized inputs.

    Args:
      inputs: An n-D tensor of the data_format specified above on which the
        transformation is performed.
      scale: A tensor up to n-D. The shape of this tensor must be broadcastable
        to the shape of `inputs`. This is the scale applied to the normalized
        inputs. This cannot be passed in if the module was constructed with
        create_scale=True.
      offset: A tensor up to n-D. The shape of this tensor must be broadcastable
        to the shape of `inputs`. This is the offset applied to the normalized
        inputs. This cannot be passed in if the module was constructed with
        `create_offset=True`.

    Returns:
      An n-d tensor of the same shape as inputs that has been normalized.
    """
    self._create_parameters(inputs)
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
      raise ValueError("The rank of the inputs cannot change between calls, the"
                       " original call was rank={} but this call was rank={}."
                       .format(self._rank, len(inputs.shape)))

    mean, var = tf.nn.moments(inputs, self._axis, keepdims=True)

    normalized = tf.nn.batch_normalization(inputs,
                                           mean=mean,
                                           variance=var,
                                           scale=scale,
                                           offset=offset,
                                           variance_epsilon=self._eps)
    return normalized

  @once.once
  def _create_parameters(self, inputs):
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
      params_shape = [inputs.shape[1]] + [1]*(self._rank - 2)

    if self._create_scale:
      self.scale = tf.Variable(self._scale_init(params_shape, dtype),
                               name="scale")
    else:
      self.scale = None

    if self._create_offset:
      self.offset = tf.Variable(self._offset_init(params_shape, dtype),
                                name="offset")
    else:
      self.offset = None


class LayerNorm(AxisNorm):
  """Normalizes inputs along the spatial and channel dimensions.

  See `snt.AxisNorm` for more details.

  Attributes:
    scale: If `create_scale`, a trainable variable holding the current scale
      after the module is connected for the first time.
    offset: If `create_offset`, a trainable variable holding the current offset
      after the module is connected for the first time.
  """

  def __init__(self, create_scale, create_offset, eps=1e-4,
               scale_init=None, offset_init=None, data_format="channels_last",
               name=None):
    """Constructs an LayerNorm module.

    This method creates a module which normalizes over the spatial and channel
    dimensions.

    Args:
      create_scale: Boolean representing whether to create a trainable scale per
        channel applied after the normalization.
      create_offset: Boolean representing whether to create a trainable offset
        per channel applied after normalization and scaling.
      eps: Small epsilon to avoid division by zero variance. Defaults to 1e-4.
      scale_init: Optional initializer for the scale variable. Can only be set
        if `create_scale` is True. By default scale is initialized to one.
      offset_init: Optional initializer for the offset variable. Can only be set
        if `create_offset` is True. By default offset is initialized to zero.
      data_format: The data format of the input. Can be either `channels_first`,
        `channels_last`, `N...C` or `NC...`. By default it is `channels_last`.
      name: Name of the module.

    Returns:
      An AxisNorm module.
    """
    super(LayerNorm, self).__init__(
        axis=slice(1, None),
        create_scale=create_scale,
        create_offset=create_offset,
        eps=eps,
        scale_init=scale_init,
        offset_init=offset_init,
        data_format=data_format,
        name=name)


class InstanceNorm(AxisNorm):
  """Normalizes inputs along the channel dimension.

  See `snt.AxisNorm` for more details.

  Attributes:
    scale: If `create_scale`, a trainable variable holding the current scale
      after the module is connected for the first time.
    offset: If `create_offset`, a trainable variable holding the current offset
      after the module is connected for the first time.
  """

  def __init__(self, create_scale, create_offset, eps=1e-4,
               scale_init=None, offset_init=None, data_format="channels_last",
               name=None):
    """Constructs an InstanceNorm module.

    This method creates a module which normalizes over the channel dimension.

    Args:
      create_scale: Boolean representing whether to create a trainable scale per
        channel applied after the normalization.
      create_offset: Boolean representing whether to create a trainable offset
        per channel applied after normalization and scaling.
      eps: Small epsilon to avoid division by zero variance. Defaults to 1e-4.
      scale_init: Optional initializer for the scale variable. Can only be set
        if `create_scale` is True. By default scale is initialized to one.
      offset_init: Optional initializer for the offset variable. Can only be set
        if `create_offset` is True. By default offset is initialized to zero.
      data_format: The data format of the input. Can be either `channels_first`,
        `channels_last`, `N...C` or `NC...`. By default it is `channels_last`.
      name: Name of the module.

    Returns:
      An AxisNorm module.
    """
    channel_index = utils.get_channel_index(data_format)
    super(InstanceNorm, self).__init__(
        axis=channel_index,
        create_scale=create_scale,
        create_offset=create_offset,
        eps=eps,
        scale_init=scale_init,
        offset_init=offset_init,
        data_format=data_format,
        name=name)
