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

"""Group normalization implementation for Sonnet."""

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


class GroupNorm(base.Module):
  """Constructs a group norm module.

  This applies group normalization to the inputs. This involves splitting the
  channels into groups before calculating the mean and variance. The default
  behaviour is to compute the mean and variance over the spatial dimensions and
  the grouped channels. The mean and variance will never be computed over the
  created groups axis.

  It transforms the input x into:

      outputs = scale * (x - mu) / sigma + offset

  Where `mu` and `sigma` are respectively the mean and standard deviation of
  `x`.

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

  def __init__(self, groups, axis=slice(1, None), create_scale=True,
               create_offset=True, eps=1e-4, scale_init=None, offset_init=None,
               data_format="channels_last", name=None):
    """Constructs a `GroupNorm` module.

    Args:
      groups: An int, the number of groups to divide the channels by. The number
        of channels must be divisible by this.
      axis: An int, slice or sequence of ints representing the axes which should
        be normalized across. By default this is all but the first dimension.
        For time series data use `slice(2, None)` to average over the none Batch
        and Time data.
      create_scale: Boolean representing whether to create a trainable scale per
        channel applied after the normalization.
      create_offset: Boolean representing whether to create a trainable offset
        per channel applied after normalization and scaling.
      eps: Small epsilon to add to the variance to avoid division by zero.
        Defaults to 1e-4.
      scale_init: Optional initializer for the scale variable. Can only be set
        if with_scale is True. By default scale is initialized to one.
      offset_init: Optional initializer for the offset variable. Can only be set
        if with_offset is True. By default offset is initialized to zero.
      data_format: The data format of the input. Can be either `channels_first`,
        `channels_last`, `N...C` or `NC...`. By default it is `channels_last`.
      name: Name of the module.
    """
    super(GroupNorm, self).__init__(name=name)

    if isinstance(axis, slice):
      self._axis = axis
    elif isinstance(axis, six.integer_types):
      self._axis = [axis]
    elif (isinstance(axis, collections.Iterable) and
          all(isinstance(ax, six.integer_types) for ax in axis)):
      self._axis = axis
    else:
      raise ValueError("`axis` should be an int, slice or iterable of ints.")

    self._groups = groups
    self._eps = eps

    self._data_format = data_format
    self._channel_index = utils.get_channel_index(data_format)

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
      inputs: An `N - D` tensor of the data_format specified above on which the
        transformation is performed.
      scale: A tensor up to `N - D`. The shape of this tensor must be
        broadcastable to the shape of `inputs`. This is the scale applied to the
        normalised inputs. This cannot be passed in if the module was
        constructed with `create_scale=True`.
      offset: A tensor up to `N - D`. The shape of this tensor must be
        broadcastable to the shape of `inputs`. This is the offset applied to
        the normalised inputs. This cannot be passed in if the module was
        constructed with `create_offset=True`.

    Returns:
      An `N - D` tensor of the same shape as inputs that has been normalized.
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
      raise ValueError("The rank of the inputs cannot change between calls, the"
                       " original call was rank={} but this call was rank={}."
                       .format(self._rank, len(inputs.shape)))

    inputs = tf.reshape(inputs, self._inputs_reshape)
    mean, var = tf.nn.moments(inputs, self._axis, keepdims=True)

    normalized = tf.nn.batch_normalization(inputs,
                                           mean=mean,
                                           variance=var,
                                           scale=None,
                                           offset=None,
                                           variance_epsilon=self._eps)
    outputs = tf.reshape(normalized, self._outputs_reshape)
    outputs = outputs * scale if scale is not None else outputs
    outputs = outputs + offset if offset is not None else outputs
    return outputs

  @once.once
  def _initialize(self, inputs):
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

    num_channels = inputs.shape[self._channel_index]
    if num_channels % self._groups != 0:
      raise ValueError(
          "The number of channels must be divisible by the number of groups, "
          "was channels = {}, groups = {}".format(num_channels, self._groups))
    if self._channel_index == -1:
      self._inputs_reshape = [-1] + list(inputs.shape[1:-1]) +[
          self._groups, num_channels//self._groups]
      self._axis = [a if a != self._rank-1 else a + 1 for a in self._axis]
    else:
      self._inputs_reshape = [-1] + [
          self._groups, num_channels//self._groups] + list(inputs.shape[2:])
      self._axis = [a if a == 0 else a + 1 for a in self._axis]
    self._outputs_reshape = [-1] + list(inputs.shape[1:])
