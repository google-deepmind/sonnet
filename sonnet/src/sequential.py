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
"""Sequential applies a linear sequence of layers."""

from typing import Any, Callable, Iterable, Optional

from sonnet.src import base


class Sequential(base.Module):
  """Sequential applies a linear chain of modules / callables.

      >>> mlp = snt.Sequential([
      ...     snt.Linear(1024),
      ...     tf.nn.relu,
      ...     snt.Linear(10),
      ... ])
      >>> mlp(tf.random.normal([8, 100]))
      <tf.Tensor: ...>

  Note that `Sequential` is limited in the range of possible architectures
  it can handle. This is a deliberate design decision; `Sequential` is only
  meant to be used for the simple case of fusing together modules/ops where
  the input of a particular module/op is the output of the previous one.

  Another restriction is that it is not possible to have extra arguments in the
  `__call__` method that are passed to the constituents of the module - for
  example, if there is a `BatchNorm` module in `Sequential` and the user wishes
  to switch the `is_training` flag. If this is the desired use case, the
  recommended solution is to subclass `snt.Module` and implement `__call__`:

      >>> class CustomModule(snt.Module):
      ...   def __init__(self, name=None):
      ...     super(CustomModule, self).__init__(name=name)
      ...     self.conv2d = snt.Conv2D(32, 4, 2)
      ...     self.bn = snt.BatchNorm()
      ...
      ...   def __call__(self, inputs, is_training):
      ...     outputs = self.conv2d(inputs)
      ...     outputs = self.bn(outputs, is_training=is_training)
      ...     outputs = tf.nn.relu(outputs)
      ...     return outputs
  """

  def __init__(self,
               layers: Optional[Iterable[Callable[..., Any]]] = None,
               name: Optional[str] = None):
    super().__init__(name=name)
    self._layers = list(layers) if layers is not None else []

  def __call__(self, inputs, *args, **kwargs):
    outputs = inputs
    for i, mod in enumerate(self._layers):
      if i == 0:
        # Pass additional arguments to the first layer.
        outputs = mod(outputs, *args, **kwargs)
      else:
        outputs = mod(outputs)
    return outputs
