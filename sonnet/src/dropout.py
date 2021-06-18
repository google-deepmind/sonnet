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

"""Sonnet dropout modules."""

from typing import Optional

from sonnet.src import base
from sonnet.src import types
from sonnet.src import utils
import tensorflow as tf


class Dropout(base.Module):
  """Randomly drop units in the input at a given rate.

  See: http://www.cs.toronto.edu/~hinton/absps/dropout.pdf

  Dropout was originally described by Hinton et al. TensorFlow deviates slightly
  from this paper by scaling activations at training time rather than test time.
  """

  def __init__(self,
               rate: types.FloatLike,
               noise_shape: Optional[types.ShapeLike] = None,
               seed: Optional[int] = None,
               name: Optional[str] = None):
    """Constructs a Dropout module.

    Args:
      rate: Probability that each element of x is discarded. Must be a scalar in
        the range `[0, 1)`.
      noise_shape: (Optional) Shape vector controlling the shape of the random
        noise used to apply dropout. If not set this will be the shape of the
        input. If set it should be broadcastable to the input shape.
      seed: (Optional) Random seed to be passed to TensorFlow ops when
        generating dropout tensor.
      name: (Optional) Name for this module.
    """
    super().__init__(name=name)
    self._rate = rate
    self._noise_shape = noise_shape
    self._seed = seed

  @utils.smart_autograph
  def __call__(self, x: tf.Tensor, is_training: types.BoolLike) -> tf.Tensor:
    if not is_training:
      return x

    # NOTE: Even if `self._seed` is a constant value (e.g. `2`) this will
    # produce a different random dropout each call (the per-op seed is used
    # in conjunction with the global seed and some persistent state to produce
    # random values).
    # c.f. https://www.tensorflow.org/api_docs/python/tf/random/set_random_seed
    return tf.nn.dropout(
        x, rate=self._rate, noise_shape=self._noise_shape, seed=self._seed)
