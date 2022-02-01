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
"""Type aliases for Sonnet."""

from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

# Parameter update type, used by optimizers.
ParameterUpdate = Optional[Union[tf.Tensor, tf.IndexedSlices]]

# Objects that can be treated like tensors (in TF2).
TensorLike = Union[np.ndarray, tf.Tensor, tf.Variable]

# Note that we have no way of statically verifying the tensor's shape.
BoolLike = Union[bool, np.bool_, TensorLike]
IntegerLike = Union[int, np.integer, TensorLike]
FloatLike = Union[float, np.floating, TensorLike]

ShapeLike = Union[int, Sequence[int], tf.TensorShape]

# Note that this is effectively treated as `Any`; see b/109648354.
TensorNest = Union[TensorLike, Iterable['TensorNest'],
                   Mapping[str, 'TensorNest'],]  # pytype: disable=not-supported-yet

ActivationFn = Callable[[TensorLike], TensorLike]
Axis = Union[int, slice, Sequence[int]]
GradFn = Callable[[tf.Tensor], Tuple[tf.Tensor, Optional[tf.Tensor]]]
