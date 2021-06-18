# Copyright 2020 The Sonnet Authors. All Rights Reserved.
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
"""Simple functional APIs for TF2."""

from sonnet import optimizers as oo_optimizers
from sonnet.src.functional import haiku
from sonnet.src.functional import jax
from sonnet.src.functional import optimizers

# Utilities for converting Sonnet code into pure functions.
variables = haiku.variables
transform = haiku.transform
transform_with_state = haiku.transform_with_state
without_state = haiku.without_state

# Utilities for working with tensors on device.
device_get = jax.device_get
device_put = jax.device_put

# Utilities for transforming pure functions.
grad = jax.grad
jit = jax.jit
value_and_grad = jax.value_and_grad

# Optimizers.
optimizer = optimizers.optimizer
sgd = optimizer(oo_optimizers.SGD)
adam = optimizer(oo_optimizers.Adam)
rmsprop = optimizer(oo_optimizers.RMSProp)
momentum = optimizer(oo_optimizers.Momentum)

# Avoid accidentally exporting the private API.
del oo_optimizers, haiku, optimizers, jax

__all__ = (
    "variables",
    "transform",
    "transform_with_state",
    "without_state",
    "device_get",
    "device_put",
    "grad",
    "jit",
    "value_and_grad",
    "optimizer",
    "sgd",
    "adam",
    "rmsprop",
    "momentum",
)
