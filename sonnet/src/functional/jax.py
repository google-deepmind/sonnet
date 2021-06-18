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
"""A subset of the JAX API in TF2."""

import functools

from sonnet.src.functional import utils
import tensorflow as tf
import tree


def device_put(t, device=None):
  return tree.map_structure(utils.run_on_device(lambda x: x, device), t)


def device_get(t):
  return tree.map_structure(lambda x: x.numpy(), t)


# TODO(tomhennigan) This should be cached.
def jit(f, device=None):
  if device is None:
    device = utils.get_first_accelerator()
  # TODO(tomhennigan) Enable XLA compilation (experimental_compile=True).
  return tf.function(utils.run_on_device(f, device))


def grad(f, argnums=0, has_aux=False):
  """Returns the gradient function for `f`."""
  value_and_grad_f = value_and_grad(f, argnums=argnums, has_aux=has_aux)
  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    if has_aux:
      (_, aux), g = value_and_grad_f(*args, **kwargs)
      return g, aux
    else:
      _, g = value_and_grad_f(*args, **kwargs)
      return g
  return wrapper


def value_and_grad(f, argnums=0, has_aux=False):
  """Returns the gradient function for `f`."""
  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    """Computes `f` and returns derivatives of the output wrt input(s)."""
    params = tree.map_structure(args.__getitem__, argnums)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tree.map_structure(tape.watch, params)
      out = f(*args, **kwargs)
    if has_aux:
      out, aux = out
    grads = tape.gradient(out, params)
    if has_aux:
      return (out, aux), grads
    else:
      return out, grads
  return wrapper
