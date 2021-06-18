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
"""Utility functions for the JAX API in TF2."""

import functools

from sonnet.src import utils
import tensorflow as tf
import tree


def get_first_accelerator():
  tpus = tf.config.experimental.list_logical_devices("TPU")
  if tpus:
    return tpus[0].name
  else:
    gpus = tf.config.experimental.list_logical_devices("GPU")
    return gpus[0].name if gpus else "/device:CPU:0"


def run_on_device(f, device):
  """Runs `f` under a tf.device context on the given device."""
  f = utils.smart_autograph(f)

  @tf.autograph.experimental.do_not_convert
  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    with tf.device(device):
      args = tree.map_structure(tf.identity, args)
      kwargs = tree.map_structure(tf.identity, kwargs)
      return f(*args, **kwargs)
  return wrapper


def get_name_scope():
  with tf.name_scope("x") as ns:
    return ns[:-2]


def first_non_none(*args):
  return next(a for a in args if a is not None)


def compose(f0, *fs):
  """Composes a sequence of functions.

  >>> f1 = lambda a, b: f"f1({a}, {b})"
  >>> f2 = lambda a: f"f2({a})"
  >>> f3 = lambda a: f"f3({a})"
  >>> f = compose(f1, f2, f3)
  >>> f("a", "b")
  'f3(f2(f1(a, b)))'

  Args:
    f0: The first function to apply.
    *fs: Other functions to apply in sequence.

  Returns:
    A function that is the composition of the input functions.
  """
  def wrapper(*args, **kwargs):
    return functools.reduce(lambda x, f: f(x), fs, f0(*args, **kwargs))
  return wrapper
