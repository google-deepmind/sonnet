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
"""Test utilities for Sonnet 2."""

import functools
import inspect
import itertools
import os
import sys
import threading
import types
from typing import Sequence, Tuple, Type, TypeVar

from absl.testing import parameterized
import tensorflow as tf

Module = TypeVar("Module")

tpu_initialized = None
tpu_initialized_lock = threading.Lock()


class TestCase(tf.test.TestCase):
  """Test case which handles TPU hard placement."""

  ENTER_PRIMARY_DEVICE = True

  def setUp(self):
    super().setUp()

    # Enable autograph strict mode - any autograph errors will trigger an error
    # rather than falling back to no conversion.
    os.environ["AUTOGRAPH_STRICT_CONVERSION"] = "1"

    self._device_types = frozenset(
        d.device_type for d in tf.config.experimental.list_logical_devices())
    self._on_tpu = "TPU" in self._device_types

    # Initialize the TPU system once and only once.
    global tpu_initialized
    if tpu_initialized is None:
      with tpu_initialized_lock:
        if tpu_initialized is None and self._on_tpu:
          tf.tpu.experimental.initialize_tpu_system()
        tpu_initialized = True

    if self.ENTER_PRIMARY_DEVICE:
      self._device = tf.device("/device:%s:0" % self.primary_device)
      self._device.__enter__()

  def tearDown(self):
    super().tearDown()
    if self.ENTER_PRIMARY_DEVICE:
      self._device.__exit__(*sys.exc_info())
      del self._device

  @property
  def primary_device(self):
    if "TPU" in self._device_types:
      return "TPU"
    elif "GPU" in self._device_types:
      return "GPU"
    else:
      return "CPU"

  @property
  def device_types(self):
    return self._device_types

  def get_atol(self):
    """Returns a good tolerance for numerical closeness tests.

    Any TPU matmuls go via bfloat16, so an assertAllClose which passes under
    some constant small tolerance on CPU will generally fail on TPU. All test
    cases can call get_atol to get an appropriate number.

    TODO(mareynolds): assess these thresholds in detail.

    Returns:
      small float, eg 1e-4 on CPU/GPU, 5se-3 on TPU.
    """
    if self._on_tpu:
      return 5e-3
    else:
      return 1e-4


def find_all_sonnet_modules(
    root_python_module: types.ModuleType,
    base_class: Type[Module],
) -> Sequence[Type[Module]]:
  """Finds all subclasses of `base_class` under `root_python_module`."""
  modules = []
  for _, python_module in find_sonnet_python_modules(root_python_module):
    for name in dir(python_module):
      value = getattr(python_module, name)
      if inspect.isclass(value) and issubclass(value, base_class):
        modules.append(value)
  return modules


def find_sonnet_python_modules(
    root_module: types.ModuleType,) -> Sequence[Tuple[str, types.ModuleType]]:
  """Returns `(name, module)` for all Sonnet submodules under `root_module`."""
  modules = set([(root_module.__name__, root_module)])
  visited = set()
  to_visit = [root_module]

  while to_visit:
    mod = to_visit.pop()
    visited.add(mod)

    for name in dir(mod):
      obj = getattr(mod, name)
      if inspect.ismodule(obj) and obj not in visited:
        if obj.__name__.startswith("sonnet"):
          to_visit.append(obj)
          modules.add((obj.__name__, obj))

  return sorted(modules)


def combined_named_parameters(*parameters):
  """Combines multiple ``@parameterized.named_parameters`` compatible sequences.

  >>> foos = ("a_for_foo", "a"), ("b_for_foo", "b")
  >>> bars = ("c_for_bar", "c"), ("d_for_bar", "d")

  >>> @named_parameters(foos)
  ... def testFoo(self, foo):
  ...   assert foo in ("a", "b")

  >>> @combined_named_parameters(foos, bars):
  ... def testFooBar(self, foo, bar):
  ...   assert foo in ("a", "b")
  ...   assert bar in ("c", "d")

  Args:
    *parameters: A sequence of parameters that will be combined and be passed
      into ``parameterized.named_parameters``.

  Returns:
    A test generator to be handled by ``parameterized.TestGeneratorMetaclass``.
  """
  combine = lambda a, b: ("_".join((a[0], b[0])),) + a[1:] + b[1:]
  return parameterized.named_parameters(
      functools.reduce(combine, r) for r in itertools.product(*parameters))


def named_bools(name) -> Sequence[Tuple[str, bool]]:
  """Returns a pair of booleans suitable for use with ``named_parameters``."""
  return (name, True), ("not_{}".format(name), False)
