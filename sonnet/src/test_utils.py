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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import os
import sys
import threading

import tensorflow as tf

tpu_initialized = None
tpu_initialized_lock = threading.Lock()


class TestCase(tf.test.TestCase):
  """Test case which handles TPU hard placement."""

  def setUp(self):
    super(TestCase, self).setUp()

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

    self._device = tf.device("/device:%s:0" % self.primary_device)
    self._device.__enter__()

  def tearDown(self):
    super(TestCase, self).tearDown()
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


def find_sonnet_python_modules(root_module):
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
