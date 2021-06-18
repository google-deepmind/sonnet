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
"""Utilities for tests working with replicator."""

from typing import Callable, Sequence, Tuple
import unittest

from absl import logging
from sonnet.src.distribute import replicator as snt_replicator
import tensorflow as tf


def _replicator_primary_device() -> snt_replicator.Replicator:
  # NOTE: The explicit device list is required since currently Replicator
  # only considers CPU and GPU devices. This means on TPU by default we only
  # mirror on the local CPU.
  for device_type in ("TPU", "GPU", "CPU"):
    devices = tf.config.experimental.list_logical_devices(
        device_type=device_type)
    if devices:
      devices = [d.name for d in devices]
      logging.info("Replicating over %s", devices)
      return snt_replicator.Replicator(devices=devices)

  assert False, "No TPU/GPU or CPU found"


def _tpu_replicator_or_skip_test() -> snt_replicator.TpuReplicator:
  tpus = tf.config.experimental.list_logical_devices(device_type="TPU")
  if not tpus:
    raise unittest.SkipTest("No TPU available.")

  logging.info("Using TpuReplicator over %s", [t.name for t in tpus])
  return snt_replicator.TpuReplicator()


Strategy = tf.distribute.Strategy


def named_replicators() -> Sequence[Tuple[str, Callable[[], Strategy]]]:
  return (("TpuReplicator", _tpu_replicator_or_skip_test),
          ("Replicator", _replicator_primary_device))
