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
"""Tests for Sonnet JAX interop layer."""

from absl.testing import parameterized
from sonnet.src import test_utils
from sonnet.src.functional import jax
import tensorflow as tf


class JaxTest(test_utils.TestCase, parameterized.TestCase):

  def test_jit_copies_to_device(self):
    accelerators = get_accelerators()
    if not accelerators:
      self.skipTest("No accelerator.")

    with tf.device("CPU"):
      x = tf.ones([])

    self.assertTrue(x.device.endswith("CPU:0"))

    for device in accelerators:
      y = jax.jit(lambda x: x, device=device)(x)
      self.assertTrue(y.device, device)

  def test_device_put(self):
    accelerators = get_accelerators()
    if not accelerators:
      self.skipTest("No accelerator.")

    with tf.device("CPU"):
      x = tf.ones([])

    for device in accelerators:
      y = jax.device_put(x, device=device)
      self.assertTrue(y.device.endswith(device))


class GradTest(test_utils.TestCase, parameterized.TestCase):

  def test_grad(self):
    f = lambda x: x ** 2
    g = jax.grad(f)
    x = tf.constant(4.)
    self.assertAllClose(g(x).numpy(), (2 * x).numpy())

  def test_argnums(self):
    f = lambda x, y: (x ** 2 + y ** 2)
    g = jax.grad(f, argnums=(0, 1))
    x = tf.constant(4.)
    y = tf.constant(5.)
    gx, gy = g(x, y)
    self.assertAllClose(gx.numpy(), (2 * x).numpy())
    self.assertAllClose(gy.numpy(), (2 * y).numpy(), rtol=1e-3)

  def test_has_aux(self):
    f = lambda x: (x ** 2, "aux")
    g = jax.grad(f, has_aux=True)
    x = tf.constant(2.)
    gx, aux = g(x)
    self.assertAllClose(gx.numpy(), (2 * x).numpy())
    self.assertEqual(aux, "aux")


def get_accelerators():
  gpus = tf.config.experimental.list_logical_devices("GPU")
  tpus = tf.config.experimental.list_logical_devices("TPU")
  return [tf.DeviceSpec.from_string(d.name).to_string() for d in gpus + tpus]

if __name__ == "__main__":
  tf.test.main()
