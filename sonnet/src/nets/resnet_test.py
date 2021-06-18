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
"""Tests for sonnet.v2.src.nets.resnet."""

from absl.testing import parameterized
from sonnet.src import test_utils
from sonnet.src.nets import resnet
import tensorflow as tf


class ResnetTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters(True, False)
  def test_simple(self, resnet_v2):
    image = tf.random.normal([2, 64, 64, 3])
    model = resnet.ResNet([1, 1, 1, 1], 10, resnet_v2=resnet_v2)

    logits = model(image, is_training=True)
    self.assertIsNotNone(logits)
    self.assertEqual(logits.shape, [2, 10])

  @parameterized.parameters(True, False)
  def test_tf_function(self, resnet_v2):
    image = tf.random.normal([2, 64, 64, 3])
    model = resnet.ResNet(
        [1, 1, 1, 1],
        10,
        resnet_v2=resnet_v2,
    )
    f = tf.function(model)

    logits = f(image, is_training=True)
    self.assertIsNotNone(logits)
    self.assertEqual(logits.shape, [2, 10])
    self.assertAllEqual(model(image, is_training=True).numpy(), logits.numpy())

  @parameterized.parameters(3, 5)
  def test_error_incorrect_args_block_list(self, list_length):
    block_list = [i for i in range(list_length)]
    with self.assertRaisesRegex(
        ValueError, "blocks_per_group_list` must be of length 4 not {}".format(
            list_length)):
      resnet.ResNet(block_list, 10, {"decay_rate": 0.9, "eps": 1e-5})

  @parameterized.parameters(3, 5)
  def test_error_incorrect_args_channel_list(self, list_length):
    channel_list = [i for i in range(list_length)]
    with self.assertRaisesRegex(
        ValueError,
        "channels_per_group_list` must be of length 4 not {}".format(
            list_length)):
      resnet.ResNet([1, 1, 1, 1], 10, {"decay_rate": 0.9, "eps": 1e-5},
                    channels_per_group_list=channel_list)

if __name__ == "__main__":
  tf.test.main()
