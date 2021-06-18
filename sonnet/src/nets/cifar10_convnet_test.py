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
"""Tests for sonnet.v2.src.nets.cifar10_convnet."""

from absl.testing import parameterized
import numpy as np
from sonnet.src import test_utils
from sonnet.src.nets import cifar10_convnet
import tensorflow as tf


class ModelTest(parameterized.TestCase, test_utils.TestCase):

  def testModelCreation(self):
    convnet = cifar10_convnet.Cifar10ConvNet()

    self.assertLen(convnet.submodules, 45)

  def testFailedModelCreation(self):
    with self.assertRaisesRegex(
        ValueError,
        'The length of `output_channels` and `strides` must be equal.'):
      cifar10_convnet.Cifar10ConvNet(strides=(1, 2, 3), output_channels=(1,))

  @parameterized.parameters({'batch_size': 1}, {'batch_size': 4},
                            {'batch_size': 128})
  def testModelForwards(self, batch_size):
    image_batch = tf.constant(
        np.random.randn(batch_size, 24, 24, 3), dtype=tf.float32)

    convnet = cifar10_convnet.Cifar10ConvNet()
    output = convnet(image_batch, is_training=True)
    self.assertLen(convnet.variables, 112)
    self.assertEqual(output['logits'].shape, [batch_size, 10])
    # One intermediate activation per conv layer, plus one after the global
    # mean pooling, before the linear.
    self.assertLen(output['activations'], 12)

  @parameterized.parameters({'batch_size': 1}, {'batch_size': 4},
                            {'batch_size': 128})
  def testModelForwardsFunction(self, batch_size):
    image_batch = tf.constant(
        np.random.randn(batch_size, 24, 24, 3), dtype=tf.float32)

    convnet = cifar10_convnet.Cifar10ConvNet()
    convnet_function = tf.function(convnet)
    output = convnet_function(image_batch, is_training=True)
    self.assertLen(convnet.variables, 112)
    self.assertEqual(output['logits'].shape, [batch_size, 10])
    # One intermediate activation per conv layer, plus one after the global
    # mean pooling, before the linear.
    self.assertLen(output['activations'], 12)

  def testDifferentSizedImages(self):
    # Due to global average pooling, different sized images should work fine
    # as long they are above some minimum size.
    convnet = cifar10_convnet.Cifar10ConvNet()

    small_image = tf.constant(np.random.randn(4, 32, 32, 3), dtype=tf.float32)
    small_output = convnet(small_image, is_training=True)
    self.assertEqual(small_output['logits'].shape, [4, 10])

    # Change height, width and batch size
    big_image = tf.constant(np.random.randn(12, 64, 64, 3), dtype=tf.float32)
    big_output = convnet(big_image, is_training=True)
    self.assertEqual(big_output['logits'].shape, [12, 10])

  def testDefunBackProp(self):

    convnet = cifar10_convnet.Cifar10ConvNet()

    @tf.function
    def do_training_step(image, labels):
      with tf.GradientTape() as tape:
        logits = convnet(image, is_training=True)['logits']
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels))
      grads = tape.gradient(loss, convnet.trainable_variables)
      return loss, grads

    image = tf.random.normal([4, 32, 32, 3])
    labels = np.random.randint(low=0, high=10, size=[4], dtype=np.int64)
    loss, grads = do_training_step(image, labels)
    self.assertEqual(loss.numpy().shape, ())
    for grad, var in zip(grads, convnet.trainable_variables):
      self.assertIsNotNone(grad)
      self.assertEqual(grad.numpy().shape, var.shape)


if __name__ == '__main__':
  tf.test.main()
