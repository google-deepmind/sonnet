# Copyright 2017 The Sonnet Authors. All Rights Reserved.
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

"""Example script using `snt.Module` to make a module with build method args.

`snt.Sequential` has been deliberately designed for simple use cases. In
particular, it assumes that the only arguments passed when called are inputs to
the first layer. As such, one cannot easily control behaviour of submodules that
do accept different arguments in their call method, such as `snt.BatchNorm` and
the `is_training` flag. One may, however, quite easily replicate the same
functionality using `snt.Module` to construct a module from a custom method, as
shown in this script.

To run this script (on CPU), use the following command:
```
bazel run -c opt module_with_build_args
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import sonnet as snt
import tensorflow as tf


def custom_build(inputs, is_training, keep_prob):
  """A custom build method to wrap into a sonnet Module."""
  outputs = snt.Conv2D(output_channels=32, kernel_shape=4, stride=2)(inputs)
  outputs = snt.BatchNorm()(outputs, is_training=is_training)
  outputs = tf.nn.relu(outputs)
  outputs = snt.Conv2D(output_channels=64, kernel_shape=4, stride=2)(outputs)
  outputs = snt.BatchNorm()(outputs, is_training=is_training)
  outputs = tf.nn.relu(outputs)
  outputs = snt.BatchFlatten()(outputs)
  outputs = tf.nn.dropout(outputs, keep_prob=keep_prob)
  outputs = snt.Linear(output_size=10)(outputs)
  return outputs


def main(unused_argv):
  inputs = tf.random_uniform(shape=[10, 32, 32, 3])
  targets = tf.random_uniform(shape=[10, 10])

  # The line below takes custom_build and wraps it to construct a sonnet Module.
  module_with_build_args = snt.Module(custom_build, name='simple_net')

  train_model_outputs = module_with_build_args(inputs, is_training=True,
                                               keep_prob=tf.constant(0.5))
  test_model_outputs = module_with_build_args(inputs, is_training=False,
                                              keep_prob=tf.constant(1.0))
  loss = tf.nn.l2_loss(targets - train_model_outputs)
  # Ensure the moving averages for the BatchNorm modules are updated.
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_step = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(
        loss)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(100):
      sess.run(train_step)
    # Check that evaluating train_model_outputs twice returns the same value.
    train_outputs, train_outputs_2 = sess.run([train_model_outputs,
                                               train_model_outputs])
    assert (train_outputs == train_outputs_2).all()
    # Check that there is indeed a difference between train_model_outputs and
    # test_model_outputs.
    train_outputs, test_outputs = sess.run([train_model_outputs,
                                            test_model_outputs])
    assert (train_outputs != test_outputs).any()

if __name__ == '__main__':
  tf.app.run()
