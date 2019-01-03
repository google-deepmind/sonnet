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

"""Example script to train a multi-layer perceptron (MLP) on MNIST."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import sonnet as snt
from sonnet.examples import dataset_mnist_cifar10 as dataset_mnist
import tensorflow as tf


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_float("learning_rate", 0.1, "Learning rate")
tf.flags.DEFINE_integer("num_hidden", 100, "Number of hidden units in MLP.")
tf.flags.DEFINE_integer("num_train_steps", 1001,
                        "How many training steps to take.")
tf.flags.DEFINE_integer("report_every", 10,
                        "Interval, in mini-batches, to report progress.")
tf.flags.DEFINE_integer("test_batch_size", 10000, "Batch size for test.")
tf.flags.DEFINE_integer("test_every", 200,
                        "Interval, in train mini-batches, to run test pass.")
tf.flags.DEFINE_integer("train_batch_size", 200, "Batch size for training.")


def train_and_eval(train_batch_size, test_batch_size, num_hidden, learning_rate,
                   num_train_steps, report_every, test_every):
  """Creates a basic MNIST model using Sonnet, then trains and evaluates it."""

  data_dict = dataset_mnist.get_data("mnist", train_batch_size, test_batch_size)
  train_data = data_dict["train_iterator"]
  test_data = data_dict["test_iterator"]

  # Sonnet separates the configuration of a model from its attachment into the
  # graph. Here we configure the shape of the model, but this call does not
  # place any ops into the graph.
  mlp = snt.nets.MLP([num_hidden, data_dict["num_classes"]])

  train_images, train_labels = train_data.get_next()
  test_images, test_labels = test_data.get_next()

  # Flatten images to pass to model.
  train_images = snt.BatchFlatten()(train_images)
  test_images = snt.BatchFlatten()(test_images)

  # Call our model, which creates it in the graph. Our build function
  # is parameterized by the source of images, and here we connect the model to
  # the training images.
  train_logits = mlp(train_images)

  # Training loss and optimizer.
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=train_labels, logits=train_logits)
  loss_avg = tf.reduce_mean(loss)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  optimizer_step = optimizer.minimize(loss_avg)

  # As before, we make a second instance of our model in the graph, which shares
  # its parameters with the first instance of the model. The Sonnet Module code
  # takes care of the variable sharing for us: because we are calling the same
  # instance of Model, we will automatically reference the same, shared
  # variables.
  test_logits = mlp(test_images)
  test_classes = tf.nn.softmax(test_logits)
  test_correct = tf.nn.in_top_k(test_classes, test_labels, k=1)

  with tf.train.SingularMonitoredSession() as sess:

    for step_idx in range(num_train_steps):
      current_loss, _ = sess.run([loss_avg, optimizer_step])
      if step_idx % report_every == 0:
        tf.logging.info("Step: %4d of %d - loss: %.02f.",
                        step_idx + 1, num_train_steps, current_loss)
      if step_idx % test_every == 0:
        sess.run(test_data.initializer)
        current_correct = sess.run(test_correct)
        correct_count = np.count_nonzero(current_correct)
        tf.logging.info("Test: %d of %d correct.",
                        correct_count, test_batch_size)


def main(unused_argv):
  train_and_eval(FLAGS.train_batch_size, FLAGS.test_batch_size,
                 FLAGS.num_hidden, FLAGS.learning_rate, FLAGS.num_train_steps,
                 FLAGS.report_every, FLAGS.test_every)


if __name__ == "__main__":
  tf.app.run()
