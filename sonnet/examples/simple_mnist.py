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

"""Trivial convnet learning MNIST."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
import sonnet as snt
import tensorflow as tf
import tensorflow_datasets as tfds


def mnist(split, batch_size):
  """Returns a tf.data.Dataset with MNIST image/label pairs."""
  @tf.function
  def map_fn(batch):
    # Mnist images are int8 [0, 255], we cast and rescale to float32 [-1, 1].
    images = tf.cast(batch["image"], tf.float32)
    images = ((images / 255.) - .5) * 2.
    labels = batch["label"]
    return images, labels

  dataset = tfds.load(name="mnist", split=split)
  dataset = dataset.map(map_fn)
  dataset = dataset.shuffle(buffer_size=4 * batch_size)
  dataset = dataset.batch(batch_size)
  # Autotune the number of prefetched records to avoid becoming input bound.
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset


@tf.function
def train_step(model, optimizer, images, labels):
  """Runs a single training step of the model on the given input."""
  with tf.GradientTape() as tape:
    logits = model(images)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                          logits=logits)
    loss = tf.reduce_mean(loss)
  variables = model.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))
  return loss


def train_epoch(model, optimizer):
  train_data = mnist("train", batch_size=128)
  for images, labels in train_data:
    loss = train_step(model, optimizer, images, labels)
  return loss


def test_accuracy(model):
  correct, total = 0, 0
  for images, labels in mnist("test", batch_size=1000):
    preds = tf.argmax(model(images), axis=1)
    correct += tf.math.count_nonzero(tf.equal(preds, labels))
    total += len(labels)
  accuracy = (correct / tf.cast(total, tf.int64)) * 100.
  return {"accuracy": accuracy, "incorrect": total - correct}


def main(unused_argv):
  del unused_argv

  model = snt.Sequential([
      snt.Conv2D(32, 3, 1), tf.nn.relu,
      snt.Conv2D(32, 3, 1), tf.nn.relu,
      snt.Flatten(),
      snt.Linear(10),
  ])

  optimizer = tf.optimizers.SGD(0.1)

  for epoch in range(5):
    train_loss = train_epoch(model, optimizer)
    test_metrics = test_accuracy(model)
    print("[Epoch %d] train loss: %.05f, test acc: %.02f%% (%d wrong)" %
          (epoch, train_loss, test_metrics["accuracy"],
           test_metrics["incorrect"]))


if __name__ == "__main__":
  # tf.enable_v2_behavior()
  app.run(main)
