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

from typing import Dict

from absl import app
import sonnet as snt
import tensorflow as tf
import tensorflow_datasets as tfds


def mnist(split: str, batch_size: int) -> tf.data.Dataset:
  """Returns a tf.data.Dataset with MNIST image/label pairs."""

  def preprocess_dataset(images, labels):
    # Mnist images are int8 [0, 255], we cast and rescale to float32 [-1, 1].
    images = ((tf.cast(images, tf.float32) / 255.) - .5) * 2.
    return images, labels

  dataset = tfds.load(
      name="mnist",
      split=split,
      shuffle_files=split == "train",
      as_supervised=True)
  dataset = dataset.map(preprocess_dataset)
  dataset = dataset.shuffle(buffer_size=4 * batch_size)
  dataset = dataset.batch(batch_size)
  # Cache the result of the data pipeline to avoid recomputation. The pipeline
  # is only ~100MB so this should not be a significant cost and will afford a
  # decent speedup.
  dataset = dataset.cache()
  # Prefetching batches onto the GPU will help avoid us being too input bound.
  # We allow tf.data to determine how much to prefetch since this will vary
  # between GPUs.
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset


def train_step(
    model: snt.Module,
    optimizer: snt.Optimizer,
    images: tf.Tensor,
    labels: tf.Tensor,
) -> tf.Tensor:
  """Runs a single training step of the model on the given input."""
  with tf.GradientTape() as tape:
    logits = model(images)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    loss = tf.reduce_mean(loss)
  variables = model.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply(gradients, variables)
  return loss


@tf.function
def train_epoch(
    model: snt.Module,
    optimizer: snt.Optimizer,
    dataset: tf.data.Dataset,
) -> tf.Tensor:
  loss = 0.
  for images, labels in dataset:
    loss = train_step(model, optimizer, images, labels)
  return loss


@tf.function
def test_accuracy(
    model: snt.Module,
    dataset: tf.data.Dataset,
) -> Dict[str, tf.Tensor]:
  """Computes accuracy on the test set."""
  correct, total = 0, 0
  for images, labels in dataset:
    preds = tf.argmax(model(images), axis=1)
    correct += tf.math.count_nonzero(tf.equal(preds, labels), dtype=tf.int32)
    total += tf.shape(labels)[0]
  accuracy = (correct / tf.cast(total, tf.int32)) * 100.
  return {"accuracy": accuracy, "incorrect": total - correct}


def main(unused_argv):
  del unused_argv

  model = snt.Sequential([
      snt.Conv2D(32, 3, 1),
      tf.nn.relu,
      snt.Conv2D(32, 3, 1),
      tf.nn.relu,
      snt.Flatten(),
      snt.Linear(10),
  ])

  optimizer = snt.optimizers.SGD(0.1)

  train_data = mnist("train", batch_size=128)
  test_data = mnist("test", batch_size=1000)

  for epoch in range(5):
    train_loss = train_epoch(model, optimizer, train_data)
    test_metrics = test_accuracy(model, test_data)
    print("[Epoch %d] train loss: %.05f, test acc: %.02f%% (%d wrong)" %
          (epoch, train_loss, test_metrics["accuracy"],
           test_metrics["incorrect"]))


if __name__ == "__main__":
  app.run(main)
