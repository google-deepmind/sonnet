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
"""Toy MLP on MNIST example of using TF2 JAX/HK shims."""

from absl import app
from absl import logging
import sonnet as snt
import tensorflow as tf
import tensorflow_datasets as tfds

fn = snt.functional


def main(unused_argv):
  del unused_argv

  with fn.variables():
    net = snt.nets.MLP([1000, 100, 10])

  def loss_fn(images, labels):
    images = snt.flatten(images)
    logits = net(images)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                       logits=logits))
    return loss

  loss_fn = fn.transform(loss_fn)

  def preprocess(images, labels):
    images = tf.image.convert_image_dtype(images, tf.float32)
    return images, labels

  #  _             _
  # | |_ _ __ __ _(_)_ __
  # | __| '__/ _` | | '_ \
  # | |_| | | (_| | | | | |
  #  \__|_|  \__,_|_|_| |_|
  #

  batch_size = 100

  dataset = tfds.load("mnist", split="train", as_supervised=True)
  dataset = dataset.map(preprocess)
  dataset = dataset.cache()
  dataset = dataset.shuffle(batch_size * 8)
  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.prefetch()

  # As before we want to unzip our loss_fn into init and apply.
  optimizer = fn.adam(0.01)

  # To get our initial state we need to pull a record from our dataset and pass
  # it to our init function. We'll also be sure to use `device_put` such that
  # the parameters are on the accelerator.
  images, labels = next(iter(dataset))
  params = fn.device_put(loss_fn.init(images, labels))
  opt_state = fn.device_put(optimizer.init(params))

  # Our training loop is to iterate through 10 epochs of the train dataset, and
  # use sgd after each minibatch to update our parameters according to the
  # gradient from our loss function.
  grad_apply_fn = fn.jit(fn.value_and_grad(loss_fn.apply))
  apply_opt_fn = fn.jit(optimizer.apply)

  for epoch in range(10):
    for images, labels in dataset:
      loss, grads = grad_apply_fn(params, images, labels)
      params, opt_state = apply_opt_fn(opt_state, grads, params)
    logging.info("[Epoch %s] loss=%s", epoch, loss.numpy())

  #  _            _
  # | |_ ___  ___| |_
  # | __/ _ \/ __| __|
  # | ||  __/\__ \ |_
  #  \__\___||___/\__|
  #

  def accuracy_fn(images, labels):
    images = snt.flatten(images)
    predictions = tf.argmax(net(images), axis=1)
    correct = tf.math.count_nonzero(tf.equal(predictions, labels))
    total = tf.shape(labels)[0]
    return correct, total

  accuracy_fn = fn.transform(accuracy_fn)

  batch_size = 10000
  dataset = tfds.load("mnist", split="test", as_supervised=True)
  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.map(preprocess)

  # Note that while we still unzip our accuracy function, we can ignore the
  # init_fn here since we already have all the state we need from our training
  # function.
  apply_fn = fn.jit(accuracy_fn.apply)

  # Compute top-1 accuracy.
  num_correct = num_total = 0
  for images, labels in dataset:
    correct, total = apply_fn(params, images, labels)
    num_correct += correct
    num_total += total
  accuracy = (int(num_correct) / int(num_total)) * 100
  logging.info("Accuracy %.5f%%", accuracy)

if __name__ == "__main__":
  app.run(main)
