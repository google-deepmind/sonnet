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

"""Classes to load textual data from Shakespeare's plays."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

# Dependency imports
import numpy as np
import sonnet as snt
import tensorflow as tf
from tensorflow.python.platform import gfile


FLAGS = tf.flags.FLAGS

SequenceDataOpsNoMask = collections.namedtuple("SequenceDataOpsNoMask",
                                               ("obs", "target"))


class TokenDataSource(object):
  """Encapsulates loading/tokenization logic for disk-based data."""

  DEFAULT_START_TOKENS = ["_unk_", "_null_", "_eos_", "|"]
  UNK, NULL, WORD_EOS, CHAR_EOS = DEFAULT_START_TOKENS

  def __init__(self, data_file, vocab_data_file):
    """Creates a TokenDataSource instance.

    Args:
      data_file: file object containing text data to be tokenized.
      vocab_data_file: file object containing text data used to initialize
        the vocabulary.
    """
    def reading_function(f):
      return list(f.read().decode().replace("\n", self.CHAR_EOS))

    self._vocab_dict = {}
    self._inv_vocab_dict = {}

    token_list = reading_function(vocab_data_file)
    self.vocab_size = 0
    for token in self.DEFAULT_START_TOKENS + token_list:
      if token not in self._vocab_dict:
        self._vocab_dict[token] = self.vocab_size
        self._inv_vocab_dict[self.vocab_size] = token
        self.vocab_size += 1

    raw_data = reading_function(data_file)
    self.flat_data = np.array(self.tokenize(raw_data), dtype=np.int32)
    self.num_tokens = self.flat_data.shape[0]

  def tokenize(self, token_list):
    """Produces the list of integer indices corresponding to a token list."""
    return [
        self._vocab_dict.get(token, self._vocab_dict[self.UNK])
        for token in token_list
    ]

  def decode(self, token_list):
    """Produces a human-readable representation of the token list."""
    return "".join([self._inv_vocab_dict[token] for token in token_list])


class TinyShakespeareDataset(snt.AbstractModule):
  """Tiny Shakespeare sequence data."""

  TRAIN = "train"
  VALID = "valid"
  TEST = "test"
  _RESOURCE_ROOT = os.path.join(
      os.path.dirname(os.path.realpath(__file__)), "data")

  def __init__(self, num_steps=1, batch_size=1,
               subset="train", random=False, dtype=tf.float32,
               name="tiny_shakespeare_dataset"):
    """Initializes a TinyShakespeare sequence data object.

    Args:
      num_steps: sequence_length.
      batch_size: batch size.
      subset: 'train', 'valid' or 'test'.
      random: boolean indicating whether to do random sampling of sequences.
        Default is false (sequential sampling).
      dtype: type of generated tensors (both observations and targets).
      name: object name.

    Raises:
      ValueError: if subset is not train, valid or test.
    """

    if subset not in [self.TRAIN, self.VALID, self.TEST]:
      raise ValueError("subset should be %s, %s, or %s. Received %s instead."
                       % (self.TRAIN, self.VALID, self.TEST, subset))

    super(TinyShakespeareDataset, self).__init__(name=name)

    # Generate vocab from train set.

    self._vocab_file = gfile.Open(
        os.path.join(self._RESOURCE_ROOT, "ts.train.txt"), mode="rb")
    self._data_file = gfile.Open(
        os.path.join(self._RESOURCE_ROOT, "ts.{}.txt".format(subset)),
        mode="rb")
    self._num_steps = num_steps
    self._batch_size = batch_size
    self._random_sampling = random
    self._dtype = dtype

    self._data_source = TokenDataSource(
        data_file=self._data_file,
        vocab_data_file=self._vocab_file)

    self._vocab_size = self._data_source.vocab_size
    self._flat_data = self._data_source.flat_data
    self._n_flat_elements = self._data_source.num_tokens

    self._num_batches = self._n_flat_elements // (self._num_steps * batch_size)
    self._reset_head_indices()

    self._queue_capacity = 10

  @property
  def vocab_size(self):
    return self._vocab_size

  def _reset_head_indices(self):
    self._head_indices = np.random.randint(
        low=0, high=self._n_flat_elements, size=[self._batch_size])

  def _one_hot(self, token):
    return tf.one_hot(token, self._vocab_size, axis=-1, dtype=self._dtype)

  def _get_batch(self):
    """Returns a batch of sequences.

    Returns:
      obs: np.int32 array of size [Time, Batch]
      target: np.int32 array of size [Time, Batch]
    """
    batch_indices = np.mod(
        np.array([
            np.arange(head_index, head_index + self._num_steps + 1) for
            head_index in self._head_indices]),
        self._n_flat_elements)

    obs = np.array([
        self._flat_data[indices[:self._num_steps]]
        for indices in batch_indices]).T
    target = np.array([
        self._flat_data[indices[1:self._num_steps + 1]]
        for indices in batch_indices]).T

    if self._random_sampling:
      self._reset_head_indices()
    else:
      self._head_indices = np.mod(
          self._head_indices + self._num_steps, self._n_flat_elements)
    return obs, target

  def _build(self):
    """Returns a tuple containing observation and target one-hot tensors."""
    q = tf.FIFOQueue(
        self._queue_capacity, [self._dtype, self._dtype],
        shapes=[[self._num_steps, self._batch_size, self._vocab_size]]*2)
    obs, target = tf.py_func(self._get_batch, [], [tf.int32, tf.int32])
    obs = self._one_hot(obs)
    target = self._one_hot(target)
    enqueue_op = q.enqueue([obs, target])
    obs, target = q.dequeue()
    tf.train.add_queue_runner(tf.train.QueueRunner(q, [enqueue_op]))
    return SequenceDataOpsNoMask(obs, target)

  def cost(self, logits, target):
    """Returns cost.

    Args:
      logits: model output.
      target: target.

    Returns:
      Cross-entropy loss for a sequence of logits. The loss will be averaged
      across time steps if time_average_cost was enabled at construction time.
    """
    logits = tf.reshape(logits, [self._num_steps * self._batch_size, -1])
    target = tf.reshape(target, [self._num_steps * self._batch_size, -1])
    xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=target)
    loss = tf.reduce_sum(xent)

    return loss / self._batch_size

  def to_human_readable(self,
                        data,
                        label_batch_entries=True,
                        indices=None,
                        sep="\n"):
    """Returns a human-readable version of a one-hot encoding of words.

    Args:
      data: A tuple with (obs, target). `obs` is a numpy array with one-hot
          encoding of words.
      label_batch_entries: bool. Whether to add numerical label before each
          batch element in the output string.
      indices: List of int or None. Used to select a subset of minibatch indices
          to print. None will print the whole minibatch.
      sep: A char separator which separates the output for each batch. Defaults
          to the newline character.

    Returns:
      String with the words from `data[0]`.
    """
    obs = data[0]
    batch_size = obs.shape[1]
    result = []
    indices = xrange(batch_size) if not indices else indices
    for b in indices:
      index_seq = np.argmax(obs[:, b], axis=1)
      prefix = "b_{}: ".format(b) if label_batch_entries else ""
      result.append(prefix + self._data_source.decode(index_seq))
    return sep.join(result)
