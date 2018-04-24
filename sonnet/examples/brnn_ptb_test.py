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
"""Tests for brnn_ptb."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import string

from sonnet.examples import brnn_ptb
import tensorflow as tf


FLAGS = tf.flags.FLAGS


def _make_random_word():
  return ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase)
                 for _ in range(random.randint(1, 15)))


def _make_random_vocab():
  # Make a limited vocab that all the sentences should be made out of, as the
  # BRNN model builds a finite vocab internally.
  return [_make_random_word() for _ in range(1000)]


def _make_sentence_with_vocab(vocab):
  return ' '.join(vocab[random.randint(0, len(vocab) - 1)]
                  for _ in range(random.randint(1, 30)))


def _make_fake_corpus_with_vocab(vocab, corpus_size):
  return '\n'.join(_make_sentence_with_vocab(vocab)
                   for _ in range(corpus_size))


class BrnnPtbTest(tf.test.TestCase):

  def testScriptRunsWithFakeData(self):
    # Make some small fake data in same format as real PTB.
    tmp_dir = tf.test.get_temp_dir()
    vocab = _make_random_vocab()
    with tf.gfile.GFile(os.path.join(tmp_dir, 'ptb.train.txt'), 'w') as f:
      f.write(_make_fake_corpus_with_vocab(vocab, 1000))
    with tf.gfile.GFile(os.path.join(tmp_dir, 'ptb.valid.txt'), 'w') as f:
      f.write(_make_fake_corpus_with_vocab(vocab, 100))
    with tf.gfile.GFile(os.path.join(tmp_dir, 'ptb.test.txt'), 'w') as f:
      f.write(_make_fake_corpus_with_vocab(vocab, 100))

    # Make model small, only run for 1 epoch.
    FLAGS.num_training_epochs = 1
    FLAGS.hidden_size = 50
    FLAGS.embedding_size = 50
    FLAGS.data_path = tmp_dir

    # Do training, test, evaluation.
    brnn_ptb.main(None)


if __name__ == '__main__':
  tf.test.main()
