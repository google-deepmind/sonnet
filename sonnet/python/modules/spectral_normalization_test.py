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
# ===================================================

"""Testing the spectral_normalization module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

# Dependency imports

import numpy as np
import sonnet as snt
from sonnet.python.modules import base
from sonnet.python.modules import basic
from sonnet.python.modules import spectral_normalization
import tensorflow.compat.v1 as tf

_ACCEPTABLE_ERROR = 1e-3


class MinimalClass(base.AbstractModule):

  def _build(self, input_):
    sn_linear = spectral_normalization.wrap_with_spectral_norm(
        basic.Linear, {'eps': 1e-4})
    linear1 = sn_linear(16)
    linear2 = sn_linear(16)
    return linear1(input_), linear2(input_)


class SpectralNormalizationTest(tf.test.TestCase):

  def test_raw_spectral_norm(self):
    with tf.Graph().as_default():
      ones_weight = 4 * tf.eye(8, 8)
      sigma = spectral_normalization.spectral_norm(ones_weight)['sigma']
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _, sigma_v = sess.run([ones_weight, sigma])
        self.assertLess(abs(4.0 - sigma_v), _ACCEPTABLE_ERROR)

  def test_raw_spectral_norm_bfloat16(self):
    with tf.Graph().as_default():
      ones_weight = 4 * tf.eye(8, 8, dtype=tf.bfloat16)
      sigma = spectral_normalization.spectral_norm(ones_weight)['sigma']
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _, sigma_v = sess.run([ones_weight, sigma])
        self.assertEqual(tf.bfloat16, sigma_v.dtype)
        self.assertLess(abs(4.0 - float(sigma_v)), _ACCEPTABLE_ERROR)

  def test_spectral_norm_creates_variables(self):
    with tf.Graph().as_default():
      ones_weight = 4 * tf.eye(8, 8)
      pre_spec_norm_vars = tf.global_variables()
      _ = spectral_normalization.spectral_norm(ones_weight)
      post_spec_norm_vars = tf.global_variables()
      self.assertEmpty(pre_spec_norm_vars)
      self.assertLen(post_spec_norm_vars, 1)
      self.assertEqual(post_spec_norm_vars[0].name.split('/')[-1], 'u0:0')

  def test_wrapper_creates_variables(self):
    with tf.Graph().as_default():
      SNLinear = functools.partial(  # pylint: disable=invalid-name
          spectral_normalization.SpectralNormWrapper, snt.Linear, {}, None)
      input_ = tf.zeros((8, 8), dtype=tf.float32)
      linear_layer_with_sn = SNLinear(16)
      _ = linear_layer_with_sn(input_)
      vars_ = tf.global_variables()
      self.assertLen(vars_, 3)

  def test_update_sn(self):
    with tf.Graph().as_default():
      SNLinear = functools.partial(  # pylint: disable=invalid-name
          spectral_normalization.SpectralNormWrapper, snt.Linear, {},
          'POWER_ITERATION_OPS')
      input_ = tf.zeros((8, 8), dtype=tf.float32)
      linear_layer_with_sn = SNLinear(16)
      output_update = linear_layer_with_sn(input_)
      output_no_update = linear_layer_with_sn(
          input_, enable_power_iteration=False)
      run_update_ops = tf.get_collection('POWER_ITERATION_OPS')
      singular_val_w = [v for v in tf.global_variables() if 'u0' in v.name][0]
      w_ph = tf.placeholder(singular_val_w.dtype, singular_val_w.shape)
      reset_sing_val = tf.assign(singular_val_w, w_ph)
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        original_sing_val_v = sess.run(singular_val_w)
        sing_val_v_implicit = sess.run(output_update)
        sess.run(reset_sing_val, {w_ph: original_sing_val_v})
        sing_val_v_explicit, _ = sess.run([output_no_update, run_update_ops])
        self.assertTrue(
            np.equal(sing_val_v_implicit, sing_val_v_explicit).all())
        self.assertFalse(
            np.equal(original_sing_val_v, sing_val_v_explicit).all())

  def test_update_sn_compatible_with_bfloat16(self):
    with tf.Graph().as_default():
      SNLinear = functools.partial(  # pylint: disable=invalid-name
          spectral_normalization.SpectralNormWrapper, snt.Linear, {},
          'POWER_ITERATION_OPS')
      input_ = tf.zeros((8, 8), dtype=tf.float32)
      linear_layer_with_sn = SNLinear(16)
      output_update = linear_layer_with_sn(input_)
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(output_update)

  def test_conflicting_names_no_scope(self):
    with tf.Graph().as_default():
      sn_linear = spectral_normalization.wrap_with_spectral_norm(
          basic.Linear, {'eps': 1e-4})
      linear1 = sn_linear(16)
      linear2 = sn_linear(16)
      input_ = tf.zeros((48, 12))  # Random [batch, dim] shape.
      linear1(input_)
      linear2(input_)

  def test_conflicting_names_with_enclosing_scope(self):
    with tf.Graph().as_default():
      input_ = tf.zeros((48, 12))  # Random [batch, dim] shape.
      MinimalClass()(input_)

if __name__ == '__main__':
  tf.test.main()
