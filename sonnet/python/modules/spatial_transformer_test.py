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

"""Tests for sonnet.python.modules.spatial_transformer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
# Dependency imports
import numpy as np
import sonnet as snt
from sonnet.testing import parameterized
import tensorflow as tf


no_constraints = snt.AffineWarpConstraints.no_constraints
scale_2d = snt.AffineWarpConstraints.scale_2d
scale_3d = snt.AffineWarpConstraints.scale_3d
translation_2d = snt.AffineWarpConstraints.translation_2d
translation_3d = snt.AffineWarpConstraints.translation_3d
translation_2d = snt.AffineWarpConstraints.translation_2d
translation_3d = snt.AffineWarpConstraints.translation_3d
shear_2d = snt.AffineWarpConstraints.shear_2d
no_shear_2d = snt.AffineWarpConstraints.no_shear_2d
no_shear_3d = snt.AffineWarpConstraints.no_shear_3d


class AffineGridWarperTest(parameterized.ParameterizedTestCase,
                           tf.test.TestCase):

  def testShapeInferenceAndChecks(self):
    output_shape2d = (2, 3)
    source_shape2d = (6, 9)
    constraints = scale_2d(y=1) & translation_2d(x=-2, y=7)
    agw2d = snt.AffineGridWarper(source_shape=source_shape2d,
                                 output_shape=output_shape2d,
                                 constraints=constraints)

    input_params2d = tf.placeholder(tf.float32,
                                    [None, constraints.num_free_params])
    warped_grid2d = agw2d(input_params2d)
    self.assertEqual(warped_grid2d.get_shape().as_list()[1:], [2, 3, 2])

    output_shape2d = (2, 3)
    source_shape3d = (100, 200, 50)
    agw3d = snt.AffineGridWarper(source_shape=source_shape3d,
                                 output_shape=output_shape2d,
                                 constraints=[[None, 0, None, None],
                                              [0, 1, 0, None],
                                              [0, None, 0, None]])

    input_params3d = tf.placeholder(tf.float32,
                                    [None, agw3d.constraints.num_free_params])
    warped_grid3d = agw3d(input_params3d)
    self.assertEqual(warped_grid3d.get_shape().as_list()[1:], [2, 3, 3])

    output_shape3d = (2, 3, 4)
    source_shape3d = (100, 200, 50)
    agw3d = snt.AffineGridWarper(source_shape=source_shape3d,
                                 output_shape=output_shape3d,
                                 constraints=[[None, 0, None, None],
                                              [0, 1, 0, None],
                                              [0, None, 0, None]])

    input_params3d = tf.placeholder(tf.float32,
                                    [None, agw3d.constraints.num_free_params])
    warped_grid3d = agw3d(input_params3d)
    self.assertEqual(warped_grid3d.get_shape().as_list()[1:], [2, 3, 4, 3])

    with self.assertRaisesRegexp(snt.Error,
                                 "Incompatible set of constraints provided.*"):
      snt.AffineGridWarper(source_shape=source_shape3d,
                           output_shape=output_shape3d,
                           constraints=no_constraints(2))

    with self.assertRaisesRegexp(snt.Error,
                                 "Output domain dimensionality.*"):
      snt.AffineGridWarper(source_shape=source_shape2d,
                           output_shape=output_shape3d,
                           constraints=no_constraints(2))

  @parameterized.NamedParameters(
      ("2d_a", [13, 17], [7, 11], no_constraints(2)),
      ("2d_b", [11, 5], [2, 8], scale_2d(x=.7)),
      ("2d_c", [9, 23], [3, 11], scale_2d(y=1.2)),
      ("2d_d", [2, 23], [9, 13], snt.AffineWarpConstraints([[1]*3, [None]*3])),
      ("3d_a", [13, 17, 3], [7, 11, 3], no_constraints(3)),
      ("3d_b", [11, 5, 6], [2, 8, 9], scale_3d(x=.7, z=2)),
      ("3d_c", [9, 23, 8], [3, 11, 2], scale_3d(y=1.2)),
      ("3d_d", [2, 23, 2], [9, 13, 33],
       snt.AffineWarpConstraints([[1]*4, [None]*4, [None, 1, None, 1]])),
      ("2d_3d_a", [13, 17], [7, 11, 3], no_constraints(3)),
      ("2d_3d_b", [11, 5], [2, 8, 9], scale_3d(y=.7, z=2)),
      ("2d_3d_c", [9, 23], [3, 11, 2], scale_3d(x=1.2)),
      ("2d_3d_d", [2, 23], [9, 13, 33],
       snt.AffineWarpConstraints([[None] * 4, [1] * 4, [1, None, None, 1]])))
  def testSameAsNumPyReference(self, output_shape, source_shape, constraints):
    def chain(x):
      return itertools.chain(*x)

    def predict(output_shape, source_shape, inputs):
      ranges = [np.linspace(-1, 1, x, dtype=np.float32)
                for x in reversed(output_shape)]
      n = len(source_shape)
      grid = np.meshgrid(*ranges, indexing="xy")
      for _ in range(len(output_shape), len(source_shape)):
        grid.append(np.zeros_like(grid[0]))
      grid.append(np.ones_like(grid[0]))
      grid = np.array([x.reshape(1, -1) for x in grid]).squeeze()
      predicted_output = []
      for i in range(0, batch_size):
        x = np.dot(inputs[i, :].reshape(n, n+1), grid)
        for k, s in enumerate(reversed(source_shape)):
          s = (s - 1) * 0.5
          x[k, :] = x[k, :] * s + s
        x = np.concatenate([v.reshape(v.shape + (1,)) for v in x], -1)
        predicted_output.append(x.reshape(tuple(output_shape) + (n,)))
      return predicted_output

    batch_size = 20
    agw = snt.AffineGridWarper(source_shape=source_shape,
                               output_shape=output_shape,
                               constraints=constraints)
    inputs = tf.placeholder(tf.float32, [None, constraints.num_free_params])
    warped_grid = agw(inputs)
    full_size = constraints.num_dim * (constraints.num_dim + 1)
    full_input_np = np.random.rand(batch_size, full_size)

    con_i = [i for i, x in enumerate(chain(constraints.mask)) if not x]
    con_val = [x for x in chain(constraints.constraints) if x is not None]
    for i, v in zip(con_i, con_val):
      full_input_np[:, i] = v
    uncon_i = [i for i, x in enumerate(chain(constraints.mask)) if x]
    with self.test_session() as sess:
      output = sess.run(warped_grid,
                        feed_dict={inputs: full_input_np[:, uncon_i]})

    self.assertAllClose(output,
                        predict(output_shape, source_shape, full_input_np),
                        rtol=1e-05,
                        atol=1e-05)

  def testIdentity(self):
    constraints = snt.AffineWarpConstraints.no_constraints()
    warper = snt.AffineGridWarper([3, 3], [3, 3], constraints=constraints)
    p = tf.placeholder(tf.float64, (None, constraints.num_free_params))
    grid = warper(p)
    with self.test_session() as sess:
      warp_p = np.array([1, 0, 0,
                         0, 1, 0]).reshape([1, constraints.num_free_params])
      output = sess.run(grid, feed_dict={p: warp_p})

    # Check that output matches expected result for a known transformation.
    self.assertAllClose(output,
                        np.array([[[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                                   [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
                                   [[0.0, 2.0], [1.0, 2.0], [2.0, 2.0]]]]))

  @parameterized.NamedParameters(
      ("2d_a", [13, 17], [7, 11], no_constraints(2)),
      ("2d_b", [11, 5], [2, 8], scale_2d(x=.7)),
      ("2d_c", [9, 23], [3, 11], scale_2d(y=1.2)),
      ("2d_d", [2, 23], [9, 13], snt.AffineWarpConstraints([[1]*3, [None]*3])))
  def testInvSameAsNumPyRef(self, output_shape, source_shape, constraints):
    def chain(x):
      return itertools.chain(*x)

    def predict(output_shape, source_shape, inputs):
      ranges = [np.linspace(-1, 1, x, dtype=np.float32)
                for x in reversed(source_shape)]
      n = len(output_shape)
      grid = np.meshgrid(*ranges, indexing="xy")
      for _ in range(len(source_shape), len(output_shape)):
        grid.append(np.zeros_like(grid[0]))
      grid.append(np.ones_like(grid[0]))
      grid = np.array([x.reshape(1, -1) for x in grid]).squeeze()
      predicted_output = []
      for i in range(0, batch_size):
        affine_matrix = inputs[i, :].reshape(n, n+1)
        inv_matrix = np.linalg.inv(affine_matrix[:2, :2])
        inv_transform = np.concatenate(
            [inv_matrix, -np.dot(inv_matrix,
                                 affine_matrix[:, 2].reshape(2, 1))], 1)
        x = np.dot(inv_transform, grid)
        for k, s in enumerate(reversed(output_shape)):
          s = (s - 1) * 0.5
          x[k, :] = x[k, :] * s + s
        x = np.concatenate([v.reshape(v.shape + (1,)) for v in x], -1)
        predicted_output.append(x.reshape(tuple(source_shape) + (n,)))
      return predicted_output

    batch_size = 20
    agw = snt.AffineGridWarper(source_shape=source_shape,
                               output_shape=output_shape,
                               constraints=constraints).inverse()
    inputs = tf.placeholder(tf.float32, [None, constraints.num_free_params])
    warped_grid = agw(inputs)
    full_size = constraints.num_dim * (constraints.num_dim + 1)
    # Adding a bit of mass to the matrix to avoid singular matrices
    full_input_np = np.random.rand(batch_size, full_size) + 0.1

    con_i = [i for i, x in enumerate(chain(constraints.mask)) if not x]
    con_val = [x for x in chain(constraints.constraints) if x is not None]
    for i, v in zip(con_i, con_val):
      full_input_np[:, i] = v
    uncon_i = [i for i, x in enumerate(chain(constraints.mask)) if x]
    with self.test_session() as sess:
      output = sess.run(warped_grid,
                        feed_dict={inputs: full_input_np[:, uncon_i]})

    self.assertAllClose(output,
                        predict(output_shape, source_shape, full_input_np),
                        rtol=1e-05,
                        atol=1e-05)


class AffineWarpConstraintsTest(tf.test.TestCase):

  def assertConstraintsEqual(self, warp_constraints, expected):
    self.assertEqual(warp_constraints.constraints, expected)

  def testCreateMasks(self):

    self.assertConstraintsEqual(no_constraints(1), ((None,) * 2,) * 1)

    self.assertConstraintsEqual(no_constraints(2), ((None,) * 3,) * 2)

    self.assertConstraintsEqual(no_constraints(3), ((None,) * 4,) * 3)

    self.assertConstraintsEqual(translation_2d(x=11, y=12), ((None, None, 11),
                                                             (None, None, 12)))

    self.assertConstraintsEqual(translation_2d(x=11), ((None, None, 11),
                                                       (None, None, None)))

    self.assertConstraintsEqual(translation_2d(y=12), ((None, None, None),
                                                       (None, None, 12)))

    self.assertConstraintsEqual(translation_3d(x=11,
                                               y=12,
                                               z=13), ((None, None, None, 11),
                                                       (None, None, None, 12),
                                                       (None, None, None, 13)))

    self.assertConstraintsEqual(translation_3d(x=11),
                                ((None, None, None, 11),
                                 (None, None, None, None),
                                 (None, None, None, None)))

    self.assertConstraintsEqual(translation_3d(y=12),
                                ((None, None, None, None),
                                 (None, None, None, 12),
                                 (None, None, None, None)))

    self.assertConstraintsEqual(translation_3d(z=13),
                                ((None, None, None, None),
                                 (None, None, None, None),
                                 (None, None, None, 13)))

    self.assertConstraintsEqual(scale_2d(x=11, y=12), ((11, None, None),
                                                       (None, 12, None)))

    self.assertConstraintsEqual(scale_2d(x=11), ((11, None, None),
                                                 (None, None, None)))

    self.assertConstraintsEqual(scale_2d(y=12), ((None, None, None),
                                                 (None, 12, None)))

    self.assertConstraintsEqual(scale_3d(x=11,
                                         y=12,
                                         z=13), ((11, None, None, None),
                                                 (None, 12, None, None),
                                                 (None, None, 13, None)))

    self.assertConstraintsEqual(scale_3d(x=11), ((11, None, None, None),
                                                 (None, None, None, None),
                                                 (None, None, None, None)))

    self.assertConstraintsEqual(scale_3d(y=12), ((None, None, None, None),
                                                 (None, 12, None, None),
                                                 (None, None, None, None)))

    self.assertConstraintsEqual(scale_3d(z=13), ((None, None, None, None),
                                                 (None, None, None, None),
                                                 (None, None, 13, None)))

    self.assertConstraintsEqual(shear_2d(x=11,
                                         y=12), ((None, 11, None),
                                                 (12, None, None)))

    self.assertConstraintsEqual(shear_2d(x=11), ((None, 11, None),
                                                 (None, None, None)))

    self.assertConstraintsEqual(shear_2d(y=12), ((None, None, None),
                                                 (12, None, None)))

    self.assertConstraintsEqual(no_shear_2d(), ((None, 0, None),
                                                (0, None, None)))

    self.assertConstraintsEqual(no_shear_3d(), ((None, 0, 0, None),
                                                (0, None, 0, None),
                                                (0, 0, None, None)))

  def testConstraintsOperations(self):

    self.assertEqual(no_constraints(2).num_free_params, 6)
    self.assertEqual(scale_2d(2, 4).num_free_params, 4)
    self.assertConstraintsEqual(scale_2d(2, 4) & translation_2d(x=2),
                                ((2, None, 2),
                                 (None, 4, None)))
    self.assertEqual(scale_2d(2, 4).mask, ((False, True, True),
                                           (True, False, True)))

    with self.assertRaisesRegexp(ValueError,
                                 "Incompatible set of constraints provided."):
      _ = scale_2d(2) & scale_2d(3)


if __name__ == "__main__":
  tf.test.main()
