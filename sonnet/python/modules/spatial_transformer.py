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

""""Implementation of Spatial Transformer networks core components."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from itertools import chain

# Dependency imports
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
from sonnet.python.modules import base
from sonnet.python.modules import basic
import tensorflow as tf


class GridWarper(base.AbstractModule):
  """Grid warper interface class.

  An object implementing the `GridWarper` interface generates a reference grid
  of feature points at construction time, and warps it via a parametric
  transformation model, specified at run time by an input parameter Tensor.
  Grid warpers must then implement a `create_features` function used to generate
  the reference grid to be warped in the forward pass (according to a determined
  warping model).
  """

  def __init__(self, source_shape, output_shape, num_coeff, name, **kwargs):
    """Constructs a GridWarper module and initializes the source grid params.

    `source_shape` and `output_shape` are used to define the size of the source
    and output signal domains, as opposed to the shape of the respective
    Tensors. For example, for an image of size `width=W` and `height=H`,
    `{source,output}_shape=[H, W]`; for a volume of size `width=W`, `height=H`
    and `depth=D`, `{source,output}_shape=[H, W, D]`.

    Args:
      source_shape: Iterable of integers determining the size of the source
        signal domain.
      output_shape: Iterable of integers determining the size of the destination
        resampled signal domain.
      num_coeff: Number of coefficients parametrizing the grid warp.
        For example, a 2D affine transformation will be defined by the 6
        parameters populating the corresponding 2x3 affine matrix.
      name: Name of Module.
      **kwargs: Extra kwargs to be forwarded to the `create_features` function,
        instantiating the source grid parameters.

    Raises:
      Error: If `len(output_shape) > len(source_shape)`.
      TypeError: If `output_shape` and `source_shape` are not both iterable.
    """
    super(GridWarper, self).__init__(name=name)

    self._source_shape = tuple(source_shape)
    self._output_shape = tuple(output_shape)
    if len(self._output_shape) > len(self._source_shape):
      raise base.Error('Output domain dimensionality ({}) must be equal or '
                       'smaller than source domain dimensionality ({})'
                       .format(len(self._output_shape),
                               len(self._source_shape)))

    self._num_coeff = num_coeff
    self._psi = self._create_features(**kwargs)

  @abc.abstractmethod
  def _create_features(self, **kwargs):
    """Generates matrix of features, of size `[num_coeff, num_points]`."""
    pass

  @property
  def n_coeff(self):
    """Returns number of coefficients of warping function."""
    return self._n_coeff

  @property
  def psi(self):
    """Returns a list of features used to compute the grid warp."""
    return self._psi

  @property
  def source_shape(self):
    """Returns a tuple containing the shape of the source signal."""
    return self._source_shape

  @property
  def output_shape(self):
    """Returns a tuple containing the shape of the output grid."""
    return self._output_shape


def _create_affine_features(output_shape, source_shape):
  """Generates n-dimensional homogenous coordinates for a given grid definition.

  `source_shape` and `output_shape` are used to define the size of the source
  and output signal domains, as opposed to the shape of the respective
  Tensors. For example, for an image of size `width=W` and `height=H`,
  `{source,output}_shape=[H, W]`; for a volume of size `width=W`, `height=H`
  and `depth=D`, `{source,output}_shape=[H, W, D]`.

  Args:
    output_shape: Iterable of integers determining the shape of the grid to be
      warped.
   source_shape: Iterable of integers determining the domain of the signal to be
     resampled.

  Returns:
    List of flattened numpy arrays of coordinates in range `[-1, 1]^N`, for
    example:
      ```
      [[x_0_0, .... , x_0_{n-1}],
       ....
       [x_{M-1}_0, .... , x_{M-1}_{n-1}],
       [x_{M}_0=0, .... , x_{M}_{n-1}=0],
       ...
       [x_{N-1}_0=0, .... , x_{N-1}_{n-1}=0],
       [1, ..., 1]]
      ```
      where N is the dimensionality of the sampled space, M is the
      dimensionality of the output space, i.e. 2 for images
      and 3 for volumes, and n is the number of points in the output grid.
      When the dimensionality of `output_shape` is smaller that that of
      `source_shape` the last rows before [1, ..., 1] will be filled with 0.
  """
  ranges = [np.linspace(-1, 1, x, dtype=np.float32)
            for x in reversed(output_shape)]
  psi = [x.reshape(-1) for x in np.meshgrid(*ranges, indexing='xy')]
  dim_gap = len(source_shape) - len(output_shape)
  for _ in xrange(dim_gap):
    psi.append(np.zeros_like(psi[0], dtype=np.float32))
  psi.append(np.ones_like(psi[0], dtype=np.float32))
  return psi


class AffineGridWarper(GridWarper):
  """Affine Grid Warper class.

  The affine grid warper generates a reference grid of n-dimensional points
  and warps it via an affine transormation model determined by an input
  parameter Tensor. Some of the transformation parameters can be fixed at
  construction time via an `AffineWarpConstraints` object.
  """

  def __init__(self,
               source_shape,
               output_shape,
               constraints=None,
               name='affine_grid_warper'):
    """Constructs an AffineGridWarper.

    `source_shape` and `output_shape` are used to define the size of the source
    and output signal domains, as opposed to the shape of the respective
    Tensors. For example, for an image of size `width=W` and `height=H`,
    `{source,output}_shape=[H, W]`; for a volume of size `width=W`, `height=H`
    and `depth=D`, `{source,output}_shape=[H, W, D]`.

    Args:
      source_shape: Iterable of integers determining the size of the source
        signal domain.
      output_shape: Iterable of integers determining the size of the destination
        resampled signal domain.
      constraints: Either a double list of shape `[N, N+1]` defining constraints
        on the entries of a matrix defining an affine transformation in N
        dimensions, or an `AffineWarpConstraints` object. If the double list is
        passed, a numeric value bakes in a constraint on the corresponding
        entry in the tranformation matrix, whereas `None` implies that the
        corresponding entry will be specified at run time.
      name: Name of module.

    Raises:
      Error: If constraints fully define the affine transformation; or if
        input grid shape and contraints have different dimensionality.
      TypeError: If output_shape and source_shape are not both iterable.
    """
    self._source_shape = tuple(source_shape)
    self._output_shape = tuple(output_shape)
    num_dim = len(source_shape)
    if isinstance(constraints, AffineWarpConstraints):
      self._constraints = constraints
    elif constraints is None:
      self._constraints = AffineWarpConstraints.no_constraints(num_dim)
    else:
      self._constraints = AffineWarpConstraints(constraints=constraints)

    if self._constraints.num_free_params == 0:
      raise base.Error('Transformation is fully constrained.')

    if self._constraints.num_dim != num_dim:
      raise base.Error('Incompatible set of constraints provided: '
                       'input grid shape and constraints have different '
                       'dimensionality.')

    super(AffineGridWarper, self).__init__(source_shape=source_shape,
                                           output_shape=output_shape,
                                           num_coeff=6,
                                           name=name,
                                           constraints=self._constraints)

  def _create_features(self, constraints):
    """Creates all the matrices needed to compute the output warped grids."""
    affine_warp_constraints = constraints
    if not isinstance(affine_warp_constraints, AffineWarpConstraints):
      affine_warp_constraints = AffineWarpConstraints(affine_warp_constraints)
    mask = affine_warp_constraints.mask
    psi = _create_affine_features(output_shape=self._output_shape,
                                  source_shape=self._source_shape)
    scales = [(x - 1.0) * .5 for x in reversed(self._source_shape)]
    offsets = scales
    # Transforming a point x's i-th coordinate via an affine transformation
    # is performed via the following dot product:
    #
    #  x_i' = s_i * (T_i * x) + t_i                                          (1)
    #
    # where Ti is the i-th row of an affine matrix, and the scalars s_i and t_i
    # define a decentering and global scaling into the source space.
    # In the AffineGridWarper some of the entries of Ti are provided via the
    # input, some others are instead fixed, according to the constraints
    # assigned in the constructor.
    # In create_features the internal dot product (1) is accordingly broken down
    # into two parts:
    #
    # x_i' = Ti[uncon_i] * x[uncon_i, :] + offset(con_var)                   (2)
    #
    # i.e. the sum of the dot product of the free parameters (coming
    # from the input) indexed by uncond_i and an offset obtained by
    # precomputing the fixed part of (1) according to the constraints.
    # This step is implemented by analyzing row by row the constraints matrix
    # and saving into a list the x[uncon_i] and offset(con_var) data matrices
    # for each output dimension.
    features = []
    for row, scale in zip(mask, scales):
      x_i = np.array([x for x, is_active in zip(psi, row) if is_active])
      features.append(x_i * scale if len(x_i) else None)

    for row_i, row in enumerate(mask):
      x_i = None
      s = scales[row_i]
      for i, is_active in enumerate(row):
        if is_active:
          continue

        # In principle a whole row of the affine matrix can be fully
        # constrained. In that case the corresponding dot product between input
        # parameters and grid coordinates doesn't need to be implemented in the
        # computation graph since it can be precomputed.
        # When a whole row if constrained, x_i - which is initialized to
        # None - will still be None at the end do the loop when it is appended
        # to the features list; this value is then used to detect this setup
        # in the build function where the graph is assembled.
        if x_i is None:
          x_i = np.array(psi[i]) * affine_warp_constraints[row_i][i] * s
        else:
          x_i += np.array(psi[i]) * affine_warp_constraints[row_i][i] * s
      features.append(x_i)

    features += offsets
    return features

  def _build(self, inputs):
    """Assembles the module network and adds it to the graph.

    The internal computation graph is assembled according to the set of
    constraints provided at construction time.

    Args:
      inputs: Tensor containing a batch of transformation parameters.

    Returns:
      A batch of warped grids.

    Raises:
      Error: If the input tensor size is not consistent with the constraints
        passed at construction time.
    """
    input_shape = tf.shape(inputs)
    input_dtype = inputs.dtype.as_numpy_dtype
    batch_size = tf.expand_dims(input_shape[0], 0)
    number_of_params = inputs.get_shape()[1]
    if number_of_params != self._constraints.num_free_params:
      raise base.Error('Input size is not consistent with constraint '
                       'definition: {} parameters expected, {} provided.'
                       .format(self._constraints.num_free_params,
                               number_of_params))
    num_output_dimensions = len(self._psi) // 3
    def get_input_slice(start, size):
      """Extracts a subset of columns from the input 2D Tensor."""
      return basic.SliceByDim([1], [start], [size])(inputs)

    warped_grid = []
    var_index_offset = 0
    number_of_points = np.prod(self._output_shape)
    for i in xrange(num_output_dimensions):
      if self._psi[i] is not None:
        # The i-th output dimension is not fully specified by the constraints,
        # the graph is setup to perform matrix multiplication in batch mode.
        grid_coord = self._psi[i].astype(input_dtype)

        num_active_vars = self._psi[i].shape[0]
        active_vars = get_input_slice(var_index_offset, num_active_vars)
        warped_coord = tf.matmul(active_vars, grid_coord)
        warped_coord = tf.expand_dims(warped_coord, 1)
        var_index_offset += num_active_vars
        offset = self._psi[num_output_dimensions + i]
        if offset is not None:
          offset = offset.astype(input_dtype)
          # Some entries in the i-th row of the affine matrix were constrained
          # and the corresponding matrix multiplications have been precomputed.
          tiling_params = tf.concat(
              [
                  batch_size, tf.constant(
                      1, shape=(1,)), tf.ones_like(offset.shape)
              ],
              0)
          offset = offset.reshape((1, 1) + offset.shape)
          warped_coord += tf.tile(offset, tiling_params)

      else:
        # The i-th output dimension is fully specified by the constraints, and
        # the corresponding matrix multiplications have been precomputed.
        warped_coord = self._psi[num_output_dimensions + i].astype(input_dtype)
        tiling_params = tf.concat(
            [
                batch_size, tf.constant(
                    1, shape=(1,)), tf.ones_like(warped_coord.shape)
            ],
            0)
        warped_coord = warped_coord.reshape((1, 1) + warped_coord.shape)
        warped_coord = tf.tile(warped_coord, tiling_params)

      warped_coord += self._psi[i + 2 * num_output_dimensions]
      # Need to help TF figuring out shape inference since tiling information
      # is held in Tensors which are not known until run time.
      warped_coord.set_shape([None, 1, number_of_points])
      warped_grid.append(warped_coord)

    # Reshape all the warped coordinates tensors to match the specified output
    # shape and concatenate  into a single matrix.
    grid_shape = self._output_shape + (1,)
    warped_grid = [basic.BatchReshape(grid_shape)(grid) for grid in warped_grid]
    return tf.concat(warped_grid, len(grid_shape))

  @property
  def constraints(self):
    return self._constraints

  def inverse(self, name=None):
    """Returns a `sonnet` module to compute inverse affine transforms.

      The function first assembles a network that given the constraints of the
      current AffineGridWarper and a set of input parameters, retrieves the
      coefficients of the corresponding inverse affine transform, then feeds its
      output into a new AffineGridWarper setup to correctly warp the `output`
      space into the `source` space.

    Args:
      name: Name of module implementing the inverse grid transformation.

    Returns:
      A `sonnet` module performing the inverse affine transform of a reference
      grid of points via an AffineGridWarper module.

    Raises:
      tf.errors.UnimplementedError: If the function is called on a non 2D
        instance of AffineGridWarper.
    """
    if self._num_coeff != 6:
      raise tf.errors.UnimplementedError('AffineGridWarper currently supports'
                                         'inversion only for the 2D case.')
    def _affine_grid_warper_inverse(inputs):
      """Assembles network to compute inverse affine transformation.

      Each `inputs` row potentially contains [a, b, tx, c, d, ty]
      corresponding to an affine matrix:

        A = [a, b, tx],
            [c, d, ty]

      We want to generate a tensor containing the coefficients of the
      corresponding inverse affine transformation in a constraints-aware
      fashion.
      Calling M:

        M = [a, b]
            [c, d]

      the affine matrix for the inverse transform is:

         A_in = [M^(-1), M^-1 * [-tx, -tx]^T]

      where

        M^(-1) = (ad - bc)^(-1) * [ d, -b]
                                  [-c,  a]

      Args:
        inputs: Tensor containing a batch of transformation parameters.

      Returns:
        A tensorflow graph performing the inverse affine transformation
        parametrized by the input coefficients.
      """
      batch_size = tf.expand_dims(tf.shape(inputs)[0], 0)
      constant_shape = tf.concat([batch_size, tf.convert_to_tensor((1,))], 0)

      index = iter(range(6))
      def get_variable(constraint):
        if constraint is None:
          i = next(index)
          return inputs[:, i:i+1]
        else:
          return tf.fill(constant_shape, tf.constant(constraint,
                                                     dtype=inputs.dtype))

      constraints = chain.from_iterable(self.constraints)
      a, b, tx, c, d, ty = (get_variable(constr) for constr in constraints)

      det = a * d - b * c
      a_inv = d / det
      b_inv = -b / det
      c_inv = -c / det
      d_inv = a / det

      m_inv = basic.BatchReshape(
          [2, 2])(tf.concat([a_inv, b_inv, c_inv, d_inv], 1))

      txy = tf.expand_dims(tf.concat([tx, ty], 1), 2)

      txy_inv = basic.BatchFlatten()(tf.matmul(m_inv, txy))
      tx_inv = txy_inv[:, 0:1]
      ty_inv = txy_inv[:, 1:2]

      inverse_gw_inputs = tf.concat(
          [a_inv, b_inv, -tx_inv, c_inv, d_inv, -ty_inv], 1)

      agw = AffineGridWarper(self.output_shape,
                             self.source_shape)


      return agw(inverse_gw_inputs)  # pylint: disable=not-callable

    if name is None:
      name = self.module_name + '_inverse'
    return base.Module(_affine_grid_warper_inverse, name=name)


class AffineWarpConstraints(object):
  """Affine warp contraints class.

  `AffineWarpConstraints` allow for very succinct definitions of constraints on
  the values of entries in affine transform matrices.
  """

  def __init__(self, constraints=((None,) * 3,) * 2):
    """Creates a constraint definition for an affine transformation.

    Args:
      constraints: A doubly-nested iterable of shape `[N, N+1]` defining
        constraints on the entries of a matrix that represents an affine
        transformation in `N` dimensions. A numeric value bakes in a constraint
        on the corresponding entry in the tranformation matrix, whereas `None`
        implies that the corresponding entry will be specified at run time.

    Raises:
      TypeError: If `constraints` is not a nested iterable.
      ValueError: If the double iterable `constraints` has inconsistent
        dimensions.
    """
    try:
      self._constraints = tuple(tuple(x) for x in constraints)
    except TypeError:
      raise TypeError('constraints must be a nested iterable.')

    # Number of rows
    self._num_dim = len(self._constraints)
    expected_num_cols = self._num_dim + 1
    if any(len(x) != expected_num_cols for x in self._constraints):
      raise ValueError('The input list must define a Nx(N+1) matrix of '
                       'contraints.')

  def _calc_mask(self):
    """Computes a boolean mask from the user defined constraints."""
    mask = []
    for row in self._constraints:
      mask.append(tuple(x is None for x in row))
    return tuple(mask)

  def _calc_num_free_params(self):
    """Computes number of non constrained parameters."""
    return sum(row.count(None) for row in self._constraints)

  @property
  def num_free_params(self):
    return self._calc_num_free_params()

  @property
  def mask(self):
    return self._calc_mask()

  @property
  def constraints(self):
    return self._constraints

  @property
  def num_dim(self):
    return self._num_dim

  def __getitem__(self, i):
    """Returns the list of constraints for the i-th row of the affine matrix."""
    return self._constraints[i]

  def _combine(self, x, y):
    """Combines two constraints, raising an error if they are not compatible."""
    if x is None or y is None:
      return x or y
    if x != y:
      raise ValueError('Incompatible set of constraints provided.')
    return x

  def __and__(self, rhs):
    """Combines two sets of constraints into a coherent single set."""
    return self.combine_with(rhs)

  def combine_with(self, additional_constraints):
    """Combines two sets of constraints into a coherent single set."""
    x = additional_constraints
    if not isinstance(additional_constraints, AffineWarpConstraints):
      x = AffineWarpConstraints(additional_constraints)
    new_constraints = []
    for left, right in zip(self._constraints, x.constraints):
      new_constraints.append([self._combine(x, y) for x, y in zip(left, right)])
    return AffineWarpConstraints(new_constraints)

  # Collection of utlities to initialize an AffineGridWarper in 2D and 3D.
  @classmethod
  def no_constraints(cls, num_dim=2):
    """Empty set of constraints for a num_dim-ensional affine transform."""
    return cls(((None,) * (num_dim + 1),) * num_dim)

  @classmethod
  def translation_2d(cls, x=None, y=None):
    """Assign contraints on translation components of affine transform in 2d."""
    return cls([[None, None, x],
                [None, None, y]])

  @classmethod
  def translation_3d(cls, x=None, y=None, z=None):
    """Assign contraints on translation components of affine transform in 3d."""
    return cls([[None, None, None, x],
                [None, None, None, y],
                [None, None, None, z]])

  @classmethod
  def scale_2d(cls, x=None, y=None):
    """Assigns contraints on scaling components of affine transform in 2d."""
    return cls([[x, None, None],
                [None, y, None]])

  @classmethod
  def scale_3d(cls, x=None, y=None, z=None):
    """Assigns contraints on scaling components of affine transform in 3d."""
    return cls([[x, None, None, None],
                [None, y, None, None],
                [None, None, z, None]])

  @classmethod
  def shear_2d(cls, x=None, y=None):
    """Assigns contraints on shear components of affine transform in 2d."""
    return cls([[None, x, None],
                [y, None, None]])

  @classmethod
  def no_shear_2d(cls):
    return cls.shear_2d(x=0, y=0)

  @classmethod
  def no_shear_3d(cls):
    """Assigns contraints on shear components of affine transform in 3d."""
    return cls([[None, 0, 0, None],
                [0, None, 0, None],
                [0, 0, None, None]])
