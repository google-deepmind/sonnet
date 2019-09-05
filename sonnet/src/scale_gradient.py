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
"""Tensorflow op that scales gradient for backwards pass."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import tensorflow as tf


@tf.custom_gradient
def scale_gradient(t, scale):
  """Scales gradients for the backwards pass.

  Args:
    t: A Tensor.
    scale: The scale factor for the gradient on the backwards pass.

  Returns:
    A Tensor same as input, with scaled backward gradient.
  """
  def grad(dy):
    """Scaled gradient."""
    return scale*dy, None
  return t, grad

