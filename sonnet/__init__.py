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

"""This python module contains Neural Network Modules for TensorFlow.

Each module is a Python object which conceptually "owns" any
variables required in that part of the Neural Network. The `__call__` function
on the object is used to connect that Module into the Graph, and this may be
called repeatedly with sharing automatically taking place.

Everything public should be imported by this top level `__init__.py` so that the
library can be used as follows:

```
import sonnet as snt

linear = snt.Linear(...)
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import sys

import semantic_version


def _ensure_dependency_available_at_version(package_name, min_version):
  """Throw helpful error if required dependencies not available."""

  try:
    pkg = importlib.import_module(package_name)
  except ImportError:
    pip_name = package_name.replace('_', '-')
    raise SystemError(
        'Sonnet requires %s (minimum version %s) to be installed. '
        'If using pip, run `pip install %s` or '
        '`pip install %s-gpu`' % (
            package_name, min_version, pip_name, pip_name))

  installed_version = semantic_version.Version(pkg.__version__)
  version_spec = semantic_version.Spec('>=' + min_version)
  if not version_spec.match(installed_version):
    raise SystemError(
        '%s version %s is installed, but Sonnet requires at least version %s.' %
        (package_name, pkg.__version__, min_version))

_ensure_dependency_available_at_version('tensorflow', '1.8.0')
_ensure_dependency_available_at_version('tensorflow_probability', '0.4.0')

# Check some version of TF is available.
from sonnet.python import custom_getters
from sonnet.python.modules import nets
from sonnet.python.modules.attention import AttentiveRead
from sonnet.python.modules.base import AbstractModule
from sonnet.python.modules.base import Module
from sonnet.python.modules.base import observe_connections
from sonnet.python.modules.base import Transposable
from sonnet.python.modules.base_errors import DifferentGraphError
from sonnet.python.modules.base_errors import Error
from sonnet.python.modules.base_errors import IncompatibleShapeError
from sonnet.python.modules.base_errors import ModuleInfoError
from sonnet.python.modules.base_errors import NotConnectedError
from sonnet.python.modules.base_errors import NotInitializedError
from sonnet.python.modules.base_errors import NotSupportedError
from sonnet.python.modules.base_errors import ParentNotBuiltError
from sonnet.python.modules.base_errors import UnderspecifiedError
from sonnet.python.modules.base_info import SONNET_COLLECTION_NAME
from sonnet.python.modules.basic import AddBias
from sonnet.python.modules.basic import BatchApply
from sonnet.python.modules.basic import BatchFlatten
from sonnet.python.modules.basic import BatchReshape
from sonnet.python.modules.basic import ConcatLinear
from sonnet.python.modules.basic import FlattenTrailingDimensions
from sonnet.python.modules.basic import Linear
from sonnet.python.modules.basic import merge_leading_dims
from sonnet.python.modules.basic import MergeDims
from sonnet.python.modules.basic import SelectInput
from sonnet.python.modules.basic import SliceByDim
from sonnet.python.modules.basic import split_leading_dim
from sonnet.python.modules.basic import TileByDim
from sonnet.python.modules.basic import TrainableVariable
from sonnet.python.modules.basic_rnn import BidirectionalRNN
from sonnet.python.modules.basic_rnn import DeepRNN
from sonnet.python.modules.basic_rnn import ModelRNN
from sonnet.python.modules.basic_rnn import VanillaRNN
from sonnet.python.modules.batch_norm import BatchNorm
from sonnet.python.modules.batch_norm_v2 import BatchNormV2
from sonnet.python.modules.clip_gradient import clip_gradient
from sonnet.python.modules.conv import CAUSAL
from sonnet.python.modules.conv import CausalConv1D
from sonnet.python.modules.conv import Conv1D
from sonnet.python.modules.conv import Conv1DTranspose
from sonnet.python.modules.conv import Conv2D
from sonnet.python.modules.conv import Conv2DTranspose
from sonnet.python.modules.conv import Conv3D
from sonnet.python.modules.conv import Conv3DTranspose
from sonnet.python.modules.conv import DepthwiseConv2D
from sonnet.python.modules.conv import FULL
from sonnet.python.modules.conv import InPlaneConv2D
from sonnet.python.modules.conv import REVERSE_CAUSAL
from sonnet.python.modules.conv import SAME
from sonnet.python.modules.conv import SeparableConv1D
from sonnet.python.modules.conv import SeparableConv2D
from sonnet.python.modules.conv import VALID
from sonnet.python.modules.embed import Embed
from sonnet.python.modules.gated_rnn import BatchNormLSTM
from sonnet.python.modules.gated_rnn import Conv1DLSTM
from sonnet.python.modules.gated_rnn import Conv2DLSTM
from sonnet.python.modules.gated_rnn import GRU
from sonnet.python.modules.gated_rnn import highway_core_with_recurrent_dropout
from sonnet.python.modules.gated_rnn import HighwayCore
from sonnet.python.modules.gated_rnn import LSTM
from sonnet.python.modules.gated_rnn import lstm_with_recurrent_dropout
from sonnet.python.modules.gated_rnn import lstm_with_zoneout
from sonnet.python.modules.gated_rnn import LSTMBlockCell
from sonnet.python.modules.gated_rnn import LSTMState
from sonnet.python.modules.layer_norm import LayerNorm
from sonnet.python.modules.moving_average import MovingAverage
from sonnet.python.modules.optimization_constraints import get_lagrange_multiplier
from sonnet.python.modules.optimization_constraints import OptimizationConstraints
from sonnet.python.modules.pondering_rnn import ACTCore
from sonnet.python.modules.relational_memory import RelationalMemory
from sonnet.python.modules.residual import Residual
from sonnet.python.modules.residual import ResidualCore
from sonnet.python.modules.residual import SkipConnectionCore
from sonnet.python.modules.rnn_core import RNNCellWrapper
from sonnet.python.modules.rnn_core import RNNCore
from sonnet.python.modules.rnn_core import trainable_initial_state
from sonnet.python.modules.rnn_core import TrainableInitialState
from sonnet.python.modules.scale_gradient import scale_gradient
from sonnet.python.modules.sequential import Sequential
from sonnet.python.modules.spatial_transformer import AffineGridWarper
from sonnet.python.modules.spatial_transformer import AffineWarpConstraints
from sonnet.python.modules.spatial_transformer import GridWarper
from sonnet.python.modules.util import check_initializers
from sonnet.python.modules.util import check_partitioners
from sonnet.python.modules.util import check_regularizers
from sonnet.python.modules.util import count_variables_by_type
from sonnet.python.modules.util import custom_getter_router
from sonnet.python.modules.util import deprecation_warning
from sonnet.python.modules.util import format_variable_map
from sonnet.python.modules.util import format_variables
from sonnet.python.modules.util import get_normalized_variable_map
from sonnet.python.modules.util import get_saver
from sonnet.python.modules.util import get_variables_in_module
from sonnet.python.modules.util import get_variables_in_scope
from sonnet.python.modules.util import has_variable_scope
from sonnet.python.modules.util import log_variables
from sonnet.python.modules.util import parse_string_to_constructor
from sonnet.python.modules.util import remove_unsupported_kwargs
from sonnet.python.modules.util import reuse_variables
from sonnet.python.modules.util import summarize_variables
from sonnet.python.modules.util import supports_kwargs
from sonnet.python.modules.util import variable_map_items
from sonnet.python.ops import nest
from sonnet.python.ops.initializers import restore_initializer


__version__ = '1.34'

