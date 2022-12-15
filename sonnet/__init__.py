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
"""Sonnet built for TensorFlow 2."""

from sonnet import distribute
from sonnet import functional
from sonnet import initializers
from sonnet import mixed_precision
from sonnet import nets
from sonnet import optimizers
from sonnet import pad
from sonnet import regularizers
from sonnet.src.axis_norm import InstanceNorm
from sonnet.src.axis_norm import LayerNorm
from sonnet.src.base import allow_empty_variables
from sonnet.src.base import Module
from sonnet.src.base import no_name_scope
from sonnet.src.base import Optimizer
from sonnet.src.batch_apply import BatchApply
from sonnet.src.batch_apply import merge_leading_dims
from sonnet.src.batch_apply import split_leading_dim
from sonnet.src.batch_norm import BaseBatchNorm
from sonnet.src.batch_norm import BatchNorm
from sonnet.src.bias import Bias
from sonnet.src.build import build
from sonnet.src.conv import Conv1D
from sonnet.src.conv import Conv2D
from sonnet.src.conv import Conv3D
from sonnet.src.conv_transpose import Conv1DTranspose
from sonnet.src.conv_transpose import Conv2DTranspose
from sonnet.src.conv_transpose import Conv3DTranspose
from sonnet.src.custom_getter import custom_variable_getter
from sonnet.src.deferred import Deferred
from sonnet.src.depthwise_conv import DepthwiseConv2D
from sonnet.src.dropout import Dropout
from sonnet.src.embed import Embed
from sonnet.src.group_norm import GroupNorm
from sonnet.src.leaky_clip_by_value import leaky_clip_by_value
from sonnet.src.linear import Linear
from sonnet.src.metrics import Mean
from sonnet.src.metrics import Metric
from sonnet.src.metrics import Sum
from sonnet.src.moving_averages import ExponentialMovingAverage
from sonnet.src.once import once
from sonnet.src.recurrent import Conv1DLSTM
from sonnet.src.recurrent import Conv2DLSTM
from sonnet.src.recurrent import Conv3DLSTM
from sonnet.src.recurrent import deep_rnn_with_residual_connections
from sonnet.src.recurrent import deep_rnn_with_skip_connections
from sonnet.src.recurrent import DeepRNN
from sonnet.src.recurrent import dynamic_unroll
from sonnet.src.recurrent import GRU
from sonnet.src.recurrent import LSTM
from sonnet.src.recurrent import lstm_with_recurrent_dropout
from sonnet.src.recurrent import LSTMState
from sonnet.src.recurrent import RNNCore
from sonnet.src.recurrent import static_unroll
from sonnet.src.recurrent import TrainableState
from sonnet.src.recurrent import UnrolledLSTM
from sonnet.src.recurrent import UnrolledRNN
from sonnet.src.recurrent import VanillaRNN
from sonnet.src.reshape import flatten
from sonnet.src.reshape import Flatten
from sonnet.src.reshape import reshape
from sonnet.src.reshape import Reshape
from sonnet.src.scale_gradient import scale_gradient
from sonnet.src.sequential import Sequential
from sonnet.src.utils import format_variables
from sonnet.src.utils import log_variables

__all__ = (
    "BaseBatchNorm",
    "BatchApply",
    "BatchNorm",
    "Bias",
    "Conv1D",
    "Conv1DLSTM",
    "Conv1DTranspose",
    "Conv2D",
    "Conv2DLSTM",
    "Conv2DTranspose",
    "Conv3D",
    "Conv3DLSTM",
    "Conv3DTranspose",
    "DeepRNN",
    "Deferred",
    "DepthwiseConv2D",
    "Dropout",
    "Embed",
    "ExponentialMovingAverage",
    "flatten",
    "Flatten",
    "GroupNorm",
    "InstanceNorm",
    "GRU",
    "LSTM",
    "LSTMState",
    "LayerNorm",
    "Linear",
    "Mean",
    "Metric",
    "Module",
    "Optimizer",
    "reshape",
    "Reshape",
    "RNNCore",
    "Sequential",
    "Sum",
    "TrainableState",
    "UnrolledLSTM",
    "UnrolledRNN",
    "VanillaRNN",
    "allow_empty_variables",
    "build",
    "custom_variable_getter",
    "deep_rnn_with_residual_connections",
    "deep_rnn_with_skip_connections",
    "distribute",
    "dynamic_unroll",
    "format_variables",
    "functional",
    "initializers",
    "log_variables",
    "lstm_with_recurrent_dropout",
    "merge_leading_dims",
    "no_name_scope",
    "nets",
    "once",
    "leaky_clip_by_value",
    "optimizers",
    "pad",
    "regularizers",
    "scale_gradient",
    "split_leading_dim",
    "static_unroll",
)

__version__ = "2.0.1"

#  ________________________________________
# / Please don't use symbols in `src` they \
# \ are not part of the Sonnet public API. /
#  ----------------------------------------
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||
#
try:
  del src  # pylint: disable=undefined-variable
except NameError:
  pass
