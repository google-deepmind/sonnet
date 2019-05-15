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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sonnet import initializers
from sonnet import nets
from sonnet import optimizers
from sonnet import pad
from sonnet.src.axis_norm import AxisNorm
from sonnet.src.axis_norm import InstanceNorm
from sonnet.src.axis_norm import LayerNorm
from sonnet.src.base import Module
from sonnet.src.base import no_name_scope
from sonnet.src.batch_norm import BaseBatchNorm
from sonnet.src.batch_norm import BatchNorm
from sonnet.src.bias import Bias
from sonnet.src.conv import Conv1D
from sonnet.src.conv import Conv2D
from sonnet.src.conv import Conv3D
from sonnet.src.conv_transpose import Conv1DTranspose
from sonnet.src.conv_transpose import Conv2DTranspose
from sonnet.src.conv_transpose import Conv3DTranspose
from sonnet.src.deferred import Deferred
from sonnet.src.dropout import Dropout
from sonnet.src.group_norm import GroupNorm
from sonnet.src.linear import Linear
from sonnet.src.metrics import Metric
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
from sonnet.src.recurrent import VanillaRNN
from sonnet.src.reshape import Flatten
from sonnet.src.reshape import Reshape
from sonnet.src.sequential import Sequential

__all__ = (
    "AxisNorm",
    "BaseBatchNorm",
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
    "Dropout",
    "Flatten",
    "GroupNorm",
    "InstanceNorm",
    "GRU",
    "LSTM",
    "LSTMState",
    "LayerNorm",
    "Linear",
    "Metric",
    "Module",
    "Reshape",
    "RNNCore",
    "Sequential",
    "TrainableState",
    "VanillaRNN",
    "deep_rnn_with_residual_connections",
    "deep_rnn_with_skip_connections",
    "dynamic_unroll",
    "initializers",
    "lstm_with_recurrent_dropout",
    "no_name_scope",
    "nets",
    "once",
    "optimizers",
    "pad",
    "static_unroll",
)

__version__ = "2.0.0a0"
