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

"""Common network architectures implemented as Sonnet modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sonnet.python.modules.nets.alexnet import AlexNet
from sonnet.python.modules.nets.alexnet import AlexNetFull
from sonnet.python.modules.nets.alexnet import AlexNetMini
from sonnet.python.modules.nets.convnet import ConvNet2D
from sonnet.python.modules.nets.convnet import ConvNet2DTranspose
from sonnet.python.modules.nets.dilation import Dilation
from sonnet.python.modules.nets.dilation import identity_kernel_initializer
from sonnet.python.modules.nets.dilation import noisy_identity_kernel_initializer
from sonnet.python.modules.nets.mlp import MLP
from sonnet.python.modules.nets.vqvae import VectorQuantizer
from sonnet.python.modules.nets.vqvae import VectorQuantizerEMA
