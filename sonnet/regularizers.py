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
"""Regularizers."""

from sonnet.src.regularizers import L1
from sonnet.src.regularizers import L2
from sonnet.src.regularizers import OffDiagonalOrthogonal
from sonnet.src.regularizers import Regularizer

__all__ = [
    "L1",
    "L2",
    "OffDiagonalOrthogonal",
    "Regularizer",
]
