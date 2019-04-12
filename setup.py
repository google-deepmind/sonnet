# pylint: disable=g-bad-file-header
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
# =============================================================================

"""Setup for pip package."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase

# This version string is semver compatible, but incompatible with pip.
# For pip, we will remove all '-' characters from this string, and use the
# result for pip.
_VERSION = '1.32'

# Use the SONNET_GPU env var to configure GPU deps.
with_gpu = os.environ.get('SONNET_GPU', '0').lower() not in ('0', 'false', 'no')

project_name = 'dm-sonnet-gpu' if with_gpu else 'dm-sonnet'

_packages = {
    'tensorflow': ['tensorflow>=1.8.0'],
    'tensorflow-gpu': ['tensorflow-gpu>=1.8.0'],
    'tensorflow-probability': ['tensorflow-probability>=0.4.0'],
    'tensorflow-probability-gpu': ['tensorflow-probability-gpu>=0.4.0'],
}

EXTRA_PACKAGES = {
    'tensorflow': _packages['tensorflow'],
    'tensorflow with gpu': _packages['tensorflow-gpu'],
    'tensorflow probability': _packages['tensorflow-probability'],
    'tensorflow probability with gpu': _packages['tensorflow-probability-gpu'],
}

REQUIRED_PACKAGES = [
    'six', 'absl-py', 'semantic_version', 'contextlib2', 'wrapt'
]

# If this is the GPU build of sonnet, tensorflow-gpu is a hard requirement.
# The CPU only version works well with both versions of tensorflow.
if with_gpu:
  REQUIRED_PACKAGES += _packages['tensorflow-gpu']
  REQUIRED_PACKAGES += _packages['tensorflow-probability-gpu']


setup(
    name=project_name,
    version=_VERSION,
    description=(
        'Sonnet is a library for building neural networks in TensorFlow.'),
    long_description='',
    url='https://github.com/deepmind/sonnet',
    author='DeepMind',
    author_email='sonnet-dev-os@google.com',
    # Contained modules and scripts.
    packages=find_packages(),
    entry_points={},
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_PACKAGES,
    # Add in any packaged data.
    include_package_data=True,
    package_data={
        '': ['*.txt', '*.rst'],
        'sonnet': ['*.so'],
    },
    zip_safe=False,
    cmdclass={
        'install': InstallCommandBase,
    },
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: {}'.format(
            '2.7' if (sys.version[0] == '2') else sys.version[0]),
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    license='Apache 2.0',
    keywords='sonnet tensorflow tensor machine learning',
)
