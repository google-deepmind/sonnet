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

import sys

from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase

_VERSION = '1.36'


EXTRA_PACKAGES = {
    'tensorflow': ['tensorflow>=1.15.0,<2.0.0'],
    'tensorflow with gpu': ['tensorflow-gpu>=1.15.0,<2.0.0'],
}

REQUIRED_PACKAGES = [
    'tensorflow-probability>=0.8.0,<0.9.0',  # Version 0.9 requires tensorflow 2
    'six',
    'absl-py',
    'semantic_version',
    'contextlib2',
    'wrapt'
]


setup(
    name='dm-sonnet',
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
