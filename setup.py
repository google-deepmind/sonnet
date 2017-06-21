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

from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase

# This version string is semver compatible, but incompatible with pip.
# For pip, we will remove all '-' characters from this string, and use the
# result for pip.
_VERSION = '1.2'

project_name = 'sonnet'

REQUIRED_PACKAGES = [
    'nose_parameterized >= 0.6.0',
]

EXTRA_PACKAGES = {
    'tensorflow': ['tensorflow>=1.0.1'],
    'tensorflow with gpu': ['tensorflow-gpu>=1.0.1']
}

setup(
    name=project_name,
    version=_VERSION,
    description=('Sonnet is a library for building neural networks in '
                 'TensorFlow.'),
    long_description='',
    url='http://www.github.com/deepmind/sonnet/',
    author='DeepMind',
    author_email='sonnet-dev-os@google.com',
    # Contained modules and scripts.
    packages=find_packages(),
    entry_points={},
    install_requires=REQUIRED_PACKAGES,
    extra_requires=EXTRA_PACKAGES,
    tests_require=REQUIRED_PACKAGES,
    # Add in any packaged data.
    include_package_data=True,
    package_data={'': ['*.txt', '*.rst'], 'sonnet': ['*.so']},
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
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
        ],
    license='Apache 2.0',
    keywords='sonnet tensorflow tensor machine learning',
    )
