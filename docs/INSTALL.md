# Installing from source

To install Sonnet from source, you will need to compile the library using bazel.
You should have installed TensorFlow by following the [TensorFlow installation
instructions](https://www.tensorflow.org/install/).

## Install bazel

Ensure you have a recent version of bazel (>= 0.4.5) and JDK (>= 1.8). If not,
follow [these directions](https://bazel.build/versions/master/docs/install.html).

## (virtualenv TensorFlow installation) Activate virtualenv

If using virtualenv, activate your virtualenv for the rest of the installation,
otherwise skip this step:

```shell
$ source $VIRTUALENV_PATH/bin/activate # bash, sh, ksh, or zsh
$ source $VIRTUALENV_PATH/bin/activate.csh  # csh or tcsh
```

## Build and run the installer

First clone the Sonnet source code:

```shell
$ git clone https://github.com/deepmind/sonnet
```

Then run the install script to create a wheel file in a temporary directory:

```shell
$ mkdir /tmp/sonnet
$ bazel build :install
$ ./bazel-bin/install /tmp/sonnet
```

To build the GPU accelerated version of Sonnet use:

```shell
$ SONNET_GPU=1 ./bazel-bin/install /tmp/sonnet
```

By default, the wheel file is built using `python`. You can optionally specify
another python binary in the previous command to build the wheel file, such as
`python3`:

```
$ ./bazel-bin/install /tmp/sonnet python3
```

`pip install` the generated wheel file:

```shell
$ pip install /tmp/sonnet/*.whl
```

If Sonnet was already installed, uninstall prior to calling `pip install` on
the wheel file:

```shell
$ pip uninstall dm-sonnet  # or dm-sonnet-gpu
```

You can verify that Sonnet has been successfully installed by, for example,
trying to instantiate and connect a Linear module:

```shell
$ cd ~/
$ python
>>> import tensorflow as tf
>>> import sonnet as snt
>>> input_ = tf.zeros(3, 5)
>>> output = snt.Linear(10)(input_)
```
