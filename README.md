# ![Sonnet](images/sonnet_logo.png)

Sonnet is a library built on top of TensorFlow for building complex neural
networks.

## Installation instructions

To install Sonnet, you will need to compile the library using bazel against
the TensorFlow header files. You should have installed TensorFlow by
following the [TensorFlow installation instructions](https://www.tensorflow.org/install/).

This installation is compatible with Linux/Mac OS X and Python 2.7 and 3.4. The version
of TensorFlow installed must be at least 1.2. Installing Sonnet supports the
[virtualenv installation mode](https://www.tensorflow.org/install/install_linux#installing_with_virtualenv)
of TensorFlow, as well as the [native pip install](https://www.tensorflow.org/install/install_linux#installing_with_native_pip).

### Install bazel

Ensure you have a recent version of bazel (>= 0.4.5) and JDK (>= 1.8). If not,
follow [these directions](https://bazel.build/versions/master/docs/install.html).

### (virtualenv TensorFlow installation) Activate virtualenv

If using virtualenv, activate your virtualenv for the rest of the installation,
otherwise skip this step:

```shell
$ source $VIRTUALENV_PATH/bin/activate # bash, sh, ksh, or zsh
$ source $VIRTUALENV_PATH/bin/activate.csh  # csh or tcsh
```

### Configure TensorFlow Headers

First clone the Sonnet source code with TensorFlow as a submodule:

```shell
$ git clone --recursive https://github.com/deepmind/sonnet
```

and then call `configure`:

```shell
$ cd sonnet/tensorflow
$ ./configure
$ cd ../
```

You can choose the suggested defaults during the TensorFlow configuration.
Note: This will not modify your existing installation of TensorFlow. This step
is necessary so that Sonnet can build against the TensorFlow headers.

### Build and run the installer

Run the install script to create a wheel file in a temporary directory:

```shell
$ mkdir /tmp/sonnet
$ bazel build --config=opt :install
$ ./bazel-bin/install /tmp/sonnet
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
$ pip uninstall sonnet
```

You can verify that Sonnet has been successfully installed by, for example,
trying out the resampler op:

```shell
$ cd ~/
$ python
>>> import sonnet as snt
>>> import tensorflow as tf
>>> snt.resampler(tf.constant([0.]), tf.constant([0.]))
```

The expected output should be:

```shell
<tf.Tensor 'resampler/Resampler:0' shape=(1,) dtype=float32>
```

However, if an `ImportError` is raised then the C++ components were not found.
Ensure that you are not importing the cloned source code (i.e. call python
outside of the cloned repository) and that you have uninstalled Sonnet prior to
installing the wheel file.

## Usage Example

The following code constructs a Linear module and connects it to multiple
inputs. The variables (i.e., the weights and biases of the linear
transformation) are automatically shared.

```python
import sonnet as snt

train_data = get_training_data()
test_data = get_test_data()

# Construct the module, providing any configuration necessary.
linear_regression_module = snt.Linear(output_size=FLAGS.output_size)

# Connect the module to some inputs, any number of times.
train_predictions = linear_regression_module(train_data)
test_predictions = linear_regression_module(test_data)
```

# Documentation

Check out the full documentation page
[here](https://deepmind.github.io/sonnet/).
