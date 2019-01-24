# ![Sonnet](docs/images/sonnet_logo.png)

Sonnet is a library built on top of TensorFlow for building complex neural
networks.


## Installation

Sonnet can be installed from pip, with or without GPU support.

This installation is compatible with Linux/Mac OS X and Python 2.7 and
3.{4,5,6}. The version of TensorFlow installed must be >= 1.5. Installing
Sonnet supports the [virtualenv installation mode](https://www.tensorflow.org/install/install_linux#installing_with_virtualenv)
of TensorFlow, as well as the [native pip install](https://www.tensorflow.org/install/install_linux#installing_with_native_pip).

To install sonnet, run:

```shell
$ pip install dm-sonnet
```

Sonnet will work with both the CPU and GPU version of tensorflow, but to allow
for that it does not list Tensorflow as a requirement, so you need to install
Tensorflow separately if you haven't already done so.

## Usage Example

The following code constructs a Linear module and connects it to multiple
inputs. The variables (i.e., the weights and biases of the linear
transformation) are automatically shared.

```python
import sonnet as snt

# Provide your own functions to generate data Tensors.
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
