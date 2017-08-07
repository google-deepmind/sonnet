# ![Sonnet](images/sonnet_logo.png)

Sonnet is a library built on top of TensorFlow for building complex neural
networks.


## Installation

Sonnet can be installed from pip, with or without GPU support.

The version with GPU depends on `tensorflow-gpu` and will install it if you have
not done so. The version without GPU support will use the version of tensorflow
you have installed, and will not work if you don't have any version installed.

To install sonnet without gpu support, run

```shell
$ pip install dm-sonnet
```

To install it with gpu support:

```shell
$ pip install dm-sonnet-gpu
```

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
