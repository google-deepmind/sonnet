![Sonnet](https://sonnet.dev/images/sonnet_logo.png)

# Sonnet

[**Documentation**](https://sonnet.readthedocs.io/) | [**Examples**](#examples)

Sonnet is a library built on top of [TensorFlow 2](https://www.tensorflow.org/)
designed to provide simple, composable abstractions for machine learning
research.

# Introduction

Sonnet has been designed and built by researchers at DeepMind. It can be used to
construct neural networks for many different purposes (un/supervised learning,
reinforcement learning, ...). We find it is a successful abstraction for our
organization, you might too!

More specifically, Sonnet provides a simple but powerful programming model
centered around a single concept: `snt.Module`. Modules can hold references to
parameters, other modules and methods that apply some function on the user
input. Sonnet ships with many predefined modules (e.g. `snt.Linear`,
`snt.Conv2D`, `snt.BatchNorm`) and some predefined networks of modules (e.g.
`snt.nets.MLP`) but users are also encouraged to build their own modules.

Unlike many frameworks Sonnet is extremely unopinionated about **how** you will
use your modules. Modules are designed to be self contained and entirely
decoupled from one another. Sonnet does not ship with a training framework and
users are encouraged to build their own or adopt those built by others.

Sonnet is also designed to be simple to understand, our code is (hopefully!)
clear and focussed. Where we have picked defaults (e.g. defaults for initial
parameter values) we try to point out why.

# Getting Started

## Examples

The easiest way to try Sonnet is to use Google Colab which offers a free Python
notebook attached to a GPU or TPU.

- [Predicting MNIST with an MLP](https://colab.research.google.com/github/deepmind/sonnet/blob/v2/examples/mlp_on_mnist.ipynb)
- [Training a Little GAN on MNIST](https://colab.research.google.com/github/deepmind/sonnet/blob/v2/examples/little_gan_on_mnist.ipynb)
- [Distributed training with `snt.distribute`](https://colab.research.google.com/github/deepmind/sonnet/blob/v2/examples/distributed_cifar10.ipynb)

## Installation

To get started install TensorFlow 2.0 and Sonnet 2:

```shell
$ pip install tensorflow tensorflow-probability
$ pip install dm-sonnet
```

You can run the following to verify things installed correctly:

```python
import tensorflow as tf
import sonnet as snt

print("TensorFlow version {}".format(tf.__version__))
print("Sonnet version {}".format(snt.__version__))
```

### Using existing modules

Sonnet ships with a number of built in modules that you can trivially use. For
example to define an MLP we can use the `snt.Sequential` module to call a
sequence of modules, passing the output of a given module as the input for the
next module. We can use `snt.Linear` and `tf.nn.relu` to actually define our
computation:

```python
mlp = snt.Sequential([
    snt.Linear(1024),
    tf.nn.relu,
    snt.Linear(10),
])
```

To use our module we need to "call" it. The `Sequential` module (and most
modules) define a `__call__` method that means you can call them by name:

```python
logits = mlp(tf.random.normal([batch_size, input_size]))
```

It is also very common to request all the parameters for your module. Most
modules in Sonnet create their parameters the first time they are called with
some input (since in most cases the shape of the parameters is a function of
the input). Sonnet modules provide two properties for accessing parameters.

The `variables` property returns **all** `tf.Variable`s that are referenced by
the given module:

```python
all_variables = mlp.variables
```

It is worth noting that `tf.Variable`s are not just used for parameters of your
model. For example they are used to hold state in metrics used in
`snt.BatchNorm`. In most cases users retrieve the module variables to pass them
to an optimizer to be updated. In this case non-trainable variables should
typically not be in that list as they are updated via a different mechanism.
TensorFlow has a built in mechanism to mark variables as "trainable" (parameters
of your model) vs. non-trainable (other variables). Sonnet provides a mechanism
to gather all trainable variables from your module which is probably what you
want to pass to an optimizer:

```python
model_parameters = mlp.trainable_variables
```

### Building your own module

Sonnet strongly encourages users to subclass `snt.Module` to define their own
modules. Let's start by creating a simple `Linear` layer called `MyLinear`:

```python
class MyLinear(snt.Module):

  def __init__(self, output_size, name=None):
    super(MyLinear, self).__init__(name=name)
    self.output_size = output_size

  @snt.once
  def _initialize(self, x):
    initial_w = tf.random.normal([x.shape[1], self.output_size])
    self.w = tf.Variable(initial_w, name="w")
    self.b = tf.Variable(tf.zeros([self.output_size]), name="b")

  def __call__(self, x):
    self._initialize(x)
    return tf.matmul(x, self.w) + self.b
```

Using this module is trivial:

```python
mod = MyLinear(32)
mod(tf.ones([batch_size, input_size]))
```

By subclassing `snt.Module` you get many nice properties for free. For example
a default implementation of `__repr__` which shows constructor arguments (very
useful for debugging and introspection):

```python
>>> print(repr(mod))
MyLinear(output_size=10)
```

You also get the `variables` and `trainable_variables` properties:

```python
>>> mod.variables
(<tf.Variable 'my_linear/b:0' shape=(10,) ...)>,
 <tf.Variable 'my_linear/w:0' shape=(1, 10) ...)>)
```

You may notice the `my_linear` prefix on the variables above. This is because
Sonnet modules also enter the modules name scope whenever methods are called.
By entering the module name scope we provide a much more useful graph for tools
like TensorBoard to consume (e.g. all operations that occur inside my_linear
will be in a group called my_linear).

Additionally your module will now support TensorFlow checkpointing and saved
model which are advanced features covered later.

# Serialization

Sonnet supports multiple serialization formats. The simplest format we support
is Python's `pickle`, and all built in modules are tested to make sure they can
be saved/loaded via pickle in the same Python process. In general we discourage
the use of pickle, it is not well supported by many parts of TensorFlow and in
our experience can be quite brittle.

## TensorFlow Checkpointing

**Reference:** https://www.tensorflow.org/alpha/guide/checkpoints

TensorFlow checkpointing can be used to save the value of parameters
periodically during training. This can be useful to save the progress of
training in case your program crashes or is stopped. Sonnet is designed to work
cleanly with TensorFlow checkpointing:

```python
checkpoint_root = "/tmp/checkpoints"
checkpoint_name = "example"
save_prefix = os.path.join(checkpoint_root, checkpoint_name)

my_module = create_my_sonnet_module()  # Can be anything extending snt.Module.

# A `Checkpoint` object manages checkpointing of the TensorFlow state associated
# with the objects passed to it's constructor. Note that Checkpoint supports
# restore on create, meaning that the variables of `my_module` do **not** need
# to be created before you restore from a checkpoint (their value will be
# restored when they are created).
checkpoint = tf.train.Checkpoint(module=my_module)

# Most training scripts will want to restore from a checkpoint if one exists. This
# would be the case if you interrupted your training (e.g. to use your GPU for
# something else, or in a cloud environment if your instance is preempted).
latest = tf.train.latest_checkpoint(checkpoint_root)
if latest is not None:
  checkpoint.restore(latest)

for step_num in range(num_steps):
  train(my_module)

  # During training we will occasionally save the values of weights. Note that
  # this is a blocking call and can be slow (typically we are writing to the
  # slowest storage on the machine). If you have a more reliable setup it might be
  # appropriate to save less frequently.
  if step_num and not step_num % 1000:
    checkpoint.save(save_prefix)

# Make sure to save your final values!!
checkpoint.save(save_prefix)
```

## TensorFlow Saved Model

**Reference:** https://www.tensorflow.org/alpha/guide/saved_model

TensorFlow saved models can be used to save a copy of your network that is
decoupled from the Python source for it. This is enabled by saving a TensorFlow
graph describing the computation and a checkpoint containing the value of
weights.

The first thing to do in order to create a saved model is to create a
`snt.Module` that you want to save:

```python
my_module = snt.nets.MLP([1024, 1024, 10])
my_module(tf.ones([1, input_size]))
```

Next, we need to create another module describing the specific parts of our
model that we want to export. We advise doing this (rather than modifying the
original model in-place) so you have fine grained control over what is actually
exported. This is typically important to avoid creating very large saved models,
and such that you only share the parts of your model you want to (e.g. you only
want to share the generator for a GAN but keep the discriminator private).

```python
@tf.function(input_signature=[tf.TensorSpec([None, input_size])])
def inference(x):
  return my_module(x)

to_save = snt.Module()
to_save.inference = inference
to_save.all_variables = list(my_module.variables)
tf.saved_model.save(to_save, "/tmp/example_saved_model")
```

We now have a saved model in the `/tmp/example_saved_model` folder:

```shell
$ ls -lh /tmp/example_saved_model
total 24K
drwxrwsr-t 2 tomhennigan 154432098 4.0K Apr 28 00:14 assets
-rw-rw-r-- 1 tomhennigan 154432098  14K Apr 28 00:15 saved_model.pb
drwxrwsr-t 2 tomhennigan 154432098 4.0K Apr 28 00:15 variables
```

Loading this model is simple and can be done on a different machine without any
of the Python code that built the saved model:

```python
loaded = tf.saved_model.load("/tmp/example_saved_model")

# Use the inference method. Note this doesn't run the Python code from `to_save`
# but instead uses the TensorFlow Graph that is part of the saved model.
loaded.inference(tf.ones([1, input_size]))

# The all_variables property can be used to retrieve the restored variables.
assert len(loaded.all_variables) > 0
```

Note that the loaded object is not a Sonnet module, it is a container object
that has the specific methods (e.g. `inference`) and properties (e.g.
`all_variables`) that we added in the previous block.

## Distributed training

**Example:** https://github.com/deepmind/sonnet/blob/v2/examples/distributed_cifar10.ipynb

Sonnet has support for distributed training using
[custom TensorFlow distribution strategies](https://sonnet.readthedocs.io/en/latest/api.html#module-sonnet.distribute).

A key difference between Sonnet and distributed training using `tf.keras` is
that Sonnet modules and optimizers do not behave differently when run under
distribution strategies (e.g. we do not average your gradients or sync your
batch norm stats). We believe that users should be in full control of these
aspects of their training and they should not be baked into the library. The
trade off here is that you need to implement these features in your training
script (typically this is just 2 lines of code to all reduce your gradients
before applying your optimizer) or swap in modules that are explicitly
distribution aware (e.g. `snt.distribute.CrossReplicaBatchNorm`).

Our [distributed Cifar-10](https://github.com/deepmind/sonnet/blob/v2/examples/distributed_cifar10.ipynb)
example walks through doing multi-GPU training with Sonnet.
