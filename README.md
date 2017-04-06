# ![Sonnet](images/sonnet_logo.png)

Sonnet is a library built on top of TensorFlow for building complex neural
networks.

## Installation instructions

To install Sonnet, you will need to compile the library using bazel against
the TensorFlow header files. You should have installed TensorFlow by
following the [TensorFlow installation instructions](https://www.tensorflow.org/install/).

This installation is compatible with Linux/Mac OS X and Python 2.7. The version
of TensorFlow installed must be at least 1.0.1. Installing Sonnet supports the
[virtualenv installation mode](https://www.tensorflow.org/install/install_linux#installing_with_virtualenv)
of TensorFlow, as well as the [native pip install](https://www.tensorflow.org/install/install_linux#installing_with_native_pip).

### Install bazel

Ensure you have a recent version of bazel (>= 0.4.5 ). If not, follow
[these directions](https://bazel.build/versions/master/docs/install.html).

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

More usage examples can be found [here](sonnet/examples/).

## General Principles

The main principle of Sonnet is to first _construct_ Python objects which
represent some part of a neural network, and then separately _connect_ these
objects into the TensorFlow computation graph. The objects are subclasses of
`sonnet.AbstractModule` and as such are referred to as `Modules`.

Modules may be connected into the graph multiple times, and any variables
declared in that module will be automatically shared on subsequent connection
calls. Low level aspects of TensorFlow which control variable sharing, including
specifying variable scope names, and using the `reuse=` flag, are abstracted
away from the user.

Separating configuration and connection allows easy construction of higher-order
Modules, i.e., modules that wrap other modules. For instance,
the `BatchApply` module merges a number of leading dimensions of a tensor into
a single dimension, connects a provided module, and then splits the leading
dimension of the result to match the input.
At construction time, the inner module is passed in as an argument to the
`BatchApply` constructor. At run time, the module first performs a reshape
operation on inputs, then applies the module passed into the constructor, and
then inverts the reshape operation.

An additional advantage of representing Modules by Python objects is that it
allows additional methods to be defined where necessary. An example of this is
a module which, after construction, may be connected in a variety of ways while
maintaining weight sharing. For instance, in the case of a generative model, we
may want to sample from the model, or calculate the log probability of a given
observation. Having both connections simultaneously requires weight sharing, and
so these methods depend on the same variables. The variables are conceptually
owned by the object, and are used by different methods of the module.

## Importing Sonnet

The recommended way to import Sonnet is to alias it to a variable named `snt`:

```python
import sonnet as snt
```

Every module is then accessible under the namespace `snt`, and the rest of this
document will use `snt` for brevity.

The following code constructs a module that is composed of other modules:

```python
import sonnet as snt

# Our data is coming in via multiple inputs, so to apply the same model to each
# we will need to use variable sharing.
train_data = get_training_data()
test_data = get_test_data()

# Make two linear modules, to form a Multi Layer Perceptron. Override the
# default names (which would end up being 'linear', 'linear_1') to provide
# interpretable variable names in TensorBoard / other tools.
lin_to_hidden = snt.Linear(output_size=FLAGS.hidden_size, name='inp_to_hidden')
hidden_to_out = snt.Linear(output_size=FLAGS.output_size, name='hidden_to_out')

# Sequential is a module which applies a number of inner modules or ops in
# sequence to the provided data. Note that raw TF ops such as tanh can be
# used interchangeably with constructed modules, as they contain no variables.
mlp = snt.Sequential([lin_to_hidden, tf.sigmoid, hidden_to_out])

# Connect the sequential into the graph, any number of times.
train_predictions = mlp(train_data)
test_predictions = mlp(test_data)
```

The following code adds initializers and regularizers to a Linear module:

```python
import sonnet as snt

train_data = get_training_data()
test_data = get_test_data()

# Initializers and regularizers for the weights and the biasses.
initializers={"w": tf.truncated_normal_initializer(stddev=1.0),
              "b": tf.truncated_normal_initializer(stddev=1.0)}
regularizers = {"w": tf.contrib.layers.l1_regularizer(scale=0.1),
                "b": tf.contrib.layers.l2_regularizer(scale=0.1)}

linear_regression_module = snt.Linear(output_size=FLAGS.output_size,
                                      initializers=initializers,
                                      regularizers=regularizers)

# Connect the module to some inputs, any number of times.
train_predictions = linear_regression_module(train_data)
test_predictions = linear_regression_module(test_data)

# ...

# Get the regularization losses and add them together.
graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
total_regularization_loss = tf.reduce_sum(graph_regularizers)

# ...

# When minimizing the loss, minimize also the regularization loss.
train_op = optimizer.minimize(loss + total_regularizer_loss)
```

## Defining your own modules

### Inherit from `snt.AbstractModule`

To define a module, create a new class which inherits from `snt.AbstractModule`.
The constructor of your class should accept any configuration which defines the
operation of that module, and store it in a member variable prefixed with an
underscore, to indicate that it is private.

### Call superclass constructor

The first thing the constructor does
should be to call the superclass constructor, passing in the name for the module
- if you forget to do this, the variable sharing will break. A `name` kwarg
should always be provided as the final one of the list, with the default value
being a `snake_case` version of the class name.

```python
class MyMLP(snt.AbstractModule):
  """Docstring for MyMLP."""
  def __init__(self, hidden_size, output_size,
               nonlinearity=tf.tanh, name="my_mlp"):
    """Docstring explaining __init__ args, including types and defaults."""
    super(MyMLP, self).__init__(name)
    self._hidden_size = hidden_size
    self._output_size = output_size
    self._nonlinearity = nonlinearity
```

### Implement `_build()` method

The only other method implementation which must be provided is `_build()`. This
will be called whenever the module is connected into the `tf.Graph`. It receives
some input, which may be empty, a single Tensor, or some arbitrary structure
containing multiple Tensors. Multiple Tensors can be provided with either a
tuple or namedtuple, the elements of which in turn can be Tensors or further
tuples / namedtuples. Most input Tensors require a batch dimension, and if a
Tensor has a color channel then it _must_ be the last dimension. While in many
cases the library will not explicitly prevent you, the use of lists and dicts is
not supported, as the mutability of these structures can lead to subtle bugs.

```python
  # Following on from code snippet above..
  def _build(self, inputs):
    """Compute output Tensor from input Tensor."""
    lin_x_to_h = snt.Linear(output_size=self._hidden_size, name="x_to_h")
    lin_h_to_o = snt.Linear(output_size=self._output_size, name="h_to_o")
    return lin_h_to_o(self._nonlinearity(lin_x_to_h(inputs)))
```

The `_build` method may include any or all of the following processes:

* Construct and use internal modules
* Use modules which already exist, and were passed into the constructor
* Create variables directly.

If you create variables yourself, it is _crucial_
to create them with `tf.get_variable`. Calling the `tf.Variable` constructor
directly will only work the first time the module is connected, but on the
second call you will receive an error message "Trainable variable created when
calling a template after the first time".

The modules in the above example are created separately, passing in various
configurations, and then the final line connects them all into the graph. The
return line should be read from right to left - the inputs Tensor is passed into
the first Linear, `lin_x_to_h`, the output of which is passed into whatever
nonlinearity was stored in the constructor, the output of which goes through
another Linear to produce the result. Note that we give short meaningful names
to the internal Linear instances.

Note that the nonlinearity above can be either a raw TF op, eg `tf.tanh` or
`tf.sigmoid`, or an instance of a Sonnet module. In keeping with Python
standards, we may choose to not check this explicitly, and so we may receive
an error when `_build` is called. It is also acceptable to add constraints and
sanity checking inside `__init__`.

Note that in the above code, new instances of `snt.Linear` are
generated each time `_build()` is called, and you may think this will create
different, unshared variables. This is not the case - only 4 variables (2 for
each `Linear`) will be created, no matter how many times the MLP instance is
connected into the graph. How this is works is a low level TF detail, and
subject to change - see
[tf.variable_op_scope]
for details.

### Where should the submodules be declared?

Note that modules may use other modules which they receive already externally
constructed - eg Sequential etc. The submodules we discuss in this section are
any Modules which are _constructed_ inside the code of another Module, which we
will refer to as the Parent Module. An example is an LSTM, where most
implementations will internally construct one or more Linear modules to contain
the weights.

It's recommended that submodules are created in `_build()`. Doing it this way
means you get the correct nesting of variable scopes, eg:

```python
class ParentModule(snt.AbstractModule):
  def __init__(self, hidden_size, name="parent_module"):
    super(ParentModule, self).__init__(name=name)
    self._hidden_size = hidden_size

  def _build(self, inputs):
    lin_mod = snt.Linear(self._hidden_size)  # Construct submodule...
    return tf.relu(lin_mod(inputs))          # then connect it.
```

The variables created by the Linear will have a name something like
`parent_module/linear/w`, which is what you probably want in this kind of
situation.

Some users prefer for practical or stylistic reasons to construct everything in
the constructor, before anything is used. This is fine, but for proper variable
nesting  *any submodules must be constructed inside a
`self._enter_variable_scope` call*.

```python
class OtherParentModule(snt.AbstractModule):
  def __init__(self, hidden_size, name="other_parent_module"):
    super(OtherParentModule, self).__init__(name=name)
    self._hidden_size = hidden_size
    with self._enter_variable_scope():  # This line is crucial!
      self._lin_mod = snt.Linear(self._hidden_size)  # Construct submodule here.

  def _build(self, inputs):
    return tf.relu(self._lin_mod(inputs))  # Connect previously constructed mod.
```

The above example is fine, and will have the same variable names etc. Different
people prefer different styles and both of the above are considered correct.

The pitfall here is forgetting to call `self._enter_variable_scope()`. Things
will still "work" but the scopes will not be nested as you might expected:

```python
class WrongModule(snt.AbstractModule):
  def __init__(self, hidden_size, name="wrong_module"):
    super(WrongModule, self).__init__(name=name)
    self._hidden_size = hidden_size
    self._lin_mod = snt.Linear(self._hidden_size)  # Construct submodule here.

  def _build(self, inputs):
    return tf.relu(self._lin_mod(inputs))  # Connect previously constructed mod.
```

The above example works okay in terms of the resulting network's calculations,
but is considered a bug due to the resulting flat (instead of hierarchical)
variable namespace. The variables in the linear will be called `"linear/w"`
which is completely disjoint from the `"wrong_module"` namespace.

### Recurrent Modules

#### Usage

Sonnet includes recurrent core modules (also called "cells" in TensorFlow
terminology), which perform one time step of computation. These are ready to
be unrolled in time using TensorFlow's [unrolling operations]

One example of an LSTM that is unrolled in time is the following:

```python
hidden_size = 5
batch_size = 20
# input_sequence should be a tensor of size
# [time_steps, batch_size, input_features]
input_sequence = ...
lstm = snt.LSTM(hidden_size)
initial_state = lstm.initial_state(batch_size)
output_sequence, final_state = tf.nn.dynamic_rnn(
    lstm, input_sequence, initial_state=initial_state, time_major=True)
```

The `batch_size` parameter passed to the `initial_state()` method can also be an
`int32` Tensor.

For a more comprehensive demonstration on the usage of recurrent modules, a
fully-documented [example of a deep LSTM with skip connections trained on PTB]
is available.

#### Defining your own recurrent modules

A recurrent module is any subclass of
[`snt.RNNCore`]
which is inherits from both `snt.AbstractModule` and `tf.RNNCell`. This
unorthodox choice of multiple inheritance allows us to use the variable sharing
model from Sonnet, but also use the cores inside TensorFlow's RNN Containers.

```python
class Add1RNN(snt.RNNCore):
  """Simple core that adds 1 to its state and produces zero outputs.

  This core computes the following:

  (`input`, (`state1`, `state2`)) -> (`output`, (`next_state1`, `next_state2`))

  where all the elements are tensors, next_statei` = `statei` + 1, and
  `output` = 0. All the outputs (`state` and `output`) are of size
  (`batch_size`, `hidden_size`), where `hidden_size` is a size that is
  specified in the constructor.
  """

  def __init__(self, hidden_size, name="add1_rnn"):
    """Constructor of the module.

    Args:
      hidden_size: an int, size of the outputs of the module (without batch
          size).
      name: the name of the module.
    """
    super(Add1RNN, self).__init__(name=name)
    self._hidden_size = hidden_size

  def _build(self, inputs, state):
    """Builds a TF subgraph that performs one timestep of computation."""
    batch_size = tf.TensorShape([inputs.get_shape()[0]])
    outputs = tf.zeros(shape=batch_size.concatenate(self.output_size))
    state1, state2 = state
    next_state = (state1 + 1, state2 + 1)
    return outputs, next_state

  @property
  def state_size(self):
    """Returns a description of the state size, without batch dimension."""
    return (tf.TensorShape([self._hidden_size]),
            tf.TensorShape([self._hidden_size]))

  @property
  def output_size(self):
    """Returns a description of the output size, without batch dimension."""
    return tf.TensorShape([self._hidden_size])

  def initial_state(self, batch_size, dtype):
    """Returns an initial state with zeros, for a batch size and data type.

    NOTE: This method is here only for illustrative purposes, the corresponding
    method in its superclass should be already doing this.
    """
    sz1, sz2 = self.state_size
    # Prepend batch size to the state shape, and create zeros.
    return (tf.zeros([batch_size] + sz1.as_list(), dtype=dtype),
            tf.zeros([batch_size] + sz2.as_list(), dtype=dtype))

```

Apart from the `_build` method from `snt.AbstractModule`, a recurrent module
must also implement the `state_size` and `output_size` properties, which provide
the expected size of the recurrent state, and an example of it, respectively.
`snt.RNNCore` defines a `initial_state` method that can be used to generate a
zero initial state or a trainable initial state (based on the aforementioned
properties). Optionally, any recurrent module can define its own `initial_state`
method. Note that the `zero_state` method is also available, inherited from
`tf.RNNCell`, to produce a correctly sized state value filled with zeros.
In some situations (LSTM, etc) it may be acceptable to begin with a state
containing all zeros, but in other situations this is too limiting, and we may
want to (eg) fill some part of the state with random noise.

A common option is to make the initial state of an RNN trainable, meaning the
state is produced from some `tf.Variable`s which are trained via
backpropagation. If a core supports this, it should provide kwargs `trainable=`
and `name=` for `initial_state()`. The `name=` kwarg can be used to provide a
prefix for the (potentially multiple) variable name(s) which will be created.

### The Transposable interface
Sonnet defines an interface for modules supporting _transposition_, called
`snt.Transposable`.
Transposition is a flexible concept (e.g. not necessarily
related to matrix transposition as defined in algebra), and in this context
it entails the definition of a new module with attributes which are somehow
related to the original module, _without_ strictly implying any form of variable
sharing. For example, given a `snt.Linear` which maps from input size _A_ to
output size _B_, via transposition we will return
another `snt.Linear` module whose weight matrix shape is the transpose of the
original one, thus mapping from input size _B_ to output size _A_; given a
`snt.Conv2D` module we will return a matching `snt.Conv2DTranspose` module.

The `snt.Transposable` interface requires that transposable modules implement a
method called `transpose`, returning a module which is the transposed version of
the one the method is called on. Whilst _not_ enforced by Sonnet, chaining the
method twice should be expected to return a module with the same specifications
as the original module.

When implementing a transposable module, special care is required to ensure that
parameters needed to instantiate the module are provided as functions whose
evaluation is _deferred_ to graph construction time. This mechanism allows for
transposed modules to be instantiated _before_ the original module is connected
to the graph. An example of this behavior can be found in `snt.Linear`,
where the `output_size` argument of the transposed module is defined as a
`lambda` returning the `input_shape` property of the original module;
upon evaluation`input_shape` will raise an error unless the module has not been
connected to the graph, but this is not an issue since the `lambda` is not
called until the transposed module is connected to the graph.

## Variable reuse with `@snt.experimental.reuse_vars` (**experimental**)

Some use cases require a `tf.VariableScope` to be shared across multiple
methods, which isn't possible with `snt.AbstractModule`.
For example, a generative model may define a `sample()` and `log_pdf()` method
that share parts of the same `tf.Graph`.

Adding the `@snt.experimental.reuse_vars` decorator to a method will
enable variable reuse in much the same manner as `_build()`. The most notable
difference is that a single `tf.VariableScope` will be used across different
decorated methods and each decorated method has its own `reuse` flag that is
used to enter the variable scope.

```python
class Reusable(object):

  def __init__(self, name):
    with tf.variable_scope(name) as vs:
      self.variable_scope = vs

  @snt.experimental.reuse_vars
  def reusable_var(self):
    return tf.get_variable("a", shape=[1])

obj = Reusable("reusable")
a1 = obj.reusable_var()
a2 = obj.reusable_var()
# a1 == a2


class NaiveAutoEncoder(snt.AbstractModule):
  def __init__(self, n_latent, n_out, name="naive_auto_encoder"):
    super(NaiveAutoEncoder, self).__init__(name)
    self._n_latent = n_latent
    self._n_out = n_out

  @experimental.reuse_vars
  def encode(self, input):
    """Builds the front half of AutoEncoder, inputs -> latents."""
    w_enc = tf.get_variable("w_enc", shape=[self._n_out, self._n_latent])
    b_enc = tf.get_variable("b_enc", shape=[self._n_latent])
    return tf.sigmoid(tf.matmul(input, w_enc) + b_enc)

  @experimental.reuse_vars
  def decode(self, latents):
    """Builds the back half of AutoEncoder, latents -> reconstruction."""
    w_rec = tf.get_variable("w_rec", shape=[self._n_latent, self._n_out])
    b_rec = tf.get_variable("b_rec", shape=[self._n_out])
    return tf.sigmoid(tf.matmul(latents, w_rec) + b_rec)

  def _build(self, input):
    """Builds the 'full' AutoEncoder, ie input -> latents -> reconstruction."""
    latents = self.encode(input)
    return self.decode(latents)


batch_size = 5
n_in = 10
n_out = n_in
n_latent = 2
nae = NaiveAutoEncoder(n_latent=n_latent, n_out=n_out)
inputs = tf.placeholder(tf.float32, shape=[batch_size, n_in])
latents = tf.placeholder(tf.float32, shape=[batch_size, n_latent])

# Connecting the default way calls build(), producing 'full' AutoEncoder.
reconstructed_from_input = nae(inputs)

# Connecting with one of the other methods might only require some subset of the
# variables, but sharing will still work.
reconstructed_from_latent = nae.decode(latents)
```

In the above example, any variables created by `obj.a()`, `obj.add_with_ab()` or
`obj.build()` exist in the same `tf.VariableScope`. In addition, since each
decorated method has its own `reuse` flag we don't need to worry about having to
create all variables on the first method call, or about the calling order at
all. We can even nest (decorated) methods within other (decorated) methods - the
`reuse` flag is always set correctly since the variable scope is re-entered for
every method call.

However every decorated method must be the *sole* owner of its variables. For
example, if we use `tf.get_variable("a", ...)` inside `obj.build()`, this will
*not* do variable sharing. Instead, TensorFlow will treat
`tf.get_variable("a", ...)` in `obj.a()` and `obj.build()` as separate variables
and an error will occur in either`obj.a()` or `obj.build()` (whichever is
called second).

See below for an example of bad variable reuse:

```python
class BadReusable(object):

  def __init__(self, name):
    with tf.variable_scope(name) as vs:
      self.variable_scope = vs

  @snt.experimental.reuse_vars
  def reusable_var(self):
    return tf.get_variable("a", shape=[1])

  @snt.experimental.reuse_vars
  def another_reusable_var(self):
    return tf.get_variable("a", shape=[1])

obj = BadReusable("bad_reusable")
obj.reusable_var()
obj.another_reusable_var()  # Raises a ValueError because `reuse=False`
```

## Wrapping functions into Sonnet modules using `snt.Module`
Whilst the recommended way of defining new Sonnet modules is to inherit from
`snt.AbstractModule`, the library also offers an alternative route to succinctly
instantiate modules wrapping user-provided functions.

The `snt.Module` class constructor takes a callable and returns a Sonnet module.
The provided function is invoked when the module is called, thus specifying how
new nodes are added to the computational graph and how to compute output Tensors
from input Tensors. Please refer to the module documentation
for more details and examples.

## FAQ

### Q: Why _another_ TF library?

A: The existing libraries were judged insufficiently flexible for the DeepMind
use case where extensive use is made of weight sharing. Making
everything use `tf.make_template`, and therefore support weight sharing from the
start, seemed to have sufficient benefits to outweight the development cost. The
paradigm of separating configuration from connection also allows easy
composability of modules.

### Q: Can I access different variables on subsequent calls to the same build()?

A: No. This is enforced by `tf.make_template`, which considers it an error to
access different / extra variables on subsequent calls.

### Q: What if I mistakenly give two modules the same name?

A: Modules which appear to be constructed with the same name will have distinct
names, and variable scopes. Under the hood, Sonnet uses `tf.make_template` which
essentially wraps a python function together with some `tf.VariableScope`,
ensuring that every call to the function happens in the same scope, and that all
calls after the first are set to reuse variables. One feature of the templating
is that it will `uniquify` any provided names, if they have already been entered
in the same scope. For example:

```python
lin_1 = snt.Linear(output_size=42, name="linear")
lin_2 = snt.Linear(output_size=84, name="linear")  # this name is already taken.

print(lin_1.name)  # prints "linear"
print(lin_2.name)  # prints "linear_1" - automatically uniquified.
```

Note that the .name property is available to see the "post-uniquification" name.

### Q: Do I _have_ to name my modules?

A: No. Modules have a default name, which should be the class name in
`snake_case`, and that will be used as the name with uniquification (see above)
if necessary. However, we recommend providing a name for modules which contain
variables, as the name provided becomes the name of the internal scope, and thus
defines the variable names. Most modules are written to declare internal weights
with names like `"w"` and `"b"` for weights and
bias - it's vastly preferable to do have a list of weights like:

```
sdae/encoder_linear/w
sdae/encoder_linear/b
sdae/decoder_linear/w
sdae/decoder_linear/b
```

rather than:

```
sdae/linear/w
sdae/linear/b
sdae/linear_1/w
sdae/linear_1/b
```

The names you choose will appear in TensorBoard.

### Q: How do I find out what variables are declared by a module?

A: You can query a module to find out all the variables in its scope using
the `get_variables()` method. Note that this will throw an error if the
module is not connected into the graph, as the variables do not exist at that
point so the relevant scope will be empty.

### Q: Should I be putting calls to `variable_scope` in my code?

A: Every module implicitly creates an internal variable_scope, which it
re-enters each time it connects to the graph. Assuming all the variables in your
model are inside Sonnet modules, it is not necessary to use scopes yourself.

### Q: Can I mix this with raw TF ops?

A: Yes. An op which doesn't declare variables internally, and so is effectively
a pure function, can be used inside module `_build` implementations, and also
to plumb together values between modules.

### Q: Should everything in Sonnet be implemented as a Module?

No, computations which do not create `tf.Variable`s and do not store internal
configurations *can* be implemented in the regular TF Op style, ie a python
function that receives input tensors, keyword arguments, and returns tensor
outputs.

If an op is going to create variables (ie call `tf.get_variable` anywhere
internally, including indirectly) it must be implemented as a subclass of
`snt.AbstractModule` so that variable sharing is correctly handled.

Note that if a computation doesn't create any Variables, it _may_ still be
desirable to implement it with a Module instead of an Op.

Aside from variable sharing, it may be convenient to use Sonnet Modules in cases
where we wish to attach configuration parameters to an op. An example of this is
the [content addressing](
modules in the Differentiable Neural Computer.
These modules receive a number of configuration parameters
(size of each word in memory, number of read heads) and some function of these
inputs defines what the valid input size is. We use a `snt.Linear` of the
correct output size before this module, in order to provide the correct
dimensionality. As a module this is easy - provide the configuration at
construction time, then a method `.param_size()` which gives the required
input dimensionality. We can then make the correct size of input tensor
and perform the connection.

```python
class CosineWeights(snt.AbstractModule):
  def __init__(self, word_size, num_heads, name="cosine_weights"):
    super(CosineWeights, self).__init__(name=name)
    self._word_size = word_size
    self._num_heads = num_heads
  def param_size(self):
    """Returns the size the 2nd dimension of `cos_params` is required to be."""
    return self._num_heads * (1 + self._word_size)
  def _build(self, memory, cos_params):
    """cos_params must be `[batch_size, self.param_size()]` shape"""
    # ...

# Construct the module, then work out the right input size
cos_weights_mod = CosineWeights(word_size=32, num_heads=3)
cosine_params_mod = snt.Linear(output_size=cos_weights_mod.param_size())

cos_params = cosine_params_mod(inputs)  # We know this is now the right shape.
weights = cos_weights_mode(memory, cos_params)
```

If the above was implemented as an op `cosine_weights(memory, cos_params,
word_size, num_heads)` then the logic to indicate the desired size of
`cos_params` would have to be stored in a separate function. Encapsulating the
related functions into one module results in cleaner code.

Another example of where this flexibility is useful is when an Op has a large
number of arguments which are conceptually configuration, along with some which
are conceptually inputs. We often want to use the same configuration in multiple
places, for different inputs, and so writing a Module which can be constructed
with the configuration and then passed around may be useful.

```python
import functools
# 1. Define our computation as some op
def useful_op(input_a, input_b,
              use_clipping=True, remove_nans=False, solve_agi='maybe'):
  # ...

# 2a). Set the configuration parameters with functools, then pass around.
useful_op_configured = functools.partial(
    useful_op,
    use_clipping=False,
    remove_nans=True,
    solve_agi='definitely')
do_something_a(... , inner_op=useful_op_configured)
do_something_else_a(..., inner_op=useful_op_configured)

# 2b). OR, set the configuration by creating kwargs and pass around both.
op_kwargs = {
    'use_clipping': False,
    'remove_nans': True,
    'solve_agi': 'definitely',
}
do_something_b(..., inner_op=useful_op, inner_op_kwargs=op_kwargs)
do_something_else_b(..., inner_op=useful_op, inner_op_kwargs=op_kwargs)
```

Either of the above approaches is valid, but many users dislike the style
of using functools or needing to pass both the Op and a dictionary around.
In which case, rewriting the Op as a Module can be a nice solution - in
particular, the difference between configuration parameters and inputs from
the Graph are now made explicit:

```python
class UsefulModule(snt.AbstractModule):
  def __init__(self, use_clipping=True, remove_nans=False,
               solve_agi='maybe', name='useful_module')
    super(UsefulModule, self).__init__(name=name)
    self._use_clipping = use_clipping
    self._remove_nans = remove_nans
    self._solve_agi = solve_agi
  def _build(self, input_a, input_b):
    #...
```

### Q: Can I mix this with other high level TF APIs, eg TF Slim?

A: Sonnet modules, once constructed, follow the Tensor-In-Tensor-Out principle,
so can be mixed with functions from TF-Slim, etc. Note that this may lead to
unexpected behaviour - TF-Slim controls sharing by passing explicit `scope=`
and `reuse=` kwargs into the layer functions - if you use a TF-Slim layer inside
the `_build()` method of a Sonnet module, then calling it multiple times is not
likely to work correctly.

### Q: Shouldn't I be overriding \_\_call\_\_ in modules?

A: No. `AbstractModule.__init__` provides an implementation of `__call__`,
which calls an internal function wrapped in a Template, which in turn wraps the
`_build` method. Overriding `__call__` yourself will likely break variable
sharing.

### Q: What is the overhead of using Sonnet vs other libraries vs raw TF?

A: None. Sonnet is only involved when constructing the computation graph.
Once you are at the stage of using `Session.run()` you are simply executing Ops,
without regard for what library was used to put that graph together.

### Q: How do I list all the variables which are used _in any way_ in a Module?

A: Currently, not easily possible. Although there is a `get_variables()` method,
it only searches the `VariableScope` defined inside a module, which
will contain any internally constructed variables or modules. However, the
actual _computation_ done by a module could use other modules - for
example, the `snt.Sequential` module in the example section above. The modules
passed into the constructor have by definition been constructed before the
`Sequential`, and so they have different variable scopes. Currently, once
the Sequential is connected into the graph, querying it with
`get_variables()` will return an empty tuple.

The DeepMind Research Engineering team is considering future additions to the
`Module` API which remedy this, without requiring extra effort from module
implementors.
