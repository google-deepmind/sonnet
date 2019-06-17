.. currentmodule:: sonnet

Modules
=======

:class:`Module` is the core abstraction provided by Sonnet.

By organising your code into :class:`Module` subclasses, it is easy to keep
track of variables and deal with common tasks such as locating model parameters
and checkpointing state. Module also helps with debugging, adding a
:tf:`name_scope` around each method, making tools like TensorBoard even more
useful.

Sonnet ships with many predefined modules (e.g. :class:`Linear`,
:class:`Conv2D`, :class:`BatchNorm`) and some predefined networks of modules
(e.g. :class:`nets.MLP`). If you can't find what you're looking for then we
encourage you to subclass :class:`Module` and implement your ideas.

Using built-in modules
----------------------

Using :doc:`built in modules <api>` is as easy as using any other Python object:

>>> linear = snt.Linear(output_size=10)
>>> linear(tf.ones([8, 28 * 28]))
<tf.Tensor: ... shape=(8, 10), dtype=float32, ... dtype=float32)>

You can get access to the modules parameters using the ``trainable_variables``
property, note that most modules only create parameters the first time they
are called with an input:

>>> linear.trainable_variables
(<tf.Variable 'linear/b:0' shape=(10,) ...>,
 <tf.Variable 'linear/w:0' shape=(784, 10) ...>)

Some modules contain references to other modules, Sonnet provides a convenient
way to find these referenced modules:

>>> mlp = snt.nets.MLP([1000, 10])
>>> mlp(tf.ones([1, 1]))
<tf.Tensor: ...>
>>> [s.name for s in mlp.submodules]
['linear_0', 'linear_1']

Writing your own modules
------------------------

To create your own module simply subclass :class:`Module` and implement
your logic. For example we can build our own simple multi-layer perceptron
module by reusing the built in :class:`Linear` modules and :tf:`nn.relu`
to add a non-linearity:

>>> class MyMLP(snt.Module):
...   def __init__(self, name=None):
...     super(MyMLP, self).__init__(name=name)
...     self.hidden1 = snt.Linear(1024, name="hidden1")
...     self.output = snt.Linear(10, name="output")
...
...   def __call__(self, x):
...     x = self.hidden1(x)
...     x = tf.nn.relu(x)
...     x = self.output(x)
...     return x

You can use your module like you would any other Python object:

>>> mlp = MyMLP()
>>> mlp(tf.random.normal([8, 28 * 28]))
<tf.Tensor: ... shape=(8, 10), ...>

Additionally, the variable and submodule tracking features of :class:`Module`
will work without any additional code:

>>> mlp.trainable_variables
(<tf.Variable 'my_mlp/hidden1/b:0' shape=(1024,) ...>,
 <tf.Variable 'my_mlp/hidden1/w:0' shape=(784, 1024) ...>,
 <tf.Variable 'my_mlp/output/b:0' shape=(10,) ...>,
 <tf.Variable 'my_mlp/output/w:0' shape=(1024, 10) ...>)
>>> mlp.submodules
(Linear(output_size=1024, name='hidden1'),
 Linear(output_size=10, name='output'))

It is often useful to defer some one-time initialization until your module is
first used. For example in a linear layer the shape of the weights matrix
depends on the input shape and the desired output shape.

Sonnet provides the :func:`once` dectorator that means a given method is
evaluated once and only once per instance, regardless of other arguments. For
example we can build a simple linear layer like so:

.. code-block:: python
  :emphasize-lines: 6-10,13

  class MyLinear(snt.Module):
    def __init__(self, output_size):
      super(MyLinear, self).__init__()
      self.output_size = output_size

    @snt.once
    def _initialize(self, inputs):
      input_size = inputs.shape[1]
      self.w = tf.Variable(tf.random.normal([input_size, self.output_size]))
      self.b = tf.Variable(tf.zeros([self.output_size]))

    def __call__(self, inputs):
      self._initialize(inputs)
      return tf.matmul(inputs, self.w) + self.b
