# Copyright 2020 The Sonnet Authors. All Rights Reserved.
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
# ============================================================================
"""Functional optimizers."""

import collections
import functools
from typing import Callable, Type

from sonnet.src import base
from sonnet.src.functional import haiku
import tensorflow as tf
import tree

TransformedOptimizer = collections.namedtuple("TransformedOptimizer",
                                              ("init", "apply"))


def optimizer(cls: Type[base.Optimizer]) -> Callable[..., TransformedOptimizer]:
  """Converts a snt.Optimizer subclass into a functional optimizer.

  To wrap a Sonnet optimizer class simply pass it to :func:`optimizer`:

  >>> adam = snt.functional.optimizer(snt.optimizers.Adam)

  This will give you back a function that drives the constructor of the
  optimizer and returns a pair of functions that give you the optimizer state
  and a way to apply it:

  >>> optimizer = adam(learning_rate=0.01)

  NOTE: We provide convenience wrappers for the builtin optimizers so you can
  just use `opt = snt.functional.adam(learning_rate=0.01)` if you prefer:

  >>> optimizer = snt.functional.adam(learning_rate=0.01)

  To make this example useful lets create a simple network to test:

  >>> with snt.functional.variables():
  ...   net = snt.nets.MLP([100, 10])

  >>> def loss_fn(images, labels):
  ...   logits = net(images)
  ...   x_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
  ...                                                          labels=labels)
  ...   loss = tf.reduce_mean(x_ent)
  ...   return loss

  >>> loss_fn = snt.functional.transform(loss_fn)

  >>> x = tf.ones([1, 1])
  >>> y = tf.constant([1])
  >>> params = loss_fn.init(x, y)

  To get the initial state of our optimizer (e.g. m/v terms in Adam) we need to
  run the `optimizer.init` function:

  >>> opt_state = optimizer.init(params)

  Now we can run a single training step by taking gradients of our network and
  applying one step of our optimizer:

  >>> grad_apply_net = snt.functional.grad(loss_fn.apply)

  >>> def train_step(x, y, params, opt_state):
  ...   grads = grad_apply_net(params, x, y)
  ...   params, opt_state = optimizer.apply(opt_state, grads, params)
  ...   return params, opt_state

  Teach the network to always predict one:

  >>> target = tf.constant([1])
  >>> dataset = [(tf.random.normal([1, 1]), target) for _ in range(10)]
  >>> for x, y in dataset:
  ...   params, opt_state = train_step(x, y, params, opt_state)

  Args:
    cls: A :class:`~sonnet.Optimizer` subclass to functionalize.

  Returns:
    A transformed optimizer with `init` and `apply`. See docstring for details.
  """
  @functools.wraps(cls.__init__)
  def wrapper(*args, **kwargs):
    with haiku.variables():
      opt = cls(*args, **kwargs)  # pytype: disable=not-instantiable
    return _wrap_optimizer(opt)
  return wrapper


def _split_on_trainable(opt_state):
  trainable = {}
  non_trainable = {}
  for param_ref, value in opt_state.items():
    if param_ref.deref().trainable:
      trainable[param_ref] = value
    else:
      non_trainable[param_ref] = value
  return trainable, non_trainable


def _merge(a, b):
  """Merges two dictionaries and returns a new one."""
  c = dict(a)
  c.update(b)
  return c


def _wrap_optimizer(opt: base.Optimizer) -> TransformedOptimizer:
  """Returns a functional optimizer."""

  def init_opt_fn(params):
    """Creates initial optimizer state."""
    def f(params):
      params = [p.deref() for p in sorted(params.keys())]
      updates = [tf.zeros_like(p) for p in params]
      for p, zero in zip(params, updates):
        p.assign(zero)
      opt.apply(updates, params)

    f = haiku.transform_with_state(f)

    trainable, non_trainable = f.init(params)
    opt_state = _merge(
        {r: v for r, v in trainable.items() if r not in params},
        {r: v for r, v in non_trainable.items() if r not in params})

    return opt_state

  def apply_opt_fn(opt_state, updates, params):
    """Applies the optimizer and returns updated parameters and opt state."""
    def f(opt_state, params, updates):
      flat_params = [p.deref() for p in sorted(params)]
      updates = tree.flatten(updates)
      opt.apply(updates, flat_params)
      params = {r: r.deref().tensor_value for r in params}
      opt_state = {r: r.deref().tensor_value for r in opt_state}
      return params, opt_state

    f = haiku.transform_with_state(f)

    trainable_opt_state, non_trainable = _split_on_trainable(opt_state)
    trainable = _merge(params, trainable_opt_state)
    (params, opt_state), _ = f.apply(trainable, non_trainable,
                                     opt_state, params, updates)
    return params, opt_state

  return TransformedOptimizer(init=init_opt_fn, apply=apply_opt_fn)
