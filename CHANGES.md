# Sonnet Changelog

## Version 1.8 - Monday 31. July 2017

* Add optional bias for the multipler in AddBias.
* Push first version of wheel files to PyPI.

## Version 1.7 - Monday 24. July 2017

* Fix install script for Python 3.
* Better error message in AbstractModule.
* Fix out of date docs about RNNCore.
* Use tf.layers.utils instead of tf.contrib.layers.utils, allowing to remove
  the use of contrib in the future, which will save on import time.
* Fixes to docstrings.

## Version 1.6 - Monday, 17. July 2017

* Support "None" entries in BatchApply's inputs.
* Add `custom_getter` option to convolution modules and MLP.
* Better error messages for BatchReshape.

## Version 1.5 - Monday, 10. July 2017

* `install.sh` now supports relative paths as well as absolute.
* Accept string values as variable scope in `snt.get_variables_in_scope` and
  `snt.get_normalized_variable_map`.
* Add IPython notebook that explains how Sonnet's `BatchNorm` module can be
  configured.

## Version 1.4 - 3rd Jul 2017

* Added all constructor arguments to `ConvNet2D.transpose` and
  `ConvNet2DTranspose.transpose`.
* *Backwards incompatible change* is_training flags of `_build` functions no
  longer default to True. They must be specified explicitly at every connection
  point.
* Added causal 1D Convolution.
* Fixed to scope name utilities.
* Added `flatten_dict_items` to `snt.nest`.
* `Conv1DTranspose` modules can accept input with undefined batch sizes.
* Apply verification to output_shape in `ConvTranspose` modules.

## Version 1.3 - 26th Jun 2017

This version is only compatible with TensorFlow 1.2.0, not the current GitHub
HEAD.

* Resampler op now tries to import from tf.contrib first and falls back to the
Sonnet op. This is in preparation for the C++ ops to be moved into tf/contrib.
* `snt.RNNCore` no longer inherits from `tf.RNNCell`. All recurrent modules
will continue to be suppoted by `tf.dynamic_rnn`, `tf.static_rnn`, etc.
* The ability to add a `custom_getter` to a module is now supported by
`snt.AbstractModule`. This is currently only available in `snt.Linear`, with
more to follow. See the documentation for `tf.get_variable` for how to use
custom_getters.
* Documentation restructured.
* Some functions and tests reorganised.

## Version 1.2 - 19th Jun 2017

* Cell & Hidden state clipping added to `LSTM`.
* Added Makefile for generating documentation with Sphinx.
* Batch Norm options for `LSTM` now deprecated to a separate class
  `BatchNormLSTM`. A future version of `LSTM` will no longer contain the batch
  norm flags.
* `@snt.experimental.reuse_vars` decorator promoted to `@snt.reuse_variables`.
* `BatchReshape` now takes a `preserve_dims` parameter.
* `DeepRNN` prints a warning if the heuristic is used to infer output size.
* Deprecated properties removed from `AbstractModule`.
* Pass inferred data type to bias and weight initializers.
* `AlexNet` now checks that dropout is disabled or set to 1.0 when testing.
* `.get_saver()` now groups partitioned variables by default.
* Docstring, variable name and comment fixes.


## Version 1.1 - 12th Jun 2017

* **breaking change**: Calling `AbstractModule.__init__` with positional
arguments is now not supported. All calls to `__init__` should be changed to use
kwargs. This change will allow future features to be added more easily.
* Sonnet modules now throw an error if pickled. Instead of serializing module
instances, you should serialize the constructor you want to call, plus the
arguments you would pass it, and recreate the module instances in each run of
the program.
* Sonnet no longer allows the possibility that `self._graph` does not exist.
This would only be the case when reloading pickle module instances, which is not
supported.
* Fix tolerance on initializers_test.
* If no name is passed to the AbstractModule constructor, a snake_case version
of the class name will be used.
* `_build()` now checks that `__init__` has been called first and throws
an error otherwise.
* Residual and Skip connection RNN wrapper cores have been added.
* `get_normalized_variable_map()` now has an option to group partitioned
variables, matching what tf.Saver expects.
* `snt.BatchApply` now support kwargs, nested dictionaries, and allows `None` to
be returned.
* Add a group_sliced_variables option to get_normalized_variable_map() that
groups partitioned variables in its return value, in line with what tf.Saver
expects to receive. This ensures that partitioned variables end up being treated
as a unit when saving checkpoints / model snapshots with tf.Saver. The option is
set to False by default, for backwards compatibility reasons.
* `snt.Linear.transpose` creates a new module which now uses the same
partitioners as the parent module.
