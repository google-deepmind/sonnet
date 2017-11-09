# Sonnet Changelog

## Version 1.14 - Thursday, 09. November 2017

This version requires TensorFlow version 1.4.0.

* Start of implementation for Eager Mode. This should be considered Alpha &
  unsupported.
* Switch parameterized tests to use Abseil.
* BatchApply passes through scalar non-Tensor inputs unmodified.
* More flexible mask argument to `Conv2D`.
* Added Sonnet `ModuleInfo` to the "sonnet" graph collection. This allows to
  keep track of which modules generated which connected sub-graphs. This
  information is serialised and available when loading a meta-graph-def. This
  can be used, for instance, to visualise the TensorFlow graph from a Sonnet
  perspective.
* Scale_gradient now handles all float dtypes.
* Fixed a bug in clip_gradient that caused clip values to be shared.
* ConvNet can now use the NCHW data format.
* Cleaned up and improved example text for `snt.custom_getters.Context`.


## Version 1.13 - Monday 25. Septebmer 2017

* Separate `BatchNormLSTM` and `LSTM` to two separate modules.
* Clarify example in Readme.

## Version 1.12 - Monday 18. September 2017

* `custom_getters` subpackage. This allows modules to be made non-trainable, or
  to completely block gradients. See documentation for `tf.get_variable` for
  more details.
* `Sequential.get_variables()` generates a warning to indicate that no
  variables will ever be returned.
* `ConvLSTM` now supports dilated convolutions.
* `utils.format_variables` allows logging Variables with non-static shape.
* `snt.trainable_initial_state` is now publicly exposed.
* Stop using private property of `tf.Graph` in `util.py`.

## Version 1.11 - Monday 21. August 2017

This version requires TensorFlow 1.3.0.

* *Backwards incompatible change*: Resampler ops removed, they are now available
  in `tf.contrib.resampler`.
* Custom getters supported in RNNs and AlexNet.
* Replace last references to `contrib.RNNCell` with `snt.RNNCore`.
* Removed Tensorflow dependencies in Bazel config files, which makes it
  unnecessary to have Tensorflow as a submodule of Sonnet.

## Version 1.10 - Monday, 14. August 2017

* First steps of AlexNet cleanup.
  * Add option to disable batch normalization on fully-connected layers.
  * Remove HALF mode.
  * Add AlexNetMini and AlexNetFull.
* Fixed bias compatibility between NHWC and NCHW data formats in Conv2D.
  Uses tf.nn.bias_add for bias addition in all convolutional layers.
* snt.BatchApply now also accepts scalar-valued inputs such as Boolean flags.

## Version 1.9 - Monday 7. August 2017

* Clean up and clarify documentation on nest's dict ordering behavior.
* Change installation instructions to use pip.

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

