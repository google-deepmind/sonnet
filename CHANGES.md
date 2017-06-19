# Sonnet Changelog


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

