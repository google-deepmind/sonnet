# Golden checkpoints

Golden checkpoints represent checkpoints generated from stable Sonnet code. We
have unit tests that ensure we don't introduce checkpoint breaking changes to
Sonnet.

To generate a new checkpoint first add an entry in `goldens.py` describing the
module you want to add. For example:

```python
@_register_golden(snt.Linear, "linear_32x64")
class Linear32x64(Golden):
  """Tests Linear without a bias."""

  def create_module(self):
    return snt.Linear(64)

  def forward(self, module):
    x = range_like(tf.TensorSpec([1, 32]))
    return module(x)

  def create_all_variables(self, module):
    self.forward(module)
    return module.w, module.b
```

Then run the `generate` binary to generate new golden checkpoints:

```shell
$ bazel run :generate -- --dry_run=false --golden_dir="$PWD" --alsologtostderr
```

At this point your golden checkpoint will be created and registered to run
whenever `goldens_test` runs:

```shell
$ bazel test :goldens_test
```

## Regenerating old checkpoints

WARNING: In general once a checkpoint is checked in it is only safe to
regenerate it if your module has zero users. If you are making an additive
change to a module (e.g. adding a new parameter) then consider making a new
checkpoint and ensure that you can load from both the old and new checkpoint.

If you absolutely need to regenerate the checkpoint and know what you're doing
then you can do so with:

```shell
$ bazel run :generate -- --dry_run=false --golden_dir="$PWD" --alsologtostderr --filter=my_checkpoint_name --regenerate
```
