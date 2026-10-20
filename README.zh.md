![Sonnet](https://sonnet.dev/images/sonnet_logo.png)

# Sonnet

[**官方文档**](https://sonnet.readthedocs.io/) | [**示例**](#示例)

**语言 (Language)**: [English](README.md) · **简体中文**

Sonnet 是一个构建在 [TensorFlow 2](https://www.tensorflow.org/) 之上的神经网络库，旨在为机器学习研究提供简洁、可组合的抽象。

# 简介

Sonnet 由 DeepMind 的研究人员设计与构建。它可用于构建面向各种不同用途（无监督/监督学习、强化学习等）的神经网络。我们发现它在 DeepMind 内部是一种非常成功的抽象范式，相信它也会对您有所助益！

具体而言，Sonnet 提供了一个围绕单一核心概念 `snt.Module` 构建的简洁而强大的编程模型。模块（Module）可以持有对参数、其他子模块以及对用户输入执行特定运算的方法的引用。Sonnet 内置了许多预定义模块（例如 `snt.Linear`、`snt.Conv2D`、`snt.BatchNorm`）以及一些预定义的模块网络（例如 `snt.nets.MLP`），同时也鼓励用户构建自己的自定义模块。

与许多框架不同，Sonnet 对于您**如何**使用模块采取了极为中立且非强制（unopinionated）的设计理念。各个模块被设计为自包含且彼此完全解耦。Sonnet 不捆绑内置的训练框架，我们鼓励用户根据自身需求构建专属的训练循环，或采用社区中已有的训练框架。

此外，Sonnet 的设计宗旨是简单易懂，我们的代码力求清晰且专注。凡是我们设定了默认值的地方（例如参数初始值的默认分布），我们都会尽量阐明背后的原因。

# 快速上手

## 示例

体验 Sonnet 最简单的方式是使用 Google Colab，它提供免费挂载 GPU 或 TPU 的在线 Python Notebook 环境。

- [使用多层感知机（MLP）预测 MNIST](https://colab.research.google.com/github/deepmind/sonnet/blob/v2/examples/mlp_on_mnist.ipynb)
- [在 MNIST 上训练一个小型生成对抗网络（GAN）](https://colab.research.google.com/github/deepmind/sonnet/blob/v2/examples/little_gan_on_mnist.ipynb)
- [使用 `snt.distribute` 进行分布式训练](https://colab.research.google.com/github/deepmind/sonnet/blob/v2/examples/distributed_cifar10.ipynb)

## 安装

首先安装 TensorFlow 2.0 和 Sonnet 2：

```shell
$ pip install tensorflow tensorflow-probability
$ pip install dm-sonnet
```

您可以运行以下代码来验证是否已正确安装：

```python
import tensorflow as tf
import sonnet as snt

print("TensorFlow version {}".format(tf.__version__))
print("Sonnet version {}".format(snt.__version__))
```

### 使用内置模块

Sonnet 随附了许多开箱即用的内置模块。例如，为了定义一个多层感知机（MLP），我们可以使用 `snt.Sequential` 模块按顺序调用一系列子模块，将前一个模块的输出作为下一个模块的输入；并使用 `snt.Linear` 与 `tf.nn.relu` 来定义具体的计算逻辑：

```python
mlp = snt.Sequential([
    snt.Linear(1024),
    tf.nn.relu,
    snt.Linear(10),
])
```

要使用该模块，我们需要“调用（call）”它。`Sequential` 模块（以及绝大多数模块）都定义了 `__call__` 方法，这意味着您可以直接通过模块实例名进行调用：

```python
logits = mlp(tf.random.normal([batch_size, input_size]))
```

获取模块中的所有参数也是一项非常常见的操作。Sonnet 中的大多数模块会在首次传入输入并被调用时延迟创建（lazy initialization）其参数（因为在大多数情况下，参数的形状取决于输入的形状）。Sonnet 模块提供了两个属性用于访问参数。

`variables` 属性会返回该模块所引用的**所有** `tf.Variable`：

```python
all_variables = mlp.variables
```

值得注意的是，`tf.Variable` 并非仅用于存储模型参数。例如，它们还用于保存 `snt.BatchNorm` 中统计指标的状态。在大多数情况下，用户获取模块变量是为了将其传递给优化器（Optimizer）进行梯度更新。此时，不可训练的变量（non-trainable variables）通常不应包含在该列表中，因为它们是通过其他机制更新的。TensorFlow 内置了将变量标记为“可训练（trainable，即模型参数）”与“不可训练（其他状态变量）”的机制。Sonnet 提供了专门的属性来收集模块中所有的可训练变量，这通常正是您需要传递给优化器的参数列表：

```python
model_parameters = mlp.trainable_variables
```

### 构建自定义模块

Sonnet 强烈鼓励用户通过继承 `snt.Module` 子类来定义自己的模块。下面我们从创建一个名为 `MyLinear` 的简单线性层（`Linear`）开始：

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

使用这个自定义模块非常简单：

```python
mod = MyLinear(32)
mod(tf.ones([batch_size, input_size]))
```

通过继承 `snt.Module`，您可以免费获得许多实用的特性。例如，默认实现的 `__repr__` 会自动展示构造函数的参数（对调试和自省非常有帮助）：

```python
>>> print(repr(mod))
MyLinear(output_size=10)
```

您还可以直接获得 `variables` 和 `trainable_variables` 属性：

```python
>>> mod.variables
(<tf.Variable 'my_linear/b:0' shape=(10,) ...)>,
 <tf.Variable 'my_linear/w:0' shape=(1, 10) ...)>)
```

您可能会注意到上述变量名带有 `my_linear` 前缀。这是因为每当调用 Sonnet 模块的方法时，模块都会自动进入其对应的名称作用域（name scope）。通过进入模块名称作用域，我们能为 TensorBoard 等可视化工具提供结构更加清晰的计算图（例如，`my_linear` 内部发生的所有运算都会归入名为 `my_linear` 的分组中）。

此外，您的模块还将自动支持 TensorFlow 检查点（Checkpointing）与 SavedModel 导出功能，下文将介绍这些高级特性。

# 序列化（Serialization）

Sonnet 支持多种序列化格式。我们支持的最简单格式是 Python 标准库的 `pickle`，所有内置模块均经过测试，确保可以在同一个 Python 进程中通过 `pickle` 保存和加载。但总体而言，我们**不建议**使用 `pickle`，因为 TensorFlow 的许多组件对其支持不佳，且根据我们的经验，这种方式往往相当脆弱。

## TensorFlow 检查点（Checkpointing）

**参考文档：** https://www.tensorflow.org/alpha/guide/checkpoints

TensorFlow 检查点（Checkpointing）可用于在训练过程中定期保存参数的值。这对于保存训练进度非常有用，以防程序意外崩溃或被中断。Sonnet 的设计能够与 TensorFlow Checkpointing 无缝配合：

```python
checkpoint_root = "/tmp/checkpoints"
checkpoint_name = "example"
save_prefix = os.path.join(checkpoint_root, checkpoint_name)

my_module = create_my_sonnet_module()  # 可以是任何继承自 snt.Module 的模块。

# `Checkpoint` 对象负责管理传入其构造函数的对象所关联的 TensorFlow 状态检查点。
# 注意：Checkpoint 支持“创建时恢复（restore on create）”，这意味着在从检查点恢复之前，
# `my_module` 的变量**无需**预先创建（它们的值会在变量首次创建时自动恢复）。
checkpoint = tf.train.Checkpoint(module=my_module)

# 大多数训练脚本都会希望在存在检查点时从中恢复状态。例如当您中断了训练
# （如临时将 GPU 用于其他任务，或在云环境中实例被抢占）时，这一机制非常实用。
latest = tf.train.latest_checkpoint(checkpoint_root)
if latest is not None:
  checkpoint.restore(latest)

for step_num in range(num_steps):
  train(my_module)

  # 在训练过程中，我们会定期保存权重的值。注意这是一个阻塞调用，速度可能较慢
  # （通常是因为正在写入机器上速度较慢的存储介质）。如果您的运行环境较为稳定，
  # 可以适当降低保存频率。
  if step_num and not step_num % 1000:
    checkpoint.save(save_prefix)

# 请务必保存最终的权重值！！
checkpoint.save(save_prefix)
```

## TensorFlow SavedModel

**参考文档：** https://www.tensorflow.org/alpha/guide/saved_model

TensorFlow SavedModel 可用于保存一份与构建它的 Python 源代码完全解耦的网络副本。这是通过同时保存描述计算过程的 TensorFlow 计算图（Graph）以及包含权重数值的检查点来实现的。

要创建 SavedModel，第一步是实例化并初始化您想要保存的 `snt.Module`：

```python
my_module = snt.nets.MLP([1024, 1024, 10])
my_module(tf.ones([1, input_size]))
```

接下来，我们需要创建另一个模块，用来描述模型中我们具体想要导出的部分。我们建议采用这种方式（而不是直接原地修改原始模型），以便您能够细粒度地控制实际导出的内容。这对于避免生成体积过大的 SavedModel 尤为重要，同时也能让您仅分享模型中希望公开的部分（例如，对于 GAN 模型，您可能只想导出生成器供他人使用，而将判别器保持私有）。

```python
@tf.function(input_signature=[tf.TensorSpec([None, input_size])])
def inference(x):
  return my_module(x)

to_save = snt.Module()
to_save.inference = inference
to_save.all_variables = list(my_module.variables)
tf.saved_model.save(to_save, "/tmp/example_saved_model")
```

现在，`/tmp/example_saved_model` 文件夹中就已经生成了保存好的模型：

```shell
$ ls -lh /tmp/example_saved_model
total 24K
drwxrwsr-t 2 tomhennigan 154432098 4.0K Apr 28 00:14 assets
-rw-rw-r-- 1 tomhennigan 154432098  14K Apr 28 00:15 saved_model.pb
drwxrwsr-t 2 tomhennigan 154432098 4.0K Apr 28 00:15 variables
```

加载该模型非常简单，并且可以在另一台完全没有构建该模型的 Python 源代码的机器上独立完成：

```python
loaded = tf.saved_model.load("/tmp/example_saved_model")

# 调用 inference 方法。注意：这里不会运行 `to_save` 中的 Python 代码，
# 而是直接执行 SavedModel 中包含的 TensorFlow 计算图。
loaded.inference(tf.ones([1, input_size]))

# 可以使用 all_variables 属性来获取恢复后的变量。
assert len(loaded.all_variables) > 0
```

请注意，加载后的对象并不是一个 Sonnet 模块，而是一个容器对象（container object），它持有我们在上一步中显式添加的特定方法（如 `inference`）和属性（如 `all_variables`）。

## 分布式训练（Distributed training）

**示例：** https://github.com/deepmind/sonnet/blob/v2/examples/distributed_cifar10.ipynb

Sonnet 支持使用[自定义 TensorFlow 分布式策略（distribution strategies）](https://sonnet.readthedocs.io/en/latest/api.html#module-sonnet.distribute)进行分布式训练。

Sonnet 与使用 `tf.keras` 进行分布式训练的一个核心区别在于：Sonnet 模块和优化器在分布式策略下运行时**不会**隐式改变自身行为（例如，我们不会在底层自动对梯度求平均，也不会自动同步 BatchNorm 的统计量）。我们认为，用户应当对训练过程中的这些细节拥有完全的掌控权，而不应将它们黑盒化地硬编码在库内部。相应的权衡是，您需要在训练脚本中显式实现这些逻辑（通常只需在应用优化器之前加上两行代码，对梯度执行 all-reduce 规约），或者显式替换为具有分布式感知能力的模块（例如 `snt.distribute.CrossReplicaBatchNorm`）。

我们的 [分布式 CIFAR-10 示例](https://github.com/deepmind/sonnet/blob/v2/examples/distributed_cifar10.ipynb)详细演示了如何使用 Sonnet 进行多 GPU 分布式训练。

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月17日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
