---
title: tensorflow
date: 2021-08-11 02:03:58
tags:
categories: python
mathjax: true
---

本篇是对python库 `tensorflow` 的学习笔记。

[tensorflow官方网站](https://www.tensorflow.org/)

<!--more -->

# tensorflow 是啥

**TensorFlow**是一个开源软件库，用于各种感知和语言理解任务的机器学习。



# 数据结构

## 张量

`tensorflow` 最重要的数据类型就是**张量** `tf.tensor`，是具有统一类型的多维数组，其实和 `numpy.arrays` 和相似。

可以用 `tf.constant()` 输入一个 list 来创建一个张量。

张量有一些基本数学运算，和 `numpy` 差不多，不赘述。

``` python
a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]]) # Could have also said `tf.ones([2,2])`

print(a + b, "\n") # element-wise addition
print(a * b, "\n") # element-wise multiplication
print(a @ b, "\n") # matrix multiplication

# Find the largest value
print(tf.reduce_max(c))
# Find the index of the largest value
print(tf.argmax(c))
# Compute the softmax
print(tf.nn.softmax(c))
```



张量有形状。下面是几个相关术语：

- **形状**：张量的每个维度的长度（元素数量）。
- **秩**：张量的维度数量。标量的秩为 0，向量的秩为 1，矩阵的秩为 2。
- **轴**或**维度**：张量的一个特殊维度。
- **大小**：张量的总项数，即乘积形状向量

索引，形状操作，广播等也不赘述。



## 变量

**变量** `tf.Variable` 是用于表示程序处理的共享持久状态的推荐方法。变量通过 `tf.Variable` 类进行创建和跟踪`tf.Variable`。表示张量，对它执行运算可以改变其值。利用特定运算可以读取和修改此张量的值。更高级的库（如`tf.keras`）使用 `tf.Variable` 来存储模型参数。

可以用 `tf.Variable()` 来初始化一个变量，都和 `tf.constant()` 很相似。

可以使用 `tf.Variable.assign` 重新分配张量。调用 `assign`（通常）不会分配新张量，而会重用现有张量的内存。

虽然变量对微分很重要，但某些变量不需要进行微分。在创建时，通过将 `trainable` 设置为 False 可以关闭梯度。

为了提高性能，TensorFlow 会尝试将张量和变量放在与其 `dtype` 兼容的最快设备上。这意味着如果有 GPU，那么大部分变量都会放置在 GPU 上。



## 自动微分

这也是 `tensorflow` 最重要的功能之一，它使得神经网络不需要手动计算梯度来更新参数。

TensorFlow 为自动微分提供了 [`tf.GradientTape`](https://tensorflow.google.cn/api_docs/python/tf/GradientTape?hl=zh_cn) API ，根据某个函数的输入变量来计算它的导数。Tensorflow 会把 'tf.GradientTape' 上下文中执行的所有操作都记录在一个磁带上 ("tape")。 然后基于这个磁带和每次操作产生的导数，用反向微分法（"reverse mode differentiation"）来计算这些被“记录在案”的函数的导数。

``` python
x = tf.ones((2, 2))

with tf.GradientTape() as t:
  t.watch(x)
  y = tf.reduce_sum(x)
  z = tf.multiply(y, y)

# Derivative of z with respect to the original input tensor x
dz_dx = t.gradient(z, x)
for i in [0, 1]:
  for j in [0, 1]:
    assert dz_dx[i][j].numpy() == 8.0
    
# Use the tape to compute the derivative of z with respect to the
# intermediate value y.
dz_dy = t.gradient(z, y)
assert dz_dy.numpy() == 8.0
```

默认情况下，调用 GradientTape.gradient() 方法时， GradientTape 占用的资源会立即得到释放。通过创建一个持久的梯度带 `tf.GradientTape(persistent=True)`，可以计算同个函数的多个导数。这样在磁带对象被垃圾回收时，就可以多次调用 'gradient()' 方法。



由于磁带会记录所有执行的操作，Python 控制流（如使用 if 和 while 的代码段）自然得到了处理。

``` python
def f(x, y):
  output = 1.0
  for i in range(y):
    if i > 1 and i < 5:
      output = tf.multiply(output, x)
  return output

def grad(x, y):
  with tf.GradientTape() as t:
    t.watch(x)
    out = f(x, y)
  return t.gradient(out, x)

x = tf.convert_to_tensor(2.0)

assert grad(x, 6).numpy() == 12.0
assert grad(x, 5).numpy() == 12.0
assert grad(x, 4).numpy() == 4.0
```



在 'GradientTape' 上下文管理器中记录的操作会用于自动微分。如果导数是在上下文中计算的，导数的函数也会被记录下来。因此，同个 API 可以用于高阶导数。

``` python
x = tf.Variable(1.0)  # Create a Tensorflow variable initialized to 1.0

with tf.GradientTape() as t:
  with tf.GradientTape() as t2:
    y = x * x * x
  # Compute the gradient inside the 't' context manager
  # which means the gradient computation is differentiable as well.
  dy_dx = t2.gradient(y, x)
d2y_dx2 = t.gradient(dy_dx, x)

assert dy_dx.numpy() == 3.0
assert d2y_dx2.numpy() == 6.0
```



# tf.keras

`keras` 是很经典的神经网络框架，也是很常用的，所以整理一下。

首先最重要的它有两个类 `Model` 和 `Sequential`，他们之间有很多联系和区别。简单来说 `Sequential` 只能往后叠层，`model` 可以搭建更复杂的模型。

[这里](https://stackoverflow.com/questions/66879748/what-is-the-difference-between-tf-keras-model-and-tf-keras-sequential)有一篇简述他们的区别的post。

## Model

有两种初始化一个 `Model` 的方法。

一个是函数式 API：

``` python
inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

一个是通过子类，可以在 `__init__` 中定义层，通过 `call` 来前向传播：

``` python
class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
    self.dropout = tf.keras.layers.Dropout(0.5)

  def call(self, inputs, training=False):
    x = self.dense1(inputs)
    if training:
      x = self.dropout(x, training=training)
    return self.dense2(x)

model = MyModel()
```

### call

```python
call(
    inputs, training=None, mask=None
)
```

进行一次前向传播。

### compile

```python
compile(
    optimizer='rmsprop', loss=None, metrics=None, loss_weights=None,
    weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs
)
```

设置训练模型。

常用参数：

- `optimizer`，使用的优化器
- `loss`，使用的损失函数
- `metrics`，在训练和测试中需要被计算的矩阵

### evaluate

```python
evaluate(
    x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None,
    callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False,
    return_dict=False, **kwargs
)
```

返回损失值和矩阵值，在测试模式下。计算是 in batches 的。

- `x`，输入数据
- `y`，目标数据
- `batch_size`，每次计算中每个 `batch ` 的样本数量

### fit

```python
fit(
    x=None, y=None, batch_size=None, epochs=1, verbose='auto',
    callbacks=None, validation_split=0.0, validation_data=None, shuffle=True,
    class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None,
    validation_steps=None, validation_batch_size=None, validation_freq=1,
    max_queue_size=10, workers=1, use_multiprocessing=False
)
```

训练模型。

会返回一个 `History` 对象，在每个epoch记录训练或验证损失和矩阵值的历史。

- `epochs`，迭代次数。

### get_layer

根据名字或编号获得一个层。

```python
get_layer(
    name=None, index=None
)
```

### predict

对于输入生成输出预测。这是对于大规模数据预测的，in batch 的，对于小规模可以用 `call`。是不计正则化的。

```python
predict(
    x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
    workers=1, use_multiprocessing=False
)
```

### summary

打印网络的框架。

```python
summary(
    line_length=None, positions=None, print_fn=None
)
```

然后有一个有关网络输出维数的一个理解，`summary()` 都会显示维数是 `(None, ...)`，[这里](https://stackoverflow.com/questions/47240348/what-is-the-meaning-of-the-none-in-model-summary-of-keras)有一个post 说的比较清楚，就是第一维其实是 batch_size。



## Sequential

整个训练、预测方法和 `Model` 差不多，但差别就在于 `Sequential` 只能简单叠层，一般通过 `add` 或直接在初始化列表中设置。

```python
# Optionally, the first layer can receive an `input_shape` argument:
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(8, input_shape=(16,)))
# Afterwards, we do automatic shape inference:
model.add(tf.keras.layers.Dense(4))
```

然后要注意的是不设置输入维度的话，这个模型就还没有建成（built），就不能 `summary` 等等，直到第一次训练或预测等，接收到了输入数据。或者用 `build` 进行手动延迟搭建。

### add

```python
add(
    layer
)
```

加一层。



# tf.data

这个主要是用来构建数据集管道的。

## Dataset

一个大数据集，提供高效的输入管道。符合几个模式：

1. 从输入数据创造源数据集。
2. 应用数据集转换对数据进行预处理。
3. 遍历数据集并处理元素。

迭代的以流的方式进行，所以完整的数据集不需要载入内存。

### as_numpy_iterator

返回一个数据集的 numpy 的迭代器。用来监视数据中的内容。

如果是查看数据集元素的形状和类型，直接打印元素就可以。



### batch

```python
batch(
    batch_size, drop_remainder=False, num_parallel_calls=None, deterministic=None
)
```

将数据集中连续的元素打包成 batch。



### list_files

``` python
@staticmethod
list_files(
    file_pattern, shuffle=None, seed=None
)
```

创建一个包含匹配该模式的所有文件的数据集。



### map

``` python
map(
    map_func, num_parallel_calls=None, deterministic=None
)
```

把 `map_func` 作用到数据集每一个元素上。



### take

``` python
take(
    count
)
```

从数据集中取出 `count` 个组成一个数据集。



## experimental

### Counter

``` python
tf.data.experimental.Counter(
    start=0, step=1, dtype=tf.dtypes.int64
)
```

创建一个从 `start` 以步长 `step` 计数的数据集。

