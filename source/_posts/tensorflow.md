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

