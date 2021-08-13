---
title: numpy
date: 2021-07-09 14:37:45
tags:
categories: python
mathjax: true
---

本文主要是对于python库`numpy`的学习使用笔记。

参考网站：

[numpy官方网站](https://numpy.org/)

[numpy中文网](https://numpy.org.cn/)

<!--more -->



# numpy是啥

NumPy是Python中科学计算的基础包。它是一个Python库，提供多维数组对象，各种派生对象（如掩码数组和矩阵），以及用于数组快速操作的各种API，有包括数学、逻辑、形状操作、排序、选择、输入输出、离散傅立叶变换、基本线性代数，基本统计运算和随机模拟等等。

`numpy` 的核心是它的 `ndarray` 数组对象，它比python原生数组更加高效，支持更多操作。

其两个特征：矢量化和广播。通过预编译的C代码达到快速。



# 数据结构

`ndarray` 是一个元素表（通常是数字），所有类型都相同，由非负整数元组索引。在NumPy维度中称为 *轴* 。

## 数组属性

它有很多重要属性，通过一个例子：

``` python
>>> import numpy as np
>>> a = np.arange(15).reshape(3, 5)
>>> a
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
>>> a.shape # 数组的维度
(3, 5)
>>> a.ndim # 数组的维度的个数
2
>>> a.dtype.name # 描述数组元素的类型
'int64'
>>> a.itemsize # 每个元素的字节大小
8
>>> a.size # 数组元素个数，维度的乘积
15
>>> type(a)
<type 'numpy.ndarray'>
>>> b = np.array([6, 7, 8])
>>> b
array([6, 7, 8])
>>> type(b)
<type 'numpy.ndarray'>
```



## 数组创建

### 直接从python元组创建

一定要是传入一个列表！

``` python
>>> a = np.array([2,3,4])
>>> b = np.array([(1.5,2,3), (4,5,6)])
>>> b
array([[ 1.5,  2. ,  3. ],
       [ 4. ,  5. ,  6. ]])
>>> c = np.array( [ [1,2], [3,4] ], dtype=complex ) # 也可以显式指定类型
>>> c
array([[ 1.+0.j,  2.+0.j],
       [ 3.+0.j,  4.+0.j]])
```



### numpy提供初始化一些函数

通常，数组的元素最初是未知的，但它的大小是已知的。因此，NumPy提供了几个函数来创建具有初始占位符内容的数组。这就减少了数组增长的必要，因为数组增长的操作花费很大。

函数`zeros`创建一个由0组成的数组，函数 `ones`创建一个完整的数组，函数`empty` 创建一个数组，其初始内容是随机的，取决于内存的状态。默认情况下，创建的数组的dtype是 `float64` 类型的。

``` python
>>> np.zeros((3,4))
array([[ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.]])
>>> np.ones( (2,3,4), dtype=np.int16 )                # dtype can also be specified
array([[[ 1, 1, 1, 1],
        [ 1, 1, 1, 1],
        [ 1, 1, 1, 1]],
       [[ 1, 1, 1, 1],
        [ 1, 1, 1, 1],
        [ 1, 1, 1, 1]]], dtype=int16)
>>> np.empty( (2,3) )                                 # uninitialized, output may vary
array([[  3.73603959e-262,   6.02658058e-154,   6.55490914e-260],
       [  5.30498948e-313,   3.14673309e-307,   1.00000000e+000]])
```

`numpy` 提供了一个类似于`range`的函数，该函数返回数组而不是列表。

``` python
>>> np.arange( 0, 2, 0.3 )                 # it accepts float arguments
array([ 0. ,  0.3,  0.6,  0.9,  1.2,  1.5,  1.8])
```

当`arange`与浮点参数一起使用时，由于有限的浮点精度，通常不可能预测所获得的元素的数量。出于这个原因，通常最好使用`linspace`函数来接收我们想要的元素数量的函数，而不是步长（step）：

``` python
>>> from numpy import pi
>>> np.linspace( 0, 2, 9 )                 # 9 numbers from 0 to 2
array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,  1.75,  2.  ])
>>> x = np.linspace( 0, 2*pi, 100 )        # useful to evaluate function at lots of points
>>> f = np.sin(x)
```



## 基本操作

数组上的操作会作用到每个元素上。

``` python
>>> a = np.array( [20,30,40,50] )
>>> b = np.arange( 4 )
>>> b
array([0, 1, 2, 3])
>>> c = a-b
>>> c
array([20, 29, 38, 47])
>>> b**2
array([0, 1, 4, 9])
>>> 10*np.sin(a)
array([ 9.12945251, -9.88031624,  7.4511316 , -2.62374854])
>>> a<35
array([ True, True, False, False])
```

需要注意的就是，乘法运算符 `*` 是逐元素对应操作而不是矩阵相乘。矩阵乘积可以用 `@` 运算符。

``` python
>>> A = np.array( [[1,1],
...             [0,1]] )
>>> B = np.array( [[2,0],
...             [3,4]] )
>>> A * B                       # elementwise product
array([[2, 0],
       [0, 4]])
>>> A @ B                       # matrix product
array([[5, 4],
       [3, 4]])
```



- ndarray 乘法操作的一些探究

事实上我在向量化操作的时候遇到了挺多问题，有一些模糊的地方，在此测试阐明一些东西。

首先 `numpy` 中是有 `matrix` 类型的，主要用于线性代数中，不过在绝大部分场合 `ndarray` 可以完全替代 `matrix`，所以我们决定基本弃用 `matrix`。

- 对于最简单的向量，也就是一个长度为 n 的 array。
  - 其转置始终是它自己，`*` 是逐元素相乘，返回相同长度的向量；
  - `@` 是矩阵乘法，返回向量内积也就是一个数。
  - 这也就要求两个向量的长度相同。
  - `A @ v` 将 `v` 视为列向量，而 `v @ A` 将 `v` 视为行向量。这可以节省键入许多转置。

``` python
>>> a
array([0, 1, 2, 3, 4])
>>> b
array([5, 6, 7, 8, 9])
>>> a * b
array([ 0,  6, 14, 24, 36])
>>> a @ b
80
>>> c = np.array([10, 20])
>>> a * c
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: operands could not be broadcast together with shapes (5,) (2,) 
>>> a @ c
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 2 is different from 5)

```

当我们尝试多维向量时，就可以把这个 `ndarray` 看作矩阵来用了，$N \times 1$ 和 $1 \times N$ 和 $N$ 都是不一样的 array。

`*` 其实是一个广播操作，两个矩阵不一定是要 $m \times k$ 和 $k \times n$，但也不是随意的。

``` python
>>> A
array([[0, 1],
       [2, 3]])
>>> B
array([4])
>>> A * B
array([[ 0,  4],
       [ 8, 12]])
# 广播到每个元素
>>> C 
array([4, 5])
>>> A * C
array([[ 0,  5],
       [ 8, 15]])
# 逐元素相乘
>>> D 
array([4, 5, 6])
>>> A * D
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: operands could not be broadcast together with shapes (2,2) (3,) 
# 广播不了了

>>> E
array([[4],
       [5]])
>>> A * E
array([[ 0,  4],
       [10, 15]])
>>> F
array([[4],
       [5],
       [6]])
>>> A * F
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: operands could not be broadcast together with shapes (2,2) (3,1) 
```

所以可以看出来其实是要符合**广播的规则**，所相乘的矩阵行或列的个数要么是1（广播到每一个元素），要么和原矩阵行列元素个数相等（逐元素操作）。这套规则其实不止是乘法，`+-/` 都是一样的。

- 当 `ndarray` 多维时，可以作为矩阵，用 `@` 进行运算，这时候就不会帮你自动转置了，要符合矩阵相乘条件也就是 $(m,k) \times (k,n)$。

``` python
>>> G
array([[4, 6],
       [5, 7]])
>>> A @ G
array([[ 5,  7],
       [23, 33]])

>>> H
array([[4, 7],
       [5, 8],
       [6, 9]])
>>> A @ H
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 3 is different from 2)
```



`ndarray` 还提供了很多一元操作

``` python
>>> a.sum()
2.5718191614547998
>>> a.min()
0.1862602113776709
>>> a.max()
0.6852195003967595
# 可以指定轴
>>> b
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>>
>>> b.sum(axis=0)                            # sum of each column
array([12, 15, 18, 21])
>>>
>>> b.min(axis=1)                            # min of each row
array([0, 4, 8])
>>>
>>> b.cumsum(axis=1)                         # cumulative sum along each row
array([[ 0,  1,  3,  6],
       [ 4,  9, 15, 22],
       [ 8, 17, 27, 38]])
```

一些**通函数**

``` python
array([0, 1, 2])
>>> np.exp(B)
array([ 1.        ,  2.71828183,  7.3890561 ])
>>> np.sqrt(B)
array([ 0.        ,  1.        ,  1.41421356])
>>> C = np.array([2., -1., 4.])
>>> np.add(B, C)
array([ 2.,  0.,  6.])
```

和python原生数组一样可以索引，切片和迭代。**多维**的数组每个轴可以有一个索引。这些索引以逗号分隔的元组给出。当提供的索引少于轴的数量时，缺失的索引被认为是完整的切片。

``` python
>>> b
array([[ 0,  1,  2,  3],
       [10, 11, 12, 13],
       [20, 21, 22, 23],
       [30, 31, 32, 33],
       [40, 41, 42, 43]])
>>> b[2,3]
23
>>> b[0:5, 1]                       # each row in the second column of b
array([ 1, 11, 21, 31, 41])
>>> b[ : ,1]                        # equivalent to the previous example
array([ 1, 11, 21, 31, 41])
>>> b[1:3, : ]                      # each column in the second and third row of b
array([[10, 11, 12, 13],
       [20, 21, 22, 23]])
>>> b[-1]                                  # the last row. Equivalent to b[-1,:]
array([40, 41, 42, 43])
```

如果想要对数组中的每个元素执行操作，可以使用`flat`属性，该属性是数组的所有元素的迭代器：

``` python
>>> for element in b.flat:
...     print(element)
...
0
1
2
...
43
```



## 形状操纵

以下三个命令都返回一个修改后的数组，但不会更改原始数组：

``` python
>>> a
array([[ 2.,  8.,  0.,  6.],
       [ 4.,  5.,  1.,  1.],
       [ 8.,  9.,  3.,  6.]])
>>> a.shape
(3, 4)
>>> a.ravel()  # returns the array, flattened
array([ 2.,  8.,  0.,  6.,  4.,  5.,  1.,  1.,  8.,  9.,  3.,  6.])
>>> a.reshape(6,2)  # returns the array with a modified shape
array([[ 2.,  8.],
       [ 0.,  6.],
       [ 4.,  5.],
       [ 1.,  1.],
       [ 8.,  9.],
       [ 3.,  6.]])
# 如果在 reshape 操作中将 size 指定为-1，则会自动计算其他的 size 大小：
>>> a.T  # returns the array, transposed
array([[ 2.,  4.,  8.],
       [ 8.,  5.,  9.],
       [ 0.,  1.,  3.],
       [ 6.,  1.,  6.]])
>>> a.T.shape
(4, 3)
>>> a.shape
(3, 4)
```

 `ndarray.resize`]方法会修改数组本身:

``` python
>>> a.resize((2,6))
>>> a
array([[ 2.,  8.,  0.,  6.,  4.,  5.],
       [ 1.,  1.,  8.,  9.,  3.,  6.]])
```

数组堆叠

``` python
>>> a
array([[ 8.,  8.],
       [ 0.,  0.]])
>>> b
array([[ 1.,  8.],
       [ 0.,  4.]])
>>> np.vstack((a,b))
array([[ 8.,  8.],
       [ 0.,  0.],
       [ 1.,  8.],
       [ 0.,  4.]])
>>> np.hstack((a,b))
array([[ 8.,  8.,  1.,  8.],
       [ 0.,  0.,  0.,  4.]])
```

拆数组

``` python
>>> a
array([[ 9.,  5.,  6.,  3.,  6.,  8.,  0.,  7.,  9.,  7.,  2.,  7.],
       [ 1.,  4.,  9.,  2.,  2.,  1.,  0.,  6.,  2.,  2.,  4.,  0.]])
>>> np.hsplit(a,3)   # Split a into 3
[array([[ 9.,  5.,  6.,  3.],
       [ 1.,  4.,  9.,  2.]]), array([[ 6.,  8.,  0.,  7.],
       [ 2.,  1.,  0.,  6.]]), array([[ 9.,  7.,  2.,  7.],
       [ 2.,  2.,  4.,  0.]])]
>>> np.hsplit(a,(3,4))   # Split a after the third and the fourth column
[array([[ 9.,  5.,  6.],
       [ 1.,  4.,  9.]]), array([[ 3.],
       [ 2.]]), array([[ 6.,  8.,  0.,  7.,  9.,  7.,  2.,  7.],
       [ 2.,  1.,  0.,  6.,  2.,  2.,  4.,  0.]])]
```



## 拷贝和视图

简单分配不会复制数组对象或其数据。

``` python
>>> a = np.arange(12)
>>> b = a            # no new object is created
>>> b is a           # a and b are two names for the same ndarray object
True
>>> b.shape = 3,4    # changes the shape of a
>>> a.shape
(3, 4)
```

不同的数组对象可以共享相同的数据。该`view`方法创建一个查看相同数据的新数组对象。

``` python
>>> c = a.view()
>>> c is a
False
>>> c.base is a                        # c is a view of the data owned by a
True
>>> c.flags.owndata
False
>>>
>>> c.shape = 2,6                      # a's shape doesn't change
>>> a.shape
(3, 4)
>>> c[0,4] = 1234                      # a's data changes
>>> a
array([[   0,    1,    2,    3],
       [1234,    5,    6,    7],
       [   8,    9,   10,   11]])
```

切片数组会返回一个视图:

``` python
>>> s = a[ : , 1:3]     # spaces added for clarity; could also be written "s = a[:,1:3]"
>>> s[:] = 10           # s[:] is a view of s. Note the difference between s=10 and s[:]=10
>>> a
array([[   0,   10,   10,    3],
       [1234,   10,   10,    7],
       [   8,   10,   10,   11]])
```

深拷贝。该`copy`方法生成数组及其数据的完整副本。

``` python
>>> d = a.copy()                          # a new array object with new data is created
>>> d is a
False
>>> d.base is a                           # d doesn't share anything with a
False
>>> d[0,0] = 9999
>>> a
array([[   0,   10,   10,    3],
       [1234,   10,   10,    7],
       [   8,   10,   10,   11]])
```

有时，如果不再需要原始数组，则应在切片后调用 `copy`。





# np.random

## np.random.choice

`random.choice(a, size=None, replace=True, p=None)`

从一个给定1维向量生成一个随机样本。

- `a` 是一个list或整数，如果是整数则 `np.arrange(a)`，从这个范围中随机生成。
- `size` 是输出的形状。



# np.expand_dims()

`np.expand_dims(a, axis)`

扩展 array 的形状，在 axis 处插入一个新轴。

``` python
>>> x = np.array([1, 2])
>>> y = np.expand_dims(x, axis=0)
>>> y
array([[1, 2]])
>>> y.shape
(1, 2)
>>> y = np.expand_dims(x, axis=1)
y
>>> array([[1],
           [2]])
>>> y.shape
(2, 1)
>>> y = np.expand_dims(x, axis=(0, 1))
>>> y
array([[[1, 2]]])
>>> y = np.expand_dims(x, axis=(2, 0))
>>> y
array([[[1],
        [2]]])
```



# np.squeeze()

`np.squeeze(a, axis)`

删除 array 的一个轴。

``` python
>>> x = np.array([[[0], [1], [2]]])
>>> x.shape
(1, 3, 1)
>>> np.squeeze(x).shape
(3,)
>>> np.squeeze(x, axis=0).shape
(3, 1)
>>> np.squeeze(x, axis=1).shape
Traceback (most recent call last):
...
ValueError: cannot select an axis to squeeze out which has size not equal to one
>>> np.squeeze(x, axis=2).shape
(1, 3)
>>> x = np.array([[1234]])
>>> x.shape
(1, 1)
>>> np.squeeze(x)
>>> array(1234)  # 0d array
>>> np.squeeze(x).shape
()
>>> np.squeeze(x)[()]
1234
```

