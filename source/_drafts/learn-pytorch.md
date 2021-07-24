---
title: pytorch
categories: python
tags:
---



本章主要是对于用于非常有名的深度学习神经网络 python 库 `pytorch` 的学习使用笔记。

参考网站：

[pytorch官方网站](https://pytorch.org/tutorials/)



# 数据结构

`pytorch` 中最重要的就是它的数据结构 `tensor` （张量）了，它很类似 `nummpy` 中的 `ndarray`，但是有很多额外的特性，比如能在 GPU 上运算、可以自动求导等。



## 基本操作

因为和 `ndarray` 很类似，一些基本操作就不在此赘述。

``` python
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```

 

我们可以将一个 tensor 的运算都交到 GPU 上进行：

``` python
# We move our tensor to the GPU if available
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
```



``` python
t1 = torch.cat([tensor, tensor, tensor], dim=1)

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
```



`torch` 中在方法后面加一个下划线 `_` 表示 **In-place operation**。

``` python
tensor.add_(5)
```

