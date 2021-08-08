---
title: scikit-learn
date: 2021-08-02 02:58:19
categories: python
tags:
---

本文主要是对 python 库 `sklearn` 的学习笔记。

参考网站：

[sklearn官方网站](https://scikit-learn.org/stable/index.html)

[官网汉化](https://scikit-learn.org.cn/)

[一个中文文档](https://sklearn.apachecn.org/)

<!--more -->

# scikit-learn 是啥

`scikit-learn` 是一个开源的机器学习库，它支持有监督和无监督的学习。它还提供了用于模型拟合，数据预处理，模型选择和评估以及许多其他实用程序的各种工具。



# 拟合和预测

`sklearn` 提供了数十种内置的机器学习算法和模型，称为估算器。每个估算器可以使用其拟合方法拟合到一些数据。

``` python
>>> from sklearn.ensemble import RandomForestClassifier
>>> clf = RandomForestClassifier(random_state=0)
# clf 就是一个分类器(classifier)
>>> X = [[ 1,  2,  3],  #2个样本，3个特征
...      [11, 12, 13]]
>>> y = [0, 1]  #每一个样本的类别
>>> clf.fit(X, y)	# 一般通过 .fit 方法来拟合
```

我们可以直接调用这些模型，给定一些超参数，获得一个分类器/估算器。然后使用 `.fit` 方法来进行拟合，所有计算、优化过程都在这个方法中，（而且基本上这个库里的实现比自己手撸的要好）。

拟合方法通常有两个输入：

- 样本矩阵（或设计矩阵）X。`X`的大小通常为`(n_samples, n_features)`，这意味着样本表示为行，特征表示为列。
- 目标值y是用于回归任务的真实数字，或者是用于分类的整数（或任何其他离散值）。对于无监督学习，`y`无需指定。`y`通常是1d数组，其中`i`对应于目标`X`的 第`i`个样本（行）。
- 虽然某些估算器可以使用其他格式（例如稀疏矩阵），但是通常，两者`X`和`y`预计都是numpy数组或等效的类似数组的数据类型。

拟合完成之后，就可以直接用它预测啦。

```python
>>> clf.predict(X)  # 预测训练数据的标签
array([0, 1])
>>> clf.predict([[4, 5, 6], [14, 15, 16]])  # 预测新数据的标签
array([0, 1])
```

