---
title: pandas
date: 2021-03-06 23:41:55
categories: python
tags:
---

本文主要是对于python库`pandas`的学习使用笔记。
参考网站：
[pandas官方网站](https://pandas.pydata.org/)
[一个不错的pandas中文网站](https://www.pypandas.cn/)
<!--more -->

# pandas是啥
这个库主要是用来干啥的？
Python在数据处理和准备方面一直做得很好，pandas主要用来做数据分析和建模而不必切换到更特定于领域的语言，如R。
方便数据的读写，智能数据对齐和丢失数据的综合处理，数据集的灵活调整和旋转，基于智能标签的切片、花式索引和大型数据集的子集。。。

---

# 数据结构
pandas最重要的就是它的数据结构：
[Series](https://pandas.pydata.org/pandas-docs/stable/reference/series.html): 带标签的一维同构数组
[DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/frame.html): 带标签的，大小可变的，二维异构表格

## Series
``` python 创建Series
>>> s = pd.Series(data, index=index) #index: 索引名字
```
- `data`是多维数组时，`index`长度必须与`data`长度一致，默认`[0, 1, ..., len(data)-1]`
``` python
>>> s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
>>> s 
a    0.469112
b   -0.282863
c   -1.509059
d   -1.135632
e    1.212112
dtype: float64

>>> s.index
Index(['a', 'b', 'c', 'd', 'e'], dtype='object')

>>>> s.dtype
dtype('float64')
```
- `data`是字典时，直接key-value，按插入顺序排列
- `data`是标量，按索引重复

- `Series`操作与`ndarray`类似，支持大多数 NumPy 函数，还支持索引切片。

### 支持矢量操作
``` python
>>> np.exp(s)
a         1.598575
b         0.753623
c         0.221118
d         0.321219
e    162754.791419
dtype: float64
```
- Series 和多维数组的主要区别在于， Series 之间的操作会自动基于标签对齐数据。因此，不用顾及执行计算操作的 Series 是否有相同的标签。
- 操作未对齐索引的 Series， 其计算结果是所有涉及索引的并集。如果在 Series 里找不到标签，运算结果标记为`NaN`，即缺失值。

### 名称
`Series` 支持 `name`属性，就是这列数据的名称。
``` python
s = pd.Series(np.random.randn(5), name='something')
s2 = s.rename("different")
```

## DataFrame
`DataFrame` 是由多种类型的列构成的二维标签数据结构，类似于 Excel 、SQL 表，或 Series 对象构成的字典。

### 用 Series 字典生成 DataFrame
- 生成的索引是每个 Series 索引的并集。先把嵌套字典转换为 Series。如果没有指定列，DataFrame 的列就是字典键的有序列表。
- index 和 columns 属性分别用于访问行、列标签：

``` python
>>> d = {'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']), 'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
>>> pd.DataFrame(d)
   one  two
a  1.0  1.0
b  2.0  2.0
c  3.0  3.0
d  NaN  4.0

>>> pd.DataFrame(d, index=['d', 'b', 'a'])
   one  two
d  NaN  4.0
b  2.0  2.0
a  1.0  1.0

>>> pd.DataFrame(d, index=['d', 'b', 'a'], columns=['two', 'three']) 
   two three
d  4.0   NaN
b  2.0   NaN
a  1.0   NaN
```

### 备选构建器
- DataFrame.from_dict
- DataFrame.from_records

### 提取、添加、删除列
就像带索引的Series字典
``` python
# 提取
>>>  df['one']
a    1.0
b    2.0
c    3.0
d    NaN
Name: one, dtype: float64
# 添加
>>> df['three'] = df['one'] * df['two']
>>> df['flag'] = df['one'] > 2
>>> df
   one  two  three   flag
a  1.0  1.0    1.0  False
b  2.0  2.0    4.0  False
c  3.0  3.0    9.0   True
d  NaN  4.0    NaN  False
# 删除
>>> del df['two']
>>> three = df.pop('three')
>>> df
   one   flag
a  1.0  False
b  2.0  False
c  3.0   True
d  NaN  False
# 标量重复
>>> df['foo'] = 'bar'
>>> df
   one   flag  foo
a  1.0  False  bar
b  2.0  False  bar
c  3.0   True  bar
d  NaN  False  bar
# 插入与 DataFrame 索引不同的 Series 时，以 DataFrame 的索引为准
>>> df['one_trunc'] = df['one'][:2]
>>> df
   one   flag  foo  one_trunc
a  1.0  False  bar        1.0
b  2.0  False  bar        2.0
c  3.0   True  bar        NaN
d  NaN  False  bar        NaN
# 可以插入原生多维数组，但长度必须与 DataFrame 索引长度一致。默认在 DataFrame 尾部插入列。insert 函数可以指定插入列的位置
>>> df.insert(1, 'bar', df['one'])
>>> df
   one  bar   flag  foo  one_trunc
a  1.0  1.0  False  bar        1.0
b  2.0  2.0  False  bar        2.0
c  3.0  3.0   True  bar        NaN
d  NaN  NaN  False  bar        NaN
```

### asign分配新链
- `assign` 返回的都是副本数据

``` python
>>> dfa = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
>>> dfa.assign(C=dfa['A'] + dfa['B'], D=lambda x: x['A'] + x['C'])
   A  B  C   D
0  1  4  5   6
1  2  5  7   9
2  3  6  9  12
```

### 索引，选择
|操作	|句法	|结果|
|:-|:-|:-|
|选择列	|df[col]	|Series|
|用标签选择行	|df.loc[label]	|Series|
|用整数位置选择行	|df.iloc[loc]	|Series|
|行切片	|df[5:10]   |DataFrame|
|用布尔向量选择行	|df[bool_vec]	|DataFrame|

选择行返回 Series，索引是 DataFrame 的列：
``` python
>>> df.loc['b']
one              2
bar              2
flag         False
foo            bar
one_trunc        2
Name: b, dtype: object
```

### 数据对齐和运算
- DataFrame 对象可以自动对齐列与索引（行标签）的数据，生成的结果是列和行标签的并集。
- DataFrame 和 Series 之间执行操作时，默认操作是在 DataFrame 的列上对齐 Series 的索引，按行执行广播操作。
- 时间序列是特例，DataFrame 索引包含日期时，按列广播

### 转置
`df[:5].T`

### DataFrame 应用 Numpy 函数
- Series 与 DataFrame 可使用 log、exp、sqrt 等多种元素级 NumPy 通用函数（ufunc）
- DataFrame 不是多维数组的替代品，它的索引语义和数据模型与多维数组都不同。



# 怎么总结呢？

我一直在思考怎么来总结这些库的技巧，毕竟不可能对于每个方法API和参数都详尽地记录下来，我们有手册可以 **rtfm**，把手册再抄一遍没有意义。但是每次在用到一些方法时还是很难很快反应过来怎么用，或者不知道一些常用的方法来解决一些问题。

所以我选择用一种问题或任务驱动的方式来进行总结，在编程的时候遇到一些任务，或者在阅读优秀代码时看到的一些数据结构和方法参数时，看不懂就查手册总结下来，得到一些常用的方法。

这个方法不止在本篇 `pandas` 中，在其他几个包的教程中也如此。



# 开始

