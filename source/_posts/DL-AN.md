---
title: 吴恩达深度学习笔记
date: 2021-08-04 22:51:29
tags:
categories: ML
mathjax: true
typora-root-url: DL-AN
---



<!--more -->



---

# Lesson 1.1

本章是对深度学习神经网络的一个介绍，课程的概览。



# Lesson 1.2

本章主要讲了一下 logisitic 回归，梯度下降，在ML中已经学习了。

然后介绍了 `python` 中的一些知识，向量化、广播、`numpy`、jupyter notebook等，都可以 **rtfm**，不在此赘述。

唯一有一点就是符号上的表示，andrew 喜欢用 $m$ 表示样本个数，$n$ 表示特征数。我习惯 mit 的课教的 $n$ 表示样本数，$d$ 表示特征维数。



# Lession 1.3

本章算是一个神经网络的引入，介绍一些基本概念，浅层神经网络（shallow neural network）。

![](L1_week3_6.png)

每个神经单元就是这么个计算过程，有输入 $x$，权重 $w$，偏移 $b$，激活函数（activation function） $h$。

- $z = w^T x + b$
- $a = h(z)$

符号：$a_1^{[1](1)}$，右上角中括号表示层数，右上角括号中表示第几个样本，右下角表示该层第几个神经元。

对于一个输入样本，避免一层内的 for 循环，向量化计算：
$$
z^{[i]} = W^{[i]} a^{[i-1]} + b^{[i]}\\
a^{[i]} = h(z^{[i]})
$$
$W^[i]$ 是第 $i$ 层每个神经元的权重排成的矩阵。

举例图示：
$$
\left[
		\begin{array}{c}
		z^{[1]}_{1}\\
		z^{[1]}_{2}\\
		z^{[1]}_{3}\\
		z^{[1]}_{4}\\
		\end{array}
		\right]
		 =
	\overbrace{
	\left[
		\begin{array}{c}
		...W^{[1]T}_{1}...\\
		...W^{[1]T}_{2}...\\
		...W^{[1]T}_{3}...\\
		...W^{[1]T}_{4}...
		\end{array}
		\right]
		}^{W^{[1]}}
		*
	\overbrace{
	\left[
		\begin{array}{c}
		x_1\\
		x_2\\
		x_3\\
		\end{array}
		\right]
		}^{input}
		+
	\overbrace{
	\left[
		\begin{array}{c}
		b^{[1]}_1\\
		b^{[1]}_2\\
		b^{[1]}_3\\
		b^{[1]}_4\\
		\end{array}
		\right]
		}^{b^{[1]}}
$$


对于所有样本，避免 for 遍历样本，向量化计算：
$$
Z^{[i]} = W^{[i]}A^{[i-1]} + b^{[i]}\\
A^{[i]} = h(Z^{[i]})
$$
其中 $Z^{[i]},A^{[i]}$ 是第 $i$ 层所有样本输出排成的矩阵。

激活函数：**sigmoid**（主要在二分类）, **tanh**（比 sigmoid 常用），**ReLU**（在神经网络中很常用），Leaky ReLU。

前两个有梯度消失的风险。

这里提到一个很重要的问题就是为什么要使用非线性函数而不是直接 $a = z$。因为全用线性激活函数（identity）会使神经网络退化成一个单层模型。

## 反向传播 Back Propagation

也就是神经网络的梯度下降（Gradient Descent）。比较重要，理解一下推倒，本质上是函数求导的链式法则。

代价函数：
$$
J(W, b) = \dfrac{1}{m}\sum\limits_{i= 1}^{m}L(\hat{y}, y)
$$
当参数初始化成某些值后，每次梯度下降都会循环计算以下预测值：$\hat{y}^{(i)},(i=1,2,…,m)$。

有 $dW^{[i]} = \dfrac{\partial J}{\partial W^{[i]}}$, $d b^{[i]} = \dfrac{\partial J}{\partial b^{[i]}}$。

在梯度下降时每一次更新：$W^{[i]}\implies{W^{[i]} - \eta dW^{[i]}},b^{[i]}\implies{b^{[i]} -\eta db^{[i]}}$, $\eta$ 为步长。

反向传播时，就是一个链式求导：
$$
\underbrace{
	\left.
	\begin{array}{l}
	x \\
	w \\
	b 
	\end{array}
	\right\}
	}_{dw=dz \cdot x, db =dz}
	\impliedby \underbrace{z=w^Tx+b}_{dz=da\cdot g^{'}(z),
	g(z)=\sigma(z),
	\frac{dL}{dz}} = \frac{dL}{da} \cdot \frac{da}{dz},
	\frac{d}{ dz} g(z)=g^{'}(z)
	\impliedby \underbrace{a = \sigma(z) 
	\impliedby L(a,y)}_{da=\frac{d}{da}L\left(a,y \right)=(-y\log{\alpha} - (1 - y)\log(1 - a))^{'}=-\frac{y}{a} + \frac{1 - y}{1 - a} }
$$
所以有：

$dz^{[L]} = A^{[L]} - Y$

$dW^{[i]} = \dfrac{1}{m} dZ^{[i]}A^{[i-1]T}$

$db^{[i]} = \dfrac{1}{m}np.sum(dZ^{[i]}, axis=1)$
$$
dz^{[i]} = \underbrace{W^{[i + 1]T} dz^{[i+1]}}_{(n^{[i]},m)}\quad \times  \underbrace{g^{[i]'}}_{activation \; function \; of \; hidden \; layer}\times  \quad\underbrace{(z^{[i]})}_{(n^{[1]},m)}
$$

- 随机初始化，不要初始化成相同的参数。



# Lesson 1.4

本章介绍深层神经网络，主要就是把前一章讲的只有两层的网络更推广一下，而我们在上一章其实已经推广过了。

## 为什么使用深层表示？

深度神经网络的这许多隐藏层中，较早的前几层能学习一些低层次的简单特征，等到后几层，就能把简单的特征结合起来，去探测更加复杂的东西。

深层的网络隐藏单元数量相对较少，隐藏层数目较多，如果浅层的网络想要达到同样的计算结果则需要指数级增长的单元数量才能达到。

> 说实话，我认为“深度学习”这个名字挺唬人的，这些概念以前都统称为有很多隐藏层的神经网络，但是深度学习听起来多高大上，太深奥了，对么？这个词流传出去以后，这是神经网络的重新包装或是多隐藏层神经网络的重新包装，激发了大众的想象力。	——Andrew



## 搭建神经网络块

其实就是对于每一层，权重矩阵，偏移值，激活函数，在前向传播的时候缓存（cache）好 $z,a$ 等值，用反向传播时计算 $dW,db$ 等。

就放一张老师的板书吧（

![building blocks](network.png)



## 参数和超参数

算法中的**learning rate** $a$（学习率）、**iterations**(梯度下降法循环的数量)、$L$（隐藏层数目）、$n^{[l]}$（隐藏层单元数目）、**choice of activation function**（激活函数的选择）都需要你来设置，这些数字实际上控制了最后的参数$W$和$b$的值，所以它们被称作超参数（Hyperparameter）。

如何寻找超参数：走**Idea—Code—Experiment—Idea**这个循环，尝试各种不同的参数，实现模型并观察是否成功，然后再迭代。

> 应用深度学习领域，一个很大程度基于经验的过程，凭经验的过程通俗来说，就是试直到你找到合适的数值。
>
> 如果你所解决的问题需要很多年时间，只要经常试试不同的超参数，勤于检验结果，看看有没有更好的超参数数值，相信你慢慢会得到设定超参数的直觉，知道你的问题最好用什么数值。

