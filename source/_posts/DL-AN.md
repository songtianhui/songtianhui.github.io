---
title: 吴恩达深度学习笔记
date: 2021-08-04 22:51:29
tags:
categories: ML
mathjax: true
typora-root-url: DL-AN
---



[视频指路](https://www.bilibili.com/video/BV1FT4y1E74V?from=search&seid=7469215768123017337)

<!--more -->

---

# Lesson 1.1

本章是对深度学习神经网络的一个介绍，课程的概览。



# Lesson 1.2

本章主要讲了一下 logisitic 回归，梯度下降，在ML中已经学习了。

然后介绍了 `python` 中的一些知识，向量化、广播、`numpy`、jupyter notebook等，都可以 **rtfm**，不在此赘述。

唯一有一点就是符号上的表示，andrew 喜欢用 $m$ 表示样本个数，$n$ 表示特征数。我习惯 mit 的课教的 $n$ 表示样本数，$d$ 表示特征维数。



# Lesson 1.3

本章算是一个神经网络的引入，介绍一些基本概念，浅层神经网络（shallow neural network）。

![](L1_week3_6.png)

每个神经单元就是这么个计算过程，有输入 $x$，权重 $w$，偏移 $b$，激活函数（activation function） $h$。

- $z = w^T x + b$
- $a = h(z)$

符号： $a_1^{[1](1)} $ ，右上角中括号表示层数，右上角括号中表示第几个样本，右下角表示该层第几个神经元。

对于一个输入样本，避免一层内的 for 循环，向量化计算：
$$
z^{[i]} = W^{[i]} a^{[i-1]} + b^{[i]}\\
a^{[i]} = h(z^{[i]})
$$
$W^{[i]}$ 是第 $i$ 层每个神经元的权重排成的矩阵。

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

$db^{[i]} = \dfrac{1}{m}$ `np.sum(dZ^{[i]}, axis=1)​`
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



---

# Lesson 2.1

本章主要讲改善神经网络，超参数调试、正则化等内容。

- 当我们有百万量级以上的数据，可以拿 99% 以上的数据来进行训练，几万条用来交叉验证（dev）和测试就可以了。

## 方差/偏差

- 高偏差（**high bias**），欠拟合（**underfitting**）。
- 高方差（**high variance**），过拟合（**overfitting**）。

通过训练集和验证集误差判断：

| Training set error | Dev set error |                      |
| :----------------: | :-----------: | :------------------: |
|         1%         |      15%      |    high variance     |
|        15%         |      16%      |      high bias       |
|        15%         |      30%      | high variance & bias |
|        0.5%        |      1%       | low variance & bias  |



## 基本方法

> 初始模型训练完成后，我首先要知道算法的偏差高不高，如果偏差较高，试着评估训练集或训练数据的性能。如果偏差的确很高，甚至无法拟合训练集，那么你要做的就是选择一个新的网络，比如含有更多隐藏层或者隐藏单元的网络，或者花费更多时间来训练网络，或者尝试更先进的优化算法，后面我们会讲到这部分内容。

- 只要正则适度，通常构建一个更大的网络便可以，在不影响方差的同时减少偏差，而采用更多数据通常可以在不过多影响偏差的同时减少方差。
- 训练网络，选择网络或者准备更多数据，现在我们有工具可以做到在减少偏差或方差的同时，不对另一方产生过多不良影响。



## 正则化（Regularization）

深度学习可能存在过拟合问题——高方差，有两个解决方法，一个是正则化，另一个是准备更多的数据，正则化通常有助于避免过拟合或减少你的网络误差。

就是在代价函数里加一个正则化项，一般用 $\dfrac{\lambda}{2m}$乘以$w$范数的平方,其中$\left\| w \right\|_2^2$是$w$的欧几里德范数的平方，$L2$ 正则化。

神经网络中的正则项就是为$\dfrac{\lambda }{2m}\sum\limits_{l = 1}^{L}| W^{[l]}|^{2}$，我们称${||W^{\left[l\right]}||}^{2}$为范数平方，这个矩阵范数${||W^{\left[l\right]}||}^{2}$（即平方范数），被定义为矩阵中所有元素的平方求和。该矩阵范数被称作“弗罗贝尼乌斯范数”，用下标$F$标注。

带正则化的梯度下降中，对$W^{[l]}$的偏导数，把$W^{[l]}$替换为$W^{[l]}$减去学习率乘以$dW$。现在我们要做的就是给$dW$加上这一项$\dfrac {\lambda}{m}W^{[l]}$，然后计算这个更新项，使用新定义的$dW^{[l]}$，它的定义含有相关参数代价函数导数和，以及最后添加的额外正则项。
$$
\begin{aligned}
W^{[l]} :&= W^{[l]}  - \alpha \times \left[(\text{from backpap}) + \dfrac{\lambda}{m}W^{[l]}\right]\\ &= (1 - \frac{\alpha \lambda}{m}) W^{[l]} - \alpha \times (\text{from backpap})
\end{aligned}
$$
*正则化预防过拟合的原因：极限思想，当lambda很大权重为0，退化成欠拟合，有个right fit 的中间态。*



## dropout 正则化

就是随机消掉一些神经元。。。

- **inverted dropout**（反向随机失活）
  - 首先要定义向量$d$，$d^{[3]}$表示网络第三层的**dropout**向量：`d3 = np.random.rand(a3.shape[0],a3.shape[1])` 。
  - 然后看它是否小于某数，我们称之为**keep-prob**，**keep-prob**是一个具体数字，它表示保留某个隐藏单元的概率。
  - 接下来要做的就是从第三层中获取$a^{[3]}$，$a^{[3]}$含有要计算的激活函数，$a^{[3]}$等于上面的$a^{[3]}$乘以 $d^{[3]}$，就是把 $d$ 中 0 对应位置的数归零。（$d$ 实际上是一个布尔数组）
  - 最后，我们向外扩展$a^{[3]}$，用它除以**keep-prob**参数。

显然在测试阶段，我们不使用**dropout**。要同时在 **fpp** 和 **bpp** 中使用 dropout。

*dropout 预防过拟合的原因，dropout 的功能类似于$L2$正则化，与$L2$正则化不同的是应用方式不同会带来一点点小变化，甚至更适用于不同的输入范围。*

- 如果你担心某些层比其它层更容易发生过拟合，可以把某些层的**keep-prob**值设置得比其它层更低，缺点是为了使用交叉验证，你要搜索更多的超级参数，另一种方案是在一些层上应用**dropout**，而有些层不用**dropout**，应用**dropout**的层只含有一个超级参数，就是**keep-prob**。

- 它在其它领域应用得比较少，主要存在于计算机视觉领域，因为我们通常没有足够的数据，所以一直存在过拟合。
- **dropout**一大缺点就是代价函数$J$不再被明确定义，每次迭代，都会随机移除一些节点，如果再三检查梯度下降的性能，实际上是很难进行复查的。



## 其他正则化方法

- 数据扩增
- **early stopping**
  - 缺点是不能独立处理梯度下降和优化代价函数。



## 归一化输入（Normalizing）

训练神经网络，其中一个加速训练的方法就是归一化输入。归一化需要两个步骤：

1. 零均值
   - $\mu = \frac{1}{m}\sum\limits_{i =1}^{m}x^{(i)}$，它是一个向量，$x$ 等于每个训练数据 $x$ 减去 $\mu$，意思是移动训练集，直到它完成零均值化。 
2. 归一化方差
   - $ \sigma^{2}= \frac{1}{m}\sum\limits_{i =1}^{m}(x^{(i)})^{2} $，$\sigma^{2}$是一个向量，它的每个特征都有方差，把所有数据除以向量$\sigma^{2}$。



## 梯度消失/梯度爆炸（Vanishing / Exploding gradients）

神经网络层数多了，激活函数就会以指数级递增或递减。



## 神经网络权重的初始化

针对梯度消失/爆炸，有一个方案就是更谨慎地选择随机初始化参数。

$z = w_{1}x_{1} + w_{2}x_{2} + \ldots +w_{n}x_{n}$，为了预防$z$值过大或过小，希望每项值更小，最合理的方法就是设置$w_{i}=\frac{1}{n}$，$n$表示神经元的输入特征数量。

实际上，你要做的就是设置某层权重矩阵 `W[l] = np.random.randn(shape) * np.sqrt(1 / n[l-1])`，$n^{[l - 1]}$ 就是我喂给第$l$层神经单元的数量（即第$l-1$层神经元数量）。

如果你是用的是**Relu**激活函数，而不是$\frac{1}{n}$，方差设置为$\frac{2}{n}$，效果会更好。

对于**tanh**函数来说，用$\sqrt{\frac{1}{n^{[l-1]}}}$。



## 梯度的数值逼近 Numerical approximation of gradients

就是导数定义，双边误差，即$\frac{f\left(\theta + \varepsilon \right) - f(\theta -\varepsilon)}{2\varepsilon}$。

先将所有的参数 $W, b$ 展开成一个大向量 $\theta$，在**bpp**中，算完梯度之后所有的梯度 $dW, db$ 就是 $d\theta$。

然后比较 $d\theta_{\text{approx}}\left[i \right] = \frac{J\left( \theta_{1},\theta_{2},\ldots\theta_{i} + \varepsilon,\ldots \right) - J\left( \theta_{1},\theta_{2},\ldots\theta_{i} - \varepsilon,\ldots \right)}{2\varepsilon}$ 和 $d\theta[i]$ 的值接不接近。

就计算它们的欧式距离再归一化，$\dfrac{||d\theta_{\text{approx}} -d\theta||_{2}}{||d\theta_{\text{approx}}||_2 + ||d\theta||_2}$。计算得到的值为$10^{-7}$或更小，这就很好；如果它的值在$10^{-5}$范围内，就要小心了，也许这个值没问题，但再次检查这个向量的所有项，确保没有一项误差过大，可能这里有**bug**。如果比$10^{-3}$大很多，就会很担心是否存在**bug**，这时应该仔细检查所有$\theta$项，看是否有一个具体的$i$值，使得$d\theta_{\text{approx}}\left[i \right]$与$ d\theta[i]$大不相同，并用它来追踪一些求导计算是否正确。



## 梯度检验的注意事项

1. 不要在训练中使用梯度检验，它只用于调试。
2. 如果算法的梯度检验失败，要检查所有项，检查每一项，并试着找出**bug**。
3. 在实施梯度检验时，如果使用正则化，请注意正则项。
4. 梯度检验不能与**dropout**同时使用，因为每次迭代过程中，**dropout**会随机消除隐藏层单元的不同子集，难以计算**dropout**在梯度下降上的代价函数$J$。



# Lesson 2.2

本节课主要讲优化算法，也就是我们如何更新参数。

## Mini-batch 梯度下降

你可以把训练集分割为小一点的子集训练，这些子集被取名为**mini-batch**，每个子集记作 $X^{\{i\}}$。就是把原来梯度下降时代入整个训练集改成代入一个mini-batch，然后多梯度下降几次。

使用**mini-batch**梯度下降法，如果作出成本函数在整个过程中的图，则并不是每次迭代都是下降的，特别是在每次迭代中，你要处理的是$X^{\{t\}}$和$Y^{\{ t\}}$。如果要作出成本函数$J$的图，你很可能会看到这样的结果，走向朝下，但有更多的噪声。

需要决定的变量之一是**mini-batch**的大小。首先，如果训练集较小，直接使用**batch**梯度下降法。样本数目较大的话，一般的**mini-batch**大小为64到512，考虑到电脑内存设置和使用的方式，如果**mini-batch**大小是2的$n$次方，代码会运行地快一些。



## 指数加权平均数 Exponentially weighted averages

递推式：
$$
v_t = \beta v_{t - 1} + (1 - \beta)\theta_t
$$
如果我们将其展开，这就是一个加权平均，是从 $0$ 到 $t$ 每个 $\theta_i$ 的平均，越远权重越小。

考虑 $\beta^{x} = \frac{1}{e}$，这个算的大约就是 $x$ 天的平均数。

指数加权平均数公式的好处之一在于，它占用极少内存，电脑内存中只占用一行数字而已，然后把最新数据代入公式，不断覆盖就可以了。

不过有可能会遇到 **偏差修正（bias corrections）的问题**

因为我们取 $v_0 = 0$，所以会使得 $i$ 较小时 $v_i$ 所占权重都很小，估算不准确。

我们可以在估测初期，不用 $v_t$，而是 $\dfrac{v_t}{1 - \beta^t}$。

不过在机器学习中，在计算指数加权平均数的大部分时候，大家不在乎执行偏差修正。



## 动量梯度下降法 Gradient descent with Momentum

有一种算法叫做**Momentum**，或者叫做动量梯度下降法，运行速度几乎总是快于标准的梯度下降算法，基本的想法就是计算梯度的指数加权平均数，并利用该梯度更新你的权重。

计算动量：
$$
v_{dW} = \beta v_{dW} + (1 - \beta) dW
$$

$$
v_{db} = \beta v_{db} + (1 - \beta) db
$$

再更新参数：
$$
W := W - \alpha v_{dW}
$$

$$
b := b - \alpha v_{db}
$$

这样就可以减缓梯度下降的幅度。*它们能够最小化碗状函数，这些微分项，想象它们为从山上往下滚的一个球，提供了加速度，**Momentum**项相当于速度。*

所以有两个超参数，学习率 $a$ 以及参数 $\beta$，$\beta$ 控制着指数加权平均数。$\beta$ 最常用的值是0.9，是很棒的鲁棒数。

有一个版本是 $v_{dW} = \beta v_{dW} + dW$，本质上没有区别。



## RMSprop

**root mean square prop**算法，它也可以加速梯度下降，通过加快损失下降的方向，减缓无关方向，减少摆动。
$$
S_{dW}= \beta S_{dW} + (1 -\beta) (dW)^{2}
$$

$$
S_{db}= \beta S_{db} + (1 - \beta)(db)^{2}
$$

再更新参数：
$$
W:= W -\alpha \dfrac{dW}{\sqrt{S_{dW}}}
$$

$$
b:=b -\alpha \dfrac{db}{\sqrt{S_{db}}}
$$

为了确保数值稳定，在实际操练的时候，要在分母上加上一个很小很小的$\varepsilon$，$\varepsilon$是多少没关系，$10^{-8}$是个不错的选择.



## Adam

把前面两个缝起来。

首先初始化：$v_{dW} = 0$，$S_{dW} =0$，$v_{db} = 0$，$S_{db} =0$。
$$
v_{dW}= \beta_{1}v_{dW} + ( 1 - \beta_{1})dW
$$

$$
v_{db}= \beta_{1}v_{db} + ( 1 -\beta_{1} )db
$$

$$
S_{dW}=\beta_{2}S_{dW} + ( 1 - \beta_{2})(dW)^{2}
$$

$$
S_{db} =\beta_{2}S_{db} + ( 1 - \beta_{2} )(db)^{2}
$$
一般使用**Adam**算法的时候，要计算偏差修正，$v_{dW}^{\text{corrected}}$，修正也就是在偏差修正之后：
$$
v_{dW}^{\text{corrected}}= \dfrac{v_{dW}}{1 - \beta_{1}^{t}}
$$

$$
v_{db}^{\text{corrected}} =\dfrac{v_{db}}{1 -\beta_{1}^{t}}
$$

$$
S_{dW}^{\text{corrected}} =\dfrac{S_{dW}}{1 - \beta_{2}^{t}}
$$

$$
S_{db}^{\text{corrected}} =\dfrac{S_{db}}{1 - \beta_{2}^{t}}
$$
最后更新权重：
$$
W:= W - \dfrac{\alpha v_{dW}^{\text{corrected}}}{\sqrt{S_{dW}^{\text{corrected}}} +\varepsilon}
$$

$$
b:=b - \frac{\alpha v_{\text{db}}^{\text{corrected}}}{\sqrt{S_{\text{db}}^{\text{corrected}}} +\varepsilon}
$$
**Adam** 是一种极其常用的学习算法，被证明能有效适用于不同神经网络，适用于广泛的结构。

本算法中有很多超参数，超参数学习率$a$很重要，也经常需要调试，你可以尝试一系列值，然后看哪个有效。$\beta_{1}$常用的缺省值为0.9。超参数$\beta_{2}$，**Adam**论文作者，也就是**Adam**算法的发明者，推荐使用0.999。关于$\varepsilon$的选择其实没那么重要，**Adam**论文的作者建议$\varepsilon$为$10^{-8}$。



## 学习率衰减 Learning rate decay

慢慢减少$a$的本质在于，在学习初期，你能承受较大的步伐，但当开始收敛的时候，小一些的学习率能让你步伐小一些。

$a= \dfrac{1}{1 + decayrate \times \text{epoch}\text{-num}}a_{0}$，（**decay-rate**称为衰减率，**epoch-num**为代数，$\alpha_{0}$为初始学习率），注意这个衰减率是另一个需要调整的超参数。

人们用到的其它公式有$a =\dfrac{k}{\sqrt{\text{epoch-num}}}a_{0}$或者$a =\dfrac{k}{\sqrt{t}}a_{0}$（$t$为**mini-batch**的数字）。



## 局部最优问题 Local optima

在深度学习研究早期，人们总是担心优化算法会困在极差的局部最优，不过随着深度学习理论不断发展，我们对局部最优的理解也发生了改变。

事实上，如果你要创建一个神经网络，通常梯度为零的点并不是这个图中的局部最优点，实际上成本函数的零梯度点，通常是鞍点。

意思就是一个最优点需要所有维度都是极小，在现在高维特征下很难遇到，所以碰到的所谓局部最优一般都不是最优，都可以跑出来。

