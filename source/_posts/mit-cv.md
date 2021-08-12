---
title: mit机器学习cv
date: 2021-08-10 17:46:40
tags:
categories: ML
mathjax: true
typora-root-url: mit-cv
---

暑假上了mit的机器学习课程，虽然时间很短内容不多，但觉得第二部分cv的老师讲得很好，所以做一篇笔记。

<!--more -->



# lecture 1

第一节课是对 deep learning 和 computer vision 的一个介绍。计算机视觉，就是指观察图像，像素，理解图像里出现了是什么，预测会发生什么。

## 神经元（Perceptron）

具体概念已经看了太多遍了，不再赘述。

对于优化的概念，$W \leftarrow W - \eta \frac{\partial J(W)}{\partial W}$，他解释的很好。这个偏导项就是代价函数随着 $W$ 的变化增长最快的方向，再加上一个小步长，就是移动一点点，因为是最小值所以是反方向移动，是负号。



## 边缘检测 Edges

图上的边特征：

- 图像梯度 Image gradient: $\nabla I = \left( \dfrac{\partial I}{\partial x}, \dfrac{\partial I}{\partial y} \right)$
- 图像导数的估算 Approximation image derivative: $\dfrac{\partial I}{\partial x} \simeq I(x, y) - I(x - 1, y)$
- 边缘强度 Edge strength: $E(x, y) = |\nabla I(x, y)|$
- 边缘取向 Edge orientation: $\theta(x, y) = \angle \nabla I = \arctan{\dfrac{\partial I /\partial y}{\partial I / \partial x}}$
- 边缘法线 Edge normal: $n = \dfrac{\nabla I}{|\nabla I|}$

如果我们使用梯度算法，并绘制出边缘的强度，我们可以看到所有的边缘都暴露出来了。我们可以将这些微分方程或公式转换成图像强度的离散线性约束，这对于定义图像变化非常重要。



## 卷积 Convolution

卷积是一个操作两个函数的数学操作，输入两个函数输出一个（函数复合）。

通过滑动函数来计算，一个函数（kernel）划过另一个函数（template），在每一步把他们相乘再相加。
$$
f[n] = h \circ g = \sum\limits_{k = 0}^{N - 1} h[n - k] g[k]
$$
我们就可以用卷积来计算图像导数。

比如用
$$
(E \otimes k)(x, y)=\sum\limits_{i=0}^{m-1} \sum\limits_{j=0}^{n-1} E(x+i, y+j) k(i, j)
$$
能用来计算
$$
\dfrac{\partial E(x, y)}{\partial x}=E(x+1, y)-E(x, y)
$$
当 $k = [-1, 1]$。

然后我们可以对图像导数的定义再扩展，运用卷积计算更加复杂的一些特征。

![](image_dev.png)

然后已知两个方向的导数，就可以用三角函数计算出任意方向的导数而不是真的创造任意方向的导数。

将导数可视化：

![](Screenshot from 2021-08-10 18-51-05.png)

甚至我们可以创造不同类型的过滤器，不只是计算导数，实现更多的图像处理技术。

![](Screenshot from 2021-08-10 18-57-36.png)

*这是我对卷积和图像处理认识的第一步，我感到非常震撼，原来我们所在ps等图像处理软件上所做的处理，其原理真的就是矩阵对于像素的计算。不由得想到jyy的名言：计算机世界没有魔法(。*

而我们在这节课学的是如何去学习卷积的参数而不是导数卷积，也就是说找到最好的 kernel。

我们可以把很多层卷基层堆积在一起，形成一个卷积块（我自己起的名字），维度是 $H \times W \times D$，$D$ 深度是过滤层的个数。

步幅（stride）。

卷基层的特点：

- 空间不变性（Spatial invariance）
- 批量处理，高效并行
- 图像过滤
- 共享参数，高效
- 多种大小输入

**gabor filters**



## 池化 Pooling

Pooling 和 convolution 不一样的地方是，卷积在于从一块像素中提取信息，而 pooling 是为了将信息压缩，使得一块像素塌缩成一个单位，也是对数据的抽样，对信息流的限制。

**max pooling**



## 结合起来

我们就可以叠buff，一层conv一层relu一层pooling，一层conv一层relu一层pooling。。。实现一个端对端模型，一层层提取结构。

General CNN architecture:

![](Screenshot from 2021-08-10 21-35-47.png)



# Lecture 2

## CNN Architectures

- LeNet: LeCun et al. 1998. 
  - 第一个视觉神经网络。
- AlexNet: Krizhevsky et al. NeurIPS 2012. 
  - 两个信息流并行，更深度。
- GoogLeNet/Inception: Szegedy et al. CVPR 2015. 
  - Inception module
  - 更小的卷积，更深，更精确。
- VGGNet: Simonyan et al. ICLR 2015.

~~CV历史~~



## 残差连接 Scaling CNNs: residual connections

普通的卷积层的代价函数过于复杂，容易陷入局部最优，很难找到全局最优。简单的卷积加池化无法使模型更好。

我们需要学习残差的变化而不是实际的端对端函数。



## 残差区块 Residual blocks

![](Screenshot from 2021-08-10 22-47-54.png)

不是只学习从 $x$ 到输出，我们学习的是变化、残差，以达到输出的目的。它能够把问题变得更简单。

优点：

- 更快的梯度传输。
- 只应用少量的残余值而不是整个函数值。
- 保持输入的结构。

当输出和输入的结构不一样时，加一个权重层来投影到正确的维度。

有各种各样的残差连接。

- ResNet: He et al. CVPR 2016.
  - 使得机器图片识别准确性开始超过人类。
  - 突破计算机可以训练的层数，$25 \to 150$。
  - 代价函数更佳平滑，容易找到全局最优。

## 数据集 Dataset

深度学习不是什么都能学的，要仔细选择数据集，不能往里面“扔垃圾”。

> Garbage in, Garbage out.

- MNIST
- Image NET
- CIFAR  10/100
  - Facet: tool for vusualization of train data
- Object Net
- MiniPlaces: scene recognition



# Lecture 3

本节主要讲图像序列，连续图像的处理。

## 循环神经网络 Recurrent neural network

为了建模序列，我们需要：

1. 处理变化长度的序列。
2. 跟踪长期的依赖关系。
3. 保持信息的时间顺序。
4. 在序列中的共享参数。

原本我们的模型都是一对一的，现在我妈们可以考虑一对多、多对一、多对多预测，多对多分类。我们需要更新我们的模型，使得能够整合连续序列信息。

### RNN

老师在这门课上并没有详细地讲解 RNN 的数学理论，简单介绍一下。

在 RNN 中，我们有时间 $t$ 这个概念，在每个时间点会有一个输入 $x_t$，然后有一个隐藏状态 $h_t$ 在每个时间点，状态序列就通过一个循环过程产生：
$$
h_t = f_W(h_{t - 1}, x_t)
$$
注意每个时间点所用的函数都是一样（保持时间对称性），也就是当前状态由前一个状态和当前输入决定，其实很像数电中状态机、时序的概念。

具体的，会由两个矩阵来线性组合出当前状态：
$$
h_t = \tanh{(W_{hh}^Th_{t-1} + W_{xh}^{T}x_t)}\\
\hat{y}_t = W_{hy}^{T}h_t
$$
这个 $\hat{y}$ 是对当前状态产生的一个输出，因为 $h$ 可以看作只是状态的一个编码。

然后其中的权重矩阵就会由学习得到。

![](Screenshot from 2021-08-11 18-07-27.png)

如果我们考虑将所有时间的信息都记录下来，长距离的依赖会导致问题：

- 需要记录的东西和内存会随着时间不断增加。
- 没有无限大的参数集来建模所有依赖。
- RNN假设：当前隐藏层状态只依赖于前一个时间点的状态。
- 想法是建立隐藏状态来建模长距离的依赖。

这些想法促使了我们需要一个有更好建模长距离依赖能力的新模型架构。



### 长短时记忆单元 Long short term memory unit

也没有仔细描述 LSTM 的细节，主要思想是 not to forget & learn to forget。

LSTM 使用**门（gates）** 来控制信息流：

- **遗忘** 来减少无关信息
- **储存** 相关信息从当前输入
- 选择性 **更新** 单元状态
- **输出**一个过滤版本状态



## RNNs + CNNs

核心思想就是 CNNs 来学习出2D图像的特征提取器，压缩后的特征喂入 RNNs 分析序列。

![](Screenshot from 2021-08-11 18-47-07.png)



## CV中的应用

### 视频分类

![](Screenshot from 2021-08-11 21-44-11.png)

有多种整合特征的方法。



### 图像标题 imgae captioning

目标是生成准确捕捉图像内容的句子。

- 生成模型

![](Screenshot from 2021-08-11 21-44-40.png)

### 动作预测

![](Screenshot from 2021-08-11 21-53-42.png)



# Lecture 4

本章主要介绍**图像生成模型（Generative Image Modeling）**。

## 生成模型 Generative modeling

在第二节课中我们学到了图像分类，我们思考能不能将这个一对一映射反过来，根据标签去生成图像。

**图像合成（image synthesis）**

**图像翻译（image translation）**

生成模型的目标就是：学习出一个代表并捕捉了一些控制相关数据的潜在概率分布深度学习模型。

在计算机视觉中：

- 学习拟合一个数据流形（data manifold）。
- 从习得的分布产生新的样例。

挑战：

- 高维。
- 潜在分布的复杂性。

所以我们用深度学习来建模因为它的数据驱动的拟合函数方式。



## 模型架构 Architecture

分布转化（distribution transformers）

![](Screenshot from 2021-08-12 18-36-59.png)

我们需要神经网络上非常聪明的方法，来实现端对端的动态模型。

我们从一个可以使低维的概率分布，习得一个神经网络模型，从原来的分布到目标数据的分布。在这个过程中我们可以生产落在目标数据中的合成图像，这就是生成模型的关键。

![](Screenshot from 2021-08-12 18-43-11.png)

关于建立这种模型可以分为两类，一种是明确估计概率函数的，另一种是不明确地学习密度函数的，从隐式的学习到实现生成。我们将会主要讨论 **VAE** 和 **GAN**。

![](Screenshot from 2021-08-12 18-46-16.png)

### 变分自编码 Variational autoencoders

我们是想要从真实的数据空间，模拟生成合成样本或真正数据空间的重构，能够概括数据的形态。

**VAE** 先将真实数据编码成低维特征空间或隐空间（latent space），由一组隐变量编码而成，我们可以去学习这个特征空间而不需要任何监督。

**VAE** 将问题分成两个部分，一个就是对于原来的高维的真实数据编码（encode）成低维的数据空间，从而来限制提取出有用的特征。基于此，我们要从这个低维的数据空间解码（decode）或习得一个重构的数据空间。

所以对于图像数据，编码和解码网络就可以由卷积结构来构建。

![](Screenshot from 2021-08-12 18-57-17.png)

我们算出隐空间 $z$ 的概率分布，根据输入数据 $x$，它将被编码网络的权重矩阵参数化，我们用 $\phi$ 来表示。然后反向地我们可以做出解码网络，根据 $z$ 计算出 $x$，解码参数为 $\theta$。把两部分连起来。

训练的方法就是将两部分整合进代价函数，参数为 $\phi$ 和 $\theta$，一项是正则化项。

为了训练这个模型，我们需要预先给定一些概率分布，这取决于我们希望这些隐变量采取怎样的形态。我们会在每一个隐变量 $z$ 上加上一个高斯分布（Gaussian prior）。

为了不是绝对重建，我们还得想办法从 $z$ 中取样，我们要参数化 $z$，用均值和标准差根据之前的正太高斯分布，把标准差按照一个噪声因子进行缩放，该因子包含了随机性、概率元素和变分元素，使我们可以用 **VAE** 生成合成的数据实例。

更具体地说，当我们施加高斯分布，并将隐变量参数化为平均值和标准差并缩放一个噪声系数，这样子我们就可以算出代价函数，根据重建项和一个正则化项，其中正则化项可以得到我们根据隐变量学习解码出的结果和根据高斯分布指定的隐变量分布形态之间的差距。
$$
\mathcal{L}(\phi, \theta, x) = \| x - \hat{x} \|^2 + D(q_\phi(z|x) | p(z))
$$
$$
z = \mu + \sigma \odot \varepsilon
$$

这是一个很好的可跟踪的方法，来检查网络并理解网络到底学了什么信息。我们要做的，就是**潜变量摄动（latent variable perturbation）**就是改变 $z$ 然后观察输出结果，得到前变量的属性，是否相互独立等。

总结：

1. 压缩数据的表示到低维空间。
2. 重建允许了无监督学习。
3. 重参数化技巧来训练。
4. 通过摄动解释潜变量。
5. 生成新实例。

### 生成对抗网络 Generative adversarial networks

核心理念是：
我们先从完全高斯噪声作为初始的 $z$ 分布开始，去生成符合的图像数据。制定一条端到端的流水线，它们由两个网络组成，一个是**生成器（generator）**，一个是**识别器（discriminator）**，它们在一个敌对博弈中相互竞争。我们从一个高斯噪声输入开始，生成器生成合成图像，然后被喂到分类网络中，这个网路被训练去思考这个实例是真实的或人造的。整个过程通过迭代，训练识别器和生成器通过相互竞争实现优化。全局最优就是识别器识别不出来了。

![](Screenshot from 2021-08-13 00-33-13.png)

从高斯噪声开始，生成器生成的合成图片被输入识别器，同时真实的图片也会被输入识别器，识别器被训练来判断输入的图像是真的还是假的，对于我们可以优化的损失这些损失都是交叉熵损失：
$$
\arg \max\limits_{D} \mathbb{E}_{\mathbf{z}, \mathbf{x}}[\log D(G(\mathbf{z}))+\log (1-D(\mathbf{x}))
$$
前一项描述的是识别器对于生成器的图像的真假判断的概率，后一项刻画的是一个给定的图片是否与一个真标签相关联。我们利用交叉熵损失和最大化分离目标来训练识别器。
$$
\arg \min\limits_{G} \mathbb{E}_{\mathbf{z}, \mathbf{x}}[\log D(G(\mathbf{z}))+\log (1-D(\mathbf{x}))]
$$
相反地，我们可以用同样的交叉熵，用最小化代替最大化，也就是说让它与识别器为敌。

我们可以通过内插，在高斯噪声中的两个数据点，实现由 GAN 生成的合成图像的扰动，观察其在学习到的数据空间流形中的距离，来调试 GAN。优点是保证了映射的条理性。

对传统 GAN 的优化：

- StyleGAN, Karras+ 2019

- 我看不懂（

![](Screenshot from 2021-08-13 00-54-23.png)



## Recent Advances

- Paired translation
  - pix2pix
- CycleGAN: domain transfer

![](Screenshot from 2021-08-13 01-05-39.png)
