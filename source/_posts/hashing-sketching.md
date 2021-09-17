---
title: 高级算法课程笔记-Hashing and Sketching
date: 2021-09-16 21:27:58
tags:
categories:
- 课程
- Advanced Algorithm
mathjax: true
---

本章主要是讲 hashing 和 sketching 技术。（这两个词确实有点难找到合适的翻译。。。）



<!--more-->

# 不同元素 Distinct Elements

一个叫数不同元素的问题：

$\Omega $ 是一个足够大的全集，输入 $n$ 个元素 $x_1, x_2, ..., x_n \in \Omega$，不一定是不同的。要输出的就是共有多少个不同种元素 $z = |\{x_1, x_2, ..., x_n\}|$。

最直接的一个方法就是维护一个字典（dictionary）的数据结构，空间复杂度是 $O(n)$ 的。当 $n$ 很大时开销是大的，但是根据信息理论，如果要计算出精确的 $z$，$O(n)$ 的空间是必要的。

所以我们又只能考虑牺牲可容忍的精度的近似答案，有一个叫 $(\epsilon, \delta)$**-estimator** 的东西：

一个随机变量 $\widehat{Z}$ 称为一个量 $z$ 的 $(\epsilon, \delta)$**-estimator** 当 $\text{Pr}[(1 - \epsilon)z \leq \widehat{Z} \leq (1 + \epsilon)z] \geq 1 - \delta$。

- 当 $\mathbb{E}[\widehat{Z}] = z$ 时称作**无偏估计（unbiased estimator）**。

- 一般的 $\epsilon$ 叫 **approximation error**，$\delta$ 叫 **confidence error**。



## 使用 hashing 的估计量

假设我们有一个理想的随机哈希函数 $h : \Omega \to [0, 1]$，是在 $\Omega $ 上的均匀分布。

对于输入序列 $x_1, ..., x_n$，我们可以维护它们的哈希值而不是原来的元素，但是在最坏的情况下我们仍然需要维护 $n$ 个不同的值。但是由于理想的随机哈希函数，区间 $[0,1]$ 将被 $z$ 个不同的哈希值分成 $z+1$ 个子区间。通过这些子区间，我们可以得到一个 $z$ 的估计量。

首先有一个命题：

{% note info %}

$\mathbb{E}[\min_{1 \leq i \leq n} h(x_i)] = \dfrac{1}{z + 1}$

{% endnote %}

所以所有哈希值中最小的那个 $Y = \min_{1 \leq i \leq n} h(x_i)$ 可以作为 $\frac{1}{z+1}$ 的无偏估计，$\min_{1 \leq i \leq n} h(x_i)$ 显然是可以在常数空间复杂度下得到的。

但是要记住 $\frac{1}{Y} - 1$ 不一定是 $z$ 的一个好的估计值。



## Flajolet-Martin 算法

上面的方法表现得不太好的原因是无偏估计量 $\min_{1 \leq i \leq n} h(x_i)$ 的方差太大了，解决方法是我们可以用多个独立的哈希函数，然后取平均值，这也就是 **Flajolet-Martin 算法**的核心思想。

假设我们可以获得 $k$ 个独立的随机哈希函数 $h_1, h_2, ..., h_k$，其中 $k$ 需要根据 $\epsilon, \delta$ 来确定。

``` pseudocode Flajolet-Martin
Scan the input sequence x_1, ..., x_n in a single pass to compute:
	Y_j = min(h_j(x_i)) for j = 1, 2, ..., k
	average value Y_bar = sum(Y_j) / k
return Z = 1 / y_bar - 1
```

这个算法仍然可以在**数据流模型**下实现，需要储存 $k$ 个哈希值。

下面这个定理保证了这个算法可以返回一个对于不同元素总数的 $(\epsilon, \delta)$-估计值，给定一个合适的 $k = O(\frac{1}{\epsilon^2 \delta})$：

{% note info %}

对于任意的 $\epsilon, \delta<1 / 2$，如果 $k \geq\left[\frac{4}{\epsilon^{2} \delta}\right]$ 则输出 $\widehat{Z}$ 总是给出一个 $z$ 的真实值的 $(\epsilon, \delta)$-estimator。

{% endnote %}

~~证略，一样有空再来填坑。~~



## 均匀哈希假设 Uniform Hash Assumption

上面的所有方法在将哈希函数换成 $h: \Omega \to [M]$  的情况下都可以获得相同好的表现，这里 $M = poly(n)$，也就是可以用 $O(\log{n})$ 个bit来表示出 $M$。

即使有这个分析，一个形式为 $h: [N] \to [M]$ 的均匀的随机离散函数都无法被高效地储存和计算。根据信息理论，表示这样一个函数至少需要 $O(N \log{M})$ 个 bit，因为这是均匀随机函数的熵（entropy）。

所以为了方便分析，我们一般有 **Uniform Hash Assumption(UHA)** 或者叫 **Simple Uniform Hash Assumption (SUHA)**：

{% note info %}

一个均匀随机函数 $h: [N] \to [M]$ 是可获得且计算高效的。

{% endnote %}



# 集合成员 Set Membership

这个基本的问题就是：

给定全集 $\Omega $ 下的一个 $n$ 元集合 $S$，我们想用一个数据结构来表示 $S$，使得每次对于元素 $x \in [N]$ 的查询（query），对于 $x$ 是否属于 $S$ 能够高效地回答。

- 有序表、平衡搜索树：$O(n\log{N})$ 的空间复杂度，$O(\log{n})$ 的时间复杂度。

- 好的哈希：$O(n\log{N})$ 的空间复杂度，$O(1)$ 的时间复杂度。

注意到 $\log \left(\begin{array}{l}N \\ n\end{array}\right)=\Theta\left(n \log \frac{N}{n}\right)$ 实际上是 $S$ 的熵，也就是说 $O(n\log{N})$ 的空间开销是不可避免的。但如果我们使用一个损失表示并容忍一个有界的查询误差就可以做的更好，这样的一种数据的有损表示称作**sketch**。



## Bloom Filter

Bloom filter 考虑一个 $cn$ 个 bit 的数组 $A$，$k$ 个哈希函数 $h_1, h_2, ..., h_k: \Omega \to [cn]$。

``` pseudocode
construction:
	initialize all cn bits of the Boolean array A to 0
	for each x in S, A[hi(x)] = 1 for 1 <= i <= k

query:
	if A[hi(x)] = 1 for 1 <= i <= k:
		return yes
	else:
		return no
```

这个布尔数组就是我们的数据结构，空间复杂度 $O(cn)$，一次查询的时间复杂度 $O(k)$。

当算法返回 `no`，也就存在 $1 \leq i \leq k$ 的 $A[h_i(x)] = 0$，就是说 $x$ 一定不属于 $S$。这说明这个算法不存在假阴性。

当算法返回 `yes`，对所有 $1 \leq i \leq k$ 的 $A[h_i(x)] = 1$，不过有假阳性，也就是存在 $x \notin S$，所有的 $A[h_i(x)]= 1$，我们需要限制这个误差，也即是当 $x\notin S$ 时：

$\operatorname{Pr}\left[\forall 1 \leq i \leq k, A\left[h_{i}(x)\right]=1\right] = \operatorname{Pr}\left[A\left[h_{1}(x)\right]=1\right]^{k}=\left(1-\operatorname{Pr}\left[A\left[h_{1}(x)\right]=0\right]\right)^{k}$。

$A[h_i(x)]$ 逃过所有 $kn$ 次的更新的概率是 $\operatorname{Pr}\left[A\left[h_{1}(x)\right]=0\right]=\left(1-\frac{1}{c n}\right)^{k n} \approx e^{-k / c}$。

然后就有：

$\begin{aligned}
\operatorname{Pr}[\text { wrongly answer "yes" }] &=\operatorname{Pr}\left[\forall 1 \leq i \leq k, A\left[h_{i}(x)\right]=1\right] \\
&=\operatorname{Pr}\left[A\left[h_{1}(x)\right]=1\right]^{k}=\left(1-\operatorname{Pr}\left[A\left[h_{1}(x)\right]=0\right]\right)^{k} \\
&=\left(1-\left(1-\frac{1}{c n}\right)^{k n}\right)^{k} \\
& \approx\left(1-e^{-k / c}\right)^{k}
\end{aligned}$

当 $k = c\ln{2}$ 时这个概率大概是 $(0.6185)^c$。

Bloom Filter 解决了这个问题，只有很小的常数单边假阳性误差，$O(n)$ bits的空间开销和 $O(1)$ 的查询时间开销。



# 频率估计 Frenquency Estimation

就是对于一串输入 $x_1, x_2, ..., x_n \in \Omega$，查询 $x$ 出现的次数 $f_x = |\{i | x_i = x\}|$。

一样的，储存整个输入开销太大了，我们要找一个 **sketch**，算法返回 $f_x$ 满足：$\operatorname{Pr}\left[\left|\hat{f}_{x}-f_{x}\right| \leq \epsilon n\right] \geq 1-\delta$。



## 最小计数 Count-min sketch

**Count-min sketch** 是对于频率估计的一个非常优雅的数据结构。

这个数据结构需要一个 $k \times m$ 的二维数组 CMS，我们仍然假设可以获得 $k$ 个独立均匀分布的哈希函数 $h_1, h_2,...,h_k: \Omega \to [m]$。

``` pseudocode count-min
construct:
	initialize all entries of CMS[k][m] to 0
	for i = 1 to n:
		for j = 1 to k:
			evaluate hj(xi) and CMS[j][hj(xi)++
			
query:
	return f_x = min(CMS[j][hj(xi)])
```

空间复杂度 $O(km)$，时间复杂度 $O(k)$。下面我们来分析它的误差边界。

很显然 $CMS[j][h_j(x_i)] \leq f_x$，因为至少加了 $f_x$ 次，即 $\hat{f}_{x}=\min _{1 \leq j \leq k} C M S[j]\left[h_{j}(x)\right] \geq f_{x}$。

所以 $\operatorname{Pr}\left[\left|\hat{f}_{x}-f_{x}\right| \geq \epsilon n\right]=\operatorname{Pr}\left[\hat{f}_{x}-f_{x} \geq \epsilon n\right]=\prod_{j=1}^{k} \operatorname{Pr}\left[C M S[j]\left[h_{j}(x)\right]-f_{x} \geq \epsilon n\right]$。

然后我们就需要界定 $\operatorname{Pr}\left[C M S[j]\left[h_{j}(x)\right]-f_{x} \geq \epsilon n\right]$，可以通过计算 $C M S[j]\left[h_{j}(x)\right]$ 的期望。

可证如下命题：

{% note info %}

对任意的 $x\in \Omega$，任意的 $1 \leq j \leq k$，$\mathbb{E}\left[C M S[j]\left[h_{j}(x)\right]\right] \leq f_{x}+\frac{n}{m}$。

{% endnote %}

所以有 $\mathbb{E}\left[C M S[j]\left[h_{j}(x)\right] - f_x \right] \leq \frac{n}{m}$。

因为 $C M S[j]\left[h_{j}(x) \right] \geq f_x$ 所以 $C M S[j]\left[h_{j}(x)\right] - f_x$ 是正随机变量，由马尔科夫不等式：

$\operatorname{Pr}\left[C M S[j]\left[h_{j}(x)\right]-f_{x} \geq \epsilon n\right] \leq \frac{1}{\epsilon m}$。

再和上面的结合一下：

$\operatorname{Pr}\left[\left|\hat{f}_{x}-f_{x}\right| \geq \epsilon n\right]=\left(\operatorname{Pr}\left[C M S[j]\left[h_{j}(x)\right]-f_{x} \geq \epsilon n\right]\right)^{k} \leq \frac{1}{(\epsilon m)^{k}}$。

令 $m = \left\lceil \frac{e}{\epsilon} \right\rceil, k = \left\lceil \ln{\frac{1}{\delta}} \right\rceil$，这个误差概率可以被限制在 $\frac{1}{(\epsilon m)^k} \leq \delta$。
