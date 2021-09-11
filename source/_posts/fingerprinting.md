---
title: 高级算法课程笔记-指纹
date: 2021-09-09 22:46:09
tags:
categories:
- 课程
- Advanced Algorithm
mathjax: true
---



# 多项式判等 Polynomial Identity Testing

**PIT问题**：

输入两个 $d$ 阶多项式 $f, g \in \mathbb{F}[x]$，输出他们是否相等 $f \equiv g$。

这里 $\mathbb{F}[x]$ 指 $x$ 在域（field）$\mathbb{F}$ 上的多项式环（ring）。

我们可以很自然地将这个问题转化为判断一个 $d$ 阶多项式（$f - g$）是否恒为 0。

如果 $f$ 被显式地给出了，我们很容易地可以通过遍历它的系数来判断。为了使问题更加 nontrivial，假设 $f$ 是一个黑盒（black box/ oracle），看不到系数，只能通过输入 $x$ 得到 $f(x)$。

<!--more-->

- 一个直接的确定性方法就是带入 $d+1$ 个不同的数计算 $f(x_1), f(x_2), ...,f(x_{d+1})$ 来计算是否全为零，若全为零则根据**代数基本定理**，能保证 $d$ 阶多项式 $f$ 一定恒为零。

{% note info %}

一个非零 $d$ 阶单变量多项式一定至多有 $d$ 个根。

{% endnote %}



然后自然地我们有如下算法：

1. suppose we have a finite subset $S \subseteq \mathbb{F}$
2. pick $r \in S$ uniformly at random
3. if $f(r) = 0$ return 'yes' else 'no'

我们很容易可以得出：

- 如果 $f \equiv 0$，算法永远返回 'yes'。
- 如果 $f \not \equiv 0$，算法可能错误地返回 'yes'（假阳性），但这只当 $r$ 是 $f$ 的根时发生。根据代数基本定理，$f$ 最多 $d$ 个根，所以算法出错的可能性有界：
  - $Pr[f(r) = 0] \leq \dfrac{d}{|S|}$。
  - 我们可以固定 $|S| = 2d$，这样出错的概率最多就是 $1/2$。然后通过独立重复 $\log_{2}{\frac{1}{\delta}}$ 次来将这个概率减小到 $\delta$。



## Equality 的通信复杂性

 **通信复杂性**（[communication complexity](http://en.wikipedia.org/wiki/Communication_complexity)）是姚期智先生（对就是那个图灵奖）提出来的多个实体的计算模型。简单来说就是有两个端，分别有两个输入 $a$ 和 $b$，我们要计算一个函数 $f(a,b)$，不考虑计算的复杂性，我们只关心**通信协议（communication protocol）**，也就是两个端传输的信息的比特。

~~这个其实在问求4讲过不过我一直没搞明白。~~

在这里我们讨论的函数是 $\mathrm{EQ}: \{0, 1\}^n \times \{0, 1\}^n \to \{0, 1\}$，对任意 $a, b \in \{0, 1\}^n$ 有：

$\mathrm{EQ}(a, b)= \begin{cases}1 & \text { if } a=b \\ 0 & \text { otherwise }\end{cases}$

- 最简单的想法就是把整个 $a$ 或 $b$ 传过去，通信复杂度是 $n$ 个 bit。
  - 可证（yao 证出来的）这是最好的确定性通信协议，也就是要确定地保证完全相等，至少需要 n bit的传输。
  - 这个证明其实远比它看起来要 nontrivial。



我们可以考虑将 $a,b$ 看成 $n$ 阶多项式 $f,g$，然后运用我们刚才的 **PIT** 来解决。

需要传输 $r, f(r)$，$r \in [2n]$ 所以需要 $O(\log{n})$ 个bit，而 $f(r)$ 实际上是 $f(r) = \sum\limits_{i = 0}^{n-1} a_i r^i = O(r^n) = O(n^n)$ 是一个指数量级，需要 $O(n\log{n})$ 个bit。这比直接传整个数据库过去还多，~~屁用没有~~。

- 所以根本原因在于我们选的**域**太大了，是和 $n$ 正比的，能不能找到小一点的，最好是一个常数大小的域。
  - 很快地我们就能够想到在模意义下取数，也就是 $\mathbb{Z}_p = \{0, 1, ..., p-1\}$。


- 我们只需要随机取一个质数 $p \in [n^2, 2n^2]$（可证一定能找到），$f,g \in \mathbb{Z}_p[x]$，然后随机取 $r \in [p]$。
  - 这样所需要传输的数据 $r, f(r) < p$，只需要 $O(\log{p}) = O(\log{n})$ 的通信复杂度。
  - 单边错概率是 $\dfrac{n}{p} = O(\dfrac{1}{n})$，w.h.p。



## Schwartz-Zippel 定理

接下来我们再来看**PIT问题**的更一般的形式，也就是推广到多元：

对于两个 $n$ 元 $d$ 阶多项式 $f, g \in \mathbb{F}[x_1, ..., x_n]$，判断它们是否相等 $f \equiv g$。

仍然我们可以将问题转化成多项式判零：

$f\left(x_{1}, \ldots, x_{n}\right)=\sum\limits_{i_{1}, \ldots, i_{n} \geq 0} a_{i_{1}, i_{2}, \ldots, i_{n}} x_{1}^{i_{1}} x_{2}^{i_{2} \ldots} x_{n}^{i_{n}} \equiv 0?$

这里注意一个多元多项式的阶（degree）是指一项中所有变量的幂次的和的最大值，所以进一步的：

$f\left(x_{1}, \ldots, x_{n}\right)=\sum\limits_{i_{1}, \ldots i_{n} \geq 0 \atop i_{1}+\cdots+i_{n} \leq d} a_{i_{1}, i_{2}, \ldots, i_{n}} x_{1}^{i_{1}} x_{2}^{i_{2} \ldots} x_{n}^{i_{n}}$

如果 $f$ 是显式地给出的，我们可以遍历所有系数，共最多 $\left(\begin{array}{c} n+d \\ d \end{array}\right) \leq(n+d)^{d}$ 个来检查是否为零（这里其实用到了一点组合知识）。

一样的 $f$ 会作为一个黑盒，给定 $\vec{x} \in \mathbb{F}^{n}$，返回 $f(\vec{x})$。

或者作为乘积形式（product form）。比如范德蒙行列式（Vandermonde determinant）：
$$
M=\left[\begin{array}{ccccc}
1 & x_{1} & x_{1}^{2} & \ldots & x_{1}^{n-1} \\
1 & x_{2} & x_{2}^{2} & \ldots & x_{2}^{n-1} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n} & x_{n}^{2} & \ldots & x_{n}^{n-1}
\end{array}\right] \quad f(\vec{x})=\operatorname{det}(M)=\prod\limits_{j<i}\left(x_{i}-x_{j}\right)
$$
乘积形式的特点就在计算一个点处的函数值非常方便，而展开算系数就会极其复杂。



如果能存在一个多项式时间的确定性算法解 PIT，那意味着两个本质性问题：$\mathbf{NEXP} \neq \mathbf{P/poly}$ 或者 $\mathbf{\sharp P} \neq \mathbf{FP}$，当然我已经不懂了。

所以依然和上面类似的我们可以有一个随机算法，在 $S \subseteq \mathbb{F}$ 中随机取点，0返回 yes 否则 no。

而它的正确性，也就是假阳性，也就是取到它的根的概率由如下 **Schwartz-Zippel 定理** 给出：

{% note info %}

对于 $d$ 阶多项式 $f \in \mathbb{F}[x_1, x_2, ..., x_n]$，如果 $f \not \equiv 0$，那么对于任何有限集 $S \in \mathbb{F}$ 和随机均匀独立选取的 $r_1, r_2, ..., r_n \in S$，有

$Pr[f(r_1, r_2, ..., r_n) = 0] \leq \dfrac{d}{|S|}$。

{% endnote %}

这个定理也说明了 $d$ 阶 $n$ 元多项式在任意 $S^n$ 中的根的个数至多 $d \times |S|^{n-1}$ 个。

原证明较复杂，Dana Moshkovitz 在之后给出了一个很简洁优雅的证明，~~我有空来写一遍~~。
