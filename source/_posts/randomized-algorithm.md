---
title: 问题求解笔记-随机算法
categories: 
- 课程
- problem-solving
date: 2021-05-26 03:55:52
mathjax: true
---


# 介绍

我们至今仍然不知道任何NP难问题的随机多项式算法。

我们只知道一些问题，满足：

- 有高效的多项式随机算法
- 没有已知的多项式时间确定算法
- 不知道它是否属于 $P$

随机算法和近似算法结合的合理性在于，算法输出错误结果的概率甚至小于确定性算法长时间运行下硬件出错的概率。

<!--more -->



---

# 随机算法的分类

## 一些概念

我们可以把随机算法就看成是在运行中需要随机数的算法，并且后续运行需要这些随机数。

- $S_{A,x} = \{ C | C \text{ 是 }A \text{ 在 }x \text{ 上的一个随机计算}\}$。（样本空间）
- $Prob $ 是 $S_{A,x} $ 的概率分布。

- $Random_A(x)$ 是 $A $ 在 $x$ 上运行中的最大随机数位数。
- $Random_A(n) = \max\{Random_A(x) | x \text{ 是大小为 } n \text{ 的输入}\}$。

复杂度的衡量是重要的因为：

- 生成伪随机数需要开销，一般和位数成正比
- “去随机化”



算法 $A$ 对于输入 $x$ 输出 $y$ 的概率 $Prob(A(x) = y)$ 是所有 $C$ 输出 $y$ 的 $Prob_{A , x}(C) $ 的和。随机算法的目标就是让输出正确的 $y$ 的 $Prob(A(x) = y)$ 尽可能大。

- $Time(C) $ 是 $C$ 的运行时间，则 $A $ 对于输入 $x$ 的期望时间复杂度是： 

  $$Exp\text{-}Time_A(x) = E[Time] = \sum\limits_{C}Prob_{A,x}(C) \cdot Time(C)$$

- $Exp\text{-}Time_A(n) = \max\{Random_A(x) | x \text{ 是大小为 } n \text{ 的输入}\}$。

- 有时我们也考虑最坏情况复杂度： $Time_A(x) = \max\{Time(C) | C \text{ 是 }A \text{ 在 }  x \text{ 上的运行} \}$。

- $Time_A(n) = \max\{Time_A(x) | x \text{ 是大小为 } n \text{ 的输入} \}$。

## 分类

### 拉斯维加斯算法（Las Vegas Algorithms）

定义一个算法 $A$ 是问题 $F$ 的拉斯维加斯算法：

1. 对于任意的输入 $x$，$Prob(A(x) = F(x)) = 1$，$F(x)$ 是问题 $F$ 对 $x$ 的解。
   - 这样 $A$ 的复杂度一般被认为是 $Exp\text{-}Time_A(n)$。

2. 我们允许输出 "?"（不能按时解决问题）。对任意的输入 $x$，$Prob(A(x) = F(x)) \geq \dfrac{1}{2}$ 且 

   $$Prob(A(x)= ?)=  1 - Prob(A(x) = F(x)) \leq \dfrac{1}{2}$$。

   - 这样 $A$ 的复杂度一般被认为是 $Time_A(n)$，因为一般需要运行到 $Time_A(x)$ 才能判断“?”。



### 单错蒙特卡罗算法（One-Sided-Error Monte Carlo Algorithm）

只对判定问题有用。

算法 $A$ 是语言 $L$ 的单错蒙特卡罗算法当：

1. 对任意的 $x \in L$，$Prob(A(x) = 1 ) \geq 1/2$；
2. 对任意的 $x \notin L$，$Prob(A(x)  = 0) = 1$。

所以单错蒙特卡罗算法不会误判错误的输入，但有小概率误判正确的输入，只有单方向错。



### 双错蒙特卡算法（Two-Sided-Error Monte Carlo Algorithm）

$F$ 是一个计算问题，一个随机算法 $A$ 是双错蒙特卡算法当：

- 存在一个实数 $0 < \varepsilon \leq 1/2$，使得对任意的输入 $x$ ， $Prob(A(x) = F(x)) \geq \dfrac{1}{2}  + \varepsilon$。
- 让算法跑 $t$ 次，取出现次数至少是 $\lceil t / 2 \rceil$ 的结果，正确的概率很大。



### 无界错蒙特卡罗算法（Unbounded-Error Monte Carlo Algorithm）

$F$ 是一个计算问题，一个随机算法 $A$ 是无界错蒙特卡算法当：

- 对任意输入 $x$，$Prob(A(x) = F(x)) > \dfrac{1}{2}$。
- 缺点在于 $Prob(A(x) = F(x))$ 和 $\dfrac{1}{2}$ 可能趋近于 $0$ 当输入很大时。
- 为了从无界错蒙特卡罗算法获得一个随机算法满足 $Prob(A_{k(|n|)} (x) = F(x)) \geq 1 - \delta, 0 \leq \delta \leq \dfrac{1}{2}$，我们必须接受 $Time_{A_{k(n)}}(n) = O(2^{2Random_A(n)} \cdot Time_A(n))$



### 随机优化算法（Randomized Optimization Algorithm）

我们一般不考虑正确解出现的频率，而直接取最优解。所以一个 $k$ 轮算法中最优解的概率是
$$
Prob(A_k(x) \in Output_U(x)) = 1 - [Prob(A(x) \notin Output_U(x))]^k,
$$
其中 $A_k$ 指运行 $k$ 次随机优化算法 $A$。

- 若 $Prob(A(x) \notin Output_U(x)) \leq \varepsilon, \varepsilon < 1$，则 $A$ 和单错蒙特卡罗效率差不多。
- 若 $Prob(A(x) \in Output_U(x)) \geq 1 / p(|x|)$，$p$ 为多项式，则 $A$ 的效率也很可观。



## 随机近似算法

- $U = (\Sigma_I, \Sigma_O, L, L_I,\mathcal{M}, cost, goal)$ 是一个优化问题，对任意的实数 $\delta > 1$，一个随机算法 $A$ 叫做 $U$ 的**随机$\delta$-近似算法**，当满足对任意的 $x\in L_I$：
  - $Prob(A(x) \in \mathcal{M}(x)) = 1$，
  - $Prob(R_A(x) \leq \delta)\geq 1/ 2$。
- 对任意的函数 $f :\mathbb{N} \to \mathbb{R}^+$，$A$ 叫做 $U$ 的**随机$f(n)$-近似算法**，当满足对任意的 $x \in L_I$：
  - $Prob(A(x) \in \mathcal{M}(x)) = 1$，
  - $Prob(R_A(x) \leq f(|x|)) \geq 1/2$。
- 一个随机算法 $A$ 叫做 $U$ 的**随机多项式时间近似方案（RPTAS）**，如果存在一个函数 $p: L_I \times \mathbb{R}^+ \to \mathbb{N}$，使得对于任何输入 $(x, \delta) \in L_I \times \mathbb{R}^+$，满足：
  - $Prob(A(x, \delta) \in  \mathcal{M}(x)) = 1$，对所有的选择 $A$ 都可以计算出一个可行解。
  - $Prob(\varepsilon_A(x, \delta) \leq \delta) \geq \dfrac{1}{2}$，一个可行解的相对误差不超过 $\delta$ 的概率大于 $1/2$。
  - $Time_A(x, \delta^{-1}) \leq p(|x| , \delta^{-1})$ 且 $p$ 是关于 $|x| $ 的多项式。
- 类似的，如果 $p$ 是同时关于 $|x| $ 和 $\delta^{-1}$ 的多项式，$A$ 叫做 $U$ 的**随机完全多项式时间近似方案（RFPTAS）**。



- $U = (\Sigma_I, \Sigma_O, L, L_I,\mathcal{M}, cost, goal)$ 是一个优化问题，对任意的实数 $\delta > 1$，一个随机算法 $A$ 叫做 $U$ 的**随机$\delta$-期望算法**，当满足对任意的 $x\in L_I$：
  - $Prob(A(x) \in \mathcal{M}(x)) = 1$，
  - $E[R_A(x)] \leq \delta$。



---

# 随机算法的设计范式

## 挫败敌人（Foiling an Adversary）

- 对手论证
- 随机算法可以看作是确定性算法上的概率分布，对手可以构造一小部分很强（大开销）的确定性算法的输入，但很难设计击败随机算法的输入。



## 丰富的证人？（Abundance of Witnesses）



## 指纹？（Fingerprinting）



## 随机样本（Random Sampling）



## （Relaxation and Random Rounding）

