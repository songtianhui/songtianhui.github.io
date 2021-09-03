---
title: 问题求解笔记-数论算法
date: 2021-04-08 09:52:18
categories: 
- 课程
- problem-solving
mathjax: true
---

---

# 数论算法

**定理 31.2.** 如果任意整数 $a$ 和 $b$ 都不为 $0$，则 $gcd(a, b)$ 是 $a$ 与 $b$ 的线性组合集 $\{ ax + by : x, y \in \mathbb{Z} \}$ 中的最小正元素。

**推论 31.3.** 任意整数 $a$ 和 $b$，$d | a \wedge d|b \Rightarrow d | gcd(a,b)$。

**推论 31.4** 对所有整数 $a,b$ 及任意非负整数 $n$，有 $$gcd(an,bn) = n~gcd(a,b)$$。

**推论 31.5.** 对所有正整数 $a,b,n$，$n | ab \wedge gcd(a, n) = 1 \Rightarrow n | b$。

**定理 31.6.** 对所有正整数 $a,b,p$，$gcd(a, p ) = 1 \wedge gcd(b, p) = 1 \Rightarrow gcd(ab, p) = 1$。

**定理 31.7.** 对所有的素数 $p$ 和所有整数 $a,b$， $p | ab \Rightarrow p |a \vee p | b$。

**定理 31.8.**（唯一因子分解定理）合数 $a$ 的素因子分解是唯一的。

<!--more -->

**定理 31.9** $\forall a, b, gcd(a, b) = gcd(b, a ~mod ~b)$.

## 欧几里得算法

![ECULID](euclid.png "eculid")

（这个图片为什么这么大。。。就这样吧反正我也不会调）

### 欧几里得算法的运行时间

**引理 31.10.** 如果 $a > b \geq 1$ 并且 ECULID$(a,b)$ 执行了 $k \geq 1$ 次递归调用，则 $a \geq F_{k+2}, b \geq F_{k+1}$。

**引理 31.11.**（Lame引理）对任意整数 $k \geq 1$，如果 $a > b \geq 1$，且 $b < F_{k+1}$，则 EUCLID$(a,b)$ 的递归调用次数少于 $k$ 次。

### 扩展欧几里得算法

![EXT-ECULID](extgcd.png "ext-eculid")

- 递归调用次数为 $O(\lg{b})$。

## 模运算

- **欧拉函数**：$$\phi(n) = n \prod\limits_{p\text{是素数且}p|n} (1 - \dfrac{1}{p})$$

- $\phi(n) > \dfrac{n}{e^\gamma \ln{\ln{n}} + \dfrac{3}{\ln{\ln{n}}}}$

  

  

  **定理31.15.** （拉格朗日定理）：如果 $(S, \oplus)$ 是一个有限群，$(S', \oplus )$ 是 $(S,\oplus)$ 的一个子群，则 $|S'|$ 是 $|S|$ 的一个约数。（我觉得比TJ上讲的通俗易懂得多。。。）

  **推论 31.16.** 如果 $S'$ 是有限群 $S$ 的真子群，则 $|S' | \leq |S| / 2$。

**定理 31.17.** 对任意有限群 $(S, \oplus)$ 和任意 $a \in S$，一个元素的阶等于它所生成的子群的规模，即 $ord(a) = |\langle a \rangle|$。

**推论 31.18.** 序列 $a^{(1)}, a^{(2)}, ...$是周期序列，其周期为 $t = ord(a)$，即 $a^{(i)} = a^{(j)} \Leftrightarrow i \equiv j~ (\text{mod }t)$。

**推论 31.19.** 如果 $(S, \oplus)$ 是具有单位元的有限群，则对所有 $a \in S$，$a^{(|S|)} = e$。



## 求解模线性方程

**定理 31.20.** 对任意正整数 $a$ 和 $n$，如果 $d = gcd(a,n)$，则在 $\mathbb{Z}_n$ 中，$\langle a \rangle = \langle d \rangle = \{ 0, d, 2d, \cdots , ((n/d) - 1)d \}$。因此，$|\langle a \rangle| = n /d$。

**推论 31.21.** $d | b \Leftrightarrow \text{方程} ax \equiv b (\text{mod } n)$ 对 $x$ 有解，$d  =gcd(a,n)$。 

**推论 31.22** 方程 $ax \equiv b (\text{mod } n)$ 要么模 $n$ 下有 $d$ 个不同的解，要么无解。

**定理 31.23.** 假设对某些整数 $x', y'$ 有 $d = ax' + ny'$。如果 $d | b$，则方程 $ax \equiv b(\text{mod } n)$ 有一个解的值为 $x_0$，$$x_0 = x'(b / d) \mod n$$

**定理 31.24.** 假设方程 $ax \equiv b(\text{mod } n)$ 有解，且 $x_0$ 是该方程的任意一个解。则该方程对模 $n$ 恰有 $d = gcd(a,n)$ 个解为 $x_i = x_0 + i(n /d), i =0, 1, \cdots, d- 1$。



## 中国剩余定理

**定理 31.27.**（中国剩余定理） 令 $n = n_1 n_2\cdots n_k$，其中因子 $n_i$ 两两互质。考虑以下对应关系：

$$a \leftrightarrow (a_1, a_2, \cdots, a_k)$$

这里 $a \in \mathbb{Z}_n, a_i \in \mathbb{Z}_{n_i}$，且对 $i = 1, 2, \cdots, k$，

$$a_i =  a \mod n_i$$

则通过在合适的系统中对每个坐标位置独立地执行操作，对 $\mathbb{Z}_n$ 中元素所执行地运算可以等价地作用于对应的 $k$ 元组。

**推论 31.28.** 如果 $n_1, n_2, \cdots, n_k$ 两两互质，则关于未知量 $x$ 地联立方程组 $$x \equiv a_i (\text{mod } n_i), i = 1, 2, \cdots,k$$ 对模 $n$ 有唯一解。

**推论 31.29.** 如果 $n_1, n_2, \cdots, n_k$ 两两互质，则对所有整数 $x$ 和 $a$，$x \equiv a(\text{mod } n_i), i = 1, 2, \cdots, k \Leftrightarrow x \equiv a (\text{mod } n)$。



## 元素的幂

**定理 31.32.** 对所有地素数 $p > 2$ 和所有的正整数 $e$，使得 $\mathbb{Z}_n^*$ 是循环群的 $n > 1$ 的值为 $2, 4, p^e, p^{2e}$。

如果 $g$ 是 $\mathbb{Z}_{b}^{\star}$ 的一个原根且 $a$ 是 $\mathbb{Z}_{b}^{\star}$ 中的任意元素，则存在一个 $z$，使得 $g^{z} \equiv a(\text{mod } n)$ 。这个 $z$ 称为对模 $n$ 到基 $g$ 上的一个**离散对数**。

**定理 31.33.**（离散对数定理）如果 $g$ 是 $\mathbb{Z}_b^*$ 的一个原根，$x \equiv y(\text{mod } \phi(n)) \Leftrightarrow g^x \equiv g^y (\text{mod } \phi(n))$。



**定理 31.34.** $p$ 是一个奇素数且 $e > 1 \Rightarrow x^2 \equiv 1(\text{mod } p^e)$ 仅有两个解 $x =1, x= -1$。

## 反复平方法

快速幂