---
title: 问题求解笔记-数论基础
date: 2021-03-20 13:39:27
categories: 
- 课程
- problem-solving
mathjax: true
---

# 数学归纳法 Mathematical Induction
没啥好讲的。
**第一第二数学归纳法**
**良序公理**
<!--more -->

# 辗转相除法 Division Algorithm

**定理 2.9.** $a$ 和 $b$ 为整数，$b > 0$，则存在**唯一整数** $q$ 和 $r$，使得
$$a = bq + r$$
其中 $0 \leq r < b$。


$a$ 整除 $b$ 记作 $a | b$。
最大公约数、互质等概念略。

**定理 2.10.** $a$ 和 $b$ 为非零整数，则存在整数 $r，s$，使得
$$gcd(a,b) = ar + bs$$
进一步，$a,b$的最大公约数是唯一的。

**推论 2.11.** $a,b$ 互质，则存在整数 $r,s$，使得 $ar + bs = 1$。（可证反之也成立）

## 欧几里得算法 Euclidean Algorithm

## 质数 Prime Number

**定理 2.14.** 存在无穷多质数。（证明挺重要的）

**定理 2.15. 算数基本定理** $n$ 为整数，则 $n$ 可以唯一表示为多个素数的乘积。
$$n = p_1 p_2 ... p_k$$

---
# 乘法逆元和最大公约数

**乘法逆元 Multipllication Inverse**: $a' \in \mathbb{Z}_n, a \in \mathbb{Z}_n, a \cdot_n a' = a$，则称 $a'$ 是 $a$ 在 $\mathbb{Z}_n$ 中的乘法逆元。

**引理 2.5.** 解模方程： 如果 $a$ 在 $\mathbb{Z}_n$ 中有逆元 $a'$，则对任意 $b \in \mathbb{Z}_n$，方程
$$a \cdot_n x = b$$
有唯一解 $x = a' \cdot_n b$。

**推论 2.6.** 如果存在 $b \in \mathbb{Z}_n$，方程
$$a \cdot_n x = b$$
无解，则 $a$ 没有逆元。

**定理 2.7.** 若有逆元则唯一。

**引理 2.8.** 方程
$$a \cdot_n x = 1$$
有解当且仅当存在整数 $x,y$，使得
$$ax + ny = 1$$

**定理 2.9.** $a \in \mathbb{Z}_n$ 有逆元 $\Leftrightarrow$ 存在整数 $x，y$，使得 $ax + ny = 1$。

**推论 2.10.** $a \in \mathbb{Z}_n$ 且存在整数 $x，y$，使得 $ax + ny = 1$，则 $a$ 的逆元为 $x$ (mod $n$)。

**引理 2.11.** 给定 $a,n$，如果存在整数 $x，y$，使得 $ax + ny = 1$，则 $a,n$ 互质，即 $gcd(a,n) = 1$。

**定理 2.12.** 同最上面一条。

**引理 2.13.** $j,k,q,r$ 是正整数，满足 $k = j q + r$，则
$$gcd(j,k) = gcd(r, j)$$

**定理 2.15.** $gcd(j,k)  = 1 \Leftrightarrow \exists x,y, jx + ky = 1$.

**推论 2.16.** $a\in \mathbb{Z}_n$ 在 $\mathbb{Z}_n$ 中有逆元 $\Leftarrow gcd(a,n) = 1$。

**推论 2.17.** 对任意质数 $p$，每个非零整数 $a \in \mathbb{Z}_p$ 都有逆元。