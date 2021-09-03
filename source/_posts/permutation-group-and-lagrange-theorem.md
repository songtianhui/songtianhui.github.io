---
title: 问题求解笔记-置换群和拉格朗日定理
date: 2021-03-07 13:05:13
categories: 
- 课程
- problem-solving
mathjax: true
---

# 置换群 Permutation Groups
## 定义和记号
- 集合$S$上的一个**置换（permutation）**：one-to-one and onto map: $\pi: S \rightarrow S$

<!--more -->

- 集合$X$的所有置换构成一个群$S_{X}$，$X = \{ 1, 2, ..., n\}$时记为$S_{n}$。该群叫**对称群（Symmetric Group）**。

**定理5.1.** $n$个letter的对成群$S_{n}$是一个有$n!$个元素的群，其二元运算为复合（composition）。

- $S_{n}$的子群称为**置换群（Permutation Group）**

### 轮换（Cycle Notation）
- 一个置换$\sigma \in S_{X}$是一个长度为$k$的**轮换（cycle）**，如果存在$a_{1}, a_{2}, ..., a_{k}\in X$，使得
$\sigma(a_{1}) = a_{2}, \sigma(a_{2}) = a_{3}, ... ,\sigma(a_{k}) = a_{1}$。

**命题5.8.** $\sigma$ 和 $\tau$ 是$S_{X}$中两个不相交轮换 $\Rightarrow \sigma \tau = \tau \sigma$

**定理5.9.** $S_{n}$中的每个置换都可以写成不相交轮换的乘积。

### 对换（Transposition）
**对换（Transposition）**：长度为2的轮换。
$(a_{1}, a_{2}, ..., a_{3}) = (a_{1}, a_{n})(a_{1}, a_{n-1})...(a_{1}a_{2})$

**命题5.12.** 每个元素个数大于2的有限集合的置换都可以写成对换的乘积。

**引理5.14.** 如果恒等变换被写成$id = \tau_{1} \tau_{2} ... \tau_{r} \Rightarrow r$是偶数。

**定理5.15.**  如果一个排列能被写成偶数个置换乘积的形式，那么另一个与之等价的排列也一定拥有偶数项置换乘积。奇数同理。
据此定理可以将排列分为奇偶两类

### 交替群 Alternating Groups
**交替群**：$S_{n}$的所有偶置换$A_{n}$。

**定理5.16.** $A_{n}$ 是 $S_{n}$ 的子群

**命题5.17.** 奇偶置换的个数相等，都是$n!/2$，即$A_{n}$的大小。

## 二面体群 Dihedral Groups
**n阶二面体群**：正n边形的刚性运动，记为$D_{n}$

**定理5.20.** $D_{n}$ 是 $S_{n}$ 的大小为$2n$的子群。

**定理5.23.** 包含所有旋转$r$ 和对称 $s$ 的乘积的群 $D_{n}$，满足以下两个关系：
$$r^{n} = 1$$
$$s^{2} = 1$$
$$srs = r^{-1}$$

## 立方体移动群 The Motion Group of a Cube
**命题5.27.** 立方体移动群有24个元素。
**定理5.28.** 立方体移动群是$S_{4}$。

---
# 陪集和拉格朗日定理

## 陪集 Cosets
$G$ 是一个群，$H$ 是 $G$ 的子群。集合$H$的代表元为$g\in G$的**左陪集（Left Coset）**：$gH = \{ gh: h \in H \}$。右陪集类似。

**引理6.3.** $H$ 为群 $G$ 的子群，$g_{1}, g_{2} \in G$，则以下条件等价：
1. $g_{1}H = g_{2} H$
2. $Hg_{1}^{-1} = Hg_{2}^{-1}$
3. $g_{1}H \subset g_{2}H$
4. $g_{2} \in g_{1}H$
5. $g_{1}^{-1}g_{2} \in H$

**定理6.4.** $H$ 为群 $G$ 的子群。$H$ 的所有左陪集分割(partition)了 $G$,即 $G$ 是 $H$ 的左陪集的disjoint union. (右陪集同理)

$H$ 的**index**：$G$ 中 $H$ 的左陪集的个数，记为$[G:H]$。

**定理6.8.** $H$ 的左陪集和右陪集个数相等。

## 拉格朗日定理 Lagrange's Theorem
**命题6.9.** map $\phi: H \rightarrow gH$是双射。所以 $H$ 和 $gH$ 的元素个数相同。

**定理6.10. Lagrange** $G$ 的元素个数是子群 $H$ 的元素个数的整数倍，为 $H$ 的所有左陪集的个数。即$|G| / |H| = [G:H]$。

**推论6.11.** $G$ 的所有元素$g$的order是$|G|$的约数。

**推论6.12.** $|G|= p$，$p$为质数，则 $G$ 是循环群且除了单位元$e$的元素都是生成元。

**推论6.13.** $K \subset H \subset G \Rightarrow [G:K] = [G:H][H:K]$.

- 拉格朗日定理的逆命题不一定成立。

**定理6.16.** $S_{n}$ 中的两个轮换 $\tau$ 和 $\mu$ 有相同的长度 $\Leftrightarrow \exists \sigma \in S_{n}, \mu =\sigma \tau \sigma^{-1}$. （相似？）

## 费马与欧拉定理 Fermat's and Euler's Theorem
**欧拉函数**$\phi$-*function*: map $\phi: \mathbb{N} \rightarrow \mathbb{N}$, 与$n$互质的整数$m(1\leq m <n)$的个数，$\phi(1) = 1$。

**定理6.17.** $U(n)$ 为 $\mathbb{Z}_{n}$ 的units的群，则 $|U(n)| = \phi(n)$。

**定理6.18. Euler's Theorem** 正整数$a, n$，$n > 0 \wedge gcd(a,n) = 1 \Rightarrow a^{\phi(n)} \equiv 1$ (mod $n$)。

**定理6.19. Fermat's Little Theorem** $p$为质数，$p \nmid a \Rightarrow a^{p-1} \equiv 1$ (mod $p$)。
$\forall b \in \mathbb{Z}, b^{p} \equiv b$ (mod $p$).
