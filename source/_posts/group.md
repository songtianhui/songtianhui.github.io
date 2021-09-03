---
title: 问题求解笔记-群论初步
date: 2021-02-26 15:53:33
categories: 
- 课程
- problem-solving
mathjax: true
---

# 群 Group

## 整数等价类 Integer Equivalence
模运算下整数等价类的性质:
1. 加法乘法交换律
2. 加法乘法结合律
3. 加法(0)乘法(1)单位元
4. 乘法分配律
5. 任意元素存在加法逆元
6. $a$为非零整数，$gcd(a,n) = 1 \Leftrightarrow a$存在乘法逆元

<!--more -->

## 定义

- 集合$G$上的一个**二元运算(binary operation)**或者一个**合成律(law of composition)**: 一个函数$f : G \times G \rightarrow G$, $f(a,b) = a \circ b$ or $ab$
- **群**$(G, \circ)$: 带有一个合成律的集合$G$, 且该合成律$(a, b) \rightarrowtail a \circ b$满足:
  + **结合律(associative)**: $\forall a, b, c \in G, (a \circ b) \circ c = a \circ (b \circ c)$
  + **单位元(identity element)**: $\exists e \in G, \forall a \in G, a \circ e = e \circ a = a$
  + **逆元(inverse element)**: $\forall a \in G, \exists a^{-1}\in G, a \circ a^{-1} = a^{-1}\circ a = e$
- **阿贝尔群(abelian)**: 群$G$, $\forall  a, b \in G, a\circ b = b \circ a$
- **凯利表(Cayley table)**或**交换群(commutative)**: 通过加法或乘法来描述一个群的表格。
- **可逆元素群(group of units)**: 拥有逆元的元素（即所有和 $n$ 互质的数）组成的乘法群
- 一般线性群(general linear group)
- 四元群(quaternion group)
- 群是有限的(finite)，或者说有有限序数(has finite order)，当它具有有限个元素，否则是无限的(infinite)或有无限序数(infinite order)

## 群的基本性质
**命题3.17.** 单位元唯一
**命题3.18.** 逆元唯一
**命题3.19.** $\forall a, b \in G, (ab)^{-1} = b^{-1}a^{-1}$

**命题3.20.** $\forall a \in G, (a^{-1})^{-1} = a$

**命题3.21.** $\forall a, b \in G,$ 方程$ax=b$和$xa=b$有唯一解

**命题3.22.** $\forall a, b,bc \in G, ab=ac \Rightarrow b=c$, $ba=ca  \Rightarrow b= c$（左右约分）

**定理3.23.** 基本指数运算都成立，$\forall g, h \in G$,
1. $g^{m}g^{n} = g ^{m+n}$
2. $(g^{m})^{n} = g^{mn}$
3. $(gh)^{n} = (h^{-1}g^{-1})^{-n}$. $G$ is an abelian $\Rightarrow (gh)^{n} = g^{n}h^{n}$

## 子群 Subgroups
- 子群是父群的子集，也是个群
- **平凡子群(trivival subgroup)**: $H = \{ e \}$
- 真子群

**定理3.30.** $H$是$G$的子群$\Leftrightarrow$
1. $G$的单位元$e \in H$
2. $\forall h_{1}, h_{2} \in H, h_{1}h_{2} \in H$
3. $\forall h \in H,  h^{-1} \in H$

**定理3.31.** $H \subset G$是$G$的子群$\Leftrightarrow H \neq \emptyset \wedge \forall g,h\in H, gh^{-1} \in H$

---
# 循环群 Cyclic Groups

**定理4.3.**  Let $G$ be a group and $a$ be any element in $G$. Then the set $\langle a\rangle =\{a^{k}:k\in\mathbb{Z}\}$ is a subgroup of $G$. Furthermore, $⟨a⟩$ is the smallest subgroup of $G$ that contains $a$.

- **循环子群(Cyclic Subgroups)**: $\langle a \rangle$
- **循环群**: 包含了某个元素$a$，使得$G=⟨a⟩$.此时$⟨a⟩$是$G$的**生成元(generator)**.
- $a$的**序数(order)**: 最小的正整数$n$, s.t. $a^{n} = e$, 记作$|a| = n$. 如果不存在，则为正无穷。

**定理4.9.** 所有循环群都是阿贝尔群。

## 循环群的子群
**定理4.10.** 循环群的子群也都是循环群。
**推论4.11.** $\mathbb{Z}$的所有子群就是$n\mathbb{Z}, n = 0,1,2,...$

**命题4.12.** $G$为一个$n$阶循环群，$a$是$G$的一个生成元。则$a^{k} = e \Leftrightarrow n | k$（$k$是$n$的倍数）。

**定理4.13.** $G$为一个$n$阶循环群，$a$是$G$的一个生成元。$b = a^{k} \Rightarrow b$的序数为$n / d, d = gcd(k,n)$。

**推论4.14.** $\mathbb{Z}_{n}$的所有生成元$r$满足$1 \leq r < n \wedge gcd(n,r) = 1$。

## 复数乘法群  Multiplicative Group of Complex Numbers
复数知识
- $\cos \theta + i \sin \theta$ 记作 $cis\theta$
**定理2.22. DeMoivre** $(r~cis\theta)^{n} = r^{n}~cis(n\theta)$

## 圆群 Circle Group
$\mathbb{T}: \{ z \in \mathbb{C}: |z| = 1 \}$
**命题4.24.** 圆群是$\mathbb{C}^{*}$的一个子群。

满足$z^{n} = 1$的复数$z$称为$n$次**单位根($n$-th roots of unity)**。
**定理4.25.** $z^{n} = 1 \Rightarrow z = cis(\dfrac{2k\pi}{n}), k = 0, 1, ... , n-1$. $n$次单位根构成$\mathbb{T}$的一个圆子群。

## 重复平方法 The Method of Repeated Squares
快速幂。
