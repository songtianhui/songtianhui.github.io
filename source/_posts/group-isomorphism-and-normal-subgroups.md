---
title: 问题求解笔记-群同态基本定理与正规子群
date: 2021-03-11 01:50:32
categories: 
- 课程
- problem-solving
mathjax: true

---

---

# 同构 Isomorphisms

## 定义

群$(G, \cdot )$ 和 $(H, \circ)$ **同构（isomorphic）**，如果存在一个双射 $\phi: G \rightarrow H$ 且满足

$$\forall a, b \in G, \phi(a \cdot b) = \phi(a) \circ \phi(b)$$

记作 $G \cong H$。映射 $\phi$ 称为**同构（isomorphism）**。

<!--more -->

**定理 9.6.** $\phi: G \rightarrow H$ 是两个群上的同构，则有

1. $\phi^{-1}: H \rightarrow G$ 也是一个同构。
2. $|G| = |H|$
3. $G$ 是阿贝尔群 $\Rightarrow$ $H$ 是阿贝尔群。
4. $G$ 是循环群 $\Rightarrow$ $H$ 是循环群。
5.  $G$有一个 $n$ 阶子群 $\Rightarrow$ $H$ 有一个 $n$ 阶子群。

**定理 9.7.** 所有无限循环群和 $\mathbb{Z}$ 同构。

**定理 9.8.** 所有 $n$ 阶循环群和 $\mathbb{Z}_{n}$ 同构。

**推论 9.9.** $G$ 是一个 $p$ 阶群，$p$ 是一个质数，则 $G$ 同构于 $\mathbb{Z}_{p}$。

**定理 9.10.** 群上的同构形成了所有群的类型的等价关系。


### 凯利定理 Cayley's Theorem

**定理 9.12. Cayley** 每个群都同构于一个置换群。


## 外直积 External Direct Products

**命题 9.13.** $(G, \cdot), (H, \circ)$ 是两个群，$(g, h) \in G \times H, g \in G, h \in H$, 定义 $G \times H$ 上的二元运算：$(g_1, h_1)(g_2, h_2) = (g_1 \cdot g_2, h_1 \circ h_2)$，是一个群。

群 $G \times H$ 叫做 $G$ 和 $H$ 的**外直积（external direct product）**。

**定理 9.17.** $(g, h) \in G \times H$，$g, h$ 都有有限的序数 $r, s$，则 $(g,h)$ 的序数是 $r$ 和 $s$ 的最小公倍数。

**定理 9.21.** $\mathbb{Z}_m \times \mathbb{Z}_n \cong \mathbb{Z}_{mn} \Leftrightarrow gcd(m,n) = 1$

## 内直积 Internal Direct Products

如果群 $G$ 的两个子群 $H$ 和 $K$ 满足以下条件：

- $ G= HK = \{ hk: h \in H, k \in K \}$
- $H \cap K = \{ e \}$
- $\forall k \in K, \forall h \in H, hk = kh$

则 $G$ 叫做 $H$ 和 $K$ 的**内直积（internal direct product）**。

**定理 9.27.** $G$ 是 $H$ 和 $K$ 的内直积 $\Rightarrow G \cong H \times K$。

---
# 正规子群和商群
## 正规子群 Normal Subgroups

群 $G$ 的一个子群 $H$ 是**正规的（normal）**，如果 $\forall g \in G, gH = Hg$

**定理 10.3.** $N$ 为群 $G$ 的一个子群，则以下命题等价：
1. $N$ 是正规的。
2. $\forall g \in G, g N g^{-1} \subseteq N$
3. $\forall g \in G, g N g^{-1} = N$

## 商群 Factor Groups
$N$ 为群 $G$ 的一个子群，则 $N$ 的所有陪集形成了一个群 $G/N$，二元运算为 $(aN)(bN) = abN$。这个群叫做 $G$ 和 $N$ 的**因子（factor）**或**商群（quotient group）**。

**定理 10.4.** $N$ 为群 $G$ 的一个子群，则 $N$ 的所有陪集形成了一个 $[G:N]$ 阶群 $G/N$。

## 对换群的simplicity

没有非平凡正规子群的群叫做**（simple group）**。

**引理 10.8.** 对换群 $A_n$ 由三元环（3-cycles）生成。

**引理 10.9.** $N$ 为群 $A_n$ 的一个子群。如果 $N$ 包含一个三元环，则 $N = A_n$。

**引理 10.10.** 对于 $n \geq 5$，$A_n$ 的每个非平凡正规子群 $N$ 包含一个三元环。

**定理 10.11.** 置换群 $A_n(n \geq 5)$ 是simple的。

---
# 同态
## 群同态 Group Homomorphisms

群 $(G, \cdot )$ 和 $(H, \circ)$ 间的**同态（homomorphisms）**是一个映射（不一定双射） $\phi: G \rightarrow H$ 且满足
$$\phi(a \cdot b) = \phi(a) \circ \phi(b)$$
$\phi$ 在 $H$ 中的值域叫做**同态像（homomorphic image）**。

**命题 11.4.** $\phi: G_1 \rightarrow G_2$ 是一个同态，则：
1. $e$ 是 $G_1$ 的单位元 $\Rightarrow \phi(e) $ 是 $G_2$ 的单位元。
2. $\forall g \in G_1, \phi(g^{-1}) = [\phi(g)]^{-1}$
3. $H_1$ 是 $G_1$ 的子群 $\Rightarrow \phi(H_1)$ 是 $G_2$ 的子群。
4. $H_2$ 是 $G_2$ 的子群 $\Rightarrow \phi^{-1}(H_2) = \{ g \in G: \phi(g) \in H_2 \}$ 是 $G_1$ 的子群。进一步，$H_2$ 是 $G_2$ 的正规子群 $\Rightarrow \phi^{-1}(H_2)$ 是 $G_1$ 的正规子群。

$\phi : G \Rightarrow H$ 是一个同态，$\phi^{-1}(\{ e \})$ 是 $G$ 的一个子集，叫做 $\phi$ 的**核（kernel）**，记作 $ker~\phi$。

**定理 11.5.** $\phi :G \to H$ 是一个同态，则 $\phi$ 的核是 $G$ 的一个正规子群。

## 同构定理 Isomorphism Theorems

$H$ 为群 $G$ 的一个正规子群，定义**自然同态（natural homomorphism）** $\phi: G \rightarrow G / H, \phi(g) = gH$。

**定理 11.10. 同构第一定理** $\psi: G \rightarrow H$ 是一个群同态，核为 $K = ker~\psi$ 是一个正规子群。$\phi:G \rightarrow G /K$ 为自然同态，则存在一个唯一同构 $\eta: G / K \rightarrow \psi(G)$，使得 $\psi = \eta \phi$。

**定理 11.12. 同构第二定理** $H$ 是群 $G$ 的子群（不一定是正规的），$N$ 是 $G$ 的正规子群，则 $HN$ 是 $G$ 的子群，$H \cap N$ 是 $H$ 的正规子群，且
$$H / (H \cap N) \cong (HN) / N$$

**定理 11.13 一致定理** $N$ 是群 $G$ 的正规子群，则 $H \rightarrow H /N$ 是一个从 $G$ 的包含 $N$ 的子群的集合到 $G /N$ 的子群的集合的满射。进一步，从 $G$ 的包含 $N$ 的**正规**子群的集合到 $G /N$ 的子群的集合是双射。

**定理 11.14. 同构第三定理** $G$ 是一个群，$N$ 和 $H$ 是正规子群且有 $N \subset H$，则 $G / H \cong \dfrac{GN}{HN}$.
