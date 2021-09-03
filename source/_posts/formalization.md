---
title: 问题求解笔记-问题的形式化描述
categories: 
- 课程
- problem-solving
mathjax: true
date: 2021-04-28 07:34:56
---




庆祝终于脱离了常规算法的学习，进入了NP问题的坑。（

然后庆祝我们终于遇到了JH这本超级无敌晦涩难懂的教科书，真的是 **太tmd难读了**！！

在此附上助教对于该书的评价：

<!--more -->

![最棒教科书](chathistory.JPG "最棒教科书")

所以在整理笔记的同时，也是一个对内容汉化的过程，希望能有助于理解。。。:cold_sweat:



---

# 一些定义

**字母表（alphabet）**：一个非空有限集 $\Sigma$ 。

**符号（symbol）**：字母表 $\Sigma$ 中的一个元素。

**字（word）**：字母表中元素组成的一个序列。**$\lambda$**：空字。所有字的集合：$\Sigma^{\star}$

**字$w$长度$|w|$**：该字中元素的个数。井$_a(w)$ ：字 $w$ 中符号 $a$ 出现的次数。

$\Sigma^n =  \{ x \in \Sigma | |x| = n\}$

**$u,v$ 的连接（concatenation）**：$u,v$ 连起来，记作$uv$。

**前缀（prefix）、后缀（suffix）**：前面/后面一截。

**子字（subword）**：$z， w = uzv$。

**语言（language）**：$L \subseteq \Sigma^{\star}$。它的补：$L^C = \Sigma^{\star} - L$。

**语言的连接**：$L_1L_2 = L_1 \circ L_2 = \{ uv \in (\Sigma_1 \cup \Sigma_2)^{\star} | u \in L_1 ~ and ~ v \in L_2\}$。

**$\Sigma^{\star} $上的序**：$\Sigma = \{s_1, s_2, \cdots, s_m \}$ 上有序 $s_1 < s_2 < \cdots < s_m$，$u < v$ 如果 $|u| < |v|$ 或 $|u| = |v|, u = xs_iu' , x = xs_jv', i < j$。先比长度，后比第一个不同的字符。



---

# 算法问题

## 判定问题（decision problem）

$A$ 是一个算法， $x$ 是输入，$A(x)$ 标记为输出。

一个**判定问题**是指三元组 $(L, U, \Sigma), L \subseteq U \subseteq\Sigma^\star$，一般情况下 $U = \Sigma$，记为 $(L, \Sigma)$。算法 $A$ 解决这个问题是指对任意 $x \in U$ 有： $A(x) = 1, x\in L$ 并且 $A(x) = 0, x \in U - L$。

## 例子

### 素性检测（primality testing）

$$(Prim, \Sigma_{bool}), Prim = \{ w\in \{0, 1\} ^{star} | Number(w) \text{ is a prime}\}.$$

### 多项式相等问题（EQ-POL）

### 一次分支相等问题 (EQ-1BP)

### 满足性问题 （SAT）

$$\text{SAT} = \{ w \in \Sigma^{\star}_{logic} | w \text{ is a code of a satisfable formula in CNF}\}.$$

### 分团问题（CLIQUE）

图中是否有大小为 $k$ 的团（clique）。

$$CLIQUE = \{ x \# w \in \{ 0, 1, \#\}^{\star} | x\in \{0,1\}^{\star}, w \text{是一个有大小为 } Number(x) \text{ 的团的图}\}.$$

### 覆盖问题（VCP）

图中是否有大小为 $k$ 的点覆盖。

### 哈密顿回路问题（HC）

图中是否有哈密顿回路。



## 优化问题（optimization problem）

一个优化问题是一个七元组 $U = (\Sigma_I, \Sigma_O, L, L_I, \mathcal{M}, cost, goal)$，其中：

1. $\Sigma_I$ 是一个字母表，叫 $U$ 的输入字母表。
2. $\Sigma_O$ 是一个字母表，叫 $U$ 的输出字母表。
3. $L \subseteq \Sigma_{I}^{\star}$ 是可满足的问题实例的语言。
4. $L_I \subseteq L$ 是 $U$ 中的问题实例的语言。
5. $\mathcal{M} $ 是一个 $L \to Pot(\Sigma_{O}^{\star})$ 的函数，对任意的 $x \in L, \mathcal{M}(x)$ 是 $x$ 的可行解的集合。
6. $cost$ 是代价函数，对任意的数对 $(u,x), u \in \mathcal{M}(x), x \in L$，赋实数值 $cost(u,v)$。
7. $goal \in \{maximum, minimum\}$。



# 复杂度理论

两个复杂度衡量：**均匀消耗（uniform cost）**，**对数消耗（logarithmic cost）**

$Time_A(x), Space_A(x)$：算法 $A$ 对输入的 $x$ 的计算时间空间复杂度。

**定理 2.3.3.3.** 对于一个判定问题 $(L , \Sigma_{bool})$，每个算法 $A$ 都存在另一个算法 $B$ 使得 $Time_B(n) = \log_2{(Time_A(n))}$

*这个定理告诉我们对于 $L$ 不存在最优的算法，对 $L$ 的复杂度的定义也是没有意义的。所以人们通常不定义算法问题的复杂度而研究问题复杂度的上下界（符号标记同TC）。*

**图灵机**是算法的直观概念的形式化，这意味着问题 $U$ 可以被算法解决当且仅当存在一个图灵机可以解决它。

对于每个递增函数 $f : \mathbb{N} \to \mathbb{R}^+$：

（1）存在一个判定问题，每个图灵机可以在 $\Omega(f(n))$ 的复杂度内解决它。

（2）也存在一个图灵机可以在 $O(f(n)\cdot \log{(f(n))})$  的复杂度内解决它。

这意味着判定问题有无穷级的难度。复杂度理论的主要内容就是*寻找一类可实际解决的问题的形式化说明* 和 *开发能够根据它们在这类问题中的关系进行分类的方法*。

## 形式化描述

对于一个图灵机（算法）$M$，$L(M)$ 是 $M$ 决定（decide）的语言。

能够在多项式时间内完成的复杂度问题类： 

$$P = \{L = L(M) |\exists c \in \mathbb{Z} , Time_M(n) \in O(n^c) \}$$

一个语言（判定问题）是**可处理的（tractable）** 当 $L \in P$，否则不可处理（intractable）。

**非确定性计算（nondeterministic computation）**：引入随机数操作。

让 $M$ 为一个非确定性图灵机，定义 $M$ **接受（accept）语言** $L$，$L = L(M)$ 如果：

（1）对于任意 $x \in L$，$M$ 中存在至少一个接受（accept）$x$ 的计算。

（2）对于任意 $y \notin L$，$M$ 的所有计算拒绝（reject） $y$ 。

对于任意输入 $w$，$M$ 的时间复杂度 $Time_M(w)$ 是 $M$ 中的最短的接受的计算（accepting computation）。

通过非确定性算法能够在多项式时间内完成的判定问题类： 

$$NP = \{L(M) | M \text{是多项式时间的非确定性算法} \}$$

算法 **B** 对于输入 $x$ 的接受计算（accepting rejecting）可以看做对于是否 $x \in L$ 的证明。

*这意味着确定性计算的复杂度是证明输出的正确性，非确定性计算的复杂度是对于一个给定的证明的确定的验证*





$L \subseteq \Sigma^{\star}$ 是一个语言，一个输入为 $\Sigma^{\star} \times \{0, 1\}^{\star}$  的算法 **A** 称为 $L$ 的**验证（verifier）**，记作 $L = V(\textbf{A})$，如果 

$$L = \{ w \in \Sigma^{\star} | A \text{ accept } (w,c) \text{，对于某个 } c \in \{0,1\}^{\star}\}$$

如果 $\textbf{A} \text{ accept } (w,c)$，我们称 $c$ 是事实（fact） $w \in L$ 的一个**证明（proof/certificate）**。

**A** 是一个**多项式时间验证（polynomial-time verifier）**如果存在一个整数 $d$ ，使得对任意 $w\in L$，$Time_A(w, c) \in O(|w|^d)$ 对于某个 $w$ 的证明 $c$。

定义**可多项式验证的语言类（class of polynomially verifiable languages）**： 

$$VP = \{VP | A \text{ 是一个多项式验证}\}$$

**定理 2.3.3.9.** $NP = VP$





$L_1 \subseteq \Sigma_1^{\star}, L_2 \subseteq \Sigma^{\star}_2$ 是两个语言，我们说 $L_1$ 对于 $L_2$ 是**多项式时间可约的（polynomial-time reducible）**，记作 $L_1 \leq_p L_2$，如果存在一个多项式时间算法 **A** 计算一个 $L_1$ 到 $L_2$ 的映射，即 

$$\forall x \in \Sigma_1^{\star}, x \in L_1 \Leftrightarrow A(x) \in L_2$$。

**A** 称作从 $L_1$ 到 $L_2$ 的多项式时间规约（reduction）。（说明 $L_2$ 至少和 $L_1$ 一样难。）

一个语言 $L$ 是 **NP-hard**，如果对于任意 $U \in NP, U \leq_p L$。

一个语言 $L$ 是 **NP-complete**，如果 $L \in NP$ 且 $L$ 是 NP-$hard$。