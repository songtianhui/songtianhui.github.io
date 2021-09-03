---
title: 问题求解笔记-代数编码
categories: 
- 课程
- problem-solving
date: 2021-04-25 23:10:21
mathjax: true
tags:
---


# 检错和纠错码

## 最大似然编码（maximum-likelihood decoding​）

所有的纠错建立在这个基础上，按最大概率出错位纠错。

**二进制对称信道**（binary symmetric channel）

<!--more -->

## 分块码 （Block Codes​）

($n,m$)-**分块码**：被编码的信息可以被分成多块 $m$ 位二进制数，每块会被编码成 $n$ 位二进制数。

- 编码函数$E:\mathbb{Z}_2^m \rightarrow \mathbb{Z}_2^n$
- 解码函数 $D: \mathbb{Z}_2^n \rightarrow \mathbb{Z}_2^m$

- 编码字(codeword)：$E$ 的像中的一个元素

**汉明距离**（Hamming distance）：$d(x,y)$，$x$ 和 $y$ 不相同的位数。

**最小距离**：$d_{min}$，所有不同有效编码字 $x,y$ 的距离最小的。

**权重**：$w(x)$，$x$ 中 $1$ 的位数。

一些性质：

- $w(x) = d(x, 0)$
- $d(x,y) \geq 0$
- $d(x, y ) = 0 \Leftrightarrow x = y$
- $d(x, y) = d(y,x)$
- $d(x,y) \leq d(x,z) + d(z,y)$

**定理 8.13.** $C$ 是 $d_{min} = 2n+1$ 的编码，则 $C$ 可以纠 $n$ 位错，可以检测 $2n$ 位错。

*所以码字空间分布越均匀，最小码字距离越大，查纠错能力越强。*



---



# 线性码（Linear Codes）

**群码**（group code）：$\mathbb{Z}_2^n$ 的子群的编码。

**引理 8.17.** $x,y$ 为 $n$ 位码字，则 $w(x  +y) = d(x,y)$。

**定理 8.18.** 群码 $C$ 的 $d_{min}$ 是所有非零码字的最小的权重。

*码字的运算，还是码字，群封闭性。*



## 线性码

**内积**：

 $\begin{aligned} \textbf{x}\cdot \textbf{y}  &= \textbf{x}^t \textbf{y} \\ &= \begin{pmatrix} x_1 & x_2 &\cdots &x_n \end{pmatrix} \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_n  \end{pmatrix} \\ &= x_1y_1 + x_2y_2 + \cdots + x_ny_n \end{aligned}$

记 $\mathbb{M}_{m\times n}(\mathbb{Z}_2)$ 为所有 $m$ 行 $n$ 列且元素属于 $\mathbb{Z}_2$ 的矩阵的集合。

$H \in \mathbb{M}_{m\times n}(\mathbb{Z}_2)$  的**空域**（null space我自己瞎起的名字）：$\{\textbf{x} \in \mathbb{Z}_2^n : H \textbf{x} = 0 \}$；记为 Null$(H)$。

**定理 8.21.** $H \in \mathbb{M}_{m\times n}(\mathbb{Z}_2)$ 的 $null ~space$ 一定是群码。

**线性码**：通过某个 $H \in \mathbb{M}_{m\times n}(\mathbb{Z}_2)$ 的 $null ~space$ 生成的编码。



*对于 $n$ 为编码，我们需要找到一个好的矩阵 $H$，使得我们的编码空间（$H$ 的 null space）的 $d_{min}$ 尽可能大。*



---



# 奇偶校验和生成矩阵（Parity-Check and Generator Matrices）

**规范奇偶校验矩阵**（canonical parity-check matrix）：$H \in \mathbb{M}_{m\times n}(\mathbb{Z}_2)$ 且 $n > m$，且最后 $m$ 列形成的 $m \times m$ 子矩阵是单位阵 $I_m$，即 $H = \left( A | I_m \right)$。

每个规范奇偶校验阵 $H$ 都有一个 $n \times (n - m)$ 阶标准**生成阵**（standard generator matrix）: $G = \left( \dfrac{I_{n-m}}{A} \right)$。

**定理 8.25.** $H$ 是一个规范奇偶校验阵，则 Null$(H)$ 包含了满足 前 $n-m$ 位随便取，但后 $m$ 位由 $H\textbf{x} = \textbf{0}$ 决定的 所有向量 $\textbf{x} \in \mathbb{Z}_2^n$。后 $m$ 位每个作为某个前 $n-m$ 位的偶校验位。所以，$H$ 可以给出一个 $(n, n-m)$ -分块码。

- 前 $n-m$ 位叫**信息位**（information bit），后 $m$ 位叫校验位

**定理 8.26.**  $G$ 是一个 $n \times k$ 阶标准生成阵，则 $C = \{ \textbf{y} : G\textbf{x} = \textbf{y}, x \in \mathbb{Z}_2^n \}$ 是一个 $(n,k)$-分块码。



**引理 8.27.** $H = (A | I_m)$ 是一个 $m \times n$ 规范奇偶校验阵, $G = \left( \dfrac{I_{n - m}}{A} \right)$ 是对应的 $n \times (n - m)$ 阶标准生成阵，则 $HG = 0$。

**定理8.28.**  $H = (A | I_m)$ 是一个 $m \times n$ 规范奇偶校验阵, $G = \left( \dfrac{I_{n - m}}{A} \right)$ 是对应的 $n \times (n - m)$ 阶标准生成阵，$C$ 是由 $G$ 生成的编码。则 $\textbf{y} \in C \Leftrightarrow H\textbf{y} = 0$，也就是说，$C$ 是有规范奇偶校验矩阵的线性码。



*偶校验矩阵的来处：在H的右侧 $m \star m$设置单位矩阵，就是要用相应位对A中相应行上的非0位进行偶校验.。*

*生成矩阵的来处：当给定任意的A，可以使用相应的偶校验矩阵来通过解线性方程的方式得到群码，但是不如构造G矩阵，对待传输数据直接进行计算得到相应的群码和编码对应。效率更高。*



*可证码字能够覆盖所有待编码位串。后m位可以完成对待编码位串中非0位的偶校验！*

*如何完成偶校验，取决于A矩阵和待编码信息。*



**定理 8.31.** $H$ 为一个 $m \times n$ 矩阵，$H$ 的空域是一个单检错码 当且仅当 $H$ 没有列是全零。（$e_i$ 不在空域中，$d_{min} > 1$）

**定理 8.34.** $H$ 为一个 $m \times n$ 矩阵，$H$ 的空域是一个单纠错码 当且仅当 $H$ 没有列是全零并且没有两列是相同的。（$e_i + e_j$ 不在空域中，$d_{min} \geq 3$）

所以一个 $m \times n$ 的规范奇偶校验阵，要检一位错、纠一位错，除了 $\textbf{0}, \textbf{e_i}$，剩余 $2^m - (1 + m)$ 个为信息列。

---

# 高效解码

**x**的**像**（syndrome）：$H\textbf{x}$

**命题 8.36.** 矩阵 $H$ 决定了一套线性码，$\textbf{x} = \textbf{c} + \textbf{e} $ 为接收到的$n$-位串，$\textbf{c}$ 为正确码字，$\textbf{e}$ 为错误，则 $H\textbf{x} = H\textbf{e}$。

- 这个命题说明接受到的信息由错误决定而不是由正确码字决定。

**定理 8.37.** $H \in \mathbb{M}_{m\times n}(\mathbb{Z}_2)$ 并且假设 $H$ 的编码是一个单检错码，$\textbf{r}$ 是接收到的 $n$-位码，如果 $H\textbf{r} = \textbf{0}$，则没有错；否则，如果 $H\textbf{r} $ 等于 $H$ 的第 $i$ 列，则第 $i$ 位出错。



## 陪集解码（Coset Decoding）

把所有陪集表示（$\textbf{e} + 	C$）列表列出来，查表解码纠错。

按最大似然，总是取权最小的出错 $\textbf{e}$ ，叫做 **coset leader**。
