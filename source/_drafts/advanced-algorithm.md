---
title: 高级算法课程笔记-Lecture1
tags:
categories:
- 课程
- Advanced Algorithm
mathjax: true
---



课程主页：[http://tcs.nju.edu.cn/wiki/](http://tcs.nju.edu.cn/wiki/)

<!--more-->

本章是 etone 老师通过 min/max-cut 来引入高级算法的概念。

**图的切（Graph Cut）**：有无向图 $G(V, E)$，子集 $C \subseteq E$ 是 $G$ 的一个切如果在去掉 $C$ 中所有边后 $G$ 不连通（disconnected）了。

显然对于 $V$ 的一个二分 $\{S, T\}$，$E(S, T)=  \{ uv \in E | u \in S, v \in T\}$ 是 $G$ 的一个切。

# 最小割 Min-Cut

我们可以推广一下问题到输入可以是 **multi-graphs**，也即可以有重边，不能自环。

## 最大流 Max flow

一个朴素的想法，也是一个可行的确定性算法就是 **最大流最小割定理**，一个图的 s-t 最大流也就是它的最小割。

- 注意到这个算法本质上是固定 s 和 t 的，也就是可能不是全局最优，不过我们只要遍历点跑最大流就可以了。
- 时间复杂度：$(n-1) \times \text{max flow time} = \tilde{O}(mn)$。
  - $\tilde{O}$ 表示忽略多项式的对数系数。
  - 实际是 $O(mn + n^2 \log{n})$

## 卡格尔算法 Karger's Contraction Algorithm

一个图的割太多了（$2^{\Omega(n)}$ 量级），但可以证明最小割最多只有 $O(n^2)$，我们希望生成随机的切集中在最小割上

**收缩（contraction）**：$contract(G, e)$ 将边 $e = uv$ 的两个端点合并（merge）。*在合适的数据结构下可以 $O(n)$ 实现。*

``` pseudocode Karger's Algorithm
Input: G(V,E)

while |V| > 2 do:
	pick random uv in E
	G = contract(G, uv)
return C = E // the edges between the only two vertices
```

显然 $O(n)$ 的 contract 操作下该算法的复杂度是 $O(n^2)$。

### 准确性的分析

因为是一个随机算法，所以输出的切是一个随机变量。

可证
