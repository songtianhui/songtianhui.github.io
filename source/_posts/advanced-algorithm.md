---
title: 高级算法课程笔记-Lecture1
categories:
  - 课程
  - Advanced Algorithm
mathjax: true
date: 2021-09-07 22:49:56
tags:
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

**收缩（contraction）**：`contract(G, e)​` 将边 $e = uv$ 的两个端点合并（merge）。*在合适的数据结构下可以 $O(n)$ 实现。*

``` pseudocode Karger's Algorithm
Input: G(V,E)

while |V| > 2 do:
	pick random uv in E
	G = contract(G, uv)
return C = E // the edges between the only two vertices
```

显然 $O(n)$ 的 `contract` 操作下该算法的复杂度是 $O(n^2)$。

### 准确性的分析

因为是一个随机算法，所以输出的切是一个随机变量。

可证如下定理：

{% note info %}
对于任意 $n$ 个顶点的 multigraph，Karger 算法的 $Pr[\text{a min-cut is returned}] \geq \dfrac{2}{n(n - 1)}$。
{% endnote %}

~~第一次发现 note 的用法，感觉好好看~~

下面来证：

- 记 $e_1, e_2, ..., e_{n-2}$ 为随机选择的收缩的边的序列。
- 记 $G_1 = G$ 为初始图，$G_{i + 1} = contract(G_i, e_i), i = 1, 2,..., n-2$。

显然可知：如果 $C$ 是 $G$ 的一个最小切且 $e \notin C$，则 $C$ 也是 `G' = contract(G, e)​` 的一个最小割。所以有：

$\begin{aligned} p_C &= Pr[C \text{ is returned}] \\ &= Pr[e_i \notin C \text{ for all } i = 1, 2, ...,n-2] \\ &= \prod\limits_{i = 1}^{n-2} Pr[e_i \notin C | \forall j < i, e_i \notin C] \end{aligned}$

$\forall j < i, e_i \notin C$ 意味着 $C$ 也是 $G_i$ 的一个最小切。

观察到一个最小切的边数 $|C|$ 一定是小于等于任意点的度数（degree），否则可以直接把这个点切开。

所以更加小于等于度数的平均数，即 $|C| \leq \dfrac{2 |E|}{|V|}$。（所有点的度数和为 $2|E|$）

所以我们可以计算

$\begin{aligned} Pr[e_i \notin C | \forall j < i, e_i \notin C]  &= 1 - \dfrac{|C|}{|E_i|} \text{（从剩下的点中均匀随机选取）} \\ & \geq 1 - \dfrac{2}{|V_i|} \\ &= 1 - \dfrac{2}{n - i + 1} \end{aligned}$

$\begin{aligned} p_C & \geq \prod\limits_{i = 1}^{n-2} (1 - \dfrac{2}{n - i + 1}) \\ &= \prod\limits_{k = 3}^{n} \dfrac{k - 2}{k} \\ &= \dfrac{2}{n(n-1)} \end{aligned}$


{% label success@得证。 %}



乍一看这个算法有啥吊用，准确率也太低了。

不过随机算法的精髓就在于有了准确性的下界，就可以通过 repeat 来提升准确性（好像是蒙特卡罗还是拉斯维加斯来着）。

Karger 算法主要就是将解空间的大小由指数级降低到了平方级。

我们可以运行算法 $t = \dfrac{n(n-1)\ln{n}}{2}$ 次并返回最小的解，该解是最小切的概率为：

$\begin{aligned} \ &1 - Pr[\text{所有 } t \text{ 次都失败}] \\ = &1 - (Pr[\text{一次失败}])^t \\ \geq & 1 - \left( 1 - \dfrac{2}{n(n-1)} \right)^{\frac{n(n-1)\ln{n}}{2}} \\ \geq & 1 - \dfrac{1}{n}  \end{aligned}$

运行一次算法的时间是 $O(n^2)$，也就是说我们可以在 $O(n^4\log{n})$ 的时间复杂度下高概率地（w.h.p）找到最小切。



### 一个推论

{% note info %}
对任意的 $n$ 阶图 $G(V, E)$，最小割的个数不会超过 $\dfrac{n(n-1)}{2}$。
{% endnote %}

这是从概率方法上推到的：

设有 $M$ 个不同的最小切，显然它们是独立的。

$\because 1 \geq Pr[\text{a min-cut is returned}] \geq M \times \dfrac{2}{n(n-1)}$

$\therefore M \leq \dfrac{n(n-1)}{2}$

{% label success@得证。 %}



## 快速最小切 Fast Min-Cut

我们知道 $p_C \geq \prod\limits_{i = 1}^{n-2} (1 - \dfrac{2}{n - i + 1})$，也就是这个概率是随 $i$ 增大递减的，也就是说当图越来越小也就越来越难成功。

我们就可以思考一个新的方法，先把图收缩的一个较小的程度，然后再递归地找这个更小的图上的最小切。

先定义一个随机收缩的函数，把图收缩到只剩 $t$ 个点：

``` pseudocode random_contract(G, t)
Input: G(V, E), integer t >= 2

while |V| > t:
	choose an edge uv in E uniformly at random
	G = contract(G, uv)
return G
```

然后定义我们的快速最小切：

``` pseudocode fast_cut(G)
Input: G(V, E)

if |V| <= 6:
	return a mincut by brute force
else:
	t = [1 + |V| / sqrt(2)]
	G1 = random_contract(G, t)
	G2 = random_contract(G, t)
	return min{fast_cut(G1), fast_cut(G2)}
```



一样的我们可以算出 `random_contract` 成功的概率：

$\begin{aligned} & Pr[C \text{ survives all contractions in random_contract}] \\ = & \prod\limits_{i=1}^{n - t} Pr[C \text{ survives the i-th contraction } | C \text{ survives the first (i−1)-th contractions}] \\ \geq & \prod\limits_{i=1}^{n - t} \left( 1 - \dfrac{2}{n - i +1} \right) \\  = & \dfrac{t(t - 1)}{n(n - 1)}  \end{aligned}$

当 $t = \lceil 1 + n / \sqrt{2} \rceil $ 时，这个概率至少是 $1/2$。

定义两个事件：

- A：$C$ survives `random_contract` 中所有的 contractions。
- B：`random_contract` 后最小切的大小保持不变。

显然 $A \to B$，所以 $Pr[B] \geq Pr[A] \geq \frac{1}{2}$。

我们用 $p(n) $ 来表示一个 $n$ 阶图的 `fast_cut` 的下界：

$\begin{aligned} p(n) & = Pr[\text{fast_cut returns a min-cut}] \\ &= Pr[\text{a min-cut is returned by fast_cut(G1) and fast_cut(G2)}] \\ & \geq 1 - (1 - Pr[B \wedge \text{ fast_cut(G1) return a min-cut in G1}])^2 \\ & \geq 1 - (1 - Pr[A \wedge \text{ fast_cut(G1) return a min-cut in G1}])^2 \\ & = 1 - (1 - Pr[A] Pr[\text{fast_cut(G1) return a min-cut in G1 | A}])^2 \\ & \geq 1 - \left( 1 - \dfrac{1}{2} p(\lceil 1 + n / \sqrt{2} \rceil) \right)^2  \end{aligned}$

base case 是 $p(n) = 1$ 当 $n \leq 6$。

归纳易证 $p(n) = \Omega(\frac{1}{\log{n}})$。

时间复杂度上由那个算法我们有 $T(n) = 2T(\lceil 1 + n / \sqrt{2} \rceil) + O(n^2)$，易证 $T(n) = O(n^2 \log{n})$。

最终我们能总结如下定理：

{% note info %}

对任意的 $n$ 阶图，`fast_cut` 算法可以在 $O(n^2 \log{n})$ 的时间复杂度下以 $\Omega(\frac{1}{\log{n}})$ 的概率返回一个最小切。

{% endnote %}

`fast_cut` 比原始的 `random_contract` 时间长，概率高。

我们可以运行它 $(\log{n})^2$ 次，在 $O(n^2 \log^3{n})$ 的时间复杂度下以 $1 - O(1/n)$ 的概率返回最小切。

回忆前面的确定算法的时间复杂度是 $O(mn + n^2 \log{n})$，也就是说在稠密图上这个随机算法会表现地更好一些。

*最终 Karger 继续改进了这个算法并获得了一个关于边数的接近线性时间的随机算法不过超出了本课程的讨论范围。*



# 最大割 Max-cut

问题就转变成了一个 **NP-hard** 的问题，如果我们相信 $P \neq NP$，就无法在多项式时间用确定性算法解决这个问题。

## 贪心 Greedy

``` pseudocode greedy_cut
Input: G(V, E)

S = T = emptyset
for i = 1, 2, ..., n:
	v_i joins one of S,T to maximize the current |E(S,T)|
```

然后讲了到**近似比（approximation ratio）**的概念，~~复习问求。~~

暂时先不假证明（懒）地给出定理：

{% note info %}

`greedy_cut` 是一个 0.5-近似的最大切算法。

{% endnote %}

这并不是多项式时间最好的最大切算法，最好的算法 [Goemans-Williamson Algorithm](http://www-math.mit.edu/~goemans/PAPERS/maxcut-jacm.pdf) 可以达到近似比 $\alpha^{\star} = \frac{2}{\pi}\inf_{x\in [-1,1]}{\frac{\arccos{(x)}}{1-x}}$，并被证明不存在近似比更好的多项式时间近似算法。



## 通过条件期望去随机化 Derandomization

对于任意顶点 $v \in V$，$X_{v} \in \{0, 1\}$ 是一个均匀随机 bit 来表示该点加入 $S$ 或 $T$。

这个随机的 cut 的大小可以被表示为：
$$
|E(S, T)| = \sum\limits_{uv\in E} I[X_u \neq X_v]。
$$
由于期望的线性（linearity），
$$
\mathbb{E}[|E(S, T)|]=\sum\limits_{u v \in E} \mathbb{E}\left[I\left[X_{u} \neq X_{v}\right]\right]=\sum\limits_{u v \in E} \operatorname{Pr}\left[X_{u} \neq X_{v}\right]=\dfrac{|E|}{2}。
$$
$|E|$ 就是最大切的 $OPT_G$ 的一个上界，所以我们有：
$$
\mathbb{E}[|E(S,T)|] \geq \dfrac{OPT_G}{2}。
$$


所以我们生成的随机的切的平均值大于等于 $\frac{OPT_G}{2}$，由于**平均原理（averaging principle）**，至少有一个二分生成的切是大于等于 $\frac{OPT_G}{2}$ 的。然后我们希望有算法来找到这个解。

我们想一个个地固定 $X_{v_{i}}$ 的值来构造一个二分 $\{\hat{S}, \hat{T}\}$ 使得：
$$
|E(\hat{S}, \hat{T})| \geq \mathbb{E}[|E(S, T)|] \geq \dfrac{O P T_{G}}{2}。
$$
我们从第一个点 $v_1$ 开始，通过全期望公式：
$$
\mathbb{E}[E(S, T)]=\dfrac{1}{2} \mathbb{E}\left[E(S, T) \mid X_{v_{1}}=0\right]+\dfrac{1}{2} \mathbb{E}\left[E(S, T) \mid X_{v_{1}}=1\right]。
$$
至少存在一个赋值 $x_1 \in \{0, 1\}$ 使得：
$$
\mathbb{E}\left[E(S, T) \mid X_{v_{1}}=x_{1}\right] \geq \mathbb{E}[E(S, T)]。
$$
我们可以继续运用这个方法：
$$
\mathbb{E}\left[E(S, T) \mid X_{v_{1}}=x_{1}, \ldots, X_{v_{i-1}}=x_{i-1}\right]= \frac{1}{2} \mathbb{E}\left[E(S, T) \mid X_{v_{1}}=x_{1}, \ldots, X_{v_{i-1}}=x_{i-1}, X_{v_{i}}=0\right]
+\frac{1}{2} \mathbb{E}\left[E(S, T) \mid X_{v_{1}}=x_{1}, \ldots, X_{v_{i-1}}=x_{i-1}, X_{v_{i}}=1\right]
$$
至少存在一个赋值 $x_i \in \{0, 1\}$ 使得：
$$
\mathbb{E}\left[E(S, T) \mid X_{v_{1}}=x_{1}, \ldots, X_{v_{i}}=x_{i}\right] \geq \mathbb{E}\left[E(S, T) \mid X_{v_{1}}=x_{1}, \ldots, X_{v_{i-1}}=x_{i-1}\right]
$$
所以我么就可以这要找到一个序列 $x_1, x_2, ..., x_n \in \{0, 1\}$ 来形成一个单调的路径：
$$
\mathbb{E}[E(S, T)] \leq \cdots \leq \mathbb{E}\left[E(S, T) \mid X_{v_{1}}=x_{1}, \ldots, X_{v_{i-1}}=x_{i-1}\right] \leq \mathbb{E}\left[E(S, T) \mid X_{v_{1}}=x_{1}, \ldots, X_{v_{i}}=x_{i}\right] \leq \cdots \leq \mathbb{E}\left[E(S, T) \mid X_{v_{1}}=x_{1}, \ldots, X_{v_{n}}=x_{n}\right]
$$
最终一个二分 $(\hat{S}, \hat{T})$ 被这个序列确定下来，并且 $E(\hat{S}, \hat{T}) \geq \frac{OPT_G}{2}$。

我们可以有如下算法：

``` pseudocode monotone_path
Input: G(V,E)

S = T = emptyset
for i = 1, 2, ..., n:
	v_i joins one of S,T to maximize the average size of cut conditioning on the choices made so far by the vertices (v_1, v_2, ..., v_i)
```

所以，上面的贪心算法实际上是通过平均情况的去随机化得来的。



## 通过 pairwise 独立的去随机化
