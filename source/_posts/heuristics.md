---
title: 问题求解笔记-启发式算法
categories: 
- 课程
- problem-solving
mathjax: true
date: 2021-06-06 08:27:48
tags:
---




---

# 模拟退火算法（Simulated Annealing）

## 基本概念

- $U = (\Sigma_I, \Sigma_O, L, L_I, \mathcal{M}, cost, goal)$ 是一个优化问题，对任意的 $x \in L_I$，$\mathcal{M}(x)$ 的一个**邻域（Neighborhood）**是一个映射 $f_x: \mathcal{M}(x) \to Pot(\mathcal{M(x)})$ 满足：
  - $\forall \alpha \in \mathcal{M}(x), \alpha \in f_x(\alpha)$
  
  - 如果 $\exists \alpha \in \mathcal{M}(x), \beta\in f_x(\alpha)$，则 $\alpha \in f_x(\beta)$
  
  - 对任意的 $\alpha, \beta \in \mathcal{M}(x)$，所在一个正整数 $k$ 和 $\gamma_1, ..., \gamma_k \in \mathcal{M}(x)$，使得 
  
    $\gamma_1 \in f_x(\alpha), \gamma_{i  +1 } \in f_x(\gamma_i), i = 1, ... k-1$ 且 $\beta \in f_x(\gamma_k)$​
- $LSS(Neigh)$

<!--more -->

模拟退火算法是从一个物理退火模型（Metropolos Algorithm）类比启发启发得到的，和 $LSS$ 的区别是有概率允许状态向恶化的方向转化，来防止局部最优解。

一些概念的类比：

|   物理退火   |     模拟退火     |
| :------: | :---------: |
|   系统状态   |    可行解集合    |
|   状态能量   |   可行解的代价   |
|   扰动机制   | 从邻居的随机选择 |
| 一个最优状态 |  一个最优可行解  |
|     温度     |     控制参数     |

所以模拟退火其实就是建立在类比 Metropolos Algorithm上，得到的一个局部搜索算法。

对于一个优化问题的固定的邻域，模拟退火可以被描述为：

![](SA.png "模拟退火算法")



## 理论和经验

**定理 6.2.2.1.**  $U$ 是一个最小化问题且让 $Neigh$ 是 $U$ 的邻域，对于输入 $x$ 模拟退火算法的渐进收敛（asymptotic convergence）被保证如果以下两个条件被满足：

- 每个 $\mathcal{M}(x)$ 的可行解是可到达的，从每个其他的 $\mathcal{M}(x)$ 的可行解。（邻域的定义条件III）
- 初始条件 $T$ 至少是最深的局部最小值。

*渐进收敛意味着随着迭代次数的增长，到达全局最优的概率趋向于1*。

*就是说模拟退火算法会 find itself at a  global optimum an increasing percentage of the time as time grows.*（我也没读懂）

一般还有要求：

- 邻域的对称性（邻域定义的条件II）
- 邻域的均匀性，$\forall \alpha, \beta \in \mathcal{M}(x),|Neigh_x (\alpha)| = |Neigh_x(\beta)|$

- 通过特定的 *cool schedules*，$T$ 增长缓慢

很遗憾在限定的迭代次数内，模拟退火并不能确保可行解的高质量，一般需要解空间的平方大小的次数的迭代。所以相比运行模拟退火到确保一个正确的近似解，还不如直接搜索全空间。。。

模拟退火的收敛率是和迭代次数成对数关系。

应该尽可能地避免以下结构的邻域：

- 毛刺结构
- 深沟
- 大高原

*因为模拟退火第一部分是在高温下完成的，使得所有恶化方向都有可能，所以可以视为对初始解的随机搜索。然后当 $T$ 很小，大的恶化基本上是不可能的。*



如果谈到 cool schedules，以下的参数一定要确定：

- 初始温度 $T$
- 温度缩减函数 $f(T, t)$
- 终止条件



### 初始温度的选择

需要很大。

一种方法是取任意两个相邻解的最大差值，比较难算，所以只要高效地找到一个估计值（上界）就足够了。

另一种实用的方法是，从任意 $T$ 值开始，在选择了初始情况 $\alpha$ 的邻居 $\beta$ 后，以一种使 $\beta$ 以靠近1的概率被接受的方式增加 $T$。进行若干次迭代，可以得到一个合理的初值 $T$。*可以看作物理类比的加热过程*



### 选择温度缩减函数（Temperature Reduction Function）

一个一般的方式是 $T$ 乘以一个系数 $0.8\leq r \leq 0.99$，工作常数 $d$ 步得到 $T := r \cdot T$。

也就是缩减 $k$ 次得到的温度 $T_k$ 为 $r_k\cdot T$。



另一个选择是 $T_k := \dfrac{T}{\log_2{(k  +2)}}$，常数 $k$ 一般为邻域的大小。



### 终止条件

一个方法是终止条件独立于 $T$，当解的代价（cost）不变时终止。

另一种是设一个常数 $term$，当 $T \leq term$ 时终止。



### 一般性质

- 模拟退火算法能够得到一个高质量解，但需要大开销的运算。
- 最终解的质量和初始解的选择关系不大。
- 重要的两个参数是缩减率和邻域，有必要做一定的工作来调参。
- 平均复杂度和最差复杂度很接近。
- 在相同的邻域下，模拟退火一般比局部搜索或多源局部搜索表现得更好。

### 一些问题的应用

- 图位置算法（最大最小割）一般表现较好。
- 工程问题（VLSI）是最好的算法。
- TSP等表现不太好。



## 随机禁忌？搜索（Randomized Tabu Search）

思想就是：保存一些之前的可行解的信息，并在决定下一个可行解时用上。

![](RTS.png)



---

# 遗传算法

## 基本概念

遗传算法是从个体种群的进化优化启发而来，进化可以被概括为以下过程：

1. 一个个体可以用一个字符串来表示（DNA序列）。
2. **交配（crossover）**，也就是交换两个父母字符串的一部分，获得子串。
3. 存在一个**适应值（fitness value）**，适应值高的串以更高的概率被选择。
4. **变异（mutation）**，随机变一位。
5. 去世。



一些概念的类比：

| 种群遗传进化 |     遗传算法     |
| :----------: | :--------------: |
|   一个个体   |    一个可行解    |
|   一个基因   |    解中的一位    |
|    适应值    |     代价函数     |
|     种群     | 可行解的一个子集 |
|     变异     |   随机局部转化   |



遗传算法和模拟退火有两个区别，一是用一组可行解而不是一个可行解，二是不能被视为纯粹的局部搜索。

在一个合适的可行解表示下，遗传算法可以被描述为：

![](GAS.png "GAS")

如果不考虑交配，只考虑变异，那么遗传算法和模拟退火很类似。



- $U = (\Sigma_I, \Sigma_O, L, L_I, \mathcal{M}, cost, goal)$ 是一个优化问题，对一个给定大小为 $n$ 的输入 $x \in L_I$，每个 $\alpha \in \{0, 1\}^n$ 代表 $x$ 的一个可行解，即 $\mathcal{M}(x) = \{0, 1\}^n$。$\mathcal{M}(x)$ 的一个 **模式（schema）** 是一个向量 $s = (s_1, s_2, ..., s_n) \in \{0, 1, \star\}^n$。$s$ 的一组可行解是
  $$
  Schema(s_1, s_2, ..., s_n) = \{\gamma_1, \gamma_2, ..., \gamma_n \in \mathcal{x} | \gamma_i = s_i , s_i \in \{0,1\} ~or~ \gamma_i \in \{0,1\}, s_i = \star \}
  $$

- 模式 $s$ 的长度 $length(s)$ 是第一位到最后一个非 $\star$ 位的距离。阶 $order(s)$ 是非 $\star$ 的位数。

- 在一个种群 $P$ 中的**适应度（Fitness）** 是 $Schema(s)$ 的可行解的平均适应度：
  $$
  Fitness(s, P) = \dfrac{1}{|Schema(s) \cap P|} \cdot \sum\limits_{\gamma \in Schema(s) \cap P} cost(\gamma).
  $$

- 在一个种群 $P$ 中的**适应率（Fitness ratio）**是：
  $$
  Fit\text{-}ratio = \dfrac{Fitness(s, P)}{\frac{1}{|P|} \sum_{\beta \in P}cost(\beta)}
  $$

- $Aver\text{-}Fit(P) = \frac{1}{|P|}\sum_{\beta\in P} cost(\beta)$ 叫做种群 $P$ 的平均适应度。

*其实模式（schema）就是在可行解的表达中固定一些基因的值，其中 $\star$ 位就是自由变量，所以 $|Schema(s)|$ 就是$\star$的数量的2次幂。一个有比种群平均适应度高很多的适应度的模式就是一个很好的基因，应该被在进化中被遗传。*



- $P_0$ 为初始种群，$P_t$ 为第 $t$ 次迭代基因算法后的种群，$X_{t + 1}(s)$ 是一个随机变量，是从 $P_t$ 中随机选择的父母且在 $Schema(s)$ 的数量。选择 $\alpha \in P_t$ 的概率是：$Pr_{par}(\alpha) = \dfrac{cost(\alpha)}{\sum_{\beta\in P_t} cost(\beta)} = \dfrac{cost(\alpha)}{|P_t| \cdot Aver\text{-}Fit(P_t)}$

**引理 6.3.1.2.** 对任意的 $t \in \mathbb{N}$ 和任意模式 $s$，$E[X_{t + 1}(s)] = Fit\text{-}ratio(s, P_t) \cdot |P_t \cap Schema(s)|$。

**引理 6.3.1.3.** 对任意的模式 $s$ 和任意 $t = 1, 2,...$，
$$
E[Y_{t+1}(s)] \geq \dfrac{|P_t|}{2} \cdot \left[2 \cdot \left( \dfrac{E[X_{t+1}(s)]}{|P_t|}\right)^2 + 2 \cdot \dfrac{n - length(s)}{n} \cdot\dfrac{E[X_{t+1}(s)]}{|P_t|} \cdot \left( 1 - \dfrac{E[X_{t+1}(s)]}{|P_t|} \right) \right].
$$
**引理 6.3.1.4.** 对任意的模式 $s$ 和任意 $t = 1, 2,...$，
$$
E[Y_{t + 1}(s)] \geq \dfrac{n - length(s)}{n} \cdot E[X_{t+1}(s)].
$$
**引理 6.3.1.5.** 对任意的模式 $s$ 和任意 $t = 1, 2,...$，
$$
E[Z_{t + 1}(s)] \geq (1 - pr_{m})^{order(s)} \cdot E[Y_{t+1}(s)] \geq (1 - order(s) \cdot pr_m) \cdot E[Y_{t+1}(s)].
$$
**定理 6.3.1.6.（GAS的模式定理）**

对任意的模式 $s$ 和任意 $t = 1, 2,...$，在第 $(t+1)-st$ 的种群 $P_{t+1}$ 中的为模式 $Schema(s)$ 的个体的数量的期望为
$$
E[Z_{t+1}] \geq Fit\text{-}ratio(s, P_t) \cdot \dfrac{n-  length(s)}{n} \cdot (1 - order(s)\cdot pr_m) \cdot |P_t \cdot Schema(s)|.
$$


## 参数调整

### 种群大小

- 小种群容易导致局部最优解。
- 反之大种群更容易到达全局最优解。

### 初始种群的选择

- 常用的是随机选择。
- 将一些预处理的高质量可行解引入初始种群可以加快收敛速度。
- 最好是结合起来，部分随机部分高适应度。

### 适应度评估和父母选择机制

- 适应度 $\alpha$ 直接用 $cost(\alpha)$，然后按概率分布 $p(\alpha) = \dfrac{cost(\alpha)}{\sum_{\beta \in P} cost(\beta)}$ 选择父母。
- 当 $max(P)$ 和 $min(P)$ 很接近时，$P$ 几乎均匀分布，一般选择 $fitness(\alpha) = cost(\alpha) - C, C < min(P)$
- 选择 $k / 2$ 个最好的个体，再随机选择 $k/2$ 个个体进行配偶。
- 通过**秩（ranking）**来随机选择，按代价排序后 $cost(\alpha_1) \leq cost(\alpha_2) \leq ...\leq cost(\alpha_n)$，然后分配概率 $prob(\alpha_i) = \dfrac{2i}{n(n+1)}$ 选择父母。

### 个体和交配的形式表示

- 允许不代表任何可行解的表示存在，但可能降低复杂度。
- 不直接在父母的表示上进行交配，而是通过一些特点参数，寻找子代的交叉混合属性。
- 在上面所有的情况，交配都是只考虑一个交配位置的简单交配，可以与更复杂的交配的概念结合起来。

### 变异的可能性

- 通常改变一位基因的概率调整为小于 $\frac{1}{100}$。
- 有时候取 $\frac{1}{n}$ 或 $\frac{1}{k^{0.93}\sqrt{n}}$，$n$ 是基因数量，$k$ 是种群规模。
- 有时候在计算过程中动态改变概率，如果种群中个体过于相似，增加突变概率，防止局部最优。

### 新种群的选择机制

- 子代完全取代父代，**en block** 策略。
- 从子代和父代中同时选择，需要调整子代对父代的偏爱程度。一种常见的是从父代中挑选一些适应度最高的，种群中绝大多数是子代。
- 在选择了新的种群后，可以立即开始交配生产新种群，也可以提高种群平均适应度，可以选择小邻域进行局部搜索或模拟退火，得到一个局部最优的个体。

### 终止标准

- 在一开始确定时间限制。
- 测量种群的平均适应度和个体之间的差异，如果平均适应度没有本质改变，或个体非常相似，遗传可以停止。
