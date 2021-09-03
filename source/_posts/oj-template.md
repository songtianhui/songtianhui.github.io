---
title: OJ模板整理
date: 2021-06-24 14:05:07
tags:
categories: 
- 课程
- problem-solving
---

快期末oj了，也是最后一学期的oj，整理一下常用的算法模板，既是复习（考试偷看），也是保留一份以后时常用得上的自己习惯的板子。

<!--more -->



# 二分查找

``` c++
int ans = 0, l, r;
    // some input ...
    while (l <= r) {
        int m = (l + r) / 2;
        if (check(m)) {
            ans = m;
            l = m + 1;
        } else {
            r = m - 1;
        }
    }
```



# 数位dp

``` c++
ll solve(int pos, bool flag, .../*cur_state*/) {	// flag 标识会不会超过最大数
    if (cond) return ;
    if (flag == false && dp[pos][...] != -1) return dp[pos][...];
    ll bound = (flag == true ? (ll)(s[pos - 1] - '0') : (ll)9);
    for (int digit = 0; digit <= bound; ++i) {
        // calculate new state
        bool new_flag = false;
        if (flag == 1 && digit == bound) new_flag = true;
        solve(pos - 1, new_flag, .../*new_state*/);
    }
}
```



# 并查集

``` c++
int fa[MAXN] = {};

void init() {
    for (int i = 1; i <= n; ++i) fa[i] = i;
}

int find(int x) {
    if (x != fa[x]) {
        fa[x] = find(fa[x]);
    }
    return fa[x];
}

// 一般不使用按秩合并就足够好了
void merge(int x, int y) {
    x = find(x);
    y = find(y);
    fa[x] = y;
} 
```



# 最小生成树

``` c++ kruskal
// 基本思想是从小到大加入边，是个贪心算法。
// 一般都用kruskal
struct Edge {
  int from, to, val;
  Edge(int from, int to, int val)
      : from(from), to(to), val(val) {
  }
  bool operator<(const struct Edge &rhs) const {
    return val < rhs.val;
  }
};

vector<Edge> edges;
int fa[MAXN] = {};	// 维护点集合

int kruskal() {
    int cnt = 0, ans = 0;
    sort(edges.begin(), edges.end());
    for (auto &e: edges) {
        if (cnt >= n) break;
        if (find(e.from) != find(e.to)) {
            merge(e.from, e.to);
            ans += e.val;
            ++cnt;
        }
    }
    return ans;
}

/* 读入
for (int i = 1; i <= m; ++i) {
    int p, q, val;
    scanf("%d%d%d", &p, &q, &val);
    edges.emplace_back(Edge(p, q, val));
    edges.emplace_back(Edge(q, p, val));
}*/
```



# 最短路

## 单源最短路

``` c++ dijkstra
vector<pair<int, int>> edges;
bool vis[MAXN] = {};
int cost[MAXN] = {};

void dijkstra(int src) {
    priority_queue<pair<int, int>> pq;	// 用于记录到达每个点的消耗，pair第一分量为cost的相反数，第二分量为到达的点的编号
    cost[src] = 0;
    pq.emplace(0, src);
    while (!pq.empty()) {
        const auto state = pq.top(); pq.pop();
        const int cur_cost = state.first;
        const int cur_idx  = state.second;
        if (vis[cur_idx]) continue;
        vis[cur_idx] = true;
        for (const auto &edge: edges[cur_idx]) {
            const int new_cost = cur_cost - edge.second;	// 因为对cost取负，所以用减法
            const int new_idx  = edge.first;
            if (new_cost > cost[new_idx]) {
                pq.push({new_cost, new_idx});
                cost[new_idx] = new_cost;
            }
        }
    }
}
```



``` c++ 多状态版dijkstra
typedef struct {...} State;
typedef int Value;

pair<Value, State> get_next(const State current, int idx) {
    //return new state
}

int dijkstra() {
    priority_queue<pair<Value, State>> pq;	// 原来second只记录点编号，现在用来表示更多的状态，相当于增加维度（分层图）
    cost[src] = 0;
    pq.emplace(0, src);
    while(!pq.empty()) {
        const auto [cur_cost, cur_state] = pq.top(); pq.pop();
        if(vis[cur_state]) continue;
        vis[cur_state] = true;
        for(const auto& edge : edges[cur_idx]) {
            const auto [new_cost, new_state] = get_next(cur_state, edge);
            if(new_cost > cost[new_state]) {
                // pq.push
                // cost[cur_state] =
            }
        }
    }
}

```



## 多源最短路

``` c++ floyd
void floyd(){
    for(int k = 1; k <= n; ++k) {
        for(int i = 1; i <= n; ++i) {
            for(int j = 1; j <= n; ++j){
                if(dis[i][j] > dis[i][k] + dis[k][j]) {
                    dis[i][j] = dis[i][k] + dis[k][j];
                }
            }
        }
    }
}
```



# 连通分量

说实话tarjan我一直没搞懂:sob:，直接搬助教的模板了。。

``` c++ tarjan
bool vis[MAXN] = {};
int low[MAXN] = {};
int dfn[MAXN] = {};
int belong[MAXN] = {};	// 第i个点属于第几个连通分量

void dfs(int pos, int pre) {                     // pos：当前位置，pre：上一个位置
    static int ts = 0;                           // 利用C++的static特性声明一个只有函数内有效的变量，也可以用全局变量
    vis[pos] = true;                             // vis也可以叫做instack，用途是判断当前点是否在栈内
    s.emplace(pos);                              // 将当前节点压入栈
    low[pos] = dfn[pos] = ++ts;                  // 访问新的节点，更新时间戳
    for (auto &nex : edges[pos]) {               // 便利所有可能的下一个节点
        if (dfn[nex] == 0) {
            dfs(nex, pos);                       // 如果还没有访问过，则递归访问并更新最小节点编号
            low[pos] = min(low[pos], low[nex]);
        } else {
            if (vis[nex] and nex != pre) {          // 这里判断vis的原因是要处理单向边的情况，如果dfn>0但!vis，则说明只可以单向访问，一定无法和这个点在同一个强连通分量中
            // 如果有向图则不需要判定 nex != pre
                low[pos] = min(low[pos], dfn[nex]); // 如果已经访问过了，则判断是否是更小的节点编号，不需要递归访问
            }
        }
    }
    if (low[pos] == dfn[pos] and vis[pos]) {        // 如果在栈内部且low=dfn，那么就找到了一个新的强连通分量
        ++cnt;                                      // 强连通分量的总数加一
        while (not s.empty() and s.top() != pos) {  // 不断弹出栈中的元素，标记当前的强连通分量的编号
            belong[s.top()] = cnt;
            vis[s.top()] = false;
            s.pop();
        }
        if (not s.empty()) s.pop();
        belong[pos] = cnt, vis[pos] = false;
    }
}
// 总之tarjan可以把所有的点分到强连通分量里去

int deg[MAXN] = {};	// 分量的度

for (int i = 0; i < n; ++i) {
    for (auto &j : edges[i]) {
        if (belong[i] != belong[j]) {  // 如果某一条边的两个顶点不在同一个分量中
            ++deg[belong[i]];          // i对应的分量出度+1
        }
    }
}
```



# 完美匹配

``` c++ hungarian
bool used[MAXN] = {};	//unordered_set<int> used;
int belong{MAXN} = {};

bool find(int x) {
    for (auto &y: edges[x]) {
        if (not used[y]) {
            used[y] = true;
            if (belong[y] == 0 or find(belong[y])) {
                belong[y] = x;
                return true;
            }
        }
    }
    return false;
}

int hungarian() {
    int tot = 0;
    for (int i = 1; i <= n; ++i) {
        used.clear();
        if (find(x)) {
            tot++;
        }
    }
    return tot;
}
```





# 网络流

网络流Dinic我其实也没搞懂qvp，照搬oiwiki了:cry:。

``` c++ Dinic
struct Edge {
  int from, to, cap, flow;
  Edge(int u, int v, int c, int f) : from(u), to(v), cap(c), flow(f) {}
};

struct Dinic {
  int n, m, s, t;
  vector<Edge> edges;
  vector<int> G[maxn];
  int d[maxn], cur[maxn];
  bool vis[maxn];

  void init(int n) {
    for (int i = 0; i < n; i++) G[i].clear();
    edges.clear();
  }

  void AddEdge(int from, int to, int cap) {
    edges.push_back(Edge(from, to, cap, 0));
    edges.push_back(Edge(to, from, 0, 0));
    m = edges.size();
    G[from].push_back(m - 2);
    G[to].push_back(m - 1);
  }

  bool BFS() {
    memset(vis, 0, sizeof(vis));
    queue<int> Q;
    Q.push(s);
    d[s] = 0;
    vis[s] = 1;
    while (!Q.empty()) {
      int x = Q.front();
      Q.pop();
      for (int i = 0; i < G[x].size(); i++) {
        Edge& e = edges[G[x][i]];
        if (!vis[e.to] && e.cap > e.flow) {
          vis[e.to] = 1;
          d[e.to] = d[x] + 1;
          Q.push(e.to);
        }
      }
    }
    return vis[t];
  }

  int DFS(int x, int a) {
    if (x == t || a == 0) return a;
    int flow = 0, f;
    for (int& i = cur[x]; i < G[x].size(); i++) {
      Edge& e = edges[G[x][i]];
      if (d[x] + 1 == d[e.to] && (f = DFS(e.to, min(a, e.cap - e.flow))) > 0) {
        e.flow += f;
        edges[G[x][i] ^ 1].flow -= f;
        flow += f;
        a -= f;
        if (a == 0) break;
      }
    }
    return flow;
  }

  int Maxflow(int s, int t) {
    this->s = s;
    this->t = t;
    int flow = 0;
    while (BFS()) {
      memset(cur, 0, sizeof(cur));
      flow += DFS(s, INF);
    }
    return flow;
  }
};
```



# 数论

``` c++ gcd
ll exgcd(ll a, ll b, ll &x, ll &y) {
  if (b == 0) {
    x = 1LL, y = 0LL;
    return a;
  }
  ll r = exgcd(b, a % b, y, x);
  y -= (a / b) * x;
  return r;
}
```



``` c++ 快速幂
ll qmul(ll a, ll b, ll p) {
  ll ret = 0;
  while (b) {
    if (b & 1) {
      ret = (ret + a) % p;
    }
    a = (a + a) % p;
    b >>= 1;
  }
  return ret;
}

ll qpow(ll x, ll n, ll p) {
  ll ret = 1;
  while (n) {
    if (n & 1) {
      ret = ret * x % p;
    }
    x = x * x % p;
    n >>= 1;
  }
  return ret;
}
```



``` c++ 乘法逆元
ll inverse(ll x, ll p) {	// p为质数
  return qpow(x, p - 2, p);
}

ll inverse(ll a, ll b) {	// b不为质数
    ll x, y;
    exgcd(a, b, x, y);
    return x;
}
```



``` c++ 组合数
ll choose(ll n, ll m, ll p) {
  if (m == 0 || m == n) {
    return 1;
  }
  ll x = 1, y = 1;
  for (ll i = n; i > n - m; --i) {
    x = x * i % p;
  }
  for (ll i = 1; i <= m; ++i) {
    y = y * i % p;
  }
  return x * inverse(y, p) % p;
}
// 也可以递推动态规划
```



``` c++ 卢卡斯定理
ll lucas(ll n, ll m, ll p) {
  if (m == 0) {
    return 1;
  } else {
    return choose(n % p, m % p, p) * lucas(n / p, m / p, p) % p;
  }
}
```



``` c++ 中国剩余定理
ll crt(const vector<ll> &a, const vector<ll> &m, ll p) {
  ll ret = 0;
  for (int i = 0, n = a.size(); i < n; ++i) {
    ll w = p / m[i], x, y;
    exgcd(w, m[i], x, y);
    x = (x % m[i] + m[i]) % m[i];
    ret = (ret + qmul(qmul(a[i], w, p), x, p)) % p;
  }
  return (ret + p) % p;
}
```



``` c++ 欧拉函数打表
int ans[MAXN] = {1};
bool fact[MAXN] = {};

for(int i = 1; i < MAXN; ++i)
	ans[i]=i;
for(int i = 2; i < MAXN; ++i) {
    if(fact[i]) continue;
    for(int j = i; j < MAXN; j += i){
        fact[j] = 1;
        ans[j] /= i;
        ans[j] *= i-1;
    }
}
```



``` c++ 素数判定
bool check(ll a,ll n,ll x,ll t){   //以a为基，n-1=x*2^t，检验n是不是合数
    ll ret=pow_mod(a,x,n),last=ret;
    for (int i=1;i<=t;i++){
        ret=muti_mod(ret,ret,n);
        if (ret==1 && last!=1 && last!=n-1) return 1;
        last=ret;
    }
    if (ret!=1) return 1;
    return 0;
}

bool Miller_Rabin(ll n) {
    ll x=n-1,t=0;
    while ((x&1)==0) x>>=1,t++;
    bool flag=1;
    if (t>=1 && (x&1)==1){
        for (int k=0;k<S;k++){
            ll a=rand()%(n-1)+1;
            if (check(a,n,x,t)) {flag=1;break;}
            flag=0;
        }
    }
    if (!flag || n==2) return 0;
    return 1;
}
```





# 字符串匹配

``` c++ kmp
char s1[]; int len1;	// 主串
char s2[]; int len2;	// 模式串
int nxt[] = {};

void build_nxt() {
    int i = 1, now = 0;
    while (i < len2) {
        if (s2[i] == s2[now]) {
            now++;
            nxt[i] = now;
            i++;
        } else if (now != 0) {
            now = nxt[now - 1];
        } else {
            i++;
            nxt[i] = now;
        }
    }
}

void kmp() {
    int tar = 0, pos = 0;
    while (tar < len1) {
        if (s1[tar] == s2[pos]) {
            tar++;
            pos++;
        } else if (pos != 0) {
            pos = nxt[pos - 1];
        } else {
            tar++;
        }
        if (pos == len2) {
            // find an answer = tar - pos + 1
            pos = nxt[pos - 1];
        }
    }
}
```



# 模拟退火

考试整不出来只能玄学调参了:joy:。。。

``` c++ simulateAnneal
#include <cmath>
#include <cstdlib>
#include <ctime>

double Rand() { return (double)rand() / RAND_MAX; }

double dis, ansx;
double cal(double x) {
    double res = 0;
    ...
    if (res < dis) dis = res, ansx = x;
    return res;
}

void simulateAnneal() {
    double T = T_init;
    double cur_x = ansx;
    while (T > T_min) {
        double new_x = get_next_state(cur_x, t);
        double delta = cal(new_x) - cal(cur_x);
        if (exp(-delta / T) > Rand()) cur_x = new_x;	// 具体看求min还是max
        //max: if (delta > 0 || exp(delta / T) > Rand())
        T *= rate;
    }
    for (int i = 0; i < 10000; ++i) {	// 小温度跑一跑找到局部最优
        double new_x = get_next_state(ans_x, T);
        cal(new_x);
    }
}

/*
int main() {
    srand(time(0));
    init(ansx); dis = cal(ansx);
    simulateAnneal();
}
*/
```



# 参考

[oiwiki](https://oiwiki.org/)

[学姐blog](https://mengzelev.github.io/2019/04/19/oj-templates/)

[oj题解](https://oj-solutions.njujb.com/)

