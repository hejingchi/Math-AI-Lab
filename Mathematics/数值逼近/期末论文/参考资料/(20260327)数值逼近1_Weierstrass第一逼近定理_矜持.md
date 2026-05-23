# 数值逼近（1） Weierstrass第一逼近定理

 **Author:** [矜持]

 **Link:** [https://zhuanlan.zhihu.com/p/2020938535826405077]

最近在研究数值逼近和概率论，里面有许多内容是相似和有关联的。数值逼近当中一个很经典的问题就是研究如何用多项式去逼近一个连续函数。Weierstrass 第一逼近定理就给出了这样一个结果：闭区间上的任何连续函数都能用一列多项式一致逼近.用数学语言描述，就是

$\forall \epsilon > 0, \exists N, \forall n \geq N, \forall x \in [a, b], |f(x) - p_n(x)| < \epsilon$

其中$\{p_n(x)\}$是一列多项式。这个定理告诉我们，闭区间上的任何连续函数可以被多项式”很好地”逼近。下面给出这个定理的具体证明。

我们先考虑简单的情形，比如$f(x)$是闭区间$[0,1]$上的连续函数.我们可以使用 Bernstein 多项式来构造逼近函数.对于每个$n$，定义

$B_n^f(x) = \sum_{k=0}^{n} f\left(\frac{k}{n}\right) \binom{n}{k} x^k (1-x)^{n-k}$

其中$\binom{n}{k}$是二项式系数.接下来，我们需要证明当$n$趋近于无穷大时，$B_n^f(x)$一致收敛于$f(x)$.在做这件事情之前，我们先看看这个构造的动机。这个构造和二项分布有着密不可分的联系。二项分布描述了进行$n$次独立重复试验的成功次数的概率。其中每次试验成功的概率为$p$.进行$n$次实验后，成功$k$次的概率如下：

$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$

我们不难发现，当$k$接近$np$时，成功的概率是最大的。如果我们把这个经验用到上面构造的多项式上面，所谓$B_n^f(x)$就是在对函数$f$进行加权平均。我们希望$B_n^f(x)$在$x$处的取值能更好的靠近$f(x)$。所以就得到了上面那个构造。

为了证明$B_n^f(x)$一致收敛于$f(x)$，我们需要以下关键恒等式：

$\sum_{k=0}^{n} \binom{n}{k} x^k (1-x)^{n-k} = 1$

这是二项式定理 $(x + (1-x))^n = 1$ 的特殊情况。下面通过二项式定理及其导数，建立三个有用的恒等式。

**引理 1**:

$\sum_{k=0}^{n} \binom{n}{k} u^k v^{n-k} = (u + v)^n$

这就是二项式定理。这里不加证明。

如果我们对上述式子两边同时关于$u$求导，我们可以得到

$\sum_{k=0}^{n} k \binom{n}{k} u^{k-1} v^{n-k} = n(u + v)^{n-1}$

然后等式两边再同时乘上$u$，我们就得到了

**引理 2**:

$\sum_{k=0}^{n} k \binom{n}{k} u^k v^{n-k} = nu(u + v)^{n-1}$

**推论**：当 $u = v = 1$ 时，$\sum_{k=0}^{n} k \binom{n}{k} = n \cdot 2^{n-1}$

同样的，我们对引理 2 所得的等式两边同时关于$u$求导，我们可以得到

$\sum_{k=0}^{n} k^2 \binom{n}{k} u^{k-1} v^{n-k} = n(u+v)^{n-1} + n(n-1)u(u+v)^{n-2}$

然后等式两边再同时乘上$u$，我们就得到了

**引理 3**:

$\sum_{k=0}^{n} k^2 \binom{n}{k} u^k v^{n-k} = nu(u+v)^{n-1} + n(n-1)u^2(u+v)^{n-2}$

**推论**：当 $u = v = 1$ 时，$\sum_{k=0}^{n} k^2 \binom{n}{k} = n(n+1)2^{n-2}$ 这里也可以不用求导的方法证明这三个恒等式，即$ k \binom{n}{k} = n\binom{n-1}{k-1}$但过程比较繁琐，并且需要考虑边界情况。这里不过多考虑。具体交给读者来完成。

接下来就可以开始证明$B_n^f(x)$一致收敛于$f(x)$了。

---

**证明**:

由于$f$在闭区间$[0,1]$上连续，根据 Cantor 定理，闭区间上的连续函数一致连续，即

$\forall \epsilon > 0, \exists \delta > 0, \forall x,y \in [0,1], |x-y| < \delta \Rightarrow |f(x) - f(y)| < \epsilon$

利用一致连续性，我们可以对 Bernstein 多项式与$f(x)$的差进行估计。因为

$\sum_{k=0}^{n} \binom{n}{k} x^k (1-x)^{n-k} = 1$

所以可以把$f(x)$改写为

$f(x) = \sum_{k=0}^{n} f(x) \binom{n}{k} x^k (1-x)^{n-k}$

因此

$B_n^f(x) - f(x) = \sum_{k=0}^{n} \left( f\left(\frac{k}{n}\right) - f(x) \right) \binom{n}{k} x^k (1-x)^{n-k}$

下面根据 $|k/n - x|$ 的大小，将求和分为两部分估计。

当 $|k/n - x| < \delta$ 时，有 $|f(k/n) - f(x)| < \epsilon$。因此

$\left| \sum_{|k/n - x| < \delta} \left( f\left(\frac{k}{n}\right) - f(x) \right) \binom{n}{k} x^k (1-x)^{n-k} \right| \leq \sum_{|k/n - x| < \delta} \left| f\left(\frac{k}{n}\right) - f(x) \right| \binom{n}{k} x^k (1-x)^{n-k}$

$\quad < \sum_{|k/n - x| < \delta} \epsilon \binom{n}{k} x^k (1-x)^{n-k} \leq \sum_{k=0}^{n} \epsilon \binom{n}{k} x^k (1-x)^{n-k} = \epsilon$

这里利用了恒等式 $\sum_{k=0}^{n} \binom{n}{k} x^k (1-x)^{n-k} = 1$

当$|k/n - x| \geq \delta$ 时，我们利用 $(k-nx)^2 \geq n^2\delta^2$ 这个不等式来放大 $|f(k/n) - f(x)|$设 $f(x)$ 在 $[0,1]$ 上的上界为 $M$（即 $|f(x)| \leq M$），则有 $|f(k/n) - f(x)| \leq 2M$

因此

$\left| \sum_{|k/n - x| \geq \delta} \left( f\left(\frac{k}{n}\right) - f(x) \right) \binom{n}{k} x^k (1-x)^{n-k} \right| \leq \sum_{|k/n - x| \geq \delta} 2M \binom{n}{k} x^k (1-x)^{n-k}$

利用不等式 $(k-nx)^2 \geq n^2\delta^2$ 在 $|k/n - x| \geq \delta$ 时成立，我们有

$\frac{(k-nx)^2}{n^2\delta^2} \geq 1$

将这个不等式乘以 $2M\binom{n}{k}x^k(1-x)^{n-k}$ 求和，可以得到

$\left| \sum_{|k/n - x| \geq \delta} \left( f\left(\frac{k}{n}\right) - f(x) \right) \binom{n}{k} x^k (1-x)^{n-k} \right| \leq \frac{2M}{n^2\delta^2} \sum_{|k/n - x| \geq \delta} (k-nx)^2 \binom{n}{k} x^k (1-x)^{n-k}$

虽然右边是 $|k/n - x| \geq \delta$ 的部分和，利用上述引理123，我们可以得到 $\sum_{k=0}^n (k-nx)^2 \binom{n}{k} x^k (1-x)^{n-k}$

$\begin{aligned} \sum_{k = 0} ^ n (k-nx)^2 \binom{n}{k} x^k (1-x)^{n-k} &= \sum_{k = 0} ^ n k^2 \binom{n}{k} x^k (1-x)^{n-k} - 2nx\sum_{k=0}^n k\binom{n}{k}x^k(1-x)^{n-k} \\ &\quad + n^2x^2 \sum_{k=0}^n \binom{n}{k} x^k (1-x)^{n-k} \end{aligned}$

这里对引理简单解释一下（代入 $u = x, v = 1-x$）：

$\begin{aligned} &= [nx + n(n-1)x^2] - 2nx \cdot (nx) + n^2x^2 \cdot 1\\ &= nx + n(n-1)x^2 - 2n^2x^2 + n^2x^2\\ &= nx + n^2x^2 - nx^2 - 2n^2x^2 + n^2x^2\\ &= nx(1-x) \end{aligned}$

其实这里的技巧有着更深刻的定理Korovkin定理，它给出了相关逼近的更一般的结论。受篇幅限制这里先不展开。

由于 $x \in [0,1]$，所以 $x(1-x) \leq 1/4$。因此，对于 $|k/n - x| \geq \delta$ 的项，我们有

$\left| \sum_{|k/n - x| \geq \delta} \left( f\left(\frac{k}{n}\right) - f(x) \right) \binom{n}{k} x^k (1-x)^{n-k} \right| \leq \frac{2M}{n^2\delta^2} \cdot nx(1-x) \leq \frac{2M}{n^2\delta^2} \cdot \frac{n}{4} = \frac{M}{2n\delta^2}$

对于任给的 $\epsilon > 0$，我们先选定 $\delta > 0$。然后选择 $N$ 满足

$N > \frac{M}{2\epsilon\delta^2}$

当 $n >N$ 时，就有 $\frac{M}{2n\delta^2} \leq \epsilon$。此时对所有 $x \in [0,1]$，

$|B_n^f(x) - f(x)| < \epsilon + \epsilon = 2\epsilon$

因此 $B_n^f(x)$ 在 $[0,1]$ 上一致收敛于 $f(x)$。

对于定义在一般闭区间 $[a, b]$ 上的连续函数 $f(x)$，我们可以通过构造一个线性变换将其转化为 $[0, 1]$ 上的问题。

构造

$g(t) = f(a + (b - a)t), \quad t \in [0, 1]$

显然 $g(t)$ 也是 $[0, 1]$ 上的连续函数。由前面已经证明的结果，存在 Bernstein 多项式序列 $\{B_n^g(t)\}$ 在 $[0, 1]$ 上一致收敛于 $g(t)$。

将变量 $t$ 替换回 $x$，即 $t = \frac{x - a}{b - a}$，可以得到 $f(x)$ 在 $[a, b]$ 上的 Bernstein 逼近多项式：

$B_n^f(x) = B_n^g\left( \frac{x - a}{b - a} \right) = \sum_{k=0}^{n} f\left( a + \frac{k}{n}(b - a) \right) \binom{n}{k} \left( \frac{x - a}{b - a} \right)^k \left( 1 - \frac{x - a}{b - a} \right)^{n-k}$

不难证明这也是一致收敛的。

因此，我们得到了 Weierstrass 第一逼近定理的完整证明。

关于它的一个推广可以浏览下文

[数值逼近（2） Baskakov线性正算子序列](https://zhuanlan.zhihu.com/p/2020956774539239731)