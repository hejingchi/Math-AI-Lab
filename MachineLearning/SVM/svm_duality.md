# 从拉格朗日方法到对偶问题

前面我们已经构造了凸二次优化问题，包括
$$
\begin{aligned}
\min_{w,b}\quad  & \frac{1}{2}\|w\|^2\\ 
\text{s.t.} \quad  &  1-y_i \left(w \cdot x_i + b\right) \leqslant 0
\end{aligned}
$$
以及
$$
\begin{aligned}
\min_{w,b,\xi}\quad
                 & \frac12\|w\|^2 + C\sum_{i=1}^N \xi_i \\ 
\text{s.t.}\quad & 1-\xi_i-y_i(w \cdot x_i+b)\le0 \\
                 & -\xi_i\le0
\end{aligned}
$$
---

考虑$$
\begin{aligned}
\min_{w,b}\quad  & \frac{1}{2}\|w\|^2\\ 
\text{s.t.} \quad  &  1-y_i \left(w \cdot x_i + b\right) \leqslant 0
\end{aligned}
$$

令 $ c_i(x) = 1-y_i(w \cdot x_i + b)$
于是我们可以构造拉格朗日函数
$$
\begin{aligned}
L(w, \alpha) &= \frac{1}{2} \|w\|^2 + \sum_{i=1}^{N} \alpha_i c_i(x) \\
&=\frac{1}{2} \|w\|^2 +\sum_{i=1}^{N} 
\alpha_i -\sum_{i=1}^{N} \alpha_i\left(y_iw \cdot x_i+b\right)
\end{aligned}
$$

关于求解此类问题，我们考虑一般的情形。对优化问题
$$
\begin{aligned}
\min_{x}\quad& f(x)\\ 
\text{s.t.}\quad &h_i(x) =0\\
                 &c_j(x) \leqslant0
\end{aligned}
$$
引入相关的拉格朗日函数
$$L(x,\alpha, \beta) = f(x) +
 \sum_{i,\alpha_i \geqslant 0} \alpha_i c_i(x) +
 \sum_{j} \beta_{j} h_j(x) 
$$

直接求解该问题，我们知道极值点的必要条件要满足以下条件：

*  $\nabla_x L=0$ 并且 $\nabla_\beta L=0$，其中后者本质上是等式约束
* 不等式约束的系数 $\alpha_i \geqslant0$
* 目标函数的极值要么在不等式约束的边界上取到等号，也就是 $c_i(x)=0$，此时 $\alpha_i>0$。
要么在不等式约束的内部取到等号，也就是 $c_i(x)<0$。此时$\alpha_i=0$。
不管是哪一种情形，我们都能得到 $\alpha_i c_i(x)=0$。

但是这样求解，由于我们并不知道某个不等式是否对目标函数的极值产生了影响，因此需要
对 $\alpha$ 的值进行讨论，过程非常复杂。那有什么简化的方法吗？

## 广义拉格朗日的极小极大问题

下面我们研究一个被称为广义拉格朗日的极小极大问题。我们延续上面的符号，考虑拉格朗日函数
$$L(x,\alpha, \beta) = f(x) +
 \sum_{i,\alpha_i \geqslant 0} \alpha_i c_i(x) +
 \sum_{j} \beta_{j} h_j(x) 
$$

定义 $$\theta_p(x) = \max_{\alpha,\beta,\alpha_i\geqslant0}
L(x, \alpha, \beta)
$$
如果 $x$ 满足约束条件，即$$
c_i(x)\le0,\quad h_j(x)=0
$$由于 $\alpha_i\ge0$，因此
$$
L(x,\alpha,\beta)
=
f(x)+\sum_i\alpha_i c_i(x)
\le f(x)
$$当取 $\alpha_i=0$ 时取等，因此
$$
\theta_p(x)
=
\max_{\alpha,\beta,\alpha_i\geqslant0}L(x,\alpha,\beta)
=
f(x)
$$
若 $x$ 不满足约束条件，则或者 $c_i(x) >0$,或者 $h_i(x) \neq 0$，两者都可以通过选取合适的参数
$\alpha$ 与 $\beta$ 使得 $L(x, \alpha, \beta)$ 趋于正无穷。于是我们有
$$
\theta_p(x) =\begin{cases}
f(x)   & x \text{符合约束} \\
+\infty & x \text{不符合约束}
\end{cases}
$$ 
从而我们对 $f$ 求极小，等价于对 $\theta_p(x)$ 求极小。

上面我们对拉格朗日函数先对参数求极大值，再求极小，也就是
$$\min_x\max_{\alpha,\beta,\alpha_i\geqslant0}L(x,\alpha,\beta)$$
我们把它称为广义拉格朗日的极小极大问题。他和原问题是等价的。我们令
$$p^*=
\min_x\max_
{\alpha,\beta,\alpha_i\geqslant0}L(x,\alpha,\beta)
=\min_x f(x)$$为原问题的解。
---

上面的我们先取极小再取极大，如果我们交换一下这个顺序呢？情况会有什么变化呢？类似地，我们可以引入
对偶问题。
## 广义拉格朗日的极大极小问题
定义$$\theta_d(\alpha, \beta) = \min_{x}
L(x, \alpha, \beta)$$
如果我们在这里考虑
$$
\max_{\alpha, \beta, \alpha_i\geqslant0}\theta_d(\alpha,\beta)
$$
其实就是
$$\max_{\alpha,\beta,\alpha_i\geqslant0}\min_xL(x,\alpha,\beta)$$
令 $$d^* =\max_{\alpha,\beta,\alpha_i\geqslant0}\min_xL(x,\alpha,\beta)$$是对偶问题的解。

我们把它称为广义拉格朗日的极大极小问题。可以看到这里与上面的
$$\min_x\max_{\alpha,\beta,\alpha_i\geqslant0}L(x,\alpha,\beta)$$
在 $\min$ 与 $\max$ 的前后顺序上有所区别。我们下面对这两个式子进行讨论。


## 弱对偶条件与强对偶条件

假设 $ x^{*}$ 是某一个可行解，于是我们有 $ c_i(x^{*}) \leqslant0, h_j(x^{*}) =0$。从而
$$\min_x L(x, \alpha, \beta) 
\leqslant L(x^{*}, \alpha, \beta)
\leqslant f(x^{*})$$
左边成立是显然的，右边是因为当 $x^{*}$是某个可行解是，自然满足 
$$
\begin{aligned}
L(x^{*},\alpha,\beta)&=f(x^{*})+\sum_{i,\alpha_i\geqslant0}\alpha_ic_i(x^{*})
+\sum_{j}\beta_jh_j(x^{*}) \\
&=f(x^{*})+\sum_{i,\alpha_i\geqslant0}\alpha_ic_i(x^{*})\\
&\leqslant f(x^{*})
\end{aligned}
$$
并由 $\theta_p(x)$ 的构造中可以看出，对在可行域中的点 $x^*$，一定有
$\theta_p(x^{*})=f(x^*)$。

因此
$$\min_x L(x, \alpha,\beta) \leqslant 
\max_{\alpha, \beta, \alpha_i\geqslant0}L(x^{*}, \alpha, \beta)$$
也就是说
$$ \theta_d(\alpha, \beta)\leqslant
\theta_p(x^{*})$$
由于左边是关于参数的函数，右边是关于自变量的函数，从而我们可以对左边取极大，对右边取极小，得到
$$
\max_{\alpha, \beta, \alpha_i\geqslant0} \theta_d(\alpha,\beta)
\leqslant \min_x\theta_p(x^{*})
$$
即对偶问题的最优解比原问题的最优解小。如果我们把原始问题的最优结果记为 $p^{*}$，把对偶问题的最优结果
记为 $d^{*}$，那么我们就有
$$d^{*}\leqslant p^{*}$$
这个就叫做弱对偶条件。弱对偶条件在任何情况下都是成立的。

如果对某个优化问题成立 $d^*=p^*$，我们就说这个是强对偶条件。这里的支撑向量机问题就是一个强对偶问题。

---

## KKT条件
 
对于带不等式约束的优化问题，最优解除了满足原始约束之外，还需要满足如下条件：
$$\begin{cases}
\nabla_x L(x,\alpha,\beta)&=0 \\
h_j(x) &=0 \\
c_i(x) &\leqslant 0\\
\alpha_i & \geqslant 0\\
\alpha_ic_i(x)&= 0 \\
\end{cases}$$
这组条件称为 KKT 条件。其中：
 
- 第一条称为驻点条件（stationarity）
- 第二、三条称为原始可行性（primal feasibility）
- 第四条称为对偶可行性（dual feasibility）
- 最后一条称为互补松弛条件（complementary slackness）
在凸优化问题中，当满足适当条件时（例如 Slater 条件），原问题与对偶问题满足强对偶性，此时 KKT 条件也是最优解的充分必要条件。
 
---
 
## 强对偶与KKT条件的联系

我们来说明，强对偶成立时，KKT条件是如何自然冒出来的。
 
假设强对偶成立，即 $d^* = p^*$，设 $(x^*, \alpha^*,\beta^*)$ 分别是原始问题与对偶问题的最优解。考虑如下不等式链：
 
$$
\underbrace{\theta_d(\alpha^*,\beta^*)}_{=\,d^*}
=
\min_x L(x,\alpha^*,\beta^*)
\leqslant
L(x^*,\alpha^*,\beta^*)
\leqslant
\theta_p(x^*)
=
\underbrace{\min_x\theta_p(x)}_{=\,p^*}
$$

由于 $d^* = p^*$，这条链两端相等，中间所有的 $\leqslant$ 都必须取等。这两个等号各自给出一个条件。
 
**第一个等号**：$\min_x L(x,\alpha^*,\beta^*) = L(x^*,\alpha^*,\beta^*)$

这说明 $x^*$ 是 $L(\,\cdot\,, \alpha^*,\beta^*)$ 关于 $x$ 的极小值点，于是驻点条件成立：
$$\nabla_x L(x^*,\alpha^*,\beta^*) = 0$$
**第二个等号**：$L(x^*,\alpha^*,\beta^*) = f(x^*)$
展开就是
$$\sum_i \alpha_i^* c_i(x^*) = 0$$其中等式约束等于 $0$，因为 $x^*$ 是可行解.
 
由于每一项都满足 $\alpha_i^* \geqslant 0$ 且 $c_i(x^*) \leqslant 0$，每一项都 $\leqslant 0$，它们加起来等于零，因此每一项必须单独为零：
$$\alpha_i^* c_i(x^*) = 0$$

这就是互补松弛条件。其直觉是：要么约束不紧（$c_i(x^*)<0$），对目标函数没有影响，此时 $\alpha_i^*=0$；要么约束紧（$c_i(x^*)=0$），$\alpha_i^*$ 可以不为零。

加上原来就有的可行性条件 $c_i(x^*)\leqslant 0$，$\alpha_i^*\geqslant 0$，合在一起就是完整的KKT条件。
从而我们如果知道一个优化问题是强对偶的，那么他的最优解就一定满足$ \text{KKT条件}$

反过来，如果 $x^*$满足 $\text{KKT}$ 条件，我们想问它是否就是最优解呢？答案一般是否定的。只有对于凸优化问题而言，才能用满足 $\text{KKT}$ 
条件的点得到这个点就是最优解。

因此，**KKT条件是强对偶成立时最优解必须满足的条件**，它从不等式链取等这一事实中自然导出，
对于凸优化问题（如SVM），在Slater条件下强对偶成立，KKT条件也从必要条件升级为充要条件：
 
$$x^*\text{ 是最优解} \iff x^*\text{ 满足KKT条件}$$

---

## 对对偶问题的进一步理解
 
现在来说明，对偶问题的求解过程中，$\nabla_x L = 0$ 这一步究竟是什么意思。
 
**第一层含义：求对偶函数的计算工具**
 
$\theta_d(\alpha) = \min_{w,b} L(w,b,\alpha)$ 是对固定的 $\alpha$，把 $L$ 关于 $w,b$ 求极小。令 $\nabla_w L = 0$，$\frac{\partial L}{\partial b}=0$，只是求极小值的必要条件，属于普通的微积分操作，此时还没有用到KKT条件，也没有用到强对偶。解出 $w,b$ 与 $\alpha$ 的关系，代回 $L$，得到只含 $\alpha$ 的 $\theta_d(\alpha)$。
 
**第二层含义：还原原始最优解时的KKT驻点条件**
 
解对偶问题得到最优 $\alpha^*$ 后，我们需要从 $\alpha^*$ 还原出原始问题的 $w^*$。由于SVM满足强对偶，最优解满足KKT条件，驻点条件给出：
$$\nabla_w L(w^*,b^*,\alpha^*) = 0$$
 
这个式子与第一步求极小时的形式完全相同，因此第一步解出的关系式
$$w = \sum_{i=1}^N \alpha_i y_i x_i$$
代入最优 $\alpha^*$ 后，直接给出原始最优解 $w^*$。
 
两层含义形式相同，含义不同。能用同一个式子做两件事，正是强对偶成立的结果。

---

## 求解对偶问题

前面已经得到拉格朗日函数

$$
L(w, b, \alpha) = \frac{1}{2}\|w\|^2 + \sum_{i=1}^{N} \alpha_i - \sum_{i=1}^{N} \alpha_i y_i (w \cdot x_i + b)
$$

其中 $\alpha_i \geq 0$。这里的参数是 $\alpha$，自变量是 $w$ 和 $b$。

延续上文，我们定义

$$
\theta_d(\alpha) = \min_{w,b} L(w, b, \alpha)
$$
并利用对偶问题的解来求原问题的解。为了求 $\theta_d(\alpha0$ 的表达式，
令 $L$ 对 $w$ 和 $b$ 的偏导为零

$$
\frac{\partial L}{\partial w} = w - \sum_{i=1}^{N} \alpha_i y_i x_i = 0
$$

$$
\frac{\partial L}{\partial b} = - \sum_{i=1}^{N} \alpha_i y_i = 0
$$

于是得到

$$
w = \sum_{i=1}^{N} \alpha_i y_i x_i
$$

$$
\sum_{i=1}^{N} \alpha_i y_i = 0
$$

将这两个结果代回 $L$，第一项 $\frac{1}{2}\|w\|^2$ 变成

$$
\frac{1}{2} \left( \sum_{i=1}^{N} \alpha_i y_i x_i \right) \cdot \left( \sum_{j=1}^{N} \alpha_j y_j x_j \right) = \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j (x_i \cdot x_j)
$$

第三项 $\sum_{i=1}^{N} \alpha_i y_i (w \cdot x_i + b)$ 变成

$$
\sum_{i=1}^{N} \alpha_i y_i \left( \left( \sum_{j=1}^{N} \alpha_j y_j x_j \right) \cdot x_i + b \right) = \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j (x_j \cdot x_i) + b \sum_{i=1}^{N} \alpha_i y_i
$$

由 $\sum_{i=1}^{N} \alpha_i y_i = 0$ 知第二项为零，而第一项与前面的 $\frac{1}{2}\|w\|^2$ 合并，得到

$$
\theta_d(\alpha) = \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j (x_i \cdot x_j)
$$

于是对偶问题就是

$$
\max_{\alpha} \quad \theta_d(\alpha)
$$

约束条件为

$$
\sum_{i=1}^{N} \alpha_i y_i = 0, \quad \alpha_i \geq 0
$$

转化为标准优化问题就是

$$
\begin{aligned}
\min_\alpha\quad & \frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}
\alpha_i\alpha_j y_i y_j \left( x_i\cdot x_j\right) -\sum_{i=1}^{N} \alpha_i\\
\text{s.t.} \quad & \sum_{i=1}^{N} \alpha_i y_i =0  \\
& \alpha_i \geqslant 0
\end{aligned}
$$

这就是 SVM 的对偶形式。
