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

关于求解此类问题，我们引入被称为原问题的对偶问题，同时引入 $\text{KKT}$ 条件。
对一般的凸二次优化问题，考虑对优化问题
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

定义 $$\theta_p(x) = \max_{\alpha,\beta}
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
\max_{\alpha,\beta}L(x,\alpha,\beta)
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

对于带不等式约束的优化问题，最优解除了满足原始约束之外，还需要满足如下条件：
$$\begin{cases}
\nabla_x L(x,\alpha,\beta)&=0 \\
h_j(x) &=0 \\
c_i(x) &\leqslant 0\\
\alpha_i & \geqslant 0\\
\alpha_ic_i(x)&= 0 \\
\end{cases}$$
这组条件称为 $\text{KKT}$ 条件。

其中：

第一条称为驻点条件（stationarity）
第二、三条称为原始可行性（primal feasibility）
第四条称为对偶可行性（dual feasibility）
最后一条称为互补松弛条件（complementary slackness）

在凸优化问题中，当满足适当条件时（例如 Slater 条件），
原问题与对偶问题满足强对偶性，此时 KKT 条件也是最优解的充分必要条件

上面我们对拉格朗日函数先对参数求极大值，再求极小，也就是
$$\min_x\max_{\alpha,\beta}L(x,\alpha,\beta)$$

---

那我们也可以先求极小，再求极大。


定义$$\theta_d(\alpha, \beta) = \min_{x}
L(x, \alpha, \beta)$$
从而考虑某个可行解 $x^{*}$，我们有
$$\min_x L(x, \alpha, \beta) 
\leqslant L(x^{*}, \alpha, \beta)
\leqslant f(x^{*})$$
左边成立是显然的，右边取等也是显然的。
因此
$$\theta_d(\alpha,\beta)\leqslant f(x^{*})$$
由于左边是关于参数的函数，右边是关于自变量的函数，从而我们可以对左边取极大，对右边取极小，得到
$$
\max_{\alpha, \beta} \theta_d(\alpha, \beta) \leqslant \min_{x} f(x)$$
即对偶问题的最优解比原问题的最优解小。而在凸二次优化问题中，对偶问题的解和原问题的解是相等的。
于是我们只需要求对偶问题的最优解就可以了。
---


由于
$$
L(x,\alpha, \beta) = f(x) +
 \sum_i \alpha_i c_i(x) +
 \sum_j \beta_j h_j(x)
$$

我们求解对偶问题，**先对 $x$ 求极小**。令

$$
\nabla_x L(x, \alpha, \beta) = 0
$$

由此解出 $x$ 与 $(\alpha, \beta)$ 的关系，代入 $L$ 得到对偶函数 $\theta_d(\alpha, \beta)$。

然后求解对偶问题：

$$
\max_{\alpha \geq 0, \beta} \theta_d(\alpha, \beta)
$$

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

为了求这个极小值，令 $L$ 对 $w$ 和 $b$ 的偏导为零

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

这就是 SVM 的对偶形式。
