# 软间隔

下面我们处理优化问题
$$
\begin{aligned}
\min_{w,b,\xi}\quad
                 & \frac12\|w\|^2 + C\sum_{i=1}^N \xi_i \\ 
\text{s.t.}\quad & 1-\xi_i-y_i(w \cdot x_i+b)\le0 \\
                 & -\xi_i\le0
\end{aligned}
$$

首先令
$$
\begin{aligned}
L(w, b, \xi, \alpha, \mu) = 
\frac12\|w\|^2+C\sum_{i=1}^{N}\xi_i &+ 
\sum_{i=1}^{N} \alpha_i\left(1-\xi_i-y_i(w\cdot x_i+b)\right) \\
&+\sum_{i=1}^{N}\mu_i(-\xi_i)
\end{aligned}
$$
其中 $\alpha_i \geqslant 0$，$\mu_i \geqslant 0$。

利用 KKT 条件，我们有驻点条件
$$
\begin{aligned}
\nabla_w L &= w - \sum_{i=1}^{N}\alpha_i y_i x_i = 0 \\
\nabla_b L &= -\sum_{i=1}^{N}\alpha_i y_i = 0 \\
\frac{\partial L}{\partial \xi_i} &= C - \alpha_i - \mu_i = 0
\end{aligned}
$$

整理得到
$$
\begin{aligned}
w &= \sum_{i=1}^{N}\alpha_i y_i x_i \\
\sum_{i=1}^{N}\alpha_i y_i &= 0 \\
\mu_i &= C - \alpha_i
\end{aligned}
$$

结合互补松弛条件
$$
\begin{aligned}
\alpha_i &\geqslant 0 \\
\mu_i &\geqslant 0 \\
\alpha_i(1-\xi_i-y_i(w\cdot x_i + b)) &= 0 \\
\mu_i(-\xi_i) &= 0
\end{aligned}
$$

由 $\mu_i = C - \alpha_i$ 且 $\mu_i \geqslant 0$，立即得到 $0 \leqslant \alpha_i \leqslant C$。

将 $w = \sum\alpha_i y_i x_i$ 和 $\sum\alpha_i y_i = 0$ 代回拉格朗日函数，所有含 $\xi_i$ 的项都消掉了，最终得到对偶问题
$$
\boxed{
\begin{aligned}
\min_{\alpha}\quad & \frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i\alpha_j y_i y_j(x_i\cdot x_j) - \sum_{i=1}^{N}\alpha_i \\
\text{s.t.}\quad & \sum_{i=1}^{N}\alpha_i y_i = 0 \\
& 0 \leqslant \alpha_i \leqslant C
\end{aligned}}
$$

可以看到，和硬间隔相比，唯一的区别就是不等式约束从 $\alpha_i \geqslant 0$ 变成了 $0 \leqslant \alpha_i \leqslant C$。换句话说，我们给每个对偶变量加上了一个上界。松弛变量 $\xi_i$ 在推导中完全消去了，它在最终的对偶问题里不留下任何痕迹——这就是所谓的“形有实无”。

这样一来，之前写的投影梯度法代码只需要改一个地方：把对 $\alpha$ 的非负截断
$$
\alpha_i = \max\{\alpha_i,\,0\}
$$
换成带上下界的截断
$$
\alpha_i = \max\{0,\,\min\{\alpha_i,\,C\}\}
$$

另外还需要注意，这里我们对 $\alpha$ 还有一个上界的限制，因此对
$b$ 这部分的代码也需要稍微修改。

结合两点就可以完成了。