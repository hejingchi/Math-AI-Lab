
## 引理一
设 $G \in \mathbb{R}^{N \times N}$ 是实对称半正定矩阵，特征值 $\lambda_1, \dots, \lambda_N \ge 0$，记 $\lambda_{\max} = \max_i \lambda_i$。定义线性变换  

 $$
F(x) = (I - \eta G)x,
 $$

其中步长 $\eta \ge 0$。则对任意 $x, y \in \mathbb{R}^N$ 有  

 $$
\|F(x) - F(y)\| \le \|x - y\|
 $$

**当且仅当**  

 $$
\eta \le \frac{2}{\lambda_{\max}} \quad (\text{当 } \lambda_{\max}=0 \text{ 时理解为无上界，即任意 } \eta \ge 0 \text{ 均成立}).
 $$

**证明**  
令 $v = x - y$，由线性性得 $F(x)-F(y) = (I-\eta G)v$。要证的不等式等价于  

 $$
\|(I-\eta G)v\|^2 \le \|v\|^2, \quad \forall v \in \mathbb{R}^N. \tag{1}
 $$

因为 $G$ 实对称，存在正交矩阵 $P$ 使  

 $$
G = P^{\mathsf T} D P, \quad D = \operatorname{diag}(\lambda_1, \dots, \lambda_N).
 $$

于是  

 $$
I - \eta G = P^{\mathsf T}(I - \eta D)P.
 $$

作正交变换 $u = Pv$，则 $\|u\| = \|v\|$，且  

 $$
\|(I-\eta G)v\|^2 = \|P^{\mathsf T}(I-\eta D)u\|^2 = \|(I-\eta D)u\|^2 = \sum_{i=1}^N (1 - \eta\lambda_i)^2 u_i^2. \tag{2}
 $$

**充分性**：设 $0 \le \eta \le 2/\lambda_{\max}$（当 $\lambda_{\max}=0$ 时 $\eta$ 任意非负）。  
对每个 $i$，由 $\lambda_i \in [0, \lambda_{\max}]$ 得 $0 \le \eta\lambda_i \le 2$，从而 $|1 - \eta\lambda_i| \le 1$，$(1 - \eta\lambda_i)^2 \le 1$。代入 (2) 得  

 $$
\|(I-\eta G)v\|^2 \le \sum_{i=1}^N u_i^2 = \|u\|^2 = \|v\|^2,
 $$

即 (1) 成立，收缩性得证。

**必要性**：设 (1) 对一切 $v \in \mathbb{R}^N$ 成立。因为 $u = Pv$ 遍历 $\mathbb{R}^N$，(1) 与 (2) 给出  

 $$
\sum_{i=1}^N (1 - \eta\lambda_i)^2 u_i^2 \le \sum_{i=1}^N u_i^2, \quad \forall u \in \mathbb{R}^N.
 $$

取标准基向量 $u = e_j$（第 $j$ 个分量为 1，其余为 0），得  

 $$
(1 - \eta\lambda_j)^2 \le 1, \quad j = 1,\dots,N.
 $$

故 $|1 - \eta\lambda_j| \le 1$，即 $-1 \le 1 - \eta\lambda_j \le 1$。移项得 $0 \le \eta\lambda_j \le 2$。  
因为对所有 $j$ 成立，特别地对最大特征值 $\lambda_{\max}$ 有 $\eta\lambda_{\max} \le 2$。  
当 $\lambda_{\max} > 0$ 时，即得 $\eta \le 2/\lambda_{\max}$；当 $\lambda_{\max} = 0$ 时，条件 $0 \le \eta\lambda_j \le 2$ 自动满足，对 $\eta \ge 0$ 无额外限制。必要性得证。

综上，引理成立。$\blacksquare$

---
注1：原问题是对目标函数 $$f(\alpha) = \frac{1}{2} \alpha^T G \alpha -\alpha$$ 
在等式约束和不等式约束下求解极小值。如果我们计算梯度 $$\nabla f(\alpha) = G\alpha - \mathbf{1}$$
就能得到利用梯度下降法得到的递推式 $$ \begin{aligned}
\alpha_{k+1} &= \alpha_k - \eta \nabla f(\alpha) \\
             &= (I - \eta G) \alpha_k  + \eta
\end{aligned}$$
是一个仿射变换，即 $$\alpha_{k+1} = F(\alpha_k)$$其中 $$F(\alpha) = (I-\eta G) \alpha +\eta$$

结合 $$ \|F(x) - F(y)\| = \|(I-\eta G)(x-y)\|$$我们利用上面的引理可以得到可以选择合适的 $\eta$ 使得
$\|F(x)-F(y)\| \leqslant \|x-y\|$。这能保证 $\ \{ \alpha_k \}$ 是有界的。
尽管其收敛性只有在 $F$ 是线性变换才能利用 $\text{Browder不动点定理}$ 给出一个合理的解释，但是由于实际模拟中 $\eta$ 的取值往往
非常小，所以也能得到相对好的精度。

---

注2：由于特征值比较难算，所以我们可以利用范数不等式 $\|A\cdot B \| \leqslant \|A\|\|B\|$ ，考虑
实对称半正定矩阵$A$ 的特征向量 $x$ 以及特征值 $\lambda \geqslant 0$，从而有
$$
\begin{aligned}
\|Ax\| &= \|\lambda x\| = \lambda \|x\| \\
 &\leqslant \|A\|\|x\|
\end{aligned}
$$
因此  $$  \lambda  \leqslant \|A\|$$
从而当
$$0 \leqslant \eta \leqslant \frac{2}{\|G\|} \leqslant  \frac{2}{\lambda_{\max}}$$
时，我们也能得到 $$
\|F(x) - F(y)\| \le \|x - y\|
 $$的结论。这在机器学习和感知机里面是一个相对重要的结果，在一些学习率的收敛性估计中需要用到。
