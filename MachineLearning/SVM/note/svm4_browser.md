# 关于学习率和收敛性
考虑这样一个数据集， 
$X=\{ \left(x_{i,1},x_{i,2}\right) \}$ 和
$ y = \{y_i:y_i^2=1\}$
我们定义 $$\|A\| = \sqrt{\sum_{i=1}^{M}\sum_{j=1}^{N} a_{ij}^2 }$$
表示 $\text{Frobenius}$ 范数。一个简单的结论是
$$ \|A\cdot B \|\leqslant \|A\|\\|B\|$$
它可以用柯西不等式证明。由上面的过程，我们分别计算 $G_x $ 和 $G_y$，利用不等式可以得到

$$ \|G_x\| = \|XX^T\|\leqslant \|X\|\|X^T\|=\|X\|^2$$
类似地,$$ \|G_y\| \leqslant \|y\|^2 = N$$
设  $$ \|X\|\leqslant M$$
因为矩阵 $G$ 是 $G_x$ 与 $G_y$ 的 $\text{Hadamard}$ 积，并且
$G_y$ 中的的元素均为 $±1$，从而
$$ \|G\| = \|G_x\| \leqslant M $$

## 投影梯度法的收敛性与稳定点严格证明

为了严格分析算法的收敛性，我们首先将一轮迭代中的三个递推步骤统一写成一个复合算子 $T: \mathbb{R}^N \to \mathbb{R}^N$。

对于空间中任意一个初始参数 $\alpha_{k-1}$，算法的单步更新步骤为：

1.  **第一步（梯度下降步 $F$）**：
    $$\beta_k = F(\alpha_{k-1}) = \alpha_{k-1} - \eta (G \alpha_{k-1} - \mathbf{1})$$
    其中可以令$$F(x) = (I - \eta G)x + \eta \mathbf{1} $$
其中这里的 $G$ 是半正定矩阵
2. **第二步（等式投影步 $P_H$）**：
    $$\gamma_k = P_H(\beta_k) = \beta_k - \frac{1}{N} y y^T \beta_k$$
3.  **第三步（非负截断步 $P_+$）**：
    $$\alpha_k = P_+(\gamma_k) = \max\{\gamma_k, \; 0\}$$

整个复合迭代算子表示为：$\alpha_k = T(\alpha_{k-1}) = (P_+ \circ P_H \circ F)(\alpha_{k-1})$。


---

### 第一部分：全局非扩张性证明（Non-expansiveness）

我们首先证明，对于空间中任意两个向量 $x, y \in \mathbb{R}^N$，算子 $T$ 满足非扩张性，即
$\|T(x) - T(y)\| \leqslant \|x - y\|$。在这个证明过程中，我们不依赖任何关于稳定点或收敛性的假设。

#### 1. 梯度步的距离变化（引用引理一）

我们首先引入关于实对称矩阵对角化导出的距离放缩引理：


>设 $G \in \mathbb{R}^{N \times N}$ 是实对称半正定矩阵，其特征值为  
$\lambda_1, \lambda_2, \dots, \lambda_N \geqslant 0$。定义线性变换$$
F(x) = (I - \eta G)x$$
对于任意两个向量 $x, y \in \mathbb{R}^N$，若 $\eta$ 满足：
$$
\eta \leqslant \frac{2}{\lambda_{\max}(G)}$$
则该线性变换满足：$$
\|F(x) - F(y)\| \leqslant \|x - y\|$$

根据前文推导，已知最大特征值受控于数据的边界常数：$\lambda_{\max}(G) \leqslant \|G\|
\leqslant M^2$。因此，我们选择学习率满足：
$$\boxed{ \eta \leqslant \frac{2}{M^2} }$$
这严格保证了梯度步的非扩张性成立：
$$\boxed{ \|F(x) - F(y)\|^2 \leqslant \|x - y\|^2 } \quad \dots \text{（式 A）}$$

#### 2. 等式投影步的距离变化

令 $v_x = F(x)$，$v_y = F(y)$。将投影后的向量相减：
$$P_H(v_x) - P_H(v_y) = (v_x - v_y) - \frac{1}{N} y y^T (v_x - v_y)$$
为了代数书写简便，令向量 $w = v_x - v_y$，上式化为：
$$P_H(v_x) - P_H(v_y) = w - \frac{y^T w}{N} y$$
计算其模长平方（展开二次型）：
$$
\begin{aligned}
\|P_H(v_x) - P_H(v_y)\|^2 &= \left( w - \frac{y^T w}{N} y \right)^T \left( w - \frac{y^T w}{N} y \right) \\
&= w^T w - \frac{2(y^T w)}{N} (y^T y)^T w + \frac{(y^T w)^2}{N^2} (y^T y)
\end{aligned}
$$
因为 $y^T y = N$，最后一项的分母消去一个 $N$，代数式简化为：
$$= \|w\|^2 - \frac{2}{N} (y^T w)^2 + \frac{1}{N} (y^T w)^2 = \|w\|^2 - \frac{(y^T w)^2}{N}$$
由于 $\frac{(y^T w)^2}{N} \geqslant 0$，代回 $w$ 得到：
$$\boxed{ \|P_H(v_x) - P_H(v_y)\|^2 \leqslant \|v_x - v_y\|^2 } \quad \dots \text{（式 B）}$$

#### 3. 非负截断步的分量对比

令 $u_x = P_H(v_x)$，$u_y = P_H(v_y)$。对比截断后各分量的平方差：
$$(P_+(u_x)_i - P_+(u_y)_i)^2 = (\max\{u_{x,i}, 0\} - \max\{u_{y,i}, 0\})^2$$
我们利用一维实数轴上的距离关系，验证不等式 $(\max\{a, 0\} - \max\{b, 0\})^2 \leqslant (a - b)^2$ 的正确性：

- **情况一**：若 $a \geqslant 0, b \geqslant 0$，不等式取等号。
- **情况二**：若 $a < 0, b < 0$，左边为 $0$，右边为 $(a-b)^2 \geqslant 0$，不等式显然成立。
- **情况三**：若 $a \geqslant 0, b < 0$，左边为 $a^2$。而右边为 $(a-b)^2 = (a+|b|)^2 > a^2$，不等式严格成立。

将所有分量相加，可得向量模长不等式：
$$\boxed{ \|P_+(u_x) - P_+(u_y)\|^2 \leqslant \|u_x - u_y\|^2 } \quad \dots \text{（式 C）}$$

#### 4. 复合算子的全局非扩张结论

将式 A、式 B、式 C 顺次相连，我们证明了对空间中任意两个向量 $x, y \in \mathbb{R}^N$：
$$\|T(x) - T(y)\|^2 \leqslant \|P_H(F(x)) - P_H(F(y))\|^2 \leqslant \|F(x) - F(y)\|^2 \leqslant \|x - y\|^2$$
由此，我们严格地证明了复合迭代算子 $T$ 满足全局非扩张性：
$$\boxed{ \|T(x) - T(y)\| \leqslant \|x - y\| \quad (\forall x, y \in \mathbb{R}^N) }$$

---

### 第二部分：稳定点（不动点） $\alpha^*$ 的存在性定理

在第一部分完全不依赖任何稳定点假设的前提下，我们已经确立了算子 $T$ 的全局非扩张性。现在，我们引入非线性分析中的经典不动点定理：

> **定理一：Browder 不动点定理**
>
> 设 $C \subset \mathbb{R}^N$ 是一个闭凸集，且映射 $T: C \to C$ 满足全局非扩张性。若迭代序列 $\{T^k(x_0)\}$ 有界，则算子 $T$ 在 $C$ 上必定存在至少一个不动点 $\alpha^*$，使得：
> $$\alpha^* = T(\alpha^*)$$

但是，很可惜，我们的映射 $T$ 不满足这个性质。也就是说我们没法使用这个定理。我们只能证明存在稳定点，不能证明收敛性。


因此，由定理一，算法稳定点 $\alpha^*$ 的存在性在数学上被严格确立。它满足：

- $\beta^* = \alpha^* - \eta (G \alpha^* - \mathbf{1})$
- $\gamma^* = \beta^* - \frac{1}{N} y y^T \beta^*$
- $\alpha^* = \max\{\gamma^*, \; 0\}$

---

### 第三部分：算法收敛性的严格证明(存疑)

既然稳定点 $\alpha^*$ 的存在性已仍然无法得知，
我们现在只能假设存在稳定点。实际上，利用 $\text{Bolzano-Weierstrass}$定理，
我们知道必然存在收敛点列，尽管不一定是最优解。d
我们现在来证明迭代序列 $\{\alpha_k\}$ 必定收敛于该点。

在第一部分已证的全局非扩张不等式中：
$$\|T(x) - T(y)\| \leqslant \|x - y\| \quad (\forall x, y \in \mathbb{R}^N)$$

因为这一不等式对空间中任意两个向量都成立，我们现在将 $x$ 特化为第 $k-1$ 步的迭代点 $\alpha_{k-1}$，将 $y$ 特化为已证实存在的稳定点 $\alpha^*$。代入上式得到：
$$\|T(\alpha_{k-1}) - T(\alpha^*)\| \leqslant \|\alpha_{k-1} - \alpha^*\|$$

根据定义，有 $\alpha_k = T(\alpha_{k-1})$，且 $T(\alpha^*) = \alpha^*$。代入上式直接得到：
$$\boxed{ \|\alpha_k - \alpha^*\| \leqslant \|\alpha_{k-1} - \alpha^*\| }$$

**结论：**

1. 几何距离序列 $d_k = \|\alpha_k - \alpha^*\|$ 是一个**单调递减且有下界（$\geqslant 0$）**的实数序列。根据实数域的单调有界原理，距离序列 $d_k$ 必定收敛。
2. 结合算子的平均性质，这严格证明了算法序列 $\{\alpha_k\}$ 必定收敛于稳定点 $\alpha^*$。 $\blacksquare$