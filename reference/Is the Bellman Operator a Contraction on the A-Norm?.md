Let $\mu$ be the stationary distribution, $D = \text{diag}(\mu)$, and $A= D(I-\gamma P)$. I want to know if $T$ is a contraction in $A$-weighted norm. Spoiler: it isn't...

Letting $\langle \cdot \rangle$ denote the Euclidean norm. $T$ is a contraction in $A$-weighted norm if for a constant $c \in (0,1)$, we have
$$\langle A T (v-u), T(v-u) \rangle \leq c \langle A (v-u), v-u \rangle.$$
Letting $x = v-u$, we can rewrite as: 
$$\gamma\langle A  Px, Px \rangle \leq c \langle Ax, x \rangle$$
Or:
$$\gamma|Px|_A - c|x|_A \leq 0$$
Recall the earlier expansion of the $A$-norm:
$$
\begin{align}
|x|_A &= \langle A x, x \rangle = \langle D(I-\gamma P) x, x \rangle \\
&= \langle Dx,x \rangle - \gamma \langle DPx,x\rangle \\
&= |x|_D - \gamma  (DPx)^\intercal x \\
&= |x|_D - \gamma \sum_{ij} \mu_i p_{ij} x_i x_j \\
&= (1-\gamma)|x|_D^2 + \gamma |x|_D^2 - \gamma \sum_{ij} \mu_i p_{ij} x_i x_j \\ &= (1-\gamma)|x|_D^2 + \gamma \left( \sum_i \mu_i x_i^2 - \sum_{ij} \mu_i p_{ij} x_i x_j \right)
\\ &= (1-\gamma)|x|_D^2 + \frac{\gamma}{2} \sum_{ij} \mu_i p_{ij}(x_i-x_j)^2
\end{align}
$$
Plugging our terms in and combining gives
$$\begin{align} 
\gamma|Px|_A - c|x|_A &=  (1-\gamma) \left(\gamma |Px|_D^2 - c|x|_D^2  \right) \\ &+ \frac{\gamma}{2} \sum_{ij} \mu_i p_{ij} \left( \gamma (Px_i-Px_j)^2 - c(x_i-x_j)^2\right) \end{align}$$
In general, the first term is negative for $c > \gamma$, but not the second term.

### What about The Lambda Operator?
$T^\lambda = v+ L_\lambda (Tv-v)$ where $L_\lambda = (I-\gamma \lambda P)^{-1}$. I drop the subscript from $L$ below.
$$\begin{align}T^\lambda u - T^\lambda v &= u +  L (Tu-u) - v -  L (Tv-v) \\&= u-v +  L T (u-v) + L(v-u) \\
T^\lambda x&= x + L Tx - Lx \\
&= x  - L (I-\gamma P )x \\
&= [I-L(I-\gamma P)]x\\
&= [LL^{-1}-L(I-\gamma P)]x\\
&= L[(I-\gamma \lambda P)-(I-\gamma P)]x\\
&= L[\gamma \lambda P+\gamma P]x\\
&= \gamma(1-\lambda)LPx
\end{align}$$
We'd like to look at:
$$|T^\lambda x|_A - c|x|_A \leq 0$$
$$|T^\lambda x|_A  = \gamma (1-\lambda) \langle A LPx, LPx \rangle$$
$\langle Ax, x \rangle = (Ax)^\intercal x = X^\intercal A x$
