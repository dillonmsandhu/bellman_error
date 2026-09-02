**Summary:**
1. The operator is  $T^\lambda = v + (I-\gamma \lambda P)^{-1} [Tv-v]$ 
2. 3. Deep TD($\lambda$) minimizes the loss $L_{TD} = \frac{1}{2}\mathbb{E}_\mu(\text{sg}(T^\lambda v) - v)^2$
3. The fixed point of $\Pi T^\lambda$ is whenever $\nabla v \perp_\mu (I-\gamma \lambda P)^{-1} \delta$.

Simple Code:
```
def T(v):
	return R_π + γ * P_π @ v

I = np.eye(len(S))
L = np.linalg.inv(I - γ * λ * P_π)

def t_lambda(v):
	return v + L @ (T(v) - v)

def td_lambda_loss(params):
	v = network.apply(params, S)
	td_errors = v - stop_gradient(t_lambda(v))
	loss = 0.5 * np.sum(mu * (td_errors ** 2))
	return loss
```
### Main Result: the TD($\lambda$) update
Given estimator $v_\theta$ and TD error $\delta \doteq Tv_\theta-v_\theta$, the TD lambda update is:
$$\theta_t \gets \theta_{t-1} + \alpha (\nabla v_\theta)^\intercal D(I-\gamma\lambda P)^{-1}\delta$$
**Derivation**
Starting from [Tagorti and Scherrer (2014)](https://proceedings.mlr.press/v37/tagorti15.pdf), the linear TD($\lambda$) fixed point is:
$$
0 = \Phi^\intercal D[LR - (I-\gamma P) (I-\gamma \lambda P)^{-1} \Phi w]
$$
Let $L \doteq (I-\gamma \lambda P)^{-1}$.  To generalize beyond linear estimators, I make the following replacements: $\Phi w \rightarrow v_\theta$ and $\Phi^\intercal \rightarrow (\nabla v_\theta)^\intercal \in \mathcal{R}^{|\theta| \times |S|}$. and simplify:
$$\begin{align}
0 &=  (\nabla v_\theta)^\intercal D[LR + \gamma PL v -L v] 
\end{align}
$$
I'd like to factor out $L$. Some basic algebra shows this can be done.
$$
\begin{align}
PL&= P(I + \gamma \lambda P + \gamma^2 \lambda^2 P^2 + \dots) \\
&= (P + \gamma \lambda P^2 + \gamma^2 \lambda^2 P^3 + \dots) \\
&= (I + \gamma \lambda P + \gamma^2 \lambda^2 P^2 + \dots) P \\
&= LP
\end{align}
$$
The fixed point can then be simplified to:
$$
\begin{align}
0 &=  (\nabla v_\theta)^\intercal DL[Tv_\theta -v_\theta]
\end{align}
$$
To find a zero, copy TD learning and apply the incremental update rule:
$$\theta_t \gets \theta_{t-1} + \alpha (\nabla v_\theta)^\intercal DL[Tv_\theta -v_\theta]$$
To get a form more standard for deep learning, we can equivalently minimize a loss constructed from the $\lambda$-Bellman Operator, $T^\lambda v \doteq v + L(Tv-v)$. 
$$L_{TD} = \frac{1}{2}\mathbb{E}_\mu(\text{sg}(T^\lambda v_\theta) - v_\theta)^2$$
 Notice that $T^\lambda v_\theta - v_\theta = L \delta$, the exact quantity we would like to be orthogonal to $\nabla v_\theta$.  TD($\lambda$) resembles standard TD learning, except that $L$ is applied to the one-step TD error, resulting in a $\lambda$-decaying sum of multi-step TD errors.
# Appendix: Extra Things about TD($\lambda$)
### A. Deriving the TD-Lambda Operator
The original definition is the exponentially-weighted average of $n$-step Bellman Operators, $T^n$.
$$T^\lambda v = (1-\lambda)\sum_{n=1}^\infty \lambda^{n-1} (T^n v)$$
We can derive the earlier definition using the $n$-step TD errors:
$$T^n v = v + \sum_{k=0}^{n-1}(\gamma P)^k (Tv-v)$$Plugging these in:
$$T^\lambda v = (1-\lambda)\sum_{n=1}^\infty \lambda^{n-1} \left(v + \sum_{k=0}^{n-1}(\gamma P)^k (Tv-v) \right)$$
Distribute:
$$T^\lambda v = (1-\lambda)\sum_{n=1}^\infty \lambda^{n-1}v + (1-
\lambda)\sum_{n=1}^\infty \lambda^{n-1}\left(\sum_{k=0}^{n-1}(\gamma P)^k (Tv-v) \right)$$
Cancelling the sum over the geometric series:
$$T^\lambda v = v + (1-\lambda)\sum_{n=1}^\infty \lambda^{n-1}\left(\sum_{k=0}^{n-1}(\gamma P)^k (Tv-v) \right)$$
Now looking at the second term, and letting the error be $\delta$,
$$
\begin{align}
&(1-\lambda)[\delta + \lambda(\delta + \gamma P \delta) + \lambda^2(\delta + \gamma P \delta  +\gamma^2 P^2 \delta) + \dots]
\end{align}$$
Grouping by powers of $\gamma P$, we get:
$$\delta(1+\lambda + \lambda^2 + \dots) + \gamma P \delta(\lambda + \lambda^2 + \dots) + \gamma^2 P^2 \delta(\lambda^2 + \dots)$$
Plugging in $\frac{1}{1-\lambda}$ for the $1+\lambda + \lambda^2 + \dots$ gives us:
$$\frac{1}{1-\lambda}\left[\delta + \gamma \lambda P \delta + (\gamma \lambda P)^2 \delta +\dots\right]$$
Finally, we can plug back in to get 
$$T^\lambda v = v + \sum_{k=0}^\infty (\gamma \lambda P)^k \delta$$
The familiar expression of $v$ plus the infinite $\lambda$-decaying sum of TD errors.
### B. Fixed Point of Projected TD-Lambda Operator:
The fixed point is defined by:
$$\begin{align}
v &= \Pi T^\lambda v \\
&= \Pi (v + L \delta) \\
&= v + \Pi L \delta
\end{align}
$$
implying it is at the point where $L\delta$ is orthogonal to the representable space:
$$0 = \Pi L \delta$$
**Linear Fixed Point**
In the linear case, $\Pi = \Phi (\Phi^\intercal D \Phi)^{-1} \Phi^\intercal D$, giving:
$$0 = \Phi (\Phi^\intercal D \Phi)^{-1} \Phi^\intercal DL  [R + \gamma Pv - v]$$
Assuming full rank features, the leading $\Phi$ has a trivial null-space and can be removed.  Since $(\Phi^\intercal D \Phi)^{-1}$ is invertible, and we get:
$$0 = \Phi^\intercal L D [R + \gamma Pv - v]$$
which can be rearranged into the original expression from [Tagorti and Scherrer (2014)](https://proceedings.mlr.press/v37/tagorti15.pdf).

## C. Vector Form of Eligibility Traces
The Eligibility trace is:

$L\mathbf{1}$ 


$T^\lambda = v + (I-\gamma \lambda P)^{-1} [Tv-v]$ 