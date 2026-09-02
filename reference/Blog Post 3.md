### Investigating Feature Learning on an Example MDP

In my last, post, I identified a condition called *alignment*, which guarantees that TD learning finds features that improve the fit of the value function. In brief, let $V \in \mathbb{R}^N$ be the true value function (there are $N$ states), $v_t \in \mathbb{R}^N$ be our estimate at iteration $t$. Denote the error vector by $e_t = V-v_t$. Finally, define the *key matrix* $A \in \mathcal{R}^{N \times N}$, as 
$$
A = D(I-\gamma P )
$$
where $D$ is a diagonal matrix with our state distribution along the diagonal and $P$ is an $N \times N$ state-transition matrix associated with the current policy.

Following Tang and Munos (2023), our weighted squared-value error $E$, can be written:
$$E(v) = \frac{1}{2} e_t^T A e_t$$
If we have a linear value estimator $v_t = \Phi_t w$, for $N \times k$ feature matrix $\Phi$ and length $k$ weight vector $w$, then the change in value error (holding the weights fixed), is given by the partial derivative $\partial_\phi E$. I derived the following result in my previous post:
$$
\begin{aligned} \partial_{\Phi} E &= \frac{1}{2} (A + A^\top) e_t w^T\end{aligned}
$$
Any square matrix can be decomposed into symmetric and asymmetric components, $S$ and $K$:

$$
\begin{align}
A &= S + K \\
S &= \frac{1}{2}(A + A^\top) \\
K &= \frac{1}{2}(A - A^\top) \\
\end{align}
$$
This emphasizes that
$$
\begin{aligned} \partial_{\Phi} E &= S e_t w^T\end{aligned}
$$
The main result is that the value error will decrease (i.e. $\dot{E} < 0$) if and only if the error weights a relatively symmetric part of $SA$:

$$

\boxed{

\begin{align}

\dot{E}(\Phi_t) < 0 \iff e_t^\top (A + A^\top ) A e_t > 0 \\

\end{align}

}

$$
This emphasizes that if $S A$ is positive definite, TD learning will improve the features, decreasing the weighted value error for all values $e_t$. If $SA$ is not positive definite, then TD learning can still lower the value error if the following inequality holds:
$$
\boxed{
\begin{aligned}
e_t^\top (S^2 + \frac{1}{2}(SK - K S)) e_t> 0
\end{aligned}
}$$
Which is obtained by rewriting $e_t^\top (A + A^\top ) A e_t > 0$.


Experiments TODO:
1. Actual Expected TD update
2. Track $\dot{E}$
3. Evaluate a fixed policy STARTING from the same value net (instead of from 0). Because E going up and then stagnating is odd, even when alighment. 

Ron's feedback:
- Redo derivation in terms of $\Phi$ and $\Phi'$.
	- think about for a fixed $\Phi_t$ remove skew-symmetric part of the update.
	- $\nabla v(s)$ and $\nabla v(s')$.
- what we get if we did LSTD with $\nabla v$ as $\phi$. 
- 
