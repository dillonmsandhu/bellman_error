Recall the following value error from Tang and Munos 2023:
$$E = e^\intercal D(I-\gamma P) e$$
where $e = V-v_\theta$ is a length $N$ vector. Similar to Proposition 3.1 from [A Tutorial on Spectral Clustering](https://arxiv.org/pdf/0711.0189) we can rewrite it as follows:
$$\begin{align}
E &= (1-\gamma) \|e\|_D^2 + \frac{\gamma}{2} \sum_{ij}\mu_i p_{ij} (e_i - e_j)^2 \\
&= (1-\gamma) \|e\|_D^2 + \frac{\gamma}{2} \sum_{ij}\mu_i p_{ij} (r + \gamma v_\theta(s_j) - v_\theta(s_i) +(1-\gamma) e_j)^2
\end{align}
$$
Looking at the expansion, the first term is the standard Monte-Carlo loss, i.e. the $\mu$-weighted value error. The second term is a smoothness term that says that for the expected on-policy transition $s_i \rightarrow s_j$, the error at the two states must be similar.

As $\gamma \rightarrow 1$ it becomes of fitting $r = v_\theta(s) - v_\theta(s')$ only.
#### Proof 
Start with the standard expression for $E$ and expand it:
$$\begin{align}
E &= e^\intercal D(I-\gamma P) e\\
&= e^\intercal D e - \gamma e^\intercal DP e\\
&= (1-\gamma) e^\intercal D e + \gamma e^\intercal D e - \gamma e^\intercal DP 
e & \quad \text{(split first term)}\\
&= (1-\gamma) e^\intercal D e + \gamma e^\intercal D e - \gamma e^\intercal D P e \\
&= (1-\gamma) \|e\|_D^2 + \gamma (e^\intercal D e - e^\intercal DP e)\\
&= (1-\gamma) \|e\|_D^2 + \gamma (\sum_i\mu_i e_i^2 - 
\sum_{ij} \mu_i p_{ij} e_i e_j  )\\
\end{align}
$$
Next, simplify the $\gamma$-weighted term in parentheses using the fact that $\mu_j$ is the stationary distribution:
$$
\begin{align}
\sum_i\mu_i e_i^2 - 
\sum_{ij} \mu_i p_{ij} e_i e_j &= \frac{1}{2} \left[ \sum_i\mu_i e_i^2 - 
2\sum_{ij} \mu_i p_{ij} e_i e_j +  \sum_j\mu_j e_j^2\right]
\end{align}
$$

Since $\sum_j p_{ij} = 1$, we have $\sum_i\mu_i e_i^2 = \sum_{ij} \mu_i p_{ij} e_i^2$.  

Since $\mu$ is the stationary distribution, $\mu^\intercal P = \mu^\intercal$, each entry $\mu_j$ satisfies $\mu_j = \sum_i \mu_i p_{ij}$. This implies $\sum_j\mu_j e_j^2 = \sum_{ij} \mu_i p_{ij} e_j^2$.

Plugging these two facts in, we get a sum over $i,j$:
$$
\begin{align}
\sum_i\mu_i e_i^2 - 
\sum_{ij} \mu_i p_{ij} e_i e_j &= \frac{1}{2} \sum_{ij}(\mu_i p_{ij} e_i^2 - 
2\mu_i p_{ij} e_i e_j + \mu_i p_{ij} e_j^2)\\
&= \frac{1}{2} \sum_{ij}\mu_i p_{ij} (e_i^2 - 2 e_i e_j +  e_j^2) \\
&= \frac{1}{2} \sum_{ij}\mu_i p_{ij} (e_i - e_j)^2
\end{align}
$$
Plugging this back into the full expression:

$$\begin{align}
E &= (1-\gamma) \|e\|_D^2 + \gamma (\sum_i\mu_i e_i^2 - 
\sum_{ij} \mu_i p_{ij} e_i e_j  )\\
 &= (1-\gamma) \|e\|_D^2 + \frac{\gamma}{2} \sum_{ij}\mu_i p_{ij} (e_i - e_j)^2 \\
\end{align}
$$
End of Proof

## Sampling
Can we estimate $E$ with samples?

Suppose we have a finite horizon task, and we collect a full trajectory on-policy: $\{s_t, a_t, r_t\}_{t=1}^T$. The MC return $G_t$ can be constructed for each $s_t$. Since the states come from $\mu$, an unbiased estimate of $\|e\|_D^2$ is $\hat{e} = \sum_{t}(G_t - v_\theta(s_t))^2$.

Now look at the second term $\sum_{ij} \mu_i p_{ij} (e_i - e_j)^2$, and recall that $p_{ij} = P(s_j | s_i)$. Since $s_{t+1} \sim P(\cdot | s_{t}$ ), then if $s_t$ is from the stationary distribution, we can construct estimators for $e_t$ and $e_{t+1}$, and plug them in. 

In summary, standard on-policy Monte-Carlo estimates $E$ as the sum of terms of the form:
$$
(1-\gamma) \sum_t(G_t - v_\theta(s_t))^2 + \frac{\gamma}{2} \sum_t(G_t - v_\theta(s_t) - G_{t+1} + v_\theta(s_{t+1}) )^2
$$
## Interpretation
So one way to understand the second term is that it is minimizing the difference in successive errors. I provide a different "one-step" interpretation below. 

When constructed from the a trajectory in this way, we can do a bit of simplification using the recurrence in $G_t$. Letting $v_t$ be shorthand for $v_\theta(s_t)$, we have:
$$
\begin{align}
e_t - e_{t'} &= G_{t} - v_i - G_{t'} + v_{t'}\\
&= r_{t} + \gamma G_{t'} - v_t - G_{t'} + v_{t'}\\
&= r_{t} + \gamma G_{t'} - v_t - G_{t'} + \gamma v_{t'} + (1-\gamma)v_{t'} \\
&= [r_{t} + \gamma v_{t'} - v_t]  + (1-\gamma)(v_{t'}-G_{t'})
\end{align}$$
The term in brackets is the standard bellman error. The second term is the monte-carlo error. Since the second term is of the form of an expection of a square, we can add up individual samples, square them, and divide by the number of samples to estimate it. There's no double sampling issue, because $p_{ij}$ is outside the square.
$$\sum_{ij}\mu_i p_{ij} (e_i - e_j)^2$$
### Final Combined Form:


#laplacian
