Linear TD doesn't minimize value error at every step - it has a bias, which I show below. Then, I consider whether we can remove the bias, which is due to an extra skew-symmetric component. First, recall some definitions:
$$A \doteq D(I-\gamma P)$$
$$e \doteq \Phi w - V$$
$$E \doteq e^\intercal A e$$
The TD update can be written:
$$\dot{w} = -\alpha \Phi^\intercal A e$$
We'd like to know if this is aligned with  $-\nabla_w E$. If they point in the same direction, then the TD update will decrease the value error (alignment).
$$-\nabla_w E = -\Phi^\intercal (A + A^\intercal ) e = -\Phi^\intercal A e - \Phi^\intercal A^\intercal e$$
Notice the first term is exactly $\frac{1}{\alpha} \dot w$.  The only difference between $\dot{w}$ and a gradient descent update due to  $-\nabla_w E$ is the second term. 

Now I examine $\langle \dot{w}, -\nabla_w E \rangle$. Since they are both length $k$ vectors, it's pretty straightforward. Rewriting $-\nabla_w E$ one more time using $S \doteq \frac{1}{2}(A + A^\intercal)$ gives.
$$-\nabla_w E = -2\Phi^\intercal S e$$
Plugging this and the earlier expression for $\dot{w}$ in gives the following. The third line expands $A^\intercal = S - K$.
$$
\begin{align} 
\langle \dot{w}, -\nabla_w E \rangle &= (-\alpha \Phi^\intercal A e)^\intercal (-2 \Phi^\intercal S e) \\ 
&= 2 \alpha e^\intercal A^\intercal \Phi \Phi^\intercal S e \\
&= 2 \alpha e^\intercal (S - K) \Phi \Phi^\intercal S e \\
&= 2 \alpha e^\intercal S \Phi \Phi^\intercal S e - 2 \alpha e^\intercal K \Phi \Phi^\intercal S e \\
&= 2 \alpha \|\Phi^\intercal S e\|_2^2 - 2 \alpha e^\intercal K \Phi \Phi^\intercal S e
\end{align}
$$
Since $\|\nabla_w E\|_2^2 = 4\|\Phi^\intercal S e\|_2^2$, we can rewrite this in terms of that magnitude:
$$
\begin{align} 
\langle \dot{w}, -\nabla_w E \rangle &= \frac{\alpha}{2} \|\nabla_wE \|_2^2 - 2 \alpha e^\intercal K \Phi \Phi^\intercal S e 
\end{align}
$$
The second term is the misalignment. It can't be directly estimated with only single-step sampled information: the value error $e$ requires knowing $V$. The $K$ matrix requires knowing $P$. However, if there are some weights $w^*$ such that $V \approx \Phi w^*$, then we can estimate a feature-space version of $K$.
### Encouraging MDP Symmetry in Feature Space with an Auxiliary Loss

What if the *MDP* *were* symmetric after matching states to features? Would this bias go away? I find that it would, if the value function is linear in $\Phi$. This suggests an auxiliary task based on the symmetry of a transition matrix in feature space.

Define the LSTD matrix $A_\Phi = \Phi^\intercal A \Phi = S_\Phi + K_\Phi$.  We might wonder, what about regularizing the features such that $K_\Phi$ becomes small? I now show that this could be done by ensuring $P_\Phi \approx P_\Phi^\intercal$ (where these will be defined below). Briefly assuming the value function is approximately in the span of $\Phi$, we can rewrite the interference term by linearizing $e = V-\Phi w \approx \Phi w^* - \Phi w = \Phi \tilde{w}$
$$
\begin{align}
\text{interference} &= -2 \alpha e^\intercal K \Phi \Phi^\intercal S e \\
 &\approx -2 \alpha (\Phi \tilde w)^\intercal K \Phi \Phi^\intercal S \Phi \tilde w \\
 &= -2 \alpha \tilde w^\intercal K_\Phi  S_ \Phi \tilde w \\
\end{align}$$
Now we can take a look at skew symmetric component, $K_\Phi$.
$$K_\Phi = A_\Phi - S_\Phi = \Phi^\intercal( D(I-\gamma P) - \frac{1}{2} D(I-\gamma P) - \frac{1}{2}(I-\gamma P^\intercal)D^\intercal)\Phi$$
$$K_\Phi = -\frac{\gamma}{2} \Phi^\intercal (DP - (DP)^\intercal) \Phi$$
Thus, defining $P_\Phi \doteq \Phi^\intercal D P \Phi$, we see that making this symmetric erases $K_\Phi$, which approximately erases TD learning's bias. This suggests an auxiliary task for a deep neural network that learns $\Phi$. However, I'm skeptical that this would be a good idea. Enforcing a A symmetric $P_\Phi$ will make the feature space unable to express the asymmetric portions of the MDP. Do we really want a backwards transition $\phi(s') \rightarrow \phi(s)$ be indistinguishable from the true transition? 
