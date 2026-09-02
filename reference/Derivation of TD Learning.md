Start with the Bellman equation (in matrix form), and isolate the value function:
$$\begin{align}V = R + \gamma P V \\
(I-\gamma P) V - R =0
\end{align}$$
We'd like this to hold for our estimate $v_\theta \approx V$. Plugging it in:
$$(I-\gamma P) v_\theta -R = 0$$
In practice, we sample data from the state distribution $\mu$. It is common to weight each state in the above equation by that distribution. This can be expressed using multiplication on the left by $D = \text{diag}[\mu]$.
$$D[(I-\gamma P) v_\theta -R]=0$$
The expression on the inside is called the (negative) TD error. $\delta_\theta = R + \gamma P v_\theta - v_\theta$. 
$$-D\delta_\theta=0$$
TD treats $R + \gamma P v_\theta$ as a fixed target, detaching its gradient, and performs gradient descent:
$$\dot{\theta} = \frac{1}{2}(\nabla v_\theta)^{\intercal} D\delta_\theta=0$$
So it's attempting to make $\delta$ and $\nabla v$ orthogonal. For a single state, 
$$\dot{\theta} = \frac{1}{2} \mu(s) \delta_\theta(s)^\intercal \nabla v_\theta(s)=0$$
----
If I want to understand what happens to the value error $V-v_\theta$, I can rewrite $D\delta$  in the above update:
$$D\delta = D(R + \gamma P_\theta - v_\theta) = D[R-(I-\gamma P) v_\theta]$$
Note that $R = (I-\gamma P)V$. Plugging this in:
$$D[R-(I-\gamma P) v_\theta] = D(I-\gamma P)(V-v_\theta)$$
Letting $A \doteq D(I-\gamma P)$, and $e_\theta \doteq (V-v_\theta)$ we have
$$D\delta_\theta = Ae_\theta$$
$$\dot{\theta} = \frac{1}{2}(\nabla v_\theta)^{\intercal} Ae_\theta=0$$

