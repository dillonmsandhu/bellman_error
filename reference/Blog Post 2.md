(This is a follow-up to my previous post [Will Deep TD Learning Result in Quality Representations?](https://dillonmsandhu.github.io/representation/learning/2026/05/20/will-deep-td-learning-result-in-quality-representations.html))

Deep temporal difference learning is the primary method for estimating the value function in reinforcement learning. The goal of TD learning is to minimize the average squared value error, averaged over a state distribution $\mu$:
$$\overline{VE}(v, \mu) = \sum \mu(s) (V(s)-v(s))^2 = 
(V-v)^\intercal D (V-v)$$
where $V$ is the true value function, $v$ is estimate, and $D = \text{diag}[\mu]$. The value estimate, obtained from a neural network, is the dot product of the state-features and value weights: $v(s) = \phi(s)^\intercal w$. Since the value is not directly observable, TD learning uses bootstrapping: fitting the value network towards a target based on its own predictions (see the linked post above for a more complete explanation of TD learning).

In the last post, I provided a geometric intuition and showed that deep TD learning can adjust the features in the *wrong direction* -- raising the inherent value error $VE$ for the best weights. This happened when the TD target and true value function are on opposite sides of the feature space, which I call *misalignment*. An example is pictured below.

![[Pasted image 20260604104120.png]]
https://excalidraw.com/#json=9mQH5PFuQEyWgUvzfeLBv,bNNOVrhFhOTtciFH726BEA
Is misalignment a problem in practice? For evaluating the random policy on Four Rooms, I found it was not. But does it become an issue in challenging domains like math or robotics? 

The literature has explored how deep TD learning will adjust the features. [Tang and Munos 2023] showed that, under the property that the MDP is reversible ($D P = P^\top D$), the TD learning update happens to perform gradient descent on $V$. This sounds great, however, rebersibility is extremely unlikely to hold in practice: It says that the probability transition $s_1 \rightarrow s_2$ equals the probability of $s_2 \rightarrow s_1$. Yet, on most tasks, strong policies don't go backwards. 

In this post, I extend the results of [Tang and Munos 2023]. I show that alignment is a necessary and sufficient condition for the value error to decrease, reinforcing the importance of alignment. I also show that an *almost reversible* property is enough to guarantee alignment.  Before that, I walk us through their original proof, which answers the important question of when TD learning is doing the right thing. So let's go!

### Main Result of Tang and Munos (2023)
Tang and Munos consider a continuous version of TD learning. The following system of differnential equations describes how continuous TD learning progresses (in expectation) with learning rates $\eta$. 
$$
\begin{align}
\dot{w}_t &= \eta_w \cdot \Phi_t^\top D(Tv_t - v_t)\\
\dot{\Phi}_t &= \eta_\Phi \cdot D(Tv_t-v_t) w^\top_t
\end{align}
$$
Where the dot notation indicates the time derivative: ($\dot{x}_t = \frac{dx}{dt}$ for a function $x(t)$). These equations describe the dynamics of TD learning when discrete algorithmic updates are brought into continuous time. For the purposes of this blog post, I'll assume that the weights are a fixed value $w$.

The key question is how the value error changes with each update. We analyze this in terms of the time-derivative of the value error, $E$ (defined below).

We now define the value error. For theoretical convenience, Tang and Munos consider a loss that weights states slightly differently. They define a value error $E$  that weights by the "key" matrix $A = D(I-\gamma P)$:
$$
\begin{align}
E(\Phi, w) &\doteq \frac{1}{2} (\Phi w - V)^T A (\Phi w - V)
\end{align}$$
The key matrix is of massive importance for the stability of TD learning. The dynamics of the features can be re-written as:
$$\begin{align}
\dot{\Phi}_t &= \eta_\Phi \cdot D(R + (I-\gamma P)v_t) w^\top_t\\
\dot{\Phi}_t &= \eta_\Phi \cdot (DR + A v_t) w^\top_t
\end{align}
$$
TD learning is based on bootstrapping, so the true the value function doesn't appear directly. A key insight is that the Bellman equation implies $DR = D(I-\gamma P)  = AV$, meaning above equation can be rewritten in terms of the true value function $V$:
$$\begin{align}
\dot{\Phi}_t &= -\eta_\Phi \cdot A(\Phi_t w_t-V) w^\top_t
\end{align}
$$
We'd like to understand how $E$ changes as TD learning progressively updates the features. Essentially this boils down to comparing $\dot{\Phi}_t$ to $-\partial_\Phi E(\Phi, w)$. If they point in the same direction, then TD learning decreases the value error $E$, exactly like gradient descent.

**Theorem (Tang and Munos 2023, Theorem 3):**
*Assume that $A$ is symmetric (i.e. $DP = P^\top D$). Then, the TD learning update just defined is the same as gradient descent on $E$, i.e.:*$$\dot{\Phi}_t \propto - \partial_{\Phi_t}E(\Phi_t, w_t)$$The conclusion is TD learning is effectively supervised learning of the value function. Sounds great! However, the premise that $A$ is symmetric is unlikely to hold. It occurs when the Markov chain is symmetric, meaning the probability of the transition $s_1 \rightarrow s_2$ is the same as $s_1 \leftarrow s_2$. 

First, I provide the proof of this theorem. Next, I relax the reversibility condition to analyze exactly when misalignment occurs in terms $A$'s symmetry.

**Proof of Tang and Munos Theorem 3:**
Define the error vector $e_t \doteq \Phi_t w - V^\pi$, which is a function of $\Phi$. The value error rewrites as:
$$E(\Phi_t, w_t) = \frac{1}{2} e_t^T A e_t$$
Since $E$ is quadratic, it can be rewritten as $e_t^\top (A + A^\top) e_t$, retaining only the symmetric portion of $A$ (this is discussed more in the next section). From this, we have:
$$\begin{aligned} \partial_{\Phi_t} E(\Phi_t, w_t) &= \frac{1}{2} (A + A^\top) e_t w^T\end{aligned}$$
Now let's consider the change in features due to TD learning. Plugging $e_t$ into the earlier expression $\dot{\Phi}_t = -\eta_\Phi \cdot A(\Phi_i w_i-V) w^\top_i$ gives the following remarkably similar, but not quadratic, formula.
$$\begin{aligned} \dot{\Phi}_t &= -\eta_\Phi Ae_tw^\top\end{aligned}$$
These formulations show that if $A$ is symmetric ($A^\top = A$), then gradient descent on $E$ gives the same feature update as TD learning. For the remainder of the proof one can refer to Tang and Munos, Appendix B.
### Without Symmetry
For the TD learning update to be beneficial, we don't need complete alignment of $\dot{\Phi}_t$ and $\partial_{\Phi_t} E$ -- just that $\dot{E} < 0$. S

ince $E$ is a function of $\Phi_t$, which is in tern a function of $t$, the chain rule states that $\dot{E}$ is the product of $\partial_{\Phi}E(\Phi)$ and $\dot{\Phi}$. Formally, the chain rule gives $\dot{E} = \langle \partial_\Phi E, \dot{\Phi} \rangle_F$ where $\langle A, B\rangle_F \doteq \text{Tr}(A^\top B)$ is the Frobenius inner product. 

Don't worry if the matrix calculus is unfamiliar, intuitions from standard calculus will be enough to understand this post. The key idea is that $\dot{\Phi}_t$ must point in a direction of high alignment with $-\partial_{\Phi_t} E$ in order to decrease $E$. 

In this section, I derive $\dot{E}$ and when it is negative. To do so, I use the decomposition of $A$ into symmetric and [skew-symmetric](https://en.wikipedia.org/wiki/Skew-symmetric_matrix) parts.
$$\begin{align}
A &= (1/2)(A + A^\top) + (1/2)(A - A^\top)\\
&= S + K
\end{align}
$$
Note that for the anti-symmetric part $K$, we have that  $u^\top K u=0$ for any vector $u$.[^1] This yields:
$$E(\Phi_t, w) = e_t^\top S e_t$$
This says that the value error only cares about the symmetric portion of its weighting matrix, which in this case is the key matrix $A$. From this we have the following expression for $\dot{E}(\Phi_t)$.
$$\dot{E}(\Phi_t) = \frac{1}{2} \left(\dot{e}_t^\top S e_t + e_t S\dot{e}_t \right) = e_t^\top S \dot{e}_t$$
Where the last equality follows from the symmetry of $S$. Substituting $\dot{e}_t = \dot{\Phi}_t w_t$, we get the following scalar derivative:
$$\dot{E}(\Phi_t) = e_t^\top S \dot{\Phi}_t w$$
To get this into the form of a Frobenius inner product, we take the trace. For a scalar $c$, $Tr(c)=c$. For matrices $A$, $B$, and $C$, we also have the cyclic property of the trace: $Tr(ABC) = Tr(CAB)$. Chaining these two together and then recalling definition of the inner product, we get:
$$
\begin{align}
\dot{E}(\Phi_t) &= \text{Tr}(e_t^\top S \dot{\Phi}_t w) \\
&= \text{Tr}(w e_t^\top S \dot{\Phi}_t)\\
&= \langle S e_tw ^\top, \dot{\Phi}_t\rangle_F\\
&= \langle \partial_\Phi E, \dot{\Phi} \rangle_F
\end{align}
$$
The last step is due to the fact that  $S e_t w_t^\top$ equals $\partial_\Phi E$ (which can be seen from the first step of the Tang and Munos proof, or the chain rule definition of $\dot{E}(\Phi_t)$. Notice that the improvement in $E$ is therefore the following notion of alignment:
$$\text{Alignment} =-\langle \partial_\Phi E, \dot{\Phi}\rangle_F$$
Next we determine when $\dot{E}$ is negative (i.e. when alignment occurs). Starting from the trace expression above, and plugging in $\dot{\Phi}_t = -\eta_\Phi A e_t w^\top$ allows us to pull out scalars:
$$\begin{align}
\dot{E}(\Phi_t) &= - \eta_\Phi \cdot \text{Tr}\left(w e_t^\top S A  e_t w^\top\right) \\
&= - \eta_\Phi \cdot \text{Tr}\left(w^\top w e_t^\top S A  e_t \right) \\
&= - \eta_\Phi \cdot \|w \|_2^2 \cdot \text{Tr}\left(e_t^\top S A  e_t \right) \\
\end{align}
$$
Since the scalar on the outside are negative, for the update to be beneficial ($\frac{dE}{d_t}<0$), the argument to the trace must be positive. That is, the following guarantees alignment:
$$\begin{align} e_t^\top S A  e_t > 0 \\ 
\end{align}
$$
In other words, the TD learning update is guaranteed to help if the matrix $SA = (A + A^\top)A$ is positive definite[^2]. Expanding the above expression and collecting terms will allow us to cancel a skew-symmetric part.
$$
\begin{align}
e_t^\top S A  e_t&=e_t^\top S (S+K)  e_t\\
&=e_t^\top S^2e_t +  e_t^\top SK e_t\\
&=e_t^\top S^2e_t  +\frac{1}{2}e_t^\top (SK + K^\top S^\top) e_t +\frac{1}{2} e_t^\top (SK - K^\top S^\top) e_t\\
\end{align}
$$
The the last term is a quadratic with skew-symetric weight, and is therefore $0$. 
Also, we have that $K^\top = -K$, which simplifies the second term.$$\begin{align}
e_t^\top \left( S^2  +\frac{1}{2} (SK -K S) \right) e_t > 0\\
\end{align}
$$Alignment occurs exactly when:
$$\begin{align}
e_t^\top S^2e_t  > -\frac{1}{2}e_t^\top (SK -K S) e_t\\
\end{align}
$$Note that the left hand side is always positive, since $S$ is positive definite. The term on the right can be positive, negative, or zero, since $SK-KS$ is indefinite ($\text{Tr}(SK-KS)=Tr(SK)-Tr(KS)=0$). This means that a beneficial update cannot be guaranteed for all error vectors $e_t$, unless $A$ is symmetric. 
#### Interpretation
This post derived when the Markov chain induced by a policy will allow TD learning to make beneficial updates. But how do we interpret the condition?

The key matrix $A = D(I-\gamma P)$ has terms like:
1. $1-\gamma \mu_i P_{ii}$ on the diagonals (guaranteed to exist in $S$, zero in $K$)
2. $-\gamma \mu_i P_{ij}$ off the diagonals

Elements of the skew-symmetric part are $K_{ij} = \frac{\gamma }{2}(\mu_j P_{ji} - \mu_i P_{ij})$.

For each pair of states, $K_{ij}$ measures the extent to which the MDP is non-reversible. When it is zero, it indicates that flow between the two states is equal. If the transition is deterministically one-way, then $|K_{ij}|=\mu_i \gamma/2$. Finally, multiplication by $e_t$ weights these terms by the value error. 

If the value error is in a region where the agent wonders back and forth equally, then TD learning will reduce it. However, in a one-way region, TD learning will raise the value error. Consider an irreversible transition $s_1 \rightarrow{} s_2$, where $\phi(s_1)^\top \phi(s_2) \neq 0$. TD Learning will update the features such that the value at $s_1$ locally satisfies the Bellman Equation, but it won't take into account how changing these features affects the value of $s_2$. In contrast, gradient descent on $E$ adjusts both $\phi(s_1)$ and $\phi(s_2)$ to adjust the total value error. 

Consider the random policy on four rooms. This is a highly symmetric markov chain. Thus, it's possible that my last example was inadvertently cherry-picked for TD learning to work. In an upcoming post, I'll take a look at whether its possible to cherry pick highly non-reversible MDPs that make deep TD learning fail, even on-policy. Thanks for reading!

-------------
[1]: $u^\top K u= (u^\top K^\top u)^\top = u^\top K^\top u = \frac{1}{2}u^\top (A^\top - A)  u = - u^\top K u$. The only way for this $K$-weighted inner product to equal its negative is for it to be zero.
[^2] Matrix $B$ is positive definite if for any vector $x$, $x^\top B x >0$.
