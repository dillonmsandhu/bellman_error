# Second-Order and Natural Gradient Methods for TD Learning
Preconditioning the error dynamics equation $\dot{\mathbf{e}} = -\Theta(S + K)\mathbf{e}$ to undo state aliasing and skew-symmetric spiraling.

---

### Motivation: Two Ways to Fix $\dot{\mathbf{e}} = -\Theta(S + K)\mathbf{e}$

In [`Investigation_into_TD_vs_E.md`](Investigation_into_TD_vs_E.md), we showed that continuous-time expected `TD(0)` obeys the error dynamics:
$$
\dot{\mathbf{e}} = -\Theta (S + K) \mathbf{e}, \qquad \Theta \doteq J J^\intercal,
$$
and that the $S$-norm value error decreases according to:
$$
\frac{d}{dt}\left(\frac{1}{2}\|\mathbf{e}\|_S^2\right) = -\|J^\intercal S \mathbf{e}\|_2^2 - \langle J^\intercal S \mathbf{e},\; J^\intercal K \mathbf{e} \rangle.
$$
TD stalls or diverges when **both** of the following happen at once:
1. The MDP is irreversible ($K \neq 0$), and
2. The function approximator aliases consecutive states ($\nabla \mathbf{v}(s) \approx \nabla \mathbf{v}(s')$), which gives $\Theta = J J^\intercal$ large off-diagonal entries, collapsing the gradient-difference term $J^\intercal S \mathbf{e}$ while amplifying the gradient-sum term $J^\intercal K \mathbf{e}$.

Notice that $E$-minimization fixes this by **changing the objective** (replacing $A = S + K$ with $S$, eliminating $K$ at the cost of requiring double sampling or an auxiliary network `v_aux`).

This document develops the complementary approach: **keeping the standard single-sample TD update $J^\intercal A \mathbf{e} = J^\intercal D \boldsymbol{\delta}$, and changing the optimizer** via a parameter-space preconditioner $M$. We build all of the necessary theory from the ground up:
1. How a preconditioner $M$ reshapes the effective NTK $\Theta_M$.
2. The duality between state-space operators ($\Theta A$) and parameter-space operators ($D_J, A_J, S_J, K_J$).
3. **Symmetric Natural TD** ($M = D_J^{-1}$): why it restores tabular geometry and eliminates $K$ from the $D$-norm contraction.
4. **Non-Symmetric Natural TD** ($M = A_J^{-1}$): why it is nonlinear LSTD, collapses all in-subspace eigenvalues to $1 + 0i$, and needs zero double sampling.
5. **From theory to sampled updates & matrix-free solvers**: clearly separating *what linear system we want to solve* (from sampled TD errors $\delta_b$) from *how we solve it* via JVPs, VJPs, CG, and Arnoldi/GMRES.
6. **Scalable layer-wise approximations (K-FAC, Sylvester Non-Symmetric K-FAC, and Muon)**: how to avoid $m$ Krylov passes per step and scale to large networks at $O(1)$ per-step cost.
7. **Why momentum breaks on $K \neq 0$**: complex eigenmodes, positive vs. negative momentum (OGDA), and extragradient.
8. **The stochastic noise trade-off and empirical ablations**: why $A_J^{-1}$ amplifies minibatch TD-error noise by $O(1/(1-\gamma))$, and how interpolation ($\alpha$), cosine LR decay, Polyak-Ruppert averaging, and Sylvester Non-Symmetric K-FAC resolve it.
9. **Do these methods "remove the skew-symmetric portion" of TD (and can a first-order method do it)?**
10. **Related work**: how this framework connects to LSTD, Pan et al. (2017) ATD, Knight & Lerner (2018) NGDQN, Ke et al. GNTD, GTD2/TDC, $E$-minimization, and rotational game optimizers.

---

### 1. Preconditioning the Error Dynamics Equation

Standard expected `TD(0)` updates parameters along the raw semi-gradient direction:
$$
\dot{\theta} = J^\intercal D \boldsymbol{\delta} = J^\intercal A \mathbf{e}.
$$
*(Note on sign convention: if we define the semi-gradient descent vector $g_{\text{TD}} \doteq \nabla_\theta \left[\frac{1}{2}\mathbb{E}_\mu\big[(v_\theta(s) - \mathrm{sg}(r + \gamma v_\theta(s')))^2\big]\right]$, then $g_{\text{TD}} = -J^\intercal A \mathbf{e}$ and $\dot{\theta} = -g_{\text{TD}} = +J^\intercal A \mathbf{e}$.)*

Now suppose instead of stepping directly along $J^\intercal A \mathbf{e}$, we multiply the update by a $k \times k$ **preconditioner matrix** $M \in \mathbb{R}^{k \times k}$ (where $k = \dim(\theta)$ is the number of parameters):
$$
\dot{\theta} = M J^\intercal A \mathbf{e} = -M g_{\text{TD}}.
$$
Using $\dot{\mathbf{e}} = -\dot{\mathbf{v}} = -J \dot{\theta}$ just as before, we get the **preconditioned error dynamics**:
$$
\bbox{\dot{\mathbf{e}} = -J M J^\intercal A \mathbf{e} = -\Theta_M (S + K) \mathbf{e}, \qquad \Theta_M \doteq J M J^\intercal.}
$$
This single equation is the foundation of everything that follows:
- Standard `TD(0)` uses $M = I$, which gives $\Theta_I = J J^\intercal = \Theta$.
- Any second-order or natural-gradient TD optimizer is simply a choice of $M$ designed to make the **effective kernel** $\Theta_M = J M J^\intercal$ behave nicely against $A = S + K$.

What would an *ideal* $\Theta_M$ look like? There are two natural targets:
1. **Undo the representation ($\Theta_M \approx D^{-1}$):** Make the neural network behave like a tabular value function (no state aliasing).
2. **Undo both the representation and the Bellman operator ($\Theta_M \approx A^{-1}$):** Make $\Theta_M A \approx I$, so every error mode contracts straight to zero at rate $1$ with no spiraling and no $1/(1-\gamma)$ slowdown.

Before deriving the exact matrices $M$ that achieve these two targets, we need a quick algebraic bridge between state space ($\mathbb{R}^{|\mathcal{S}|}$) and parameter space ($\mathbb{R}^k$).

---

### 2. Parameter-Space Operators and State-Space Duality

In tabular theory, we work with $|\mathcal{S}| \times |\mathcal{S}|$ state-space matrices ($D, A, S, K, \Theta$). In deep RL, $|\mathcal{S}|$ may be huge or continuous, so an optimizer can only work with $k \times k$ parameter-space matrices (or matrix-vector products in $\mathbb{R}^k$).

Define the **parameter-space projected operators** ($k \times k$ matrices):
$$
D_J \doteq J^\intercal D J, \qquad A_J \doteq J^\intercal A J, \qquad S_J \doteq J^\intercal S J, \qquad K_J \doteq J^\intercal K J.
$$
Since $A = S + K$, we immediately have the same symmetric/skew-symmetric decomposition in parameter space:
$$
A_J = S_J + K_J, \qquad S_J^\intercal = S_J \succeq 0, \qquad K_J^\intercal = -K_J.
$$

#### Key Duality Fact: Same Nonzero Eigenvalues
The $|\mathcal{S}| \times |\mathcal{S}|$ state-space operator $\Theta A = J J^\intercal A$ and the $k \times k$ parameter-space operator $A_J = J^\intercal A J$ have the **exact same nonzero eigenvalues**.

*Proof:* Suppose $A_J u = \nu u$ with $\nu \neq 0$ (so $Ju \neq \mathbf{0}$). Multiplying both sides on the left by $J$ gives:
$$
J (J^\intercal A J u) = \nu (J u) \implies (\Theta A)(J u) = \nu (J u).
$$
Conversely, if $\Theta A \mathbf{e} = \nu \mathbf{e}$ with $\nu \neq 0$, then multiplying on the left by $J^\intercal A$ gives $A_J (J^\intercal A \mathbf{e}) = \nu (J^\intercal A \mathbf{e})$. $\square$

This means the entire spectral picture of TD learning—both the real decay rates $\alpha_m = \mathrm{Re}(\nu_m)$ and the rotational speeds $\beta_m = \mathrm{Im}(\nu_m)$—lives identically inside the parameter-space matrix $A_J = S_J + K_J$.

#### Transition-Level Form (No State-Space Matrices Needed)
Because the $s$-th row of $J$ is $\nabla \mathbf{v}(s)^\intercal$, each parameter-space operator is simply an expectation over single transitions $(s \sim \mu,\; s' \sim P(\cdot \mid s))$:
$$
\begin{aligned}
D_J &= \mathbb{E}_s\big[\nabla \mathbf{v}(s)\,\nabla \mathbf{v}(s)^\intercal\big], \\
A_J &= \mathbb{E}_{s, s'}\big[\nabla \mathbf{v}(s)\big(\nabla \mathbf{v}(s) - \gamma \nabla \mathbf{v}(s')\big)^\intercal\big], \\
S_J &= \mathbb{E}_{s, s'}\!\left[\nabla \mathbf{v}(s)\nabla \mathbf{v}(s)^\intercal - \frac{\gamma}{2}\big(\nabla \mathbf{v}(s)\nabla \mathbf{v}(s')^\intercal + \nabla \mathbf{v}(s')\nabla \mathbf{v}(s)^\intercal\big)\right], \\
K_J &= \frac{\gamma}{2}\,\mathbb{E}_{s, s'}\big[\nabla \mathbf{v}(s')\nabla \mathbf{v}(s)^\intercal - \nabla \mathbf{v}(s)\nabla \mathbf{v}(s')^\intercal\big].
\end{aligned}
$$
Notice that **$D_J$ is the standard Gauss-Newton / Fisher matrix** of a regression network predicting $v_\theta(s)$, whereas **$A_J$ is the exact Jacobian of the TD semi-gradient** $g_{\text{TD}}(\theta)$ with respect to $\theta$ (ignoring second derivatives $\nabla^2 v_\theta$).

---

### 3. Symmetric Natural TD ($M = D_J^{-1}$): Restoring Tabular Geometry

#### What Is a Natural Gradient for a Value Function?
Standard gradient descent chooses the parameter step $\Delta\theta$ that best aligns with $J^\intercal A \mathbf{e}$ subject to a **parameter-space** step-size budget $\|\Delta\theta\|_2^2 \le \epsilon^2$. Because neural network parameterizations warp distances (e.g., `AvgPool` makes many states share the same gradient), a small step in $\theta$ can move $\mathbf{v}(s)$ and $\mathbf{v}(s')$ together in lockstep.

**Symmetric Natural TD** instead measures step size in **function space** using the $D$-norm of the value change $\Delta \mathbf{v} \approx J \Delta\theta$:
$$
\|\Delta \mathbf{v}\|_D^2 = \Delta\theta^\intercal (J^\intercal D J) \Delta\theta = \Delta\theta^\intercal D_J \Delta\theta \le \epsilon^2.
$$
Solving $\max_{\Delta\theta^\intercal D_J \Delta\theta \le \epsilon^2} \langle \Delta\theta,\, J^\intercal A \mathbf{e} \rangle$ gives the Symmetric Natural TD update:
$$
\dot{\theta}_{\text{Sym-NTD}} = D_J^{-1} J^\intercal A \mathbf{e} = (J^\intercal D J)^{-1} J^\intercal A \mathbf{e}.
$$
*(In practice, we add damping $(D_J + \lambda I)^{-1}$ so the inverse always exists even when $D_J$ is rank-deficient.)*

#### Effect on the Error Dynamics: The $K$-Term Vanishes in the $D$-Norm!
Plugging $M = D_J^{-1}$ into $\dot{\mathbf{e}} = -J M J^\intercal A \mathbf{e}$ yields:
$$
\dot{\mathbf{e}}_{\text{Sym-NTD}} = -J (J^\intercal D J)^{-1} J^\intercal A \mathbf{e} = -\Pi_{D, J}\, D^{-1} (S + K) \mathbf{e},
$$
where
$$
\Pi_{D, J} \doteq J (J^\intercal D J)^{-1} J^\intercal D
$$
is the **$D$-orthogonal projection** onto the tangent space $\mathrm{col}(J)$ (meaning $\Pi_{D, J} \mathbf{u} = \mathbf{u}$ for every $\mathbf{u} \in \mathrm{col}(J)$, and $\Pi_{D, J}^\intercal D = D \Pi_{D, J}$).

Now compare this directly to the **Warm-Up (Tabular TD)** in `Investigation_into_TD_vs_E.md`! Suppose the value error lies in the tangent space, $\mathbf{e} \in \mathrm{col}(J)$ (which is exact for overparameterized networks on finite state spaces, and holds for the representable component $\mathbf{e}_\parallel = \Pi_{D, J}\mathbf{e}$ in general). Compute the rate of change of the **$D$-norm** squared error $\frac{1}{2}\|\mathbf{e}\|_D^2$:
$$
\frac{d}{dt}\left(\frac{1}{2}\|\mathbf{e}\|_D^2\right) = \mathbf{e}^\intercal D \dot{\mathbf{e}} = -\mathbf{e}^\intercal D \Pi_{D, J} D^{-1}(S + K)\mathbf{e}.
$$
Because $\mathbf{e} \in \mathrm{col}(J)$ and $D \Pi_{D, J} = \Pi_{D, J}^\intercal D$, we have $\mathbf{e}^\intercal D \Pi_{D, J} = (\Pi_{D, J} \mathbf{e})^\intercal D = \mathbf{e}^\intercal D$. Therefore $D$ and $D^{-1}$ cancel cleanly:
$$
\bbox{\frac{d}{dt}\left(\frac{1}{2}\|\mathbf{e}\|_D^2\right) = -\mathbf{e}^\intercal (S + K)\mathbf{e} = -\mathbf{e}^\intercal S \mathbf{e} - \underbrace{\mathbf{e}^\intercal K \mathbf{e}}_{=\,0} = -\|\mathbf{e}\|_S^2 < 0.}
$$

**Why this is a major insight:**
- In **standard TD**, $\Theta = J J^\intercal$ has large off-diagonal entries whenever $\nabla \mathbf{v}(s) \approx \nabla \mathbf{v}(s')$. That off-diagonal coupling breaks the cancellation of $K$, allowing $\tilde{T}_2$ to cancel $\tilde{T}_1$.
- **Symmetric Natural TD** replaces $\Theta$ with $\Theta_{D_J^{-1}} = \Pi_{D, J} D^{-1}$. Inside $\mathrm{col}(J)$, it **completely removes the representation's state aliasing** and restores the exact continuous-time contraction guarantee of **tabular TD**!
- Even for a `Pooled CNN` or `(x, y) + MLP`, $D_J^{-1}$ orthogonalizes the effective state features so that consecutive states stop interfering with each other.

#### What Symmetric Natural TD Does *Not* Fix
Look at $\dot{\mathbf{e}}_{\text{Sym-NTD}} = -\Pi_{D, J}\, D^{-1}(S + K)\mathbf{e} = -\Pi_{D, J}(I - \gamma P)\mathbf{e}$:
1. **Horizon ill-conditioning remains:** The smallest eigenvalues of $I - \gamma P$ are $O(1 - \gamma)$, so when $\gamma = 0.99$, slow global value modes still decay $\sim 100\times$ slower than fast local modes.
2. **Tabular spiraling remains:** $K$ no longer *increases* $\|\mathbf{e}\|_D^2$ in continuous time, because $K \mathbf{e}$ is $D$-orthogonal to $\mathbf{e}$. However, $K$ still rotates $\mathbf{e}$ sideways at speed $\beta_m$, which in **discrete time** forces a smaller step size $\eta$ and causes spiraling on cyclic/directed MDPs (such as the Ring MRP).
3. **Out-of-subspace error ($\mathbf{e}_\perp$):** When the network cannot represent $\mathbf{V}$ exactly ($\mathbf{e} = \mathbf{e}_\parallel + \mathbf{e}_\perp$), $\dot{\mathbf{e}}_\parallel$ still has a cross-term $-\Pi_{D,J} D^{-1} A \mathbf{e}_\perp$ that drives $\mathbf{v}_\theta$ toward the projected Bellman fixed point (not the true $\mathbf{V}$), which is the standard on-policy vs. off-policy TD fixed-point issue.

---

### 4. Non-Symmetric Natural TD ($M = A_J^{-1}$): Inverting Both $J$ and $S + K$

If $M = D_J^{-1}$ removes $J$ but leaves $A = S + K$ intact, the natural next question is: **what if we invert the full non-symmetric projected Bellman operator $A_J = J^\intercal A J = S_J + K_J$?**

Define the **Non-Symmetric Natural TD** update:
$$
\dot{\theta}_{\text{NS-NTD}} = A_J^{-1} J^\intercal A \mathbf{e} = (J^\intercal A J)^{-1} J^\intercal A \mathbf{e} = -(S_J + K_J)^{-1} g_{\text{TD}}.
$$

#### 4.1 Effect on the Error Dynamics: Exact Straight-Line Contraction
Plug $M = A_J^{-1}$ into $\dot{\mathbf{e}} = -J M J^\intercal A \mathbf{e}$:
$$
\bbox{\dot{\mathbf{e}}_{\text{NS-NTD}} = -J (J^\intercal A J)^{-1} J^\intercal A \mathbf{e} = -\Pi_{A, J}\, \mathbf{e}, \qquad \Pi_{A, J} \doteq J (J^\intercal A J)^{-1} J^\intercal A.}
$$
$\Pi_{A, J}$ is an **oblique projection** onto $\mathrm{col}(J)$ (orthogonal with respect to the bilinear form $A$). For **any** error vector in the tangent space $\mathbf{e} = J w \in \mathrm{col}(J)$:
$$
\Pi_{A, J}(J w) = J (J^\intercal A J)^{-1} (J^\intercal A J) w = J w = \mathbf{e}.
$$
Therefore, on every representable direction:
$$
\dot{\mathbf{e}}_{\text{NS-NTD}} = -\mathbf{e} \quad \Longrightarrow \quad \text{every eigenvalue is } \nu_m = 1 + 0i \quad (\alpha_m = 1,\; \beta_m = 0).
$$
Look at what just happened:
- **State aliasing ($\Theta$)**: completely eliminated.
- **Horizon slowdown ($1/(1-\gamma)$)**: completely eliminated.
- **Skew-symmetric spiraling ($K$)**: completely eliminated ($\beta_m = 0$).

In discrete time with step size $\eta = 1$, a linear model converges in **1 step**. For a nonlinear neural network, $J(\theta)$ changes slightly as $\theta$ moves, so this is a **Gauss-Newton / Newton-like iteration** that converges quadratically/superlinearly in ~10–20 steps (as observed in the exact MRP experiments in `forks/td_optimizer/ns_natural_td.png`).

#### 4.2 Connection to LSTD
Suppose our value function is linear, $\mathbf{v}_\theta = \Phi \theta$, so $J = \Phi$ is constant. Then:
- $A_J = \Phi^\intercal D (I - \gamma P) \Phi$,
- $g_{\text{TD}} = \Phi^\intercal D (\Phi \theta - \mathbf{r} - \gamma P \Phi \theta) = A_J \theta - \Phi^\intercal D \mathbf{r}$.

Taking one Non-Symmetric Natural TD step with $\eta = 1$ gives:
$$
\theta_{\text{new}} = \theta - A_J^{-1}(A_J \theta - \Phi^\intercal D \mathbf{r}) = \big(\Phi^\intercal D (I - \gamma P)\Phi\big)^{-1} \Phi^\intercal D \mathbf{r} = \theta_{\text{LSTD}}.
$$
Thus, **Non-Symmetric Natural TD is literally local-linearized (nonlinear) LSTD**! At each step $\theta_t$, it linearizes the neural network around $\theta_t$ ($\mathbf{v}_{\theta_t + \Delta\theta} \approx \mathbf{v}_{\theta_t} + J(\theta_t)\Delta\theta$) and solves the LSTD fixed-point equation in that tangent space.

#### 4.3 Why $A_J + \lambda I$ Is Guaranteed Invertible
A worry with non-symmetric matrices is that they might be singular or have negative-real-part eigenvalues (causing divergence). Your positive-definiteness result for $S$ guarantees this cannot happen on-policy!

For any nonzero parameter vector $u \in \mathbb{R}^k$:
$$
u^\intercal (A_J + \lambda I) u = u^\intercal S_J u + \underbrace{u^\intercal K_J u}_{=\,0} + \lambda \|u\|_2^2 = (Ju)^\intercal S (Ju) + \lambda \|u\|_2^2 \ge \lambda \|u\|_2^2 > 0.
$$
By the Bendixson/Rayleigh bound, **every eigenvalue $\nu$ of $A_J + \lambda I$ satisfies $\mathrm{Re}(\nu) \ge \lambda > 0$**. Therefore $A_J + \lambda I$ is strictly positive-real (accretive), always invertible, and well-behaved for iterative linear solvers.

#### 4.4 Why It Requires Zero Double Sampling (Unlike $E(0)$)
This is one of the cleanest theoretical advantages of Non-Symmetric Natural TD over $E(0)$:
- **Why $E(0)$ has double-sampling bias:** The $E(0)$ gradient is $\mathbb{E}_{s, s'}[(\nabla \mathbf{v}(s) - \gamma \nabla \mathbf{v}(s'))\,(v(s) - r - \gamma v(s'))]$. Because the *same* sampled next state $s'$ appears inside both $\nabla \mathbf{v}(s')$ and $v(s')$, their product produces a spurious variance term $\gamma^2 \mathrm{Cov}(\nabla \mathbf{v}(s'), v(s') \mid s) \neq 0$ whenever $P(\cdot \mid s)$ is stochastic.
- **Why Non-Symmetric Natural TD does not:**
  1. The right-hand side $g_{\text{TD}} = \mathbb{E}_{s, s'}[\nabla \mathbf{v}(s)\,(v(s) - r - \gamma v(s'))]$ only multiplies the random target $r + \gamma v(s')$ by $\nabla \mathbf{v}(s)$ (which depends only on $s$, not $s'$), so it is unbiased from single transitions.
  2. The operator $\hat{A}_J = \frac{1}{B}\sum_{b=1}^B \nabla \mathbf{v}(s_b)\big(\nabla \mathbf{v}(s_b) - \gamma \nabla \mathbf{v}(s'_b)\big)^\intercal$ contains **no $r_b$ and no $v(s'_b)$**—it is linear in $P(s' \mid s)$, so $\mathbb{E}_{s' \sim P(\cdot \mid s)}[\hat{A}_J] = A_J$ is **100% unbiased from single-trajectory transitions $(s_t, r_t, s_{t+1})$**.

*(Caveat: while $\hat{A}_J$ is unbiased, the **inverse** of a noisy matrix is slightly biased by Jensen's inequality, $\mathbb{E}[\hat{A}_J^{-1}] \neq A_J^{-1}$, and $\hat{A}_J^{-1} \hat{g}_{\text{TD}}$ correlates the two noises if computed on the same minibatch—which we analyze carefully in Section 8.)*

---

### 5. From Theory to Practice: Sampled Updates vs. Matrix-Free Solvers

Sections 1–4 analyzed the population dynamics in terms of the true value error $\mathbf{e} = \mathbf{V} - \mathbf{v}_\theta$, which we never observe in practice. From here on, we work directly with **sampled transitions** $\{(s_b, r_b, s'_b, \text{done}_b)\}_{b=1}^B$ and their **sampled TD errors**:
$$
\delta_b \doteq r_b + \gamma (1 - \text{done}_b)\, v_\theta(s'_b) - v_\theta(s_b).
$$
Recall the exact bridge between the theoretical error $\mathbf{e}$ and the sampled TD error $\delta_b$: because $D \boldsymbol{\delta} = A \mathbf{e}$, the sampled semi-gradient descent vector
$$
\hat{g}_{\text{TD}} \doteq -\frac{1}{B}\sum_{b=1}^B \delta_b \,\nabla \mathbf{v}(s_b)
$$
is an unbiased estimator of $-J^\intercal D \boldsymbol{\delta} = -J^\intercal A \mathbf{e}$. (Taking a step $\theta \leftarrow \theta - \eta \hat{g}_{\text{TD}} = \theta + \frac{\eta}{B}\sum_b \delta_b \nabla \mathbf{v}(s_b)$ is standard `TD(0)`.)

#### Separating *What* Update We Want from *How* We Solve It
To keep the algorithmic landscape crystal clear, it helps to separate two completely independent choices:

1. **Axis 1 — Which update equation we want to solve (the operator):**
   Every natural-gradient TD update solves a $k \times k$ linear system for the parameter step $\Delta\theta \in \mathbb{R}^k$, followed by $\theta_{t+1} = \theta_t - \eta\,\Delta\theta$:
   $$
   \bbox{(\hat{M}_{\text{op}} + \lambda I)\,\Delta\theta = \hat{g}_{\text{TD}} = -\frac{1}{B}\sum_{b=1}^B \delta_b\,\nabla \mathbf{v}(s_b),}
   $$
   where the operator $\hat{M}_{\text{op}} \in \mathbb{R}^{k \times k}$ is estimated from the minibatch transitions (using **no** $r_b$ or $\delta_b$):
   - **Symmetric Natural TD ($\alpha = 0$):** $\hat{M}_{\text{op}} = \hat{D}_J \doteq \frac{1}{B}\sum_{b=1}^B \nabla \mathbf{v}(s_b)\,\nabla \mathbf{v}(s_b)^\intercal$.
   - **Non-Symmetric Natural TD ($\alpha = 1$):** $\hat{M}_{\text{op}} = \hat{A}_J \doteq \frac{1}{B}\sum_{b=1}^B \nabla \mathbf{v}(s_b)\big(\nabla \mathbf{v}(s_b) - \gamma(1-\text{done}_b)\nabla \mathbf{v}(s'_b)\big)^\intercal$.
   - **Interpolated Natural TD ($\alpha \in [0, 1]$):** $\hat{M}_{\text{op}} = \hat{A}_J^{(\alpha)} \doteq \frac{1}{B}\sum_{b=1}^B \nabla \mathbf{v}(s_b)\big(\nabla \mathbf{v}(s_b) - \alpha\gamma(1-\text{done}_b)\nabla \mathbf{v}(s'_b)\big)^\intercal$, which smoothly interpolates between Symmetric ($\alpha = 0$) and Non-Symmetric ($\alpha = 1$).

2. **Axis 2 — How we solve $(\hat{M}_{\text{op}} + \lambda I)\,\Delta\theta = \hat{g}_{\text{TD}}$ in practice (the solver):**
   A neural network can have $k = 10^4$ to $10^7$ parameters, so forming and inverting a $k \times k$ matrix directly ($O(k^2)$ memory, $O(k^3)$ compute) is impossible. There are two main solver families:
   - **Exact Matrix-Free Krylov Solvers (this section):** Never approximate $\hat{M}_{\text{op}}$; solve the exact linear system iteratively using **CG** (when $\alpha = 0$) or **Arnoldi / GMRES** (when $\alpha > 0$), costing $m$ JVP+VJP passes per step.
   - **Structured Layer-Wise Approximations (Section 6):** Approximate $\hat{M}_{\text{op}}$ block-diagonally per layer (**K-FAC**, **Sylvester Non-Symmetric K-FAC**, and **Muon**), reducing the per-step cost to $O(1)$ passes.

---

#### 5.1 What We Are Computing: The Matrix-Vector Product Oracle $u \mapsto (\hat{A}_J^{(\alpha)} + \lambda I) u$

Why do iterative linear solvers (CG and GMRES) avoid forming a $k \times k$ matrix? Because to solve $(\hat{A}_J^{(\alpha)} + \lambda I)\,\Delta\theta = \hat{g}_{\text{TD}}$, they only ever need a black-box function (an **oracle**) that takes a vector $u \in \mathbb{R}^k$ and returns the matrix-vector product $(\hat{A}_J^{(\alpha)} + \lambda I) u \in \mathbb{R}^k$.

**What is $u$ in practice?** During each optimization step, the iterative solver feeds a sequence of **candidate parameter-update directions** $u \in \mathbb{R}^k$ into this oracle:
- On the very first call, $u$ is simply the normalized TD semi-gradient $q_1 = \hat{g}_{\text{TD}} / \|\hat{g}_{\text{TD}}\|_2$.
- On calls $2, \dots, m$, $u$ is each new orthogonalized search direction $q_j \in \mathbb{R}^k$ built by CG or Arnoldi/GMRES to refine the solution $\Delta\theta$.

**What does $\hat{A}_J^{(\alpha)} u$ mean physically?** Suppose we nudge the network weights slightly along direction $u$: $\theta \to \theta + \epsilon u$.
1. On each sampled transition $b$, the predicted values shift to first order by:
   $$
   \Delta v(s_b) \approx \epsilon \langle \nabla \mathbf{v}(s_b),\, u \rangle, \qquad \Delta v(s'_b) \approx \epsilon \langle \nabla \mathbf{v}(s'_b),\, u \rangle.
   $$
2. Consequently, the sampled TD error $\delta_b = r_b + \gamma (1-\text{done}_b) v(s'_b) - v(s_b)$ shifts (with interpolation $\alpha$) by $-\epsilon\big(\Delta v(s_b) - \alpha\gamma(1-\text{done}_b)\Delta v(s'_b)\big)$.
3. Multiplying that shift by $\nabla \mathbf{v}(s_b)$ and averaging over the batch gives **$\hat{A}_J^{(\alpha)} u$ — the first-order change in the TD semi-gradient $\hat{g}_{\text{TD}}$ caused by moving $\theta$ in direction $u$!**

Solving $(\hat{A}_J + \lambda I)\,\Delta\theta = \hat{g}_{\text{TD}}$ therefore asks a direct Newton-like question: *"What parameter step $\Delta\theta$ produces a first-order change in the TD semi-gradient that cancels out the current $\hat{g}_{\text{TD}}$?"*

**Computing $(\hat{A}_J^{(\alpha)} + \lambda I) u$ via 1 JVP + 1 VJP:**
Think of the network evaluated on the batch of $B$ states $\{s_b\}_{b=1}^B$ as a map $f_s: \mathbb{R}^k \to \mathbb{R}^B$, $f_s(\theta) = \big(v_\theta(s_1), \dots, v_\theta(s_B)\big)^\intercal$, with $B \times k$ Jacobian $J_s$. Autodiff (JAX / PyTorch) provides two linear primitives, each costing roughly one forward/backward pass:
- **JVP (Jacobian-Vector Product, forward-mode):** Given a parameter direction $u \in \mathbb{R}^k$, computes the $B$-dimensional vector of directional value changes:
  $$
  j_s(u) \doteq \mathrm{JVP}(f_s, \theta, u) = J_s u \in \mathbb{R}^B, \qquad \text{with } b\text{-th entry } \langle \nabla \mathbf{v}(s_b),\, u \rangle.
  $$
- **VJP (Vector-Jacobian Product, reverse-mode / backprop):** Given any $B$-dimensional weight vector $w \in \mathbb{R}^B$, pulls it back to parameter space:
  $$
  \mathrm{VJP}(f_s, \theta, w) = J_s^\intercal w = \sum_{b=1}^B w_b\,\nabla \mathbf{v}(s_b) \in \mathbb{R}^k.
  $$

Expanding $\hat{A}_J^{(\alpha)} u$ shows it is literally one JVP followed by one VJP:
$$
\hat{A}_J^{(\alpha)} u = \frac{1}{B}\sum_{b=1}^B \nabla \mathbf{v}(s_b)\,\Big(\underbrace{\langle \nabla \mathbf{v}(s_b), u \rangle}_{j_s(u)_b} - \alpha\gamma(1-\text{done}_b) \underbrace{\langle \nabla \mathbf{v}(s'_b), u \rangle}_{j_{s'}(u)_b}\Big) = J_s^\intercal w.
$$
So for any search direction $u \in \mathbb{R}^k$ supplied by the solver:
1. Run two forward-mode JVPs: $j_s(u) = \mathrm{JVP}(f_s, \theta, u)$ and $j_{s'}(u) = \mathrm{JVP}(f_{s'}, \theta, u)$. *(When $\alpha = 0$, skip $j_{s'}$.)*
2. Form the $B$-vector of linearized TD-error shifts: $w = \frac{1}{B}\big(j_s(u) - \alpha\gamma(1 - \text{done}) \odot j_{s'}(u)\big) \in \mathbb{R}^B$.
3. Pull back via reverse-mode VJP and add damping: $\mathrm{VJP}(f_s, \theta, w) + \lambda u \in \mathbb{R}^k$.

---

#### 5.2 Matrix-Free Krylov Solvers: CG vs. Arnoldi / GMRES

Now plug the oracle $u \mapsto \hat{M}_{\text{op}} u$ (where $\hat{M}_{\text{op}} = \hat{A}_J^{(\alpha)} + \lambda I$) and right-hand side $b = \hat{g}_{\text{TD}}$ into a **Krylov subspace solver**. In $m$ oracle calls, a Krylov method searches for the best approximate update $\Delta\theta_m$ inside the $m$-dimensional subspace:
$$
\mathcal{K}_m(\hat{M}_{\text{op}}, \hat{g}_{\text{TD}}) \doteq \mathrm{span}\big\{\hat{g}_{\text{TD}},\; \hat{M}_{\text{op}} \hat{g}_{\text{TD}},\; \hat{M}_{\text{op}}^2 \hat{g}_{\text{TD}},\; \dots,\; \hat{M}_{\text{op}}^{m-1} \hat{g}_{\text{TD}}\big\}.
$$
- **For Symmetric Natural TD ($\alpha = 0$, so $\hat{M}_{\text{op}} = \hat{D}_J + \lambda I$):** Because $\hat{D}_J + \lambda I$ is **symmetric positive-definite**, we can use **Conjugate Gradient (CG)**, which minimizes $\frac{1}{2} \Delta\theta^\intercal (\hat{D}_J + \lambda I) \Delta\theta - \hat{g}_{\text{TD}}^\intercal \Delta\theta$ over $\mathcal{K}_m$ using only $O(1)$ memory (3 parameter vectors) and $m$ oracle calls.
- **For Non-Symmetric / Interpolated Natural TD ($\alpha > 0$, so $\hat{M}_{\text{op}} = \hat{A}_J^{(\alpha)} + \lambda I$):** Because $\hat{A}_J^{(\alpha)}$ is **non-symmetric**, CG breaks down (its orthogonality recurrences rely on symmetry). Instead, we use **GMRES (Generalized Minimal Residual)** via the **Arnoldi iteration**:
  1. Initialize the first basis vector with the normalized TD semi-gradient: $q_1 = \hat{g}_{\text{TD}} / \|\hat{g}_{\text{TD}}\|_2$.
  2. For $j = 1, \dots, m$:
     - Call the JVP+VJP oracle on $u = q_j$: $w = (\hat{A}_J^{(\alpha)} + \lambda I) q_j$.
     - Orthogonalize $w$ against $q_1, \dots, q_j$ (Gram-Schmidt), storing the dot products in column $j$ of an $(m+1) \times m$ upper-Hessenberg matrix $H$, and set $q_{j+1} = w / \|w\|_2$.
  3. By construction, $\hat{M}_{\text{op}} Q_m = Q_{m+1} H$, where the columns of $Q_m = [q_1, \dots, q_m] \in \mathbb{R}^{k \times m}$ are orthonormal.
  4. Minimizing the residual $\|\hat{M}_{\text{op}} \Delta\theta - \hat{g}_{\text{TD}}\|_2$ over $\Delta\theta = Q_m y$ reduces to a **tiny $(m+1) \times m$ least-squares problem**:
     $$
     y^* = \arg\min_{y \in \mathbb{R}^m} \|H y - \|\hat{g}_{\text{TD}}\|_2\, e_1\|_2, \qquad \Delta\theta_m = Q_m y^*.
     $$
  5. Update the parameters: $\theta_{t+1} = \theta_t - \eta\,\Delta\theta_m$.

With $m = 5\text{--}8$, the $(m+1)\times m$ least-squares solve is negligible; the entire cost is $m$ JVP+VJP calls ($\approx 2m$ forward/backward passes).

---

### 6. Scalable Layer-Wise Approximations: K-FAC, Sylvester Non-Symmetric K-FAC, and Muon

Matrix-free GMRES is exact, but running $m = 8$ JVP+VJP passes on every gradient step can be expensive for large networks. An alternative solver family approximates the preconditioner **one layer at a time** (ignoring cross-layer blocks of $D_J$ or $A_J$), reducing the per-step cost to $O(1)$ forward/backward passes while tracking curvature with an exponential moving average (EMA).

#### 6.1 K-FAC (Symmetric Layer-Wise Preconditioning for $D_J$)
Consider a single dense layer $h = W a$ with weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ and input activation $a(s) \in \mathbb{R}^{d_{\text{in}}}$. Let $\psi(s) \doteq \nabla_h v_\theta(s) \in \mathbb{R}^{d_{\text{out}}}$ denote the backpropagated pre-activation sensitivity of the value output with respect to $h$. *(We use $\psi$ rather than $\delta$ so it is never confused with the TD error $\delta_b$!)*

By the chain rule, the gradient of $v_\theta(s)$ with respect to $W$ is the outer product $\nabla_W v_\theta(s) = \psi(s)\, a(s)^\intercal$, which in vectorized form is $\mathrm{vec}(\nabla_W v_\theta(s)) = a(s) \otimes \psi(s)$. Plugging this into the layer's diagonal block of $D_J = \mathbb{E}_s[\nabla \mathbf{v}(s)\nabla \mathbf{v}(s)^\intercal]$ gives:
$$
D_{J, l} = \mathbb{E}_s\Big[\big(a(s) \otimes \psi(s)\big)\big(a(s) \otimes \psi(s)\big)^\intercal\Big] = \mathbb{E}_s\Big[\big(a(s)a(s)^\intercal\big) \otimes \big(\psi(s)\psi(s)^\intercal\big)\Big].
$$
**K-FAC (Kronecker-Factored Approximate Curvature)** makes one structural approximation—replacing the expectation of the Kronecker product with the Kronecker product of the expectations:
$$
D_{J, l} \;\approx\; A_0 \otimes G_0, \qquad A_0 \doteq \mathbb{E}_s[a(s)a(s)^\intercal] \in \mathbb{R}^{d_{\text{in}}\times d_{\text{in}}}, \qquad G_0 \doteq \mathbb{E}_s[\psi(s)\psi(s)^\intercal] \in \mathbb{R}^{d_{\text{out}}\times d_{\text{out}}}.
$$
Using the Kronecker identity $(A_0 \otimes G_0)\,\mathrm{vec}(\Delta W) = \mathrm{vec}(G_0\,\Delta W\,A_0^\intercal)$, solving the massive $(d_{\text{in}} d_{\text{out}}) \times (d_{\text{in}} d_{\text{out}})$ Symmetric Natural TD system $(D_{J,l} + \lambda I)\,\mathrm{vec}(\Delta W) = \mathrm{vec}(G_{\text{TD}})$ collapses to **two small matrix solves**:
$$
\Delta W_{\text{K-FAC}} = (G_0 + \lambda I)^{-1}\, G_{\text{TD}}\, (A_0 + \lambda I)^{-1},
$$
where $G_{\text{TD}} = -\frac{1}{B}\sum_{b=1}^B \delta_b\,\psi(s_b)\,a(s_b)^\intercal$ is simply the standard TD semi-gradient for layer $W$! Both $A_0$ and $G_0$ are maintained as smooth EMAs across minibatches, giving an $O(1)$-pass approximation to Symmetric Natural TD.

#### 6.2 Non-Symmetric K-FAC via the Generalized Sylvester Equation
Now apply the exact same Kronecker approximation to the **Interpolated / Non-Symmetric Bellman operator** $A_{J, l}^{(\alpha)} = \mathbb{E}_{s, s'}[\nabla_W v(s)(\nabla_W v(s) - \alpha\gamma \nabla_W v(s'))^\intercal]$:
$$
A_{J, l}^{(\alpha)} = \mathbb{E}_s\big[(a(s)a(s)^\intercal) \otimes (\psi(s)\psi(s)^\intercal)\big] - \alpha\gamma\,\mathbb{E}_{s, s'}\big[(a(s)a(s')^\intercal) \otimes (\psi(s)\psi(s')^\intercal)\big].
$$
Factoring each expectation separately yields a **two-term Kronecker sum**:
$$
\bbox{A_{J, l}^{(\alpha)} \;\approx\; A_0 \otimes G_0 \;-\; \alpha\gamma\, A_1 \otimes G_1,}
$$
where the two new cross-transition factors are:
$$
A_1 \doteq \mathbb{E}_{s, s'}\big[(1 - \text{done})\,a(s)\,a(s')^\intercal\big], \qquad G_1 \doteq \mathbb{E}_{s, s'}\big[(1 - \text{done})\,\psi(s)\,\psi(s')^\intercal\big].
$$
Applying $(A_0 \otimes G_0 - \alpha\gamma\, A_1 \otimes G_1)\,\mathrm{vec}(\Delta W) = \mathrm{vec}(G_{\text{TD}})$ gives a **generalized Sylvester matrix equation** for $\Delta W$:
$$
(G_0 + \lambda I)\,\Delta W\,(A_0 + \lambda I)^\intercal - \alpha\gamma\, G_1\,\Delta W\,A_1^\intercal = G_{\text{TD}}.
$$
Because $\alpha\gamma < 1$, we can solve this cleanly without any expensive matrix decomposition! Left-multiply by $\tilde{G}_0^{-1} \doteq (G_0 + \lambda I)^{-1}$ and right-multiply by $\tilde{A}_0^{-\intercal} \doteq (A_0 + \lambda I)^{-\intercal}$:
$$
\Delta W = \underbrace{\tilde{G}_0^{-1} G_{\text{TD}} \tilde{A}_0^{-\intercal}}_{\Delta W^{(0)}\;\text{(Sym. K-FAC)}} \;+\; \alpha\gamma\,(\tilde{G}_0^{-1} G_1)\,\Delta W\,(\tilde{A}_0^{-1} A_1)^\intercal.
$$
Starting from the Symmetric K-FAC solution $\Delta W^{(0)}$, just **5–6 fixed-point iterations**
$$
\Delta W^{(t+1)} = \Delta W^{(0)} + \alpha\gamma\, P_G\,\Delta W^{(t)}\, P_A^\intercal, \qquad P_G \doteq \tilde{G}_0^{-1} G_1,\quad P_A \doteq \tilde{A}_0^{-1} A_1,
$$
converge rapidly to the full Non-Symmetric K-FAC update! *(And for the final scalar value head where $d_{\text{out}} = 1$, we have $\psi(s) \equiv 1$, so $G_0 = G_1 = 1$ and the equation simplifies to a single exact linear solve $(A_0 - \alpha\gamma A_1 + \lambda I)\,\Delta W^\intercal = G_{\text{TD}}^\intercal$.)*

#### 6.3 Muon and Non-Symmetric Muon (Sylvester NS-Muon)
**What is Muon?** *Muon* (MomentUm Orthogonalized by Newton-Schulz; Jordan et al., 2024) is a lightweight matrix-aware optimizer designed specifically for 2D hidden weight matrices $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ (while falling back to standard Adam for 1D biases and small input/conv layers):
1. Given a layer's gradient (or momentum buffer) $G \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ with SVD $G = U \Sigma V^\intercal$, Muon replaces all singular values with $1$ and updates along the **nearest orthogonal matrix**:
   $$
   \Delta W_{\text{Muon}} = U V^\intercal = (G G^\intercal)^{-1/2} G.
   $$
2. Rather than running an expensive SVD, Muon computes $U V^\intercal$ in **5 fast GPU matrix multiplications** using a polynomial **Newton-Schulz iteration** $X \leftarrow a X + b(XX^\intercal)X + c(XX^\intercal)^2 X$.

**How does Muon relate to TD learning?**
- By equalizing all singular values of $G_{\text{TD}}$, Muon acts as a cheap, memoryless, **symmetric** layer-wise preconditioner (loosely akin to $(D_{J,l})^{-1/2}$). This helps undo ill-conditioned representations without storing curvature matrices.
- **Limitation 1 (Momentum):** Standard Muon uses aggressive Nesterov momentum (`b1 = 0.95`), which—as we show in Section 7—destabilizes TD when $K \neq 0$. Setting `b1 = 0.0` immediately improves Muon on irreversible MDPs.
- **Limitation 2 (Symmetry):** Because $U V^\intercal$ only orthogonalizes $G_{\text{TD}}$ symmetrically, standard Muon cannot invert the non-symmetric Bellman operator $I - \gamma P$.

**Sylvester Non-Symmetric Muon (`Sylvester NS-Muon`):** Following Muon's architectural split—applying a matrix-structured preconditioner to 2D dense layers and zero-momentum Adam (`b1 = 0.0`) to 1D biases and conv filters—we replace Muon's symmetric Newton-Schulz orthogonalization on 2D layers with the **EMA-tracked Generalized Sylvester K-FAC solve** from Section 6.2! As shown in the ablations in Section 8.2, this gives the best of both worlds: $O(1)$ per-step compute, EMA-smoothed curvature factors, and full inversion of $A_J^{(\alpha)}$.

---

### 7. Why Standard Momentum Breaks on $K \neq 0$ (and First-Order Fixes)

Before or alongside second-order preconditioning, your $S + K$ theory also reveals something surprising about **first-order optimizers** like Adam and Muon!

Consider a single complex eigenmode $z_t \in \mathbb{C}$ of the linearized TD error dynamics $\Theta(S + K)$ (or $A_J = S_J + K_J$) with eigenvalue $\nu = \alpha + i\beta$ ($\alpha > 0$, $\beta \neq 0$). In discrete time, plain TD is:
$$
z_{t+1} = (1 - \eta \nu) z_t = \big((1 - \eta \alpha) - i \eta \beta\big) z_t.
$$
Its per-step contraction factor is $|1 - \eta\nu|^2 = 1 - 2\eta\alpha + \eta^2(\alpha^2 + \beta^2)$, which is $< 1$ if and only if:
$$
\bbox{\eta < \frac{2\alpha}{\alpha^2 + \beta^2}, \qquad \min_\eta |1 - \eta\nu|^2 = 1 - \frac{1}{1 + (\beta/\alpha)^2}.}
$$
When $\beta \gg \alpha$ (strong skew-symmetry + state aliasing), the step size $\eta$ must be tiny, and even at the *optimal* $\eta$, contraction slows down by a factor of $1 + (\beta/\alpha)^2$.

#### Why Positive Momentum ($\beta_1 = 0.9$ in Adam, $\beta_1 = 0.95$ in Muon) Hurts TD
Standard Heavy-Ball / Adam momentum adds a fraction of the previous step:
$$
z_{t+1} = z_t - \eta \nu z_t + \mu (z_t - z_{t-1}), \qquad \mu > 0.
$$
- When $K = 0$ ($\beta = 0$, symmetric optimization), consecutive gradients point in the same inward direction, so $\mu > 0$ accelerates convergence.
- When $K \neq 0$ ($\beta \gg \alpha$, spiraling dynamics), the previous step $(z_t - z_{t-1})$ was mostly **tangential** (perpendicular to the origin). Adding positive momentum pushes the iterate **outward along the tangent**, shrinking the stability region and amplifying oscillations!

This directly explains why, across our benchmarks in `forks/td_optimizer/`, simply setting **Adam's $\beta_1 = 0.0$** cut TD's AUC by $3\text{--}5\times$ on irreversible chains (`p_fwd = 0.75, 1.00`), while hurting slightly on the reversible chain (`p_fwd = 0.50`, where $K \approx 0$ and positive momentum helps).

#### Negative Momentum / OGDA / Extragradient
In game theory and minimax optimization (which are also governed by $S + K$ operators!), the standard first-order fix is **Optimistic Gradient Descent (OGDA)**—equivalently, **negative momentum**:
$$
z_{t+1} = z_t - \eta (2 g_t - g_{t-1}).
$$
Taylor-expanding $g_{t-1} \approx g_t + \eta \nu^2 z_t$ gives the effective operator $\nu_{\text{eff}} = \nu - \eta \nu^2$. Plugging in $\nu = \alpha + i\beta$:
$$
\nu_{\text{eff}} = \big(\alpha - \eta \alpha^2 + \mathbf{\eta \beta^2}\big) + i\beta(1 - 2\eta\alpha).
$$
Because $i^2 = -1$, the $-\eta(i\beta)^2$ term becomes **$+\eta\beta^2$**, converting rotational energy directly into inward contraction! **Extragradient TD** ($\theta_{t+1} = \theta_t - \eta \hat{g}_{\text{TD}}(\theta_t - \tau \hat{g}_{\text{TD}}(\theta_t))$) achieves the same $+\tau \beta^2$ stabilization with one extra gradient call.

Finally, using the same JVP + VJP oracle from Section 5.1 on $u = \hat{g}_{\text{TD}}$, an optimizer can **measure online** whether the current problem is reversible ($K \approx 0$) or spiraling ($K \neq 0$) via the Rayleigh quotients:
$$
\alpha_{\text{eff}} \doteq \frac{u^\intercal \hat{S}_J u}{\|u\|_2^2}, \qquad \beta_{\text{eff}} \doteq \frac{\|\hat{K}_J u\|_2}{\|u\|_2},
$$
and turn off momentum (e.g., $\beta_{1, t} = \frac{0.9}{1 + (\beta_{\text{eff}}/\alpha_{\text{eff}})^2}$) only when rotation is present.

---

### 8. The Stochastic Noise Trade-Off and Empirical Ablations

On the **exact** MRP (`ns_natural_td.png`), Non-Symmetric Natural TD ($\alpha = 1$) converges to $10^{-4}$ in ~15 steps, beating `TD(0) + Adam`, `Symmetric Natural TD`, and even `E(0) + Adam`.

However, when we switch to **sampled minibatches** ($B = 64$, `p_fwd = 0.75`, `1-Hot + CNN (AvgPool)`) with a **constant learning rate**, a striking reversal occurs: full Non-Symmetric Natural TD ($\alpha = 1$) drops fast initially but hits a high noise floor ($\sim 9 \times 10^{-3}$), whereas Symmetric Natural TD ($\alpha = 0$) reaches a much lower noise floor ($\sim 3.5 \times 10^{-4}$).

#### 8.1 The Noise-Amplification Trade-Off ($\hat{D}_J^{-1}$ vs. $\hat{A}_J^{-1}$)
Write the minibatch TD semi-gradient explicitly in terms of sampled TD errors $\delta_b = r_b + \gamma(1-\text{done}_b) v_\theta(s'_b) - v_\theta(s_b)$:
$$
\hat{g}_{\text{TD}} = -\frac{1}{B}\sum_{b=1}^B \delta_b\,\nabla \mathbf{v}(s_b) = \underbrace{-J^\intercal D \boldsymbol{\delta}}_{-\,J^\intercal A \mathbf{e}} + \xi,
$$
where $\xi$ is zero-mean minibatch sampling noise with covariance:
$$
\Sigma_\xi \doteq \mathrm{Cov}(\xi) = \frac{1}{B}\, J^\intercal D\,\mathrm{diag}(\sigma_\delta^2)\, J \;\approx\; \frac{\sigma_\delta^2}{B}\, D_J,
$$
and $\sigma_\delta^2(s) \doteq \mathrm{Var}(\delta(s, s') \mid s)$ is the conditional variance of the sampled TD error at state $s$. Notice the crucial factor: **the TD-error noise covariance $\Sigma_\xi$ is proportional to $D_J$, NOT $A_J$!**

Now apply a general preconditioner $M$ to $\hat{g}_{\text{TD}}$ with a constant step size $\eta$:
$$
\theta_{t+1} = \theta_t - \eta M \hat{g}_{\text{TD}} = \theta_t + \eta M J^\intercal D \boldsymbol{\delta} - \eta M \xi.
$$
The covariance of the injected noise step $-\eta M \xi$ is $\eta^2 M \Sigma_\xi M^\intercal \propto \frac{\eta^2 \sigma_\delta^2}{B} M D_J M^\intercal$. Compare our two preconditioners:

1. **Symmetric Natural TD ($M = D_J^{-1}$):**
   $$
   M D_J M^\intercal = D_J^{-1} D_J D_J^{-1} = D_J^{-1}.
   $$
   In function space ($\Delta \mathbf{v} = J \Delta\theta$), the injected noise variance is:
   $$
   \mathrm{Cov}(\Delta \mathbf{v}_{\text{noise}}) \propto \frac{\eta^2 \sigma_\delta^2}{B}\, J D_J^{-1} J^\intercal = \frac{\eta^2 \sigma_\delta^2}{B}\, \Pi_{D, J} D^{-1}.
   $$
   Every representable direction gets the **exact same $O(1)$ noise floor**, completely independent of $\gamma$!

2. **Non-Symmetric Natural TD ($M = A_J^{-1}$):**
   $$
   M D_J M^\intercal = A_J^{-1} D_J A_J^{-\intercal}.
   $$
   Along a slow global mode where $A_J \approx (1 - \gamma) D_J$, the operator $A_J^{-1}$ is **$O\!\left(\frac{1}{1-\gamma}\right)$ times larger** than $D_J^{-1}$! Therefore, the injected noise variance along slow modes is amplified by:
   $$
   \bbox{\frac{\mathrm{Var}_{\text{NS-NTD}}}{\mathrm{Var}_{\text{Sym-NTD}}} \;\sim\; \frac{1}{(1 - \gamma)^2} \quad \text{(up to } 10{,}000\times \text{ at } \gamma = 0.99\text{)}!}
   $$
   In other words, $A_J^{-1}$ is a **high-gain inverse filter**: it cancels the $(1-\gamma)$ slowdown on the *signal* $J^\intercal D \boldsymbol{\delta}$, so the initial transient drops in 10 steps, **but** with a constant step size $\eta$ it amplifies the minibatch TD-error *noise* $\xi$ by $1/(1-\gamma)$, creating a high steady-state noise floor!

*(Note: In classical stochastic approximation theory, $M = A_J^{-1}$ actually achieves the **optimal asymptotic Cramér-Rao / Godambe variance** when paired with a **decaying step size** $\eta_t \to 0$ or **Polyak-Ruppert iterate averaging**—that is the famous optimality theorem for LSTD!)*

#### 8.2 Resolving the Trade-Off: Empirical Ablations (`exp_ablations.py`)
Our theory predicts three clean levers to get the fast transient of $A_J^{-1}$ **without** the constant-LR noise floor:
1. **Interpolation $\alpha \in [0, 1]$** in $(\hat{A}_J^{(\alpha)} + \lambda I)^{-1}$, trading off Symmetric ($\alpha = 0$) vs. Non-Symmetric ($\alpha = 1$).
2. **Step-size decay (Cosine LR) or Polyak-Ruppert iterate averaging**, which suppresses the injected noise $\eta_t^2 A_J^{-1} \Sigma_\xi A_J^{-\intercal}$ as training progresses.
3. **EMA-tracked curvature factors (`Sylvester NS-Muon`)**, which eliminate per-minibatch operator noise in $\hat{A}_J^{(\alpha)}$ at $O(1)$ per-step cost.

The figure below tests all three levers on the sampled 1D Chain (`p_fwd = 0.75`, $B = 64$, `1-Hot + CNN (AvgPool)`, $\gamma = 0.99$, 5 seeds; script: `forks/td_optimizer/exp_ablations.py`):

![Ablations of Interpolation Alpha, LR Schedule, Polyak Averaging, and Sylvester NS-Muon](../forks/td_optimizer/ablations.png)

Every prediction of the theory shows up cleanly in the data:
- **Left panel (AUC vs. $\alpha$):** All second-order methods crush `Adam (b1=0.9)` (dashed grey, $\sim 5\times 10^{-2}$) and `Adam (b1=0.0)` (dotted black, $\sim 1.6\times 10^{-2}$). For GMRES, moving from Symmetric ($\alpha = 0$, $\text{AUC} \approx 1.5\times 10^{-2}$) to **$\alpha = 0.8$** cuts AUC by **$\sim 3\times$** (down to $\sim 5\times 10^{-3}$ with Polyak averaging or Cosine LR—nearly **$10\times$ better than Adam `b1=0.9`**).
- **Right panel (Final Noise Floor vs. $\alpha$):**
  - `GMRES Const LR` (blue circles) climbs monotonically from $3.5\times 10^{-4}$ ($\alpha = 0$) to $9\times 10^{-3}$ ($\alpha = 1$), directly visualizing the $O(1/(1-\gamma))$ noise amplification!
  - Adding **Cosine LR** (green squares) or **Polyak-Ruppert averaging** (purple triangles) crushes the final noise floor down to **$10^{-5}\text{--}6\times 10^{-5}$** for $\alpha \in [0, 0.8]$ (a $200\times\text{--}1000\times$ improvement over Adam `b1=0.9`).
  - **`Sylvester NS-Muon Cosine`** (red triangles) is the standout practical method: using EMA-tracked two-term Kronecker factors + 6 fixed-point Sylvester steps + Cosine LR, it achieves **both near-optimal AUC ($\sim 5.5\times 10^{-3}$) and a $6\times 10^{-5}$ final noise floor even at $\alpha = 1.0$**, all at $O(1)$ per-step cost!

---

### 9. Do These Methods "Remove the Skew-Symmetric Portion" of TD?

A natural question is whether these second-order proposals literally "remove the skew-symmetric portion $K$" of TD learning, and whether a first-order method could do the same thing. The $S + K$ framework gives a precise answer:

#### 9.1 Three Distinct Ways to Deal with $K$
1. **Literally deleting $K$ from the Bellman operator ($E$-minimization):**
   $E$-minimization replaces $A = S + K$ with $S$, giving $\dot{\mathbf{e}}_E = -\Theta S \mathbf{e}$. It keeps the raw metric $\Theta = J J^\intercal$ intact and deletes $K$ directly—which is why it requires knowing (or estimating via `v_aux` / double sampling) the backward update $J^\intercal A^\intercal \mathbf{e}$.
2. **Neutralizing $K$'s ability to break contraction (Symmetric Natural TD, $M = D_J^{-1}$):**
   Symmetric Natural TD does **not** remove $K$ from the dynamics: $\dot{\mathbf{e}}_{\text{Sym-NTD}} = -\Pi_{D, J} D^{-1}(S + K)\mathbf{e}$ still contains $K$, so discrete-time iterates can still spiral. However, by removing the off-diagonal aliasing in $\Theta$, it restores $\mathbf{e}^\intercal K \mathbf{e} = 0$ in the $D$-norm, so $K$ can **no longer cancel $S$** or break continuous-time contraction.
3. **Cancelling $K$ by inverting $S + K$ on the left (Non-Symmetric Natural TD, $M = A_J^{-1}$):**
   Non-Symmetric Natural TD **does** completely eliminate the skew-symmetric rotation from the error dynamics—not by *subtracting* $K$ from $A$, but by *inverting* $S_J + K_J$ on the left:
   $$
   \dot{\mathbf{e}}_{\text{NS-NTD}} = -J\big(J^\intercal(S + K)J\big)^{-1} J^\intercal(S + K)\mathbf{e} = -\Pi_{A, J}\mathbf{e}.
   $$
   On $\mathrm{col}(J)$, every eigenvalue becomes $\nu_m = 1 + 0i$ (zero imaginary / skew-symmetric part, $\beta_m = 0$). Crucially, unlike $E$-minimization, inverting $S_J + K_J$ cancels $K$ **using only single-trajectory samples** (zero double-sampling bias and no `v_aux`).

#### 9.2 Is There a First-Order Way to Remove $K$?
It depends on what extra machinery you allow:
- **Literal removal of $K$ in first-order (requires extra structure):**
  - **$E$-minimization** ($J^\intercal S \mathbf{e} = \frac{1}{2}(J^\intercal A \mathbf{e} + J^\intercal A^\intercal \mathbf{e})$) removes $K$ cleanly, but requires double sampling or an auxiliary network `v_aux` because $J^\intercal A^\intercal \mathbf{e}$ cannot be estimated from single-sample $\delta_b$ alone.
  - **Residual Gradient / MSBE minimization** multiplies by $A^\intercal D^{-1}$ to get the symmetric PSD operator $J^\intercal A^\intercal D^{-1} A \mathbf{e}$ (zero skew-symmetry), but suffers from double-sampling bias and squares the horizon condition number to $O((1-\gamma)^2)$.
  - **Two-timescale Gradient TD (GTD2 / TDC)** avoids double-sampling bias by maintaining a fast auxiliary vector $w \approx D_J^{-1} J^\intercal A \mathbf{e}$ to follow the MSPBE gradient $A_J^\intercal D_J^{-1} J^\intercal A \mathbf{e}$ (symmetric PSD), at the cost of a two-timescale min-max optimization and a squared condition number on $A_J$.
- **Single-timescale, single-sample first-order methods (mitigate $K$, do not remove it):**
  - **OGDA / negative momentum** ($2\hat{g}_t - \hat{g}_{t-1}$), **Extragradient TD**, and **Explicit Skew-Lookahead** ($\hat{g}_{\text{TD}} - \tau \hat{K}_J \hat{g}_{\text{TD}}$) do not remove $K$ completely; instead, they act as a **first-order Taylor approximation** to inverting $I + \tau K_J$, turning leading-order skew rotation $-\tau(i\beta)^2 = +\tau\beta^2$ into inward contraction. They help significantly when $\beta / \alpha$ is moderate, but still trail $A_J^{-1}$ and $E(0)$ when $\beta \gg \alpha$.

---

### 10. Related Work and How These Methods Fit

These proposals sit at the intersection of five lines of RL and optimization research. Below is how each relates to our $S + K$ theory and algorithms:

1. **LSTD and Preconditioned Linear TD:**
   - **Bradtke & Barto (1996); Boyan (2002) (LSTD / LSTD($\lambda$)):** Solves $(\Phi^\intercal D(I - \gamma P)\Phi)\,\theta = \Phi^\intercal D \mathbf{r}$ in closed form. Non-Symmetric Natural TD ($A_J^{-1} g_{\text{TD}}$) reduces to LSTD in one step for linear features; our formulation generalizes LSTD to deep nonlinear networks via local tangent-space linearization without forming $O(k^2)$ matrices.
   - **Pan, White & White (2017) (Accelerated Gradient TD / ATD):** Probably the closest classical precursor to Non-Symmetric Natural TD. ATD preconditions linear TD updates using a regularized pseudo-inverse of the non-symmetric $A$-matrix ($\alpha_t \hat{A}_t^\dagger + \eta I$), approximated in sub-quadratic time via incremental low-rank SVD. *How our work differs:* ATD is formulated for linear function approximation with a global low-rank SVD, whereas we (i) extend non-symmetric $A_J$ preconditioning to deep nonlinear networks via exact matrix-free Arnoldi/GMRES and layer-wise **Sylvester Non-Symmetric K-FAC (`Sylvester NS-Muon`)**, (ii) connect the preconditioner directly to the $\dot{\mathbf{e}} = -\Theta(S + K)\mathbf{e}$ aliasing-vs-rotation mechanism, and (iii) identify and resolve the $O(1/(1-\gamma))$ minibatch noise-amplification trade-off via interpolation $\alpha$, cosine LR, and Polyak averaging.
   - **Geist & Pietquin (2010) (Kalman TD / KTD):** Uses nonlinear Kalman filtering (Unscented Transform) to maintain a $k \times k$ parameter covariance matrix for online value estimation, which is $O(k^2)$ in memory and impractical for modern neural networks.

2. **Symmetric Natural Gradient and Gauss-Newton TD in Deep RL:**
   - **Knight & Lerner (2018) (Natural Gradient DQN / NGDQN); Wu et al. (2017) (ACKTR):** Apply Symmetric Natural Gradient / K-FAC ($D_J^{-1}$) to deep Q-networks and actor-critic value functions, demonstrating strong stability gains (even without target networks in NGDQN). *How our work differs:* Our $S + K$ analysis gives a sharp theoretical explanation for *why* $D_J^{-1}$ stabilizes deep TD (it restores tabular geometry and makes $\mathbf{e}^\intercal K \mathbf{e} = 0$ in the $D$-norm), while also showing what $D_J^{-1}$ leaves behind ($I - \gamma P$ conditioning and tabular spiraling), motivating Non-Symmetric / Interpolated $A_J^{(\alpha)}$ preconditioning.
   - **Ke, Zhang & Wen (GNTD):** Proposes Gauss-Newton TD (and GNTD-KFAC) for nonlinear function approximation with improved sample-complexity bounds for ReLU networks. *How our work differs:* GNTD preconditions with a symmetric Gauss-Newton metric rather than inverting the non-symmetric semi-gradient Jacobian $A_J = S_J + K_J$ via GMRES or the two-term Sylvester K-FAC equation.

3. **First-Order Objective-Level Symmetrization:**
   - **Baird (1995) (Residual Gradient / MSBE); Sutton et al. (2009) (GTD2 / TDC); Maei et al. (2009) (Nonlinear GTD):** Remove skew-symmetry by minimizing squared (projected) Bellman errors ($A^\intercal D^{-1} A$ or $A_J^\intercal D_J^{-1} A_J$), *except* MSBE has double-sampling bias, GTD2/TDC requires delicate two-timescale optimization, and both *square* the horizon condition number to $O((1-\gamma)^2)$ instead of inverting it away.
   - **Tang & Munos (2025) ($E$-minimization):** Literally removes $K$ by replacing $A = S + K$ with $S$ (preserving linear $(1-\gamma)$ conditioning), *except* it requires double sampling or an auxiliary network `v_aux` and leaves $\Theta$-aliasing untouched.

4. **Interference-Aware TD Analysis:**
   - **Achiam et al. (2019); Fellows et al. (2023):** Identify off-diagonal NTK spillover $\Theta(s, s') = \langle \nabla \mathbf{v}(s), \nabla \mathbf{v}(s') \rangle$ as the mechanism behind TD divergence and target-network stabilization, *except* they treat high $\Theta(s, s')$ as uniformly harmful rather than showing its interaction with skew-symmetry $K \neq 0$, and focus on target networks / step-size bounds rather than non-symmetric preconditioning.
   - **Durugkar & Stone (2018) (CATD):** Projects out the component of $\nabla \mathbf{v}(s)$ aligned with $\nabla \mathbf{v}(s')$ per transition, *except* unconditional projection introduces double-sampling bias and degrades on stochastic/reversible transitions.

5. **Rotational Dynamics and Game Optimization:**
   - **Korpelevich (1976) (Extragradient); Daskalakis et al. (2018) (OGDA); Gidel et al. (2019) (Negative Momentum):** Show that for differentiable games with skew-symmetric Jacobians ($K \neq 0$, complex eigenvalues $\alpha + i\beta$), positive momentum destabilizes spirals while negative momentum / OGDA / extragradient converts rotation into $+\eta\beta^2$ contraction. Our $S + K$ theory shows that irreversible TD with aliased features lives in the exact same rotational regime—explaining why standard Adam (`b1 = 0.9`) and Muon (`b1 = 0.95`) underperform zero-momentum variants on directed MDPs.

---

### 11. Summary Comparison Table

| Method | Preconditioner / Solver | Error Dynamics on $\mathrm{col}(J)$ | Fixes Aliasing ($\Theta$)? | Fixes $K$-Cancellation? | Fixes $1/(1-\gamma)$ & Spiral? | Double-Sampling Bias? | Minibatch Noise & Cost |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Standard `TD(0)` + Adam** | $M = I$ (+ `b1=0.9`) | $-\Theta(S+K)\mathbf{e}$ | No | No | No | No | $O(1)$ noise; $O(1)$ passes |
| **Zero-Mom. / OGDA TD** | $M = I$ (`b1=0` / neg. mom.) | $-\Theta(S+K)\mathbf{e}$ | No | Partially (no overshoot / $+\eta\beta^2$) | Partially | No | $O(1)$ noise; $O(1)$ passes |
| **$E(0)$ Minimization** | Diff. objective ($A \to S$) | $-\Theta S \mathbf{e}$ | No | **Yes** ($K$ removed) | No ($S$ still has $1-\gamma$) | **Yes** (needs `v_aux`) | $O(1)$ noise; needs `v_aux` |
| **Symmetric Natural TD** | $(\hat{D}_J + \lambda I)^{-1}$ (CG / K-FAC) | $-\Pi_{D, J} D^{-1}(S+K)\mathbf{e}$ | **Yes** | **Yes** ($\mathbf{e}^\intercal K \mathbf{e} = 0$ in $\|\mathbf{e}\|_D^2$) | No (tabular $I-\gamma P$ remains) | No | **$O(1)$ noise**; $m$ passes (CG) or $O(1)$ (K-FAC) |
| **Non-Sym. / Interp. Natural TD (GMRES)** | $(\hat{A}_J^{(\alpha)} + \lambda I)^{-1}$ (GMRES) | $-\Pi_{A, J} \mathbf{e}$ ($\alpha=1$) | **Yes** | **Yes** ($\beta_m = 0$) | **Yes** ($\nu_m = 1 + 0i$) | No | Needs $\alpha \approx 0.8$ + Cosine/Polyak; $m$ passes |
| **Sylvester NS-Muon (NS K-FAC)** | Layer-wise $A_0\otimes G_0 - \alpha\gamma A_1\otimes G_1$ | Approx. $-\Pi_{A, J} \mathbf{e}$ | **Yes** | **Yes** | **Yes** | No | **Low noise ($6\times 10^{-5}$ w/ Cosine)**; **$O(1)$ passes** |
