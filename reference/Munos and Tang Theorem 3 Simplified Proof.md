### **Theorem 3 (Standard RL Setup)**

**Statement:** Assume the Markov chain is reversible, i.e., $D^\pi P^\pi = (P^\pi)^T D^\pi$, then the end-to-end linear TD dynamics is effectively gradient descent on the error function $E(\Phi_t, w_t)$, i.e.,

$$\begin{aligned} \frac{dw_t}{dt} &= -\eta_w \cdot \partial_{w_t} E(\Phi_t, w_t) \\ \frac{d\Phi_t}{dt} &= -\eta_\Phi \cdot \partial_{\Phi_t} E(\Phi_t, w_t) \end{aligned}$$

As a result, the weighted value approximation error is non-increasing $dE(\Phi_t, w_t)/dt \le 0$. If $(\Phi_t, w_t)$ is not at a critical point of the learning dynamics, i.e. when $\frac{d\Phi_t}{dt} \neq 0$ or $\frac{dw_t}{dt} \neq 0$, then $dE(\Phi_t, w_t)/dt$ is strictly negative.

### **Proof**

For notational simplicity, let $A = D^\pi(I - \gamma P^\pi)$. Under the reversibility assumption, $A$ is symmetric $A^T = A$. We also define the prediction error vector $v_t := \Phi_t w_t - V^\pi$. As a result, the weighted error rewrites as:

$$E(\Phi_t, w_t) = \frac{1}{2} v_t^T A v_t$$

Recall by definition of the end-to-end linear TD dynamics, we have:

$$\begin{aligned} \frac{d\Phi_t}{dt} &= -\eta_\Phi \cdot A (\Phi_t w_t - V^\pi) w_t^T = -\eta_\Phi A v_t w_t^T \\ \frac{dw_t}{dt} &= -\eta_w \cdot \Phi_t^T A (\Phi_t w_t - V^\pi) = -\eta_w \Phi_t^T A v_t \end{aligned}$$

In order to derive $\partial_{\Phi_t} E(\Phi_t, w_t)$ and $\partial_{w_t} E(\Phi_t, w_t)$, note a few useful facts as follows,

$$\begin{aligned} \partial_{\Phi_t} E(\Phi_t, w_t) &= \frac{1}{2}(A + A^T) v_t w_t^T = A v_t w_t^T \\ \partial_{w_t} E(\Phi_t, w_t) &= \frac{1}{2} \Phi_t^T (A + A^T) v_t = \Phi_t^T A v_t \end{aligned}$$

As a result, we can verify

$$\begin{aligned} \frac{d\Phi_t}{dt} &= -\eta_\Phi \partial_{\Phi_t} E(\Phi_t, w_t) \\ \frac{dw_t}{dt} &= -\eta_w \partial_{w_t} E(\Phi_t, w_t) \end{aligned}$$

where this comes from the fact that $A = A^T$. Now with chain rule, we have

$$\frac{dE(\Phi_t, w_t)}{dt} = \text{Tr}\left((\partial_{\Phi_t} E(\Phi_t, w_t))^T \frac{d\Phi_t}{dt}\right) + (\partial_{w_t} E(\Phi_t, w_t))^T \frac{dw_t}{dt}$$

$$= -\left(\frac{1}{\eta_\Phi} \left\|\frac{d\Phi_t}{dt}\right\|_2^2 + \frac{1}{\eta_w} \left\|\frac{dw_t}{dt}\right\|_2^2\right) \le 0$$

which is strictly negative if $(\Phi_t, w_t)$ is not at a critical point. The proof is hence concluded.