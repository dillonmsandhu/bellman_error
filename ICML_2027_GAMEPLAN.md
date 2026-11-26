# ICML 2027 Paper Gameplan: Bellman-Smoothed Regression for Value Function Estimation

**Target:** ICML 2027 (Deadline: ~Late January 2027 — ~8–9 weeks remaining from Nov 24, 2026)  
**Core Rule:** Ruthless scope discipline. Every experiment run in the next 4 weeks must directly serve **one unified, predictive scientific story**.

---

## 1. The Unified Paper Narrative

Replace the older `||K|| / ||S||` (skew-symmetry alone) framing in `reference/paper/main.tex` with the predictive theory from `investigation/Investigation_into_TD_vs_E.pdf`:

1. **Mechanism — Why & When Deep TD Fails (Contraction Cancellation):**
   - Under continuous-time NTK error dynamics $\dot{\mathbf{e}} = -\Theta(S + K)\mathbf{e}$, TD's $S$-norm contraction rate splits into:
     - **Contraction ($\text{Term 1}$):** $\|J^\top S \mathbf{e}\|_2^2$, driven by **gradient differences** $\nabla v(s) - \nabla v(s')$.
     - **Rotation / Cancellation ($\text{Term 2}$):** $-\langle J^\top S \mathbf{e}, J^\top K \mathbf{e}\rangle$, driven by **gradient sums** $\nabla v(s) + \nabla v(s')$.
   - **Key Predictive Insight:** TD does *not* fail from irreversibility ($K \neq 0$) alone, nor from feature aliasing alone—it fails from their **interaction** (irreversible dynamics + consecutive-state gradient aliasing $\nabla v(s) \approx \nabla v(s')$).
   - This predicts **both** where $E$-minimization should win (continuous-state MLPs like Brax, `(x, y)` representations, pooled CNNs) **and** where TD already contracts cleanly (DAGs, sharp/localized grid features like standard MinAtar CNNs), where $v_{\text{aux}}$ pays pure lag overhead for little contraction benefit.
2. **Method — Unbiased Sampled $E(\lambda)$ via Surgical $v_{\text{aux}}$:**
   - $E(\lambda)$ decomposes into a weighted MC anchor + multi-step geometric-jump Dirichlet energy ($m \sim \text{Geom}(1 - \gamma\lambda)$).
   - Surgical $v_{\text{aux}}$ resolves double-sampling bias in Term (1) while preserving the exact SPD operator $S$ and keeping variance $\mathcal{O}(1)$ independent of horizon.
3. **Validation — Exact Diagnostics $\to$ Deep RL Scaling:**
   - **Exact/Tabular:** 1D Chain ($3 \times 3$ sweep) + 6 environments $\times$ 3 representations showing Contraction Cancellation predicts `AUC(E) / AUC(TD)` ($\rho = -0.60$), plus Exact PPO.
   - **Estimator Quality:** Sampled $E(0)$ and $E(\lambda)$ gradient cosine similarity / MSE vs. analytical $\nabla E$, plus BSMC vs. $E(\lambda)$ ablation.
   - **Deep RL (V-PPO):** Brax (9–10 envs, where $E(\lambda)$ achieves ~138% Mean / ~125% IQM over $\mathrm{TD}(\lambda)$) + MJX locomotion + TDRC baseline + Deep RL $\bar{r}_{\text{rot}}$ diagnostic (and optionally Rotation-Gated $E$ or a controlled CNN vs. Pooled-CNN visual aliasing test).

---

## 2. Scope Triage: Keep, Timebox, Cut

### KEEP (Core Paper)
- **$V$-PPO $E(\lambda)$ with surgical $v_{\text{aux}}$** (primary Deep RL algorithm).
- **Contraction Cancellation Theory** (`investigation/Investigation_into_TD_vs_E.pdf`) as Section 3.
- **BSMC vs. $E(\lambda)$** as a 1-figure ablation (already complete; disentangles MC anchor weight $(1 - \tilde{\gamma})$ from geometric jumps $P_\lambda$).
- **Double-Sampling Baselines (MSBE Baird Double, MSPBE Tangent CG+HVP)** from existing Brax sweep (compact table in main text or appendix).

### ADD (High ROI, Low Cost)
- **$\bar{r}_{\text{rot}}$ Online Rotation Diagnostic** (`compute_online_rotation_ratio` from `random_policy/rotation_gated_E.py`) measured across Brax, MinAtar, and MJX to verify in Deep RL whether $\bar{r}_{\text{rot}}$ predicts $E(\lambda)$ vs. $\mathrm{TD}(\lambda)$ win margins—and specifically to diagnose why MJX Hopper Stand / Finger Spin and MinAtar favor $\mathrm{TD}(\lambda)$.
- **TDRC Baseline** (Ghiassian et al., 2020) on Brax V-PPO: canonical auxiliary-head gradient-TD baseline ($h \approx \delta$ with $\ell_2$ regularization). Directly neutralizes the *"you added an auxiliary network"* critique by contrasting $h \approx \delta$ (MSPBE / TD fixed point) against $v_{\text{aux}} \approx V$ ($E$-minimization / removing $K$).

### TIMEBOX (Strict 4–5 Day Cap)
- **Rotation-Gated $E$ in V-PPO** (`random_policy/rotation_gated_E.py` ported to Brax/MJX V-PPO):
  - *Goal:* Automatically set $\omega \to 0$ when $\bar{r}_{\text{rot}}$ is low (eliminating $v_{\text{aux}}$ lag on easy-for-TD tasks like MinAtar or MJX Hopper Stand / Finger Spin) and $\omega \to 1$ when $\bar{r}_{\text{rot}}$ is high.
  - *Decision Rule:* If it cleanly improves the Brax + MJX win/loss record within 1 week, include as Section 5. Otherwise, **stop**, keep $\bar{r}_{\text{rot}}$ as a diagnostic tool only, and defer adaptive gating to future work.

### CUT (Do Not Spend More Time Before Deadline)
- **Q-PPO (Deterministic $Q$-network):**
  - Trails $V$-PPO $E(\lambda)$ on 8/10 Brax environments (`E_min/paper/results_section.tex`).
  - Even if probe count $K$ is increased (at $K=2$, only 1 probe action estimates $\mathbb{E}_{a'}[Q(s', a')]$), Q-PPO still suffers from (i) higher-dimensional $(s, a)$ function approximation (hurting Ant/Humanoid), and (ii) losing multi-step $E(\lambda)$ without an auxiliary head (since intermediate actions $a_{t+1..t+m-1}$ are stochastic even in deterministic MDPs).
  - *Action:* Drop from main paper; at most a 2-sentence remark in Discussion/Appendix.
- **On-Policy SAC:** Custom non-standard setup where $\mathrm{TD}(\lambda)$ wins 8/10 on Brax due to 16-epoch target recomputation. Cut to keep the paper focused on PPO.
- **Atari:** Too slow/tuning-heavy for a 2-month window.

---

## 3. Open Puzzles to Diagnose Immediately (Week 1)

1. **Why did $E(\lambda)$ win on Brax but lose on simple MJX tasks (Hopper Stand, Finger Spin)?**
   - *Check 1 (Rotation Diagnostic):* Compute $\bar{r}_{\text{rot}}$ on Brax vs. MJX Hopper Stand / Finger Spin. Does MJX's different observation normalization (or DM Control's dense $[0,1]$ reward / no-termination structure) produce lower $\bar{r}_{\text{rot}}$?
   - *Check 2 ($v_{\text{aux}}$ Lag / Normalization Bug):* Log `eval_gap_v_v_aux` and `aux_loss` on MJX. If observation/reward normalization or `CRITIC_LR` / `AUX_VF_COEF` differs from Brax, $v_{\text{aux}}$ may simply be lagging behind a shifting reward/value scale.
2. **MinAtar (and Controlled Visual Aliasing):**
   - Standard MinAtar CNN likely has *low* $\bar{r}_{\text{rot}}$ (sharp 10×10 binary grids, just like tabular Space Invaders had 0% contraction cancelled), so $E(\lambda)$ losing slightly to $\mathrm{TD}(\lambda)$ due to $v_{\text{aux}}$ lag is actually **consistent with the theory**.
   - *Test:* Measure $\bar{r}_{\text{rot}}$ on MinAtar with **Standard CNN vs. Pooled CNN** (or pick *one* of Craftax/Octax that already runs cleanly in `E_min`). Only run a full visual PPO sweep if $\bar{r}_{\text{rot}}$ confirms high contraction cancellation!

---

## 4. Target 8-Page ICML Paper Outline

- **Section 1: Introduction** *(Currently empty in `main.tex`)* — The puzzle of when deep TD succeeds vs. fails; preview Contraction Cancellation and $E(\lambda)$ minimization.
- **Section 2: Preliminaries & Bellman-Smoothed Value Error** — $A = S + K$, $E = \mathbf{e}^\top S \mathbf{e}$, $E(\lambda)$.
- **Section 3: When Does Deep TD Fail? (Contraction Cancellation)** — NTK error dynamics, gradient difference vs. gradient sum decomposition, Chain $3 \times 3$, 6-env tabular scatter ($\rho = -0.60$), Exact PPO.
- **Section 4: Unbiased Sampled $E(\lambda)$ via Surgical $v_{\text{aux}}$** — Double sampling, surgical $v_{\text{aux}}$ loss, geometric jumps $P_\lambda$, gradient alignment table, BSMC vs. $E(\lambda)$ ablation.
- **Section 5: Deep RL Experiments (PPO)** — Brax V-PPO (+ MJX locomotion) comparing $E(\lambda)$, $\mathrm{TD}(\lambda)$, TDRC, and BSMC (Mean / IQM / Performance Profiles) + Deep RL $\bar{r}_{\text{rot}}$ diagnostic (+ Rotation-Gated $E$ *only if* Week 1–2 test succeeds + controlled visual aliasing experiment).
- **Section 6: Related Work & Discussion** *(Currently `[TODO]` in `main.tex`)* — Tang et al. (2023), Tsitsiklis & Van Roy (1997), Baird (1995), GTD/TDC/TDRC (Sutton, Maei, Ghiassian et al. 2020), deep TD feature/capacity literature, limitations ($v_{\text{aux}}$ lag).

---

## 5. 8-Week Execution Timeline

| Weeks | Focus | Concrete Deliverables & Exit Criteria |
| :--- | :--- | :--- |
| **Weeks 1–2** | **Diagnostics + TDRC + Timeboxed Gate** | 1. Run $\bar{r}_{\text{rot}}$ + `v_aux` lag diagnostic on Brax, MinAtar, and MJX (explain Hopper Stand / Finger Spin).<br>2. Implement & run **TDRC** on Brax V-PPO (9–10 envs).<br>3. 4–5 day timeboxed test of **Rotation-Gated $E$** on Brax/MJX V-PPO (keep or revert to diagnostic-only at end of Week 2). |
| **Weeks 3–4** | **Final Experiments & Data Freeze** | 1. Finalize MJX locomotion sweep.<br>2. Run **at most one** controlled visual aliasing experiment (CNN vs. Pooled CNN on MinAtar or Craftax/Octax, gated by $\bar{r}_{\text{rot}}$ check).<br>3. Lock all seeds (5–10 seeds), tables, and publication figures (`rliable` IQM + performance profiles). **Hard experiment freeze at end of Week 4.** |
| **Weeks 5–6** | **Full Draft Writing** | 1. Integrate `Investigation_into_TD_vs_E.pdf` as Section 3 of `main.tex`.<br>2. Write Section 1 (Introduction), Section 5 (Deep RL Experiments), and Section 6 (Related Work & Discussion).<br>3. Resolve all `[TODO]`s in `main.tex`. |
| **Weeks 7–8** | **Polish, Proofs & Advisor Passes** | 1. Complete and verify all Appendix proofs & hyperparameter tables.<br>2. Tighten main body to 8 pages (ICML 2027 style file).<br>3. Run two full revision passes with Ronald Parr before submission. |
