# Covariant Bitemporal Geometrodynamics (CBG) — Cycle I Foundations
### Deductive Architecture (`[A1]–[A3]` $`\to`$ `[D]` $`\to`$ `[H1]–[H7]`, `[H_aux]` $`\to`$ `[D|Hn]` $`\to`$ `[N]`) & Symbolic/Numerical Verification Suites

[![CBG Verification Suite](https://github.com/CBG-physics/cbg-cycle1-foundations/actions/workflows/verify_cbg.yml/badge.svg)](https://github.com/CBG-physics/cbg-cycle1-foundations/actions/workflows/verify_cbg.yml)
[![Paper 0 DOI](https://img.shields.io/badge/Zenodo%20Paper%200-10.5281%2Fzenodo.23829404-blue)](https://doi.org/10.5281/zenodo.23829404)
[![Paper I DOI](https://img.shields.io/badge/Zenodo%20Paper%20I-10.5281%2Fzenodo.23443511-blue)](https://doi.org/10.5281/zenodo.23443511)
[![Paper II DOI](https://img.shields.io/badge/Zenodo%20Paper%20II-10.5281%2Fzenodo.23451973-blue)](https://doi.org/10.5281/zenodo.23451973)
[![Paper III DOI](https://img.shields.io/badge/Zenodo%20Paper%20III-10.5281%2Fzenodo.23458461-blue)](https://doi.org/10.5281/zenodo.23458461)

**Author:** Manuel Jiménez Velasco (*M.Sc. in Civil Engineering, UPC & Independent Researcher in Theoretical Physics, Madrid, Spain — `manujimevela@gmail.com`*)

---

## 1. Overview & Deductive Epistemology

This repository archives the LaTeX monographs, compiled PDFs, and self-contained 64-bit Python symbolic (`SymPy`) and numerical (`SciPy` `solve_bvp` / `DOP853` / `RK45` / `brentq`) verification suites for **Cycle I** of **Covariant Bitemporal Geometrodynamics (CBG)** on the 5-dimensional warped orbifold bundle:

```math
\mathcal{M}_{3,2} = \left(\mathbb{R}_{t_1} \times_{\Phi} S^1_{t_2}/\mathbb{Z}_2\right) \times \Sigma_3, \qquad \boldsymbol{\eta}_5 = \mathrm{diag}(-1, -1, +1, +1, +1), \qquad (d_t, d_s, D) = (2, 3, 5).
```

The corpus is organized as a **Directed Acyclic Graph (DAG)** under an explicit **Anti-Promotion Lock** across five epistemological tiers. A statement never carries a higher tier than the weakest statement it depends on.

1. **`[A1]–[A3]` — Closed Axiomatic Core.** The three structural axioms of *Paper 0*. Axioms are primitives and are never claimed as derived.
2. **`[D]` — Unconditional Mathematical Deductions.** Exact symbolic theorems proved from `[A1]–[A3]` with no constitutive hypothesis, no empirical constant and no $`\hbar`$. This tier includes identities that hold *on the class of $`t_2`$-independent configurations*: they are theorems about that class. The separate physical statement that only this class is realised is root hypothesis `H1`.
3. **`[H1]–[H7]` and `[H_aux.1]–[H_aux.3]` — Constitutive Hypotheses.**
   - **`[H1]–[H7]`: seven groups of root hypotheses.** Each group collects the bridge postulates of one physical sector; a group may contain more than one independent postulate. None is derived from `[A1]–[A3]`. For `H1`, `H2` and `H7` an explicit obstruction is proved; for `H3`–`H6` the axioms leave the stated choice undetermined.
   - **`[H_aux.1]–[H_aux.3]`: auxiliary ansätze and initial-data benchmarks** disclosed in *Papers I–III*. They are hypotheses of the same standing as the roots; results that use them are tagged accordingly.
4. **`[D | Hn, …]` — Conditional Mathematical Deductions.** Exact symbolic theorems that follow once the listed hypotheses are assumed. Every hypothesis a result depends on is listed in its tag. A cascade is promoted to `[D]` only if *all* hypotheses in its tag are derived.
5. **`[N]` and `[N | Hn, …]` — Empirical Inputs & Numerical Evaluations.** External CODATA 2022 and survey inputs (`[N]`), and every floating-point result computed under the listed hypotheses (`[N | Hn, …]`). This includes machine-precision BVP/ODE solutions (`solve_bvp` / `DOP853`, constraint and finite-difference residuals $`\le 10^{-9}`$): a numerical solution is evidence at stated precision, not an exact deduction. The monograph suites label some of these solves `[D]` internally; this README applies the stricter rule throughout.

```text
        [A1] 5D Orbifold Arena M_{3,2}     [A2] Spectral Action     [A3] Horizon Regularity
                          \                         |                         /
                           \                        v                        /
                            +--------->  [D] Unconditional Theorems  <------+
                                         (Paper 0: 24 THMs, 94 CORs)
                                                    |
                +-----------------------------------+-----------------------------------+
                |                                                                       |
                v                                                                       v
   [H1]-[H7] Root Hypothesis Groups                                     [N] CODATA & Survey Inputs
   [H_aux.1-3] Auxiliary Ansaetze                                       (c, hbar, G, alpha_EM, H_0, z_*)
                |                                                                       |
                v                                                                       |
   [D | Hn, ...] Conditional Cascades  -------------------------------------------------+
   (Exact SymPy Identities)                                                             |
                                                                                        v
                                                            [N | Hn, ...] BVP/ODE Solutions & Predictions
```

---

## 2. Tier 0: The Closed Axiomatic Core (`[A1]–[A3]`)

- **`[A1]` (Axiom 1.1 — The 5D Bitemporal Orbifold Arena).** Spacetime is the 5-dimensional warped orbifold bundle $`\mathcal{M}_{3,2} = (\mathbb{R}_{t_1} \times_{\Phi} \mathcal{S}_{t_2}) \times \Sigma_3`$ over a spatial 3-manifold $`\Sigma_3`$, with compact secondary temporal orbifold fiber $`\mathcal{S}_{t_2} \equiv S^1_{t_2}/\mathbb{Z}_2`$ at 1-based coordinate index $`2`$ (0-based index 1), tangent signature $`\boldsymbol{\eta}_5 = \mathrm{diag}(-1, -1, +1, +1, +1)`$, and orbifold reflection $`\mathcal{R}: t_2 \mapsto -t_2`$ with $`\mathbf{R}_{\mathbb{Z}_2} = \mathrm{diag}(1, -1, 1, 1, 1)`$.
- **`[A2]` (Axiom 4.1 — Spectral Action & UV Winding Finiteness).** The effective action is generated by the Wick-rotated spectral trace $`\mathrm{Tr}\,f(\widehat{\mathcal{D}}_{\text{JV}}^2 / \Lambda_{\text{cut}}^2)`$ of the 5D Dirac–Kostant operator $`\widehat{\mathcal{D}}_{\text{JV}}`$ with non-vanishing cutoff moments $`f_{4-2k} \neq 0`$ ($`k \in \{0, 1, 2\}`$). Untwisted $`\mathrm{Sym}^2_+`$ modes at $`k \in \{0, 1\}`$ and twisted $`\mathrm{Sym}^2_-`$ modes at $`k = 2`$ give finite non-zero winding moments $`0 < |\mathcal{M}_k| < \infty`$ spanning $`\mathcal{W}_3 = \mathrm{Span}\{e^{-2\psi}, e^{-4\psi}, e^{-6\psi}\}`$, with $`\psi \equiv \ln(\Phi/\Phi_\infty)`$.
- **`[A3]` (Axiom 5.3 — Killing Horizon Curvature Regularity & Equilibrium-Root Coupling).** Every stationary geometry with $`\psi(r) \neq \text{const}`$ across a Killing horizon $`r_h > 0`$ satisfies $`|\mathcal{K}_5(r_h)| < \infty`$, and the khronon norm

  $`\displaystyle \mathcal{A} \equiv \left(-g^{AB}\partial_A\Theta^{(1)}\partial_B\Theta^{(1)}\right)^{-1}`$

  enters $`\mathcal{V}_{\text{spec}}(\psi, \mathcal{A})`$ solely through its equilibrium root $`\psi_t(\mathcal{A})`$.

---

## 3. Tier 1: Unconditional Mathematical Theorems (`[D]`)

Proved symbolically in `Paper_0_Principia_Geometrica/` (**24 Theorems, 94 Corollaries**) and in the `Module 0` engines of *Papers I–III*.

### 3.1 Algebra of the Tangent Split, Gradings, $`\mathrm{Cl}(3,2)`$ and the DeWitt–Weyl Identity
- **Dimensions and symplectic multiplicity (`THM #01, #03`).** `[A1]` fixes $`(d_t, d_s, D) = (2, 3, 5)`$, hence $`\dim\mathrm{Sym}^2(V_5^*) = 15`$ and $`\mathcal{N}_{\text{Cartan}} = \mathrm{Tr}(\mathbf{J}_{30}^T\mathbf{J}_{30}) = 30`$ with $`\mathbf{J}_{30}^2 = -\mathbf{I}_{30}`$. Two consistency relations are satisfied by this triple and by no other: $`\dim\Lambda^2(\mathbb{R}^{d_s}) = d_s`$ and $`d_t d_s D = D(D+1)`$, the latter equivalent to $`(d_t - 1)(d_s - 1) = 2`$.
- **Two distinct gradings (`THM #02, #04`).**
  - Under the time/space involution $`\mathbf{F} = \mathrm{diag}(-1,-1,1,1,1)`$: $`\mathrm{End}(V_5) = 13_+ \oplus 12_-`$, with $`13 = d_t^2 + d_s^2`$ and $`12 = 2 d_t d_s`$.
  - Under the orbifold reflection $`\mathbf{R}_{\mathbb{Z}_2} = \mathrm{diag}(1,-1,1,1,1)`$: $`\mathrm{Sym}^2(V_5^*) = 11_+ \oplus 4_-`$ (ten $`g_{\mu\nu}`$ and $`g_{22}`$ even; four $`g_{\mu 2}`$ odd), and $`V_5^* \otimes V_5^* = 17_+ \oplus 8_-`$ (11 symmetric and 6 antisymmetric even components).
- **Real Clifford structure (`THM #05, #06`).** $`\mathrm{Cl}(3,2) \simeq \mathrm{Mat}(4,\mathbb{R}) \oplus \mathrm{Mat}(4,\mathbb{R})`$. It has an irreducible real $`4 \times 4`$ representation, corresponding to one of the two factors, with central volume element $`\Gamma_{\text{vol}} = +\mathbf{I}_4`$ and $`\mathrm{Spin}_0(3,2) \simeq \mathrm{Sp}(4,\mathbb{R})`$.
- **DeWitt–Weyl identity (`THM #04, #09, #12–#14`).** Inverting the $`15 \times 15`$ DeWitt supermetric at $`\lambda = 1/2`$ gives the inverse trace coefficient $`c_{\text{DW}}^{-1} = -\frac{1}{D-2} = -\frac{1}{3}`$. The fiber reduction of the 5D curvature gives $`c_{\text{Weyl}} = 1`$, $`\kappa_\psi = -\frac{3}{2}`$ and $`c_V = \frac{3}{4}`$; the zero-mode 5D geodesic slope is $`\beta_0 = +\frac{1}{2}`$. These satisfy

  $`\displaystyle \alpha_0^2 = \frac{\beta_0^2}{-\kappa_\psi/2} = \frac{c_{\text{Weyl}}^2}{-2\kappa_\psi} = \frac{1}{3} = -c_{\text{DW}}^{-1}.`$

  The number $`\alpha_0^2 = 1/3`$ is `[D]`. Its realisation as a force on matter belongs to `[D | H1, H2]`.

### 3.2 Spectral Geometry, 5D Spinor Parity and Horizon Regularity
- **Spectral invariants (`THM #15–#19, #22`).** Curved spinorial tetrad postulates $`\nabla_M^{\text{spin}}\Gamma^N = \mathbf{0}_4`$; Lichnerowicz weights $`(c_{R_5}, c_{\text{Pauli}}) = (\frac{1}{4}, \frac{1}{2})`$; $`c_{\text{GB}} = 8`$; $`n`$-reflection boundary holonomy factor $`\Phi_\infty^{2n}`$; $`-\zeta_{\text{orb}}'(0) = \ln(2\pi L_0\Phi)`$; twisted moment $`\mathcal{M}_2^{(-)} = \frac{\ln 2}{16\pi^2}`$; 2-jet determinant $`\det\mathbf{M}_{\text{jet}}^{\text{raw}} = -16`$ (scaled $`-\frac{32}{3}`$); barrier polynomial $`\Pi_2(\psi) = \frac{1}{8}e^{-6\psi}(e^{2\psi}-1)^2`$ with maximum $`\Pi_2(\frac{1}{2}\ln 3) = \frac{1}{54}`$; ball-overlap jump of the normalised mutual-energy kernel $`\Delta f_{\text{mut}}^{(4)}(1) = -30`$ (equivalently $`\Delta f^{(4)}(1) = +36`$ for the overlap function $`f = \frac{6}{5}[1 - f_{\text{mut}}]`$ of *Paper I*); supertrace $`\mathrm{Str}(e^{-s\Delta}) = +1`$.
- **Zero-mode Dirac reduction and bulk-mass parity.** On $`ds_5^2 = -dt_1^2 - \Phi(t_1, x)^2 dt_2^2 + d\mathbf{x}^2`$, the rescaled zero-mode spinor $`\Psi_5 = \Phi^{-1/2}\chi_4`$ satisfies $`\widehat{\mathcal{D}}_5(\Phi^{-1/2}\chi_4) = \Phi^{-1/2}\widehat{\mathcal{D}}_4\chi_4`$: the fiber function $`\Phi`$ drops out. (The verification script proves this for the stated metric; the extension to a general curved 4D block is not yet in the scripts.) Under $`\mathcal{R}: t_2 \mapsto -t_2`$, a constant bulk mass bilinear $`\bar{\Psi}_5\Psi_5`$ is $`\mathbb{Z}_2`$-odd relative to the kinetic bilinear and is projected out. Since `[A2]` contains no fermionic action, `[A1]–[A3]` do not fix how matter couples to the fiber: this is why the matter coupling is placed in `H2`.
- **Two-sided horizon regularity (`THM #20, #22`).** The stationary off-diagonal equation gives $`h'(r) \equiv 0`$ wherever $`\partial_{\mathcal{A}}V \neq 0`$. For any smooth $`V(\psi, \mathcal{A})`$, finiteness of $`\psi''(r_h)`$ fixes the regular slope $`\psi'(r_h) = \frac{2r_h(V - V_\psi)}{r_h^2(2V + V_\psi) - 3}`$ on both sides of the horizon and cancels the double pole of the $`B'(r)`$ equation ($`\mathcal{C}_{B^2} + 1/b_{-1} \equiv 0`$).
- **Chern–Simons identity on zero modes.** For $`t_2`$-independent, $`\mathbb{Z}_2`$-even potentials ($`A_2 = 0`$): $`\epsilon^{ABCDE}A_A F_{BC}F_{DE} \equiv 0`$.

### 3.3 Homogeneous Identities and the Khronon Charge
- **Noether identity and conserved khronon charge (`THM #21`).** From the minisuperspace Lagrangian $`L_{\text{mini}}[N, a, \psi, \Theta^{(1)}]`$ with $`\mathcal{A} = N^2/\dot{\Theta}^2`$: $`\dot{\Theta}\mathcal{E}_\Theta - N\dot{\mathcal{E}}_N + \dot{a}\mathcal{E}_a + \dot{\psi}\mathcal{E}_\psi \equiv 0`$, and since $`\Theta^{(1)}`$ is cyclic,

  $`\displaystyle J_\Theta \equiv \frac{\partial L_{\text{mini}}}{\partial\dot{\Theta}^{(1)}} = 4\,a^3 e^\psi \mathcal{A}^{3/2}\,\frac{\partial V}{\partial\mathcal{A}} = \text{const}, \qquad 2\mathcal{A}\frac{\partial V}{\partial\mathcal{A}} = \frac{J_\Theta}{2\,a^3 e^\psi \sqrt{\mathcal{A}}}.`$

- **Turning-point identity.** With a radiation fluid ($`\rho_r \propto a^{-4}e^{-\psi}`$), at a point where $`\dot{a} = \dot{\psi} = 0`$ (lapse $`N = 1`$):

  $`\displaystyle \frac{\ddot{a}}{a} = \frac{1}{3}\left[V_\psi - \rho_r - 2\mathcal{A}V_{\mathcal{A}}\right], \qquad \ddot{\psi} = \frac{2}{3}\left[V - V_\psi + \mathcal{A}V_{\mathcal{A}}\right].`$

- **Shear first integral (`THM #21`).** On $`k = 0`$ Bianchi-I slices, $`\frac{d}{dt_1}(a^3\Phi\sigma^i{}_j) = 0`$, hence $`\rho_\sigma = \Sigma_0^2 a^{-6}e^{-2\psi}`$.
- **Anisotropic closed slices.** On homogeneous anisotropic $`S^3`$ the spatial curvature is $`{}^3R = \frac{2(a_1^2a_2^2 + a_2^2a_3^2 + a_3^2a_1^2) - a_1^4 - a_2^4 - a_3^4}{2a_1^2a_2^2a_3^2}`$, which depends on the anisotropy.

---

## 4. Tiers 2 & 3: Constitutive Hypotheses and Conditional Cascades

### 4.1 The Seven Root Hypothesis Groups (`[H1]–[H7]`)

| Root | Hypothesis group | Why it is not `[D]` | Conditional cascade |
| :--- | :--- | :--- | :--- |
| **`H1`** | **Projective cylinder restriction.** Physical fields are $`t_2`$-independent: $`\mathcal{L}_{\xi_{(2)}}g_{AB} = 0`$. | *Obstruction:* `[A1]–[A3]` fix the $`\mathbb{Z}_2`$ parity, not the mode number. The $`n \ge 1`$ tower has $`m_n^2 = -n^2/(\Phi_0 L_0)^2 < 0`$. | **`[D\|H1]`:** removal of the $`n \ge 1`$ modes; zero-mode propagation $`\omega^2 = k^2`$ ($`c_T = 1`$); no Chern–Simons photon term. |
| **`H2`** | **Matter coupling and fiber bridge.** (a) Matter couples to $`g_{\mu\nu}`$ (Branch A) rather than to $`e^\psi g_{\mu\nu}`$ (Branch B); (b) $`\Phi_\infty = \frac{\alpha_{\text{EM}}}{30}(1 - \frac{5}{3}\alpha_{\text{EM}})`$; (c) configuration count $`\mathcal{D}_{\text{config}} = 15 + 2 = 17`$; (d) scalar range $`\lambda_\psi = \sqrt{L_P R_H}`$. | *Obstruction:* the zero-mode Dirac operator is $`\Phi`$-blind, the bulk mass is $`\mathbb{Z}_2`$-odd, and `[A2]` has no fermionic action. $`\alpha_{\text{EM}}`$ is `[N]`. The count $`15 + 2`$ is not the `[D]` integer $`\dim(V_5^*\otimes V_5^*)_+ = 17`$. $`L_P`$ contains $`\hbar`$. | **`[D\|H1,H2]`:** composition-independent coupling $`\alpha_{\text{Yuk}} = \frac{1}{3}`$; $`\mu_{\text{eff}} \in [1, \frac{4}{3})`$, $`\eta_J \in (\frac{1}{2}, 1]`$; inversion $`\alpha_{\text{EM}} = \frac{3}{10}(1 - \sqrt{1 - 200\Phi_\infty})`$; ratio $`\frac{17}{15}`$; $`\Omega_\infty = c\Phi_\infty`$. |
| **`H3`** | **Potential and kernel.** $`V = \frac{3}{4}m_\psi^2[\psi - \psi_t(\mathcal{A})]^2`$ with $`\psi_t(\mathcal{A}) = \psi_{\max}[e^{-\mathcal{A}^2/A_{\text{crit}}^2} - e^{-1/A_{\text{crit}}^2}]`$ and $`(A_{\text{crit}}, \psi_{\max}, \hat{m}_\psi^{\text{(ref)}}) = (\frac{1}{4}, \frac{2}{25}, 4)`$. | *Undetermined:* the cutoff moments $`f_{4-2k}`$ are free, `[A3]` asserts only that a root $`\psi_t(\mathcal{A})`$ exists, and the exponential dependence on $`\mathcal{A}^2`$ is a choice. | **`[D\|H1,H3]`:** $`J_\Theta = 0 \iff \psi = \psi_t(\mathcal{A}) \implies V = V_\psi = 0`$; screening $`[\psi_t'(1)]^2 = 4\psi_{\max}^2 e^{-2/A_{\text{crit}}^2}/A_{\text{crit}}^4`$. **`[N\|H1,H3]`:** stiff-limit black-hole hair $`f_{\text{hair}} = 4.162\%`$, $`\Delta\mathcal{A}_H/\mathcal{A}_H = -8.15\%`$, $`r_{\text{ADM}}/r_h = 1.0434`$; interior turnover $`r_{\text{turn}} = 0.9081\,r_h`$; screening value $`8.30 \times 10^{-14}`$. |
| **`H4`** | **UV saturation and collapse gates.** $`\chi_{\max} = 2/L_P^2`$, $`r_{\text{core}} = (r_s L_P^2/2)^{1/3}`$, $`w_{\text{UV}} = 1 - \rho/\rho_P`$, regularised horizon temperature, $`\xi_{t_2}(3r_s) = \frac{1}{4}`$, and mesoscopic gates in $`M/m_{\text{crit}}`$ with $`m_{\text{crit}} = \Phi_\infty m_P`$. | *Undetermined:* the classical 5D field equations contain neither $`\hbar`$ nor $`L_P`$. | **`[D\|H1,H4]`:** $`\mathcal{K}_5(0) = 96/L_P^4`$ (core) and $`192/L_P^4`$ (throat); temperature peak at mass $`M_{\text{peak}} = \frac{\sqrt{3}}{2}m_P`$ with $`T_{\max} = \frac{T_P}{6\sqrt{3}\pi}`$; remnant root $`M_{\text{rem}} = \frac{1}{2}m_P`$; accretion time factor $`\frac{3}{4}`$. |
| **`H5`** | **Director-cosine acceleration law.** $`\mu_{\text{CBG}}(g_{\text{tot}}/a_0)\,g_{\text{tot}} = g_{\text{bar}}`$ with $`\mu_{\text{CBG}}(x) = x/\sqrt{1+x^2}`$; $`a_0^{\text{(dS)}} = \frac{cH_0}{2\pi}`$; $`a_0^{\text{(CBG)}} = \frac{17}{15}a_0^{\text{(dS)}}`$; $`H_0^{\text{(local)}} = H_0(1 + \Omega_\infty/v_{\text{vir}})`$. | *Undetermined:* $`\mu(x) = x/\sqrt{1+x^2}`$ is a `[D]` director identity (`THM #02`); its use as a force law, and the scale $`a_0`$, are not fixed by the axioms. | **`[D\|H5]`:** inversion $`g_{\text{tot}} = g_{\text{bar}}\sqrt{\frac{1}{2} + \sqrt{\frac{1}{4} + (a_0/g_{\text{bar}})^2}}`$; deep limit $`v_f^4 = GM_b a_0`$; elasticity $`\varepsilon_2(x) = \frac{1}{1+x^2}`$. |
| **`H6`** | **Five-sector system, IR exhaustion and apogee reflection.** The constitutive system of *Paper III* Eqs. 4.6–4.8, with exhaustion sector $`-\alpha_\Lambda(x/x_{\text{ap}})^2`$ ($`a_{\text{ap}} = \Phi_\infty^{-2}`$) **and** velocity reflection $`(v_x, v_\psi) \mapsto (-v_x, -v_\psi)`$ at $`H = 0`$. | *Undetermined:* the system differs from *Paper 0* Eqs. 118–122 in four disclosed respects; curvature alone cannot stop the expansion while the vacuum sector is positive; the $`\mathbb{Z}_2`$ of `[A1]` acts on $`t_2`$, not $`t_1`$. Without the reflection the contraction reaches a kinetic collapse (Check `#153`). | **`[D\|H6]`:** identity $`\frac{d\mathcal{E}_N}{d\tau} + 2H_d\mathcal{E}_N \equiv 0`$; time-reversal invariance; $`\lambda_{\text{UV}} = \frac{\sqrt{19/7}-1}{2}`$ and $`\Delta_{\text{IR}} = -17`$ (at $`V_0 = 54`$); smooth turnaround $`x_{\text{ap}}\mathrm{sech}(\sqrt{\alpha_\Lambda}\tau)`$ with $`\ddot{a}/a = -\alpha_\Lambda`$; mirror cycle $`T_{\text{cycle}} = 2t_{\text{apogee}}`$. **`[N\|H6,H_aux.3]`:** $`C_{P_2} = 1.1666666665 \simeq \frac{7}{6}`$; chained 18-step `DOP853` trajectory with residuals $`\le 10^{-9}`$. |
| **`H7`** | **Horizon polarization holonomy.** $`d\beta = +\frac{\pi}{2}\chi_E(S^2_H)\Phi_\infty\,d\ln a = +\pi\Phi_\infty\,d\ln a`$, and sub-ring holonomy $`\Delta\chi_{\text{pol}} = \Phi_\infty`$ rad. | *Obstruction:* on zero-mode photons the Chern–Simons density vanishes, and $`\mathrm{Tr}\,f(\widehat{\mathcal{D}}_{\text{JV}}^2)`$ is even in $`\widehat{\mathcal{D}}_{\text{JV}}`$: the axioms contain no parity-odd photon term. | **`[D\|H2,H7]`:** $`\beta(z) = \pi\Phi_\infty\ln(1+z)`$; $`C_\ell^{EB}/C_\ell^{EE} = \frac{1}{2}\sin(4\beta)`$; $`\beta_{\text{reion}} > 0`$. |

### 4.2 Auxiliary Ansätze and Initial-Data Benchmarks (`[H_aux.1]–[H_aux.3]`)

These are hypotheses of the same standing as the roots. They are listed separately because each serves one calculation.

- **`H_aux.1` (initial data and truncations).** Interior fiber-flux amplitudes $`C_\Omega \in \{1, 2, 4\}\psi_{\max}^2`$ (*Paper II*); the small-$`J_\Theta`$ truncation $`\mathcal{A} = 1`$ in homogeneous evolution ($`|2\partial_{\mathcal{A}}V|/(\rho_r + V) \le 1.38 \times 10^{-5}`$ in *Paper I*, $`\le 7.03 \times 10^{-6}`$ in *Paper III*); the shear law $`\rho_\sigma \propto a^{-6}e^{-2\psi}`$ applied on closed $`S^3`$ slices, which is not the exact anisotropic curvature of Section 3.3.
- **`H_aux.2` (strong-field and galactic-centre models).** Spinning extension of the core metric; basal $`\psi = 0`$ optical ansatz; ringdown shift $`+\frac{1}{2}\Phi_\infty^2`$; stellar-orbit shift; cluster transition radius $`R_{\text{tr}} = r_s/\Phi_\infty^2`$; mass ceiling $`M_{\max} = 5.13 \times 10^{10}\,M_\odot`$.
- **`H_aux.3` (cosmological closure and perturbation ansätze).** Closure expressions for $`\Omega_{b,0} = 0.049475`$, $`\Omega_{m,0}^{\text{(mode)}} = 0.315886`$, $`N_{\text{vac}} = 468`$, $`H_0^{\text{(CBG)}} = 67.3460`$ km/s/Mpc, $`T_{\text{CMB},0} = 2.7246`$ K, $`\eta_{\text{bitemp}} = 13.6432`$; throat diameter $`D_{\text{obs},b} = \lambda_{\text{micro}}/e = 17.33\,\mu\text{m}`$; energy rungs $`E_n = \Phi_\infty^{2n}E_P`$; late-time relaxation ansatz $`w_\Phi(a) = -1 + \frac{2\pi}{3}\Omega_{b,0}a^3`$, which is evaluated separately from `H6` and is not used in the chained trajectory; perturbation pump ansatz $`z(\eta) \propto |\eta|^{-(1+1/N)}`$; acoustic envelope with $`\Omega_{m,\text{eff}} = 0.32133`$; two-slope gravitational-wave transfer curve ($`+9/5`$, $`-1`$).

Results that rest on these carry the tag. Under the truncation $`\mathcal{A} = 1`$ the turning-point identity of Section 3.3 reduces to $`\ddot{a}/a = \frac{1}{3}[V_\psi - \rho_r]`$, so the $`k = +1`$ bounce condition $`V_\psi(\psi_b) > \rho_r(a_b)`$ and the absence of a bounce for $`k \le 0`$ are `[D | H1, H_aux.1]`. The oscillating-$`\psi`$ dust behaviour ($`\langle w_\psi\rangle \to 0`$, $`a^3\rho_\psi`$ constant to $`10^{-7}`$) is a `DOP853` result, `[N | H1, H3]`.

---

## 5. Tier 4: Numerical Evaluations (`[N]` / `[N | Hn, …]`)

*Convention for $`H_0`$: Paper I Eq. 90 evaluates horizon scales at the observed input $`H_0 = 67.40`$ km/s/Mpc `[N]`, giving $`a_0^{\text{(dS)}} = 1.0422 \times 10^{-10}`$ m/s². The closure value of `H_aux.3`, $`H_0^{\text{(CBG)}} = 67.3460`$ km/s/Mpc, gives $`1.0414 \times 10^{-10}`$ m/s².*

| Quantity | Tag | CBG value | Reference value (as quoted in the papers) | Status |
| :--- | :---: | :--- | :--- | :---: |
| Zero-mode wave speed $`c_T/c`$ | `[D \| H1]` | $`1`$ | $`\lvert c_T/c - 1\rvert \lesssim 10^{-15}`$ | Holds by construction under `H1` |
| Short-range scalar force | `[N \| H1,H2]` | $`\alpha_{\text{Yuk}} = 1/3`$ at $`47.10\,\mu\text{m}`$ ($`4.19`$ meV) | Torsion-balance bounds about $`1.5`$–$`1.8\times`$ above | Testable |
| Black-hole hair and shadow shift (stiff limit) | `[N \| H1,H3]` | $`f_{\text{hair}} = 4.162\%`$, shadow $`-0.071\%`$ | Shadow measurements at the $`\sim 10\%`$ level | Consistent |
| Tracking shift $`\gamma - 1`$ | `[N \| H1,H3]` | $`-5.762 \times 10^{-7}`$ ($`\beta - 1 = -8.643 \times 10^{-6}`$) | $`\lvert\gamma - 1\rvert \lesssim 2.3 \times 10^{-5}`$ | Within bound |
| Acceleration scales $`a_0^{\text{(dS)}}, a_0^{\text{(local)}}, a_0^{\text{(CBG)}}`$ | `[N \| H2,H5]` | $`1.0422,\ 1.1321,\ 1.1812 \times 10^{-10}`$ m/s² | $`(1.20 \pm 0.02) \times 10^{-10}`$ m/s² | Comparison: $`-7.9\sigma`$, $`-3.4\sigma`$, $`-0.9\sigma`$ |
| Chained age and apogee | `[N \| H6,H_aux.3]` | $`t_0 = 13.7909`$ Gyr; $`t_{\text{ap}} = 317.39`$ Gyr; $`T_{\text{cycle}} = 634.78`$ Gyr | Age $`13.787 \pm 0.020`$ Gyr | Comparison |
| Polarization rotation $`\beta`$ ($`z_* = 1089.8`$) | `[N \| H2,H7]` | $`+0.3025^\circ`$; $`C_\ell^{EB}/C_\ell^{EE} = 1.056 \times 10^{-2}`$ | $`\approx +0.30^\circ`$ to $`+0.34^\circ`$ ($`\pm 0.09^\circ`$) | Comparison |
| Reionization rotation ($`z = 7.7`$) | `[N \| H2,H7]` | $`+0.0936^\circ`$ | No measurement yet | Testable |
| Sub-ring polarization shift | `[N \| H2,H7]` | $`+49.56''`$ | No measurement yet | Testable |

---

## 6. Repository Structure

```text
cbg-cycle1-foundations/
├── .github/workflows/
│   └── verify_cbg.yml                 # CI pipeline running run_all_audits.py
├── Paper_0_Principia_Geometrica/      # Paper 0: axioms and the 24 theorems / 94 corollaries
├── Paper_I_Foundations/               # Paper I: 5D variational principles and field equations
├── Paper_II_Black_Hole_Mechanics/     # Paper II: horizon continuation and interior
├── Paper_III_Cosmology/               # Paper III: homogeneous cosmology
├── CITATION.cff
├── LICENSE                            # MIT (software) + CC BY 4.0 (monographs and figures)
├── README.md
├── cbg_readme_architecture_audit.py   # tier-by-tier audit of this README (runs the Paper III suite for delegated items)
├── requirements.txt                   # numpy, scipy, sympy
└── run_all_audits.py                  # runs the tier audit, then the newest suite in each Paper_* folder
```

| Monograph | Directory | Suite status | Zenodo DOI |
| :--- | :--- | :---: | :--- |
| **Paper 0:** *Principia Geometrica of Covariant Bitemporal Geometrodynamics* | [`Paper_0_Principia_Geometrica/`](Paper_0_Principia_Geometrica/) | 24 theorems, 94 corollaries pass | [`10.5281/zenodo.23829404`](https://doi.org/10.5281/zenodo.23829404) |
| **Paper I:** *Covariant Bitemporal Geometrodynamics (CBG): Variational Principles, 5D Laplacian Field Equations, and the Informational Resolution of Cosmological and Quantum Singularities* | [`Paper_I_Foundations/`](Paper_I_Foundations/) | all checks pass | [`10.5281/zenodo.23443511`](https://doi.org/10.5281/zenodo.23443511) |
| **Paper II:** *Relativistic Black Hole Mechanics in Covariant Bitemporal Geometrodynamics* | [`Paper_II_Black_Hole_Mechanics/`](Paper_II_Black_Hole_Mechanics/) | 184/184 checks pass | [`10.5281/zenodo.23451973`](https://doi.org/10.5281/zenodo.23451973) |
| **Paper III:** *Relativistic Bitemporal Cosmology in Covariant Bitemporal Geometrodynamics* | [`Paper_III_Cosmology/`](Paper_III_Cosmology/) | 157/157 checks pass | [`10.5281/zenodo.23458461`](https://doi.org/10.5281/zenodo.23458461) |

---

## 7. Quick Start

```bash
git clone https://github.com/CBG-physics/cbg-cycle1-foundations.git
cd cbg-cycle1-foundations
pip install -r requirements.txt

# Tier-by-tier audit of this README, including the Paper III suite (about one minute):
python run_all_audits.py --core-only

# Full verification: tier audit and all four monograph suites (the Paper 0 and Paper I suites take several minutes each):
python run_all_audits.py
```

---

## 8. Citation & License

```bibtex
@misc{JimenezVelasco2026CBGPaper0,
  author    = {Jim{\'e}nez Velasco, Manuel},
  title     = {{Principia Geometrica of Covariant Bitemporal Geometrodynamics (Paper 0)}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23829404},
  url       = {https://doi.org/10.5281/zenodo.23829404}
}
```

- **Verification software (`*.py`):** [MIT License](LICENSE).
- **Monographs and LaTeX sources (`*.pdf`, `*.tex`):** © 2026 Manuel Jiménez Velasco, [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
