# Covariant Bitemporal Geometrodynamics (CBG) — Cycle I Foundations
### Deductive Architecture (`[A1]–[A3]` + quotient postulate `[A4]` $`\to`$ `[D]` $`\to`$ `[H1]–[H6]`, `[H_aux]` $`\to`$ `[D | Hn]` $`\to`$ `[N]`) & Symbolic/Numerical Verification Suites

[![CBG Verification Suite](https://github.com/CBG-physics/cbg-cycle1-foundations/actions/workflows/verify_cbg.yml/badge.svg)](https://github.com/CBG-physics/cbg-cycle1-foundations/actions/workflows/verify_cbg.yml)
[![Paper 0 DOI](https://img.shields.io/badge/Zenodo%20Paper%200-10.5281%2Fzenodo.23829404-blue)](https://doi.org/10.5281/zenodo.23829404)
[![Paper I DOI](https://img.shields.io/badge/Zenodo%20Paper%20I-10.5281%2Fzenodo.23443511-blue)](https://doi.org/10.5281/zenodo.23443511)
[![Paper II DOI](https://img.shields.io/badge/Zenodo%20Paper%20II-10.5281%2Fzenodo.23451973-blue)](https://doi.org/10.5281/zenodo.23451973)
[![Paper III DOI](https://img.shields.io/badge/Zenodo%20Paper%20III-10.5281%2Fzenodo.23458461-blue)](https://doi.org/10.5281/zenodo.23458461)

**Author:** Manuel Jiménez Velasco (*M.Sc. in Civil Engineering, UPC & Independent Researcher in Theoretical Physics, Madrid, Spain — `manujimevela@gmail.com`*)

---

## 1. Overview & Deductive Epistemology

This repository holds the four monographs of **Covariant Bitemporal Geometrodynamics (CBG), Cycle I**, their verification suites, and a set of standalone audits that locate, statement by statement, where each result sits in the deductive chain. The scope of every script is **internal deductive coherence**: what follows from what. None of them adjudicates empirical truth.

Methodology note: Manuscript drafting, LaTeX formatting, and Python verification/audit scripting in this repository utilized LLM assistance under human architectural direction; every mathematical identity and numerical benchmark claimed here is enforced deterministically by the executable SymPy/SciPy suites below

Every statement carries exactly one tag:

1. **`[A]` — Axioms.** `[A1]–[A3]` are the closed three-axiom core of *Paper 0*. The exact quotient postulate $`\partial_{t_2} \equiv 0`$ is listed with them as `[A4]` in this README (see the crosswalk below).
2. **`[D]` — Exact deductions.** Symbolic theorems with zero residual. Section 3.1 lists those that hold from `[A1]–[A3]` alone; Section 3.2 those that also use the quotient postulate.
3. **`[H1]–[H6]`, `[H_aux.1]–[H_aux.3]` — Hypotheses.** Constitutive postulates and initial data. None is derived from the axioms.
4. **`[D | Hn, …]` — Conditional deductions.** Exact symbolic statements that assume the named hypotheses.
5. **`[N]`, `[N | Hn, …]` — Numerical items.** Empirical inputs and every floating-point result, including machine-precision ODE solutions. The monograph suites label some of these `[D]` internally; this README does not.

```text
   [A1]-[A3]  Paper 0 axioms            [A4] exact quotient postulate  d/dt2 = 0  (all scales)
        |                                        |
        +------------------+---------------------+
                           v
   [D]  exact theorems (3.1: from [A1]-[A3] alone;  3.2: with the quotient)
                           |
        [H1]-[H6] root hypothesis groups      [H_aux.1-3] auxiliary closures and initial data
                           |                                   |
                           v                                   v
   [D | Hn]  conditional exact deductions  --->  [N | Hn]  numerical evaluations   <---  [N] CODATA / survey inputs
```

### Terminology & Tag Crosswalk

The published Zenodo v1 PDFs and their scripts use an earlier numbering. Nothing in them has been changed; this table is the only bridge.

| Content | This README | Strict 3-axiom numbering | *Paper 0* (Zenodo v1) | *Papers I–III* (Zenodo v1) |
| :--- | :---: | :---: | :---: | :---: |
| Arena, spectral action, horizon regularity | `[A1]`, `[A2]`, `[A3]` | `[A1]`, `[A2]`, `[A3]` | Axioms 1.1, 4.1, 5.3 | `[A1]`, `[A2]`, `[A3]` |
| Exact quotient $`\mathcal{L}_{\xi_{(2)}}\mathcal{F}_{\text{phys}} = 0`$ | **`[A4]`** | `[H0]` | cylinder hypothesis `H` | `H1` |
| Matter coupling and fiber scale bridge | **`H1`** | `H1` | `[H]` items | `H2` |
| Equilibrium kernel and local Hooke 2-jet | **`H2`** | `H2` | `[H]` items | `H3` |
| Planckian UV saturation and gates | **`H3`** | `H3` | — | `H4` |
| Director-cosine acceleration law | **`H4`** | `H4` | — | `H5` |
| Five-sector cyclic cosmos | **`H5`** | `H5` | — | `H6` |
| Horizon and sub-ring polarization holonomy | **`H6`** | `H6` | — | `H7` |
| Auxiliary closures and initial data | `H_aux.1–3` | `H_aux.1–3` | — | `H_aux.1–3` |

A tag `[D | H1]` in a Zenodo v1 PDF of *Papers I–III* therefore reads `[D]` here; `[N | H1,H3]` there reads `[N | H2]` here; and so on, one index down.

---

## 2. Tier `[A]` — Foundational Mathematical Axioms (Closed Core of *Paper 0*)

| Tag | Name | Exact Mathematical Statement | Upstream Source |
| :--- | :--- | :--- | :--- |
| **`[A1]`** | **The Bitemporal Arena** | 5D warped orbifold bundle $`\mathcal{M}_{3,2} = (\mathbb{R}_{t_1} \times_\Phi S^1_{t_2}/\mathbb{Z}_2) \times \Sigma_3`$, metric signature $`(-1,-1,+1,+1,+1)`$, secondary temporal fiber at coordinate $`x^2 = t_2`$, orbifold reflection $`\mathbf{R}_{\mathbb{Z}_2} = \mathrm{diag}(1,-1,1,1,1)`$ with two fixed boundaries. | *Paper 0* Axiom 1.1 |
| **`[A2]`** | **Spectral Action & UV Winding Finiteness** | Effective action generated by the Wick-rotated spectral trace $`\mathrm{Tr}\,f(\widehat{\mathcal{D}}_{\text{JV}}^2/\Lambda_{\text{cut}}^2)`$ over the compact fiber $`\mathcal{S}_{t_2}`$ with finite non-zero Mellin winding moments $`0 < \lvert\mathcal{M}_k\rvert < \infty`$ ($`k \in \{0,1,2\}`$) spanning $`\mathcal{W}_3 = \mathrm{Span}\{e^{-2\psi}, e^{-4\psi}, e^{-6\psi}\}`$ (no constant term $`c_0`$ in $`\mathcal{W}_3`$). The cutoff moments $`f_{4-2k}`$ are non-zero and otherwise free. | *Paper 0* Axiom 4.1 |
| **`[A3]`** | **Killing Horizon Curvature Regularity** | Stationary geometries with $`\psi(r) \neq \text{const}`$ across a Killing horizon $`r_h > 0`$ satisfy $`\lvert\mathcal{K}_5(r_h)\rvert < \infty`$, with the primary temporal foliation norm $`\mathcal{A} = N^2/\dot{\Theta}^2`$ entering $`\mathcal{V}_{\text{spec}}(\psi,\mathcal{A})`$ solely via its equilibrium root $`\psi_t(\mathcal{A})`$. | *Paper 0* Axiom 5.3 |

> **Exact Well-Posedness Quotient Postulate (`[A4]` in this README / `[H0]` in strict 3-axiom numbering / `H` in *Paper 0* / `H1` in the Zenodo v1 PDFs of *Papers I–III*):**
>
> $`\mathcal{L}_{\xi_{(2)}}\mathcal{F}_{\text{phys}} = 0`$ ($`\partial_{t_2} = 0`$) on all physical classical fields, bosonic and spinorial, **identically at all energy scales**; the virtual traces of `[A2]` sample the full compact fiber.
>
> It is independent of `[A1]–[A3]`: they fix the $`\mathbb{Z}_2`$ parity of a field, not its mode number. It cannot be weakened to a low-energy approximation: the temporal block of `[A1]` has the real elliptic principal part $`e^{-2\nu}\partial_1^2 + \Phi^{-2}\partial_2^2`$ (`t2_audit` `S4`), so every $`n \ge 1`$ mode obeys $`\ddot{\psi}_n = +(n/\Phi_0 L_0)^2\psi_n`$ (`S14`), with a growth rate unbounded in $`n`$. Without the quotient there is no well-posed evolution in $`t_1`$ at any scale.

---

## 3. Tier `[D]` — Exact Mathematical Deductions (Zero Free Parameters)

### 3.1 Unconditional `[D]` from `[A1]–[A3]` Alone (Before Imposing $`\partial_{t_2} = 0`$)

| Tag | Proved Theorem / Identity | Exact Content & Physical Consequence | Proof Script Node |
| :--- | :--- | :--- | :--- |
| **`[D]`** | **Unique Dimension & Signature $`(2,3,5)`$** | From $`\dim\Lambda^2(\mathbb{R}^{d_s}) = d_s`$ and $`d_t d_s D = D(D+1)`$: $`(d_t,d_s,D) = (2,3,5)`$; $`\mathcal{N}_{\text{Cartan}} = 30`$; $`\mathrm{Cl}(3,2) \simeq \mathrm{Mat}(4,\mathbb{R}) \oplus \mathrm{Mat}(4,\mathbb{R})`$, $`\mathrm{Spin}_0(3,2) \simeq \mathrm{Sp}(4,\mathbb{R})`$; polar conic $`Q = 2/3`$. The two relations solved are those stated in *Paper 0*; the theorem is their unique positive root and its agreement with `[A1]`. | `verify_cbg_paper0_v5.py` (`THM #01–#07`) |
| **`[D]`** | **DeWitt Supermetric & Orbifold Parity Split** | $`15\times 15`$ DeWitt inverse trace weight $`c_{\text{DW}}^{-1} = -\frac{1}{D-2} = -\frac{1}{3}`$ at $`D = 5`$; $`\mathrm{End}(V_5) = 13 \oplus 12`$ under the time/space grading and $`25 = 17_+ \oplus 8_-`$ under $`\mathbf{R}_{\mathbb{Z}_2}`$; $`g_{\mu 2}(t_1,-t_2) = -g_{\mu 2}(t_1,t_2)`$, so $`g_{\mu 2} = 0`$ on both fixed orbifold walls (`S5`). | `THM #02–#04`; `t2_audit` `S5`; `a1a4` `b.1–b.3` |
| **`[D]`** | **General 2-Time Bitemporal Geometry & Hadamard Theorem** | Exact $`G^{(5)}_{02} = -\frac{3}{a}(\partial_1\partial_2 a - \nu_{,2}\partial_1 a - \psi_{,1}\partial_2 a)`$ (`S1`); real elliptic $`(t_1,t_2)`$ symbol $`e^{-2\nu}\xi_1^2 + \Phi^{-2}\xi_2^2 > 0`$ (`S4`); strict $`t_1`$-concavity of the fiber-averaged $`a`$ in vacuum with Neumann walls (`S6`); for vacuum or a constant potential, generalized Birkhoff Killing vector $`K^a = \epsilon^{ab}\partial_b a`$ and the explicit family with $`\mathcal{K}_5 = \frac{72\mu^2}{a^8} + \frac{10}{9}V^2`$, with no two-wall $`t_2`$-profile for $`k = +1`$ (`S8–S12`); $`n \ge 1`$ exponential growth $`\ddot{\psi}_n = +(n/\Phi_0 L_0)^2\psi_n`$ and, in the two-mode reduction, volume-preserving flow with negative mode energy (`S14–S15`). | `cbg_bitemporal_t2_throat_audit.py` (`S1–S15`) |
| **`[D]`** | **Lichnerowicz–Gilkey Curvature Weights** | Curved 5-bein $`\nabla_M^{\text{spin}}\Gamma^N = \mathbf{0}_4`$, $`c_{R_5} = 1/4`$, $`c_{\text{Pauli}} = 1/2`$; Gilkey weights $`(-\frac{7}{360}\mathcal{K}_5,\ -\frac{1}{45}\mathcal{R}_2^{(5)},\ +\frac{1}{72}(R^{(5)})^2)`$, reducing on $`\mathrm{FRW}(k=+1)\times\mathcal{S}_{t_2}`$ to the pure Gauss–Bonnet density $`\frac{11}{15}\frac{\ddot{a}}{a}(H^2 + \frac{1}{a^2})`$ (`S16`). | `THM #15–#16`; `t2_audit` `S16` |

### 3.2 Quotient `[D]` from `[A1]–[A3]` + Exact Quotient $`\mathcal{L}_{\xi_{(2)}} = 0`$ (`[D | H0]` in strict numbering / `[D]` in the 4-axiom view)

| Tag | Proved Theorem / Identity | Exact Content & Physical Consequence | Proof Script Node |
| :--- | :--- | :--- | :--- |
| **`[D]`** | **Dual-Branch Coexistence Theorem** | Coexistence on $`\mathcal{M}_{3,2}/\xi_{(2)}`$ of **Branch A** (Jordan metric $`g_{\mu\nu}`$, bare scalar coupling $`\alpha_0^2 = \frac{1}{3} = -c_{\text{DW}}^{-1}`$), **Branch B** (Einstein metric $`\tilde{g}_{\mu\nu} = e^\psi g_{\mu\nu}`$, $`\kappa_\psi = -\frac{3}{2}`$, exact contracted-Bianchi conservation $`\tilde{\nabla}^\mu\tilde{T}_{\mu\nu}^{(\psi)} \equiv 0`$, $`\alpha = 0`$, $`c_T = 1`$), and **`Branch A ∩ B`** (null cones $`ds^2 = 0`$, massless chiral Weyl spinors $`\Psi_5 = \Phi^{-1/2}\chi_4`$ with $`\bar{\Psi}\Psi \equiv 0`$, and $`d_{\text{macro}} = 4`$ Weyl invariance of $`\sqrt{-g_4}F_{\mu\nu}F^{\mu\nu}`$). Which matter follows which branch is `H1(a)`, not part of this theorem. | `THM #08–#14`; `a1a4` `a.1–a.13` |
| **`[D]`** | **Exact Elimination of $`g_{\mu 2}`$ & CS Vanishing** | Only the $`11_+`$ even metric components $`(g_{\mu\nu}, g_{22})`$ survive ($`g_{\mu 2} \equiv 0`$, zero graviphoton); the 5D Maxwell Chern–Simons density $`\epsilon^{ABCDE}A_A F_{BC}F_{DE} \equiv 0`$ identically. Both need the $`\mathbb{Z}_2`$ parity of `[A1]` together with the quotient; the Chern–Simons density does not vanish under the quotient alone. | `THM #08, #11`; `a1a4` `b.2, b.4, d.2, d.3` |
| **`[D]`** | **Spectral Winding, Horizon & Overlap Identities** | Ray–Singer $`-\zeta_{\text{orb}}'(0) = \ln(2\pi L_0\Phi)`$, $`\mathcal{M}_2^{(-)} = \frac{\ln 2}{16\pi^2}`$, Bekenstein crest $`\Pi_2(\frac{1}{2}\ln 3) = \frac{1}{54}`$ (single well + single crest in $`\mathcal{W}_3`$, `S1–S3`), 3D ball jump $`\Delta f_{\text{mut}}^{(4)}(1) = -30`$ ($`\Delta f^{(4)}(1) = +36`$), stationary locking $`h'(r) \equiv 0`$, Frobenius pole cancellation $`\mathcal{C}_{B^2} + 1/b_{-1} \equiv 0`$, $`\psi_t \in C^2(0)`$ *(stiff limit)*. Items from *Paper 0* `THM #14, #17, #18, #21, #23` keep the regime conditions and sign conventions declared in that suite. | `THM #17–#24`; `18stage` `S1–S3`; `a1a4` `c.1–c.2` |
| **`[D]`** | **Homogeneous Bitemporal & Cosmological Bounds** | Time-reversal symmetry; khronon dust density $`\rho_\Theta = 2\mathcal{A}V_{\mathcal{A}} = \frac{J_\Theta}{2a^3 e^\psi\sqrt{\mathcal{A}}}`$; Jordan still-point acceleration $`\frac{\ddot{a}}{a} = \frac{1}{3}[V_\psi - \rho_r - \rho_m - \rho_\Theta]`$; clock rotation leaves $`a(t_1), \rho(a)`$ invariant (`S13`); $`\mathcal{W}_3`$ fiber-suction turnaround ($`V < 0`$ / $`V_\psi < 0`$; no bounce for $`k \le 0`$); anti-friction $`\frac{d}{dt}[\frac{3}{4}\dot{\psi}^2 + V] = -\frac{9}{2}H\dot{\psi}^2`$; adiabatic radiation–curvature bound $`(a_0/a_b)^2 \le (\Omega_K/\Omega_r)e^{\psi_b-\psi_0}`$ at a still point with $`V + \rho_m + \rho_\Theta \ge 0`$; constant-$`\Omega_\Lambda`$ apogee bound $`\Omega_\Lambda \le \frac{4\Omega_K^3}{27\Omega_m^2}`$ (`S6`). | `THM #20–#21`; `suction`; `autonomous`; `18stage` (`S6`); `t2_audit` (`S13`) |

---

## 4. Tiers `[H]`, `[D | H]`, `[N]`, `[N | H]` — Six Root Hypothesis Groups & Conditional Cascades

*(Numbered `H1–H6` under this README's numbering, where the exact quotient is `[A4]`/`[H0]`; the Zenodo v1 PDF equivalents `H2–H7` are shown in parentheses.)*

| Group | Explicit Root Postulates `[H]` | Conditional Mathematical Deductions `[D \| H]` | Empirical Inputs `[N]` & Numerical Predictions `[N \| H]` |
| :--- | :--- | :--- | :--- |
| **`H1`** *(Zenodo `H2`)* **Dual-Branch Matter & Fiber Scale Bridge** | **(a)** Universal complex opposite-parity bulk fermion pairs $`(\Psi_+,\Psi_-)`$ with constant 5D mass + canonical $`\Phi^0`$ 4D Maxwell normalization $`-\frac{1}{4}\sqrt{-g_4}F^2`$; **(b)** $`\Phi_\infty = \frac{\alpha_{\text{EM}}}{30}(1 - \frac{5}{3}\alpha_{\text{EM}})`$; **(c)** bare ratio $`\eta_{\text{EM}}^{-1} = \frac{17}{15}`$ (a constitutive weighting; not derived from $`25 = 17_+ \oplus 8_-`$); **(d)** $`\lambda_\psi = \sqrt{L_P R_H}`$. | **`[D \| H1a]`:** Composition-independent **Branch A** baryonic coupling ($`\alpha_{\text{Yuk}} = 1/3`$, $`\Delta a/a = 0`$) & **`Branch A ∩ B`** physical radiation ($`\alpha = 0`$, $`\dot{\alpha}_{\text{EM}} = 0`$); Yukawa crossover $`\mu_{\text{eff}}(k) \in [1, 4/3)`$, $`\eta_J(k) \in (1/2, 1]`$ (`Branch A` response at $`r \ll \lambda_\psi`$, `Branch B` response at $`r \gg \lambda_\psi`$).<br>**`[D \| H1]`:** Exact quadratic inversion $`\alpha_{\text{EM}}(\Phi_\infty) = \frac{3}{10}(1 - \sqrt{1 - 200\Phi_\infty})`$, $`\Omega_\infty = c\Phi_\infty`$. | **`[N]`:** CODATA $`\alpha_{\text{EM}}, G, \hbar, c`$; $`H_0`$.<br>**`[N \| H1]`:** $`\Phi_\infty = 2.402867\times 10^{-4}`$; $`\lambda_\psi = 47.10\,\mu\text{m}`$ at the *Paper I* input $`H_0 = 67.40`$ km/s/Mpc ($`47.12\,\mu\text{m}`$ at $`H_0^{\text{(CBG)}}`$), $`m_\psi = 4.1880\text{ meV}`$. |
| **`H2`** *(Zenodo `H3`)* **Equilibrium Kernel & Local Hooke 2-Jet** | Evenness + semigroup law in $`\mathcal{A}^2`$; local Hooke 2-jet $`V(\psi,\mathcal{A}) = \frac{3}{4}m_\psi^2[\psi - \psi_t(\mathcal{A})]^2`$ around $`\psi_t(\mathcal{A})`$ (the frame in which it is written, Jordan or Einstein, must be declared; the two differ by $`e^{-\psi}`$); thresholds $`(A_{\text{crit}} = \frac{1}{4},\ \psi_{\max} = \frac{2}{25},\ \hat{m}_\psi^{\text{(ref)}} = 4)`$. | **`[D \| H2]`:** Unique Gaussian $`\psi_t(\mathcal{A}) = \psi_{\max}[e^{-\mathcal{A}^2/A_{\text{crit}}^2} - e^{-1/A_{\text{crit}}^2}]`$; $`J_\Theta = 0 \iff \psi = \psi_t(\mathcal{A}) \implies V = V_\psi = 0`$; analytical $`[\psi_t'(1)]^2`$; **Branch B** WKB oscillating condensate $`\langle w_\psi\rangle \to 0`$ (CDM geodesics); khronon energy bound $`\lvert\rho_\Theta\rvert \le \frac{6}{e}m_\psi^2\psi_{\max}\lvert\psi-\psi_t\rvert`$. | **`[N \| H2]`:** Stiff BH hair $`4.162\%`$, shadow $`-0.071\%`$, turnover $`0.9081\,r_h`$, $`[\psi_t'(1)]^2 = 8.30\times 10^{-14}`$; isolated reflection-free Jordan toy periodic cycles (unstable; offset growth $`\sim 10^2`$ per cycle).<br>**`[N \| H1, H2]`:** PPN $`\gamma - 1 = -5.76\times 10^{-7}`$ (inside the Cassini bound by a factor $`40`$). |
| **`H3`** *(Zenodo `H4`)* **Planckian UV Saturation & Quantum Gates** | Source-density UV saturation dressing $`w_{\text{UV}}(\rho) = 1 - \rho/\rho_P`$ ($`\chi_{\max} = 2/L_P^2`$), i.e. $`H^2 = \frac{\rho}{3}(1 - \frac{\rho}{\rho_P}) - \frac{k}{\tilde{a}^2}`$ with ordinary continuity; regularised Hawking temperature; ISCO saturation $`\xi_{t_2}(3r_s) = \frac{1}{4}`$; mesoscopic collapse gate $`m_{\text{crit}} = \Phi_\infty m_P`$. | **`[D \| H3]`:** Closed conservation $`\frac{d}{dt}[H^2 - \frac{\rho}{3}(1-\frac{\rho}{\rho_P}) + \frac{k}{\tilde{a}^2}] \equiv 0`$ (`S4`); **non-singular Bounce #1 & Bounce #2** ($`\rho = \rho_P`$, $`H = 0`$, $`\dot{H} > 0`$) without $`x^{-6}`$ recoil or velocity reflection (`S5`; numerically `N1`, `N6`); $`r_{\text{core}} = (r_s L_P^2/2)^{1/3}`$; 1-channel BH core $`\mathcal{K}_5(0) = 96/L_P^4`$ vs. 2-channel bounce throat $`192/L_P^4`$; $`M_{\text{peak}} = \frac{\sqrt{3}}{2}m_P`$, $`M_{\text{rem}} = \frac{1}{2}m_P`$; $`\frac{3}{4}`$ Salpeter factor. | **`[N \| H1, H3]`:** $`m_{\text{crit}} = 5.23\,\text{ng}`$ interferometry benchmark.<br>**`[N \| H1, H2, H3]`:** $`a_b = 1.9723\times 10^{-32}`$; reproduction of the *Paper III* stage times without the $`x^{-4}`$ divisor (Stages 4–14 to three or four digits, Stages 1–3 within $`8\%`$), given the initial datum $`\psi_b = 5.64533\times 10^{-7}`$ (`H_aux.1`; equivalently a Gilkey coefficient $`\xi = 3.308\times 10^{-6}`$, `[N \| H3, ξ]`, `t2_audit` `N2`); 40-digit `mpmath` mirror retrace (return error $`3.73\times 10^{-6}`$ against $`6\times 10^{2}`$ in float64). |
| **`H4`** *(Zenodo `H5`)* **Galactic Director-Cosine Law** | $`\mu_{\text{CBG}}(g_{\text{tot}}/a_0)\,g_{\text{tot}} = g_{\text{bar}}`$ with $`\mu_{\text{CBG}}(x) = x/\sqrt{1+x^2}`$, $`a_0^{\text{(dS)}} = \frac{cH_0}{2\pi}`$, $`a_0^{\text{(CBG)}} = \frac{17}{15}a_0^{\text{(dS)}}`$, and virial $`H_0^{\text{(local)}}`$. | **`[D \| H4]`:** Exact algebraic root $`g_{\text{tot}}(g_{\text{bar}})`$, baryonic Tully–Fisher $`v_f^4 = GM_b a_0`$, external-field screening $`\varepsilon_2(x) = \frac{1}{1+x^2}`$. | **`[N \| H4]`:** $`a_0^{\text{(dS)}} = 1.0422\times 10^{-10}\text{ m/s}^2`$.<br>**`[N \| H1, H4]`:** $`a_0^{\text{(local)}} = 1.1321\times 10^{-10}`$, $`a_0^{\text{(CBG)}} = 1.1812\times 10^{-10}\text{ m/s}^2`$ (vs. SPARC $`(1.20 \pm 0.02)\times 10^{-10}`$). |
| **`H5`** *(Zenodo `H6`)* **5-Sector Constitutive Cyclic Cosmos** | *Paper III* 5-sector constitutive ODEs ($`x^{-4}`$ barrier conformal dilution, $`-C_{P_2}/x^6`$ throat recoil) + IR vacuum-sector exhaustion $`-\alpha_\Lambda(x/x_{\text{ap}})^2`$ at $`a_{\text{ap}} = \Phi_\infty^{-2}`$ + apogee time-symmetric reflection at $`H = 0`$. *Not replaceable inside the audited 2-derivative system:* the constant-sector bound $`\Omega_\Lambda \le 2.86\times 10^{-22}`$ (`18stage` `S6`), the pinned fiber with no autobalancing (`N2`), and the generic-phase crest runaway after Bounce #2 (`N7`); see Section 5. | **`[D \| H5]`:** Exact constraint propagation $`\dot{\mathcal{E}}_N + 2H_d\mathcal{E}_N \equiv 0`$, time-reversal symmetry, IR anomalous scaling $`\Delta_{\text{IR}} = 1 - \frac{54}{3} = -17`$, $`\mathrm{sech}`$ apogee envelope, $`T_{\text{cycle}} = 2t_{\text{ap}}`$. | **`[N \| H1, H5, H_aux.3]`:** Full 18-stage trajectory (`157/157` checks in *Paper III v13*): $`t_0 = 13.7909\text{ Gyr}`$, $`H_0 = 67.346\text{ km/s/Mpc}`$, $`t_{\text{ap}} = 317.39\text{ Gyr}`$, $`T_{\text{cycle}} = 634.78\text{ Gyr}`$, acoustic scale $`\theta_*`$. |
| **`H6`** *(Zenodo `H7`)* **Horizon & Sub-Ring Polarization Holonomy** | Topological cosmic horizon phase $`d\beta = +\frac{\pi}{2}\chi_E(S^2_H)\Phi_\infty\,d\ln a`$ and black-hole photon sub-ring polarization step $`\Delta\chi_{\text{pol}} = \Phi_\infty\text{ rad}`$. Independent of the Chern–Simons term, which vanishes on physical photons (Section 3.2). | **`[D \| H6]`:** Redshift law $`\beta(z) = \pi\Phi_\infty\ln(1+z)`$, $`E`$-$`B`$ cross-spectrum $`C_\ell^{EB}/C_\ell^{EE} = \frac{1}{2}\sin(4\beta)`$ *(for $`C_\ell^{BB,\text{prim}} = 0`$)*, strictly positive $`\beta_{\text{reion}} > 0`$. | **`[N \| H1, H6]`:** Zero-parameter targets $`\beta_{\text{CMB}} = +0.3025^\circ`$ (vs. Planck/ACT $`+0.30^\circ`$ to $`+0.34^\circ`$, $`\pm 0.09^\circ`$), $`\beta_{\text{reion}} = +0.0936^\circ`$ (LiteBIRD), $`\Delta\chi_{\text{pol}} = +49.56''`$ (ngEHT). |
| **`H_aux.1–3`** **Auxiliary Closures** | **`H_aux.1`:** $`k=+1`$ closed $`S^3`$ spatial topology & $`\mathcal{A}=1`$ synchronous slice; initial data and truncations of *Papers I–III* (interior flux amplitudes, shear law), including the post-bounce displacement $`\psi_b`$.<br>**`H_aux.2`:** Astrophysical galaxy/BH environmental priors and strong-field models.<br>**`H_aux.3`:** Cosmological closure expressions of *Paper III* and recombination inputs ($`\Omega_b = 0.049475`$, $`\Omega_m^{\text{(mode)}} = 0.315886`$, $`H_0^{\text{(CBG)}} = 67.3460`$ km/s/Mpc, $`T_{\text{CMB},0} = 2.7246\text{ K}`$). | **`[D \| H_aux.1]`:** Turning-point bounce criterion $`V_\psi(\psi_b) > \rho_r(a_b) + \rho_m(a_b)`$. | **`[N \| H_aux.2–3]`:** Empirical astrophysical & Planck/CODATA boundary inputs feeding *Papers I–III*. |

---

## 5. Why the 2-Derivative Master Equations Are the Adiabatic Core, and Why `H3` and `H5` Are Separate Closures

The four coupled Master Equations (I: Einstein–Friedmann–Raychaudhuri on $`\tilde{g}_{\mu\nu} = e^\psi g_{\mu\nu}`$; II: the scalar-modulus wave equation with $`\kappa_\psi = -\frac{3}{2}`$; III: conservation of the cyclic fiber momentum $`P_2`$; IV: the Bianchi–Noether constraint with the khronon charge $`J_\Theta`$) are closed and consistent: the constraint propagates exactly, and initial data determine the evolution. They are the classical two-derivative theory, valid for $`L_P \ll L \ll R_H`$. The audits establish what they deliver and where they stop.

**What they deliver.** Given the initial displacement $`\psi_b`$, the hot history of *Paper III* is reproduced stage by stage without the $`x^{-4}`$ divisor (`18stage` Part N2); the fiber is released at $`H = m_\psi`$ and then behaves as cold matter; an apogee exists by $`S^3`$ curvature alone when there is no constant sector (`18stage` `N3`); and the contraction mirrors the expansion through the matter era.

**Where they stop, with the theorem that shows it.**

| Missing piece | Exact statement | Node | Closure that supplies it |
| :--- | :--- | :--- | :--- |
| A turn-round at Planck density | At the *Paper III* bounce data the Master constraint is violated by a factor $`2.5\times 10^{66}`$; a still point with conserved radiation needs $`(a_0/a_b)^2 \le (\Omega_K/\Omega_r)e^{\psi_b-\psi_0}`$. | `probe` [1]; `autonomous` `S8–S9` | **`H3`** |
| No help from the second time | Relaxing $`\partial_{t_2} = 0`$ gives an elliptic problem (`S4`); in vacuum the fiber-averaged radius is concave (`S6`); every vacuum or constant-potential solution keeps a temporal Killing vector (`S8`) and none has a two-wall profile for $`k=+1`$ (`S11`); the one non-singular member needs a constant and $`\Phi \to 0`$ (`S12`). | `t2_audit` | **`H3`** |
| A late vacuum sector that later switches off | $`\mathcal{W}_3`$ has one well and one crest and no plateau (`S3`); $`m_\psi/H_0 = 2.9\times 10^{30}`$ pins the fiber; with a constant, an apogee needs $`\Omega_\Lambda \le 4\Omega_K^3/(27\Omega_m^2) = 2.86\times 10^{-22}`$ (`S6`). Without a constant: $`37.86`$ km/s/Mpc and $`17.21`$ Gyr. | `18stage` | **`H5`** (exhaustion sector) |
| The initial displacement | From the bounce to $`H = m_\psi`$ the fiber does not move; the condensate is fixed by $`\psi_b^2`$ (`N2`). The Gilkey term supplies it only through a free coefficient $`\xi`$ (`S16`, `N2`). | `18stage`; `t2_audit` | **`H_aux.1`** |
| A reset at the bounce | Under `H3` a second bounce always occurs (`N6`), but for a generic condensate phase it is kinetic-dominated at $`\sim 10^{25}`$ eV and throws the fiber past the crest (`N7`); the equations are time-reversible, and $`n \ge 1`$ modes pump, not damp (`S14–S15`, `N1`). | `18stage`; `t2_audit` | **`H5`** (reflection / five-sector closure) |

So `H3` and `H5` are not derivable from the audited two-derivative system; they are the physical closures that the cyclic cosmology of *Paper III* rests on, and they are tagged `[H]` accordingly.

**Scope of these statements.** They are exact where a node `S…` is cited and numerical where a node `N…` is cited. Three limits are declared: (i) in the 18-stage runs $`J_\Theta = 0`$ and $`P_2`$ is not evolved; the khronon bound of `H2` and the $`a^{-3}`$ scaling of a $`P_2`$ gas indicate they do not change the outcome, but no run has them active; (ii) with radiation or dust and $`\partial_{t_2} \neq 0`$ only the averaged identity `S6` is available, not a full no-bounce theorem; (iii) the two-mode model behind `N1` is a reduced system, not the full Einstein equations.

---

## 6. Falsifiability & Zero-Parameter Observational Benchmarks

*Convention for $`H_0`$: Paper I evaluates horizon scales at the observed input $`H_0 = 67.40`$ km/s/Mpc `[N]`, giving $`a_0^{\text{(dS)}} = 1.0422 \times 10^{-10}`$ m/s². The closure value of `H_aux.3`, $`H_0^{\text{(CBG)}} = 67.3460`$ km/s/Mpc, gives $`1.0414 \times 10^{-10}`$ m/s². "Zero-parameter" means: no parameter beyond the stated hypotheses and the `[N]` inputs.*

| Quantity | Tag | CBG value | Reference value (as quoted in the papers) | Status |
| :--- | :---: | :--- | :--- | :---: |
| Zero-mode wave speed $`c_T/c`$ | `[D]` | $`1`$ | $`\lvert c_T/c - 1\rvert \lesssim 10^{-15}`$ | Holds by construction under the quotient |
| Short-range scalar force | `[N \| H1]` | $`\alpha_{\text{Yuk}} = 1/3`$ at $`47.10\,\mu\text{m}`$ ($`4.19`$ meV) | Torsion-balance bounds about $`1.5`$–$`1.8\times`$ above | Testable |
| Composition dependence of free fall | `[D \| H1a]` | $`\Delta a/a = 0`$ | Equivalence-principle tests | Consistent by construction |
| Drift of $`\alpha_{\text{EM}}`$ | `[D \| H1a]` | $`\dot{\alpha}_{\text{EM}} = 0`$ | Clock and quasar bounds | Consistent by construction |
| Black-hole hair and shadow shift (stiff limit) | `[N \| H2]` | $`f_{\text{hair}} = 4.162\%`$, shadow $`-0.071\%`$ | Shadow measurements at the $`\sim 10\%`$ level | Consistent |
| Tracking shift $`\gamma - 1`$ | `[N \| H1, H2]` | $`-5.762 \times 10^{-7}`$ ($`\beta - 1 = -8.643 \times 10^{-6}`$) | $`\lvert\gamma - 1\rvert \lesssim 2.3 \times 10^{-5}`$ | Within bound |
| Mesoscopic collapse gate | `[N \| H1, H3]` | $`m_{\text{crit}} = \Phi_\infty m_P = 5.23`$ ng | Matter-wave interferometry | Testable |
| Acceleration scales $`a_0^{\text{(dS)}}, a_0^{\text{(local)}}, a_0^{\text{(CBG)}}`$ | `[N \| H1, H4]` | $`1.0422,\ 1.1321,\ 1.1812 \times 10^{-10}`$ m/s² | $`(1.20 \pm 0.02) \times 10^{-10}`$ m/s² | Comparison: $`-7.9\sigma`$, $`-3.4\sigma`$, $`-0.9\sigma`$ |
| Chained age and apogee | `[N \| H1, H5, H_aux.3]` | $`t_0 = 13.7909`$ Gyr; $`t_{\text{ap}} = 317.39`$ Gyr; $`T_{\text{cycle}} = 634.78`$ Gyr | Age $`13.787 \pm 0.020`$ Gyr | Comparison |
| Polarization rotation $`\beta`$ ($`z_* = 1089.8`$) | `[N \| H1, H6]` | $`+0.3025^\circ`$; $`C_\ell^{EB}/C_\ell^{EE} = 1.056 \times 10^{-2}`$ | $`\approx +0.30^\circ`$ to $`+0.34^\circ`$ ($`\pm 0.09^\circ`$) | Comparison |
| Reionization rotation ($`z = 7.7`$) | `[N \| H1, H6]` | $`+0.0936^\circ`$ | No measurement yet | Testable |
| Sub-ring polarization shift | `[N \| H1, H6]` | $`+49.56''`$ | No measurement yet | Testable |

---

## 7. Repository Structure

```text
cbg-cycle1-foundations/
├── .github/workflows/
│   └── verify_cbg.yml                         # CI pipeline running run_all_audits.py
├── Paper_0_Principia_Geometrica/              # Paper 0: axioms, 24 theorems, 94 corollaries   (Zenodo v1, unchanged)
├── Paper_I_Foundations/                       # Paper I: variational principles, field equations (Zenodo v1, unchanged)
├── Paper_II_Black_Hole_Mechanics/             # Paper II: horizon continuation and interior    (Zenodo v1, unchanged)
├── Paper_III_Cosmology/                       # Paper III: homogeneous cosmology, v13           (Zenodo v1, unchanged)
├── audits/                                    # standalone report-only audits (they import no suite and write no suite file)
│   ├── cbg_h01_h22_sympy_audit.py             #   H01-H22: which hypotheses reduce to which
│   ├── cbg_a1_a4_derivability_audit.py        #   "a1a4": spinor parity, mode counts, kernel, CS density
│   ├── cbg_fiber_suction_audit.py             #   "suction": turnaround and contraction in W_3
│   ├── cbg_autonomous_cycle_audit.py          #   "autonomous": khronon, dust, still points, radiation-curvature bound
│   ├── cbg_paper0_18stage_cycle_probe.py      #   "probe": 18 stages without H3 (report)
│   ├── cbg_complete_18stage_dualbranch_audit.py   # "18stage": 18 + 18 stages with H1, H2, H3
│   ├── cbg_bitemporal_t2_throat_audit.py      #   "t2_audit": relaxing d/dt2 = 0
│   ├── cbg_mirror_retrace_mpmath.py           #   40-digit mirror retrace (tens of minutes)
│   └── mp_result.txt                          #   its recorded result
├── CITATION.cff
├── LICENSE                                    # MIT (software) + CC BY 4.0 (monographs and figures)
├── README.md
├── cbg_readme_architecture_audit.py           # tier-by-tier audit of this README
├── requirements.txt                           # numpy, scipy, sympy, mpmath
└── run_all_audits.py                          # tier audit, standalone audits, then the newest suite in each Paper_* folder
```

| Monograph | Directory | Suite status | Zenodo DOI |
| :--- | :--- | :---: | :--- |
| **Paper 0:** *Principia Geometrica of Covariant Bitemporal Geometrodynamics* | [`Paper_0_Principia_Geometrica/`](Paper_0_Principia_Geometrica/) | 24 theorems, 94 corollaries pass | [`10.5281/zenodo.23829404`](https://doi.org/10.5281/zenodo.23829404) |
| **Paper I:** *Covariant Bitemporal Geometrodynamics (CBG): Variational Principles, 5D Laplacian Field Equations, and the Informational Resolution of Cosmological and Quantum Singularities* | [`Paper_I_Foundations/`](Paper_I_Foundations/) | all checks pass | [`10.5281/zenodo.23443511`](https://doi.org/10.5281/zenodo.23443511) |
| **Paper II:** *Relativistic Black Hole Mechanics in Covariant Bitemporal Geometrodynamics* | [`Paper_II_Black_Hole_Mechanics/`](Paper_II_Black_Hole_Mechanics/) | 184/184 checks pass | [`10.5281/zenodo.23451973`](https://doi.org/10.5281/zenodo.23451973) |
| **Paper III:** *Relativistic Bitemporal Cosmology in Covariant Bitemporal Geometrodynamics* | [`Paper_III_Cosmology/`](Paper_III_Cosmology/) | 157/157 checks pass | [`10.5281/zenodo.23458461`](https://doi.org/10.5281/zenodo.23458461) |

---

## 8. Quick Start

```bash
git clone https://github.com/CBG-physics/cbg-cycle1-foundations.git
cd cbg-cycle1-foundations
pip install -r requirements.txt

# 1. Tier-by-tier audit of this README (runs the Paper III suite and the standalone audits it cites; a few minutes):
python cbg_readme_architecture_audit.py
python cbg_readme_architecture_audit.py --no-delegate --no-audits      # symbolic tiers only, under a minute

# 2. Everything: tier audit, standalone audits, and all four monograph suites (Paper 0 and Paper I take several minutes each):
python run_all_audits.py
python run_all_audits.py --core-only                                   # tier audit only

# 3. The four published suites, one by one (run the verify_cbg_*.py file shipped in each folder):
python Paper_0_Principia_Geometrica/verify_cbg_paper0.py
python Paper_I_Foundations/verify_cbg_paper1.py
python Paper_II_Black_Hole_Mechanics/verify_cbg_paper2.py
python Paper_III_Cosmology/verify_cbg_paper3.py

# 4. The standalone audits, one by one:
python audits/cbg_h01_h22_sympy_audit.py
python audits/cbg_a1_a4_derivability_audit.py
python audits/cbg_fiber_suction_audit.py
python audits/cbg_autonomous_cycle_audit.py
python audits/cbg_paper0_18stage_cycle_probe.py
python audits/cbg_complete_18stage_dualbranch_audit.py
python audits/cbg_bitemporal_t2_throat_audit.py
python audits/cbg_mirror_retrace_mpmath.py                             # 40 digits; tens of minutes
```

---

## 9. Citation & License

If you use this framework or its verification suites, please cite:

> Jiménez Velasco, M. (2026). *Principia Geometrica of Covariant Bitemporal Geometrodynamics (Paper 0)*. Zenodo. [doi:10.5281/zenodo.23829404](https://doi.org/10.5281/zenodo.23829404)

<details>
<summary>BibTeX</summary>

```bibtex
@misc{JimenezVelasco2026CBGPaper0,
  author    = {Jiménez Velasco, Manuel},
  title     = {Principia Geometrica of Covariant Bitemporal Geometrodynamics (Paper 0)},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23829404}
}
```

</details>

Citation metadata for all four monographs is in [`CITATION.cff`](CITATION.cff).

| Content | License |
| :--- | :--- |
| Verification software (`*.py`) | [MIT](LICENSE) |
| Monographs and LaTeX sources (`*.pdf`, `*.tex`) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), © 2026 Manuel Jiménez Velasco |
