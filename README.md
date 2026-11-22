# Covariant Bitemporal Geometrodynamics (CBG) — Cycle I Foundations
### Euclidean-Deductive Monographs & Symbolic/Numerical Verification Suites (`[A]` / `[D]` / `[H]` / `[N]`)

[![CBG Verification Suite](https://github.com/CBG-physics/cbg-cycle1-foundations/actions/workflows/verify_cbg.yml/badge.svg)](https://github.com/CBG-physics/cbg-cycle1-foundations/actions/workflows/verify_cbg.yml)
[![Paper 0 DOI](https://img.shields.io/badge/Zenodo%20Paper%200-10.5281%2Fzenodo.23829404-blue)](https://doi.org/10.5281/zenodo.23829404)
[![Paper I DOI](https://img.shields.io/badge/Zenodo%20Paper%20I-10.5281%2Fzenodo.23443511-blue)](https://doi.org/10.5281/zenodo.23443511)
[![Paper II DOI](https://img.shields.io/badge/Zenodo%20Paper%20II-10.5281%2Fzenodo.23451973-blue)](https://doi.org/10.5281/zenodo.23451973)
[![Paper III DOI](https://img.shields.io/badge/Zenodo%20Paper%20III%20(v12)-10.5281%2Fzenodo.23458461-blue)](https://doi.org/10.5281/zenodo.23458461)

**Author:** Manuel Jiménez Velasco (*M.Sc. in Civil Engineering, UPC & Independent Researcher in Theoretical Physics, Madrid, Spain — `manujimevela@gmail.com`*)

---

## Overview

This repository hosts the foundational LaTeX monographs, compiled PDFs, and self-contained 64-bit Python symbolic (`SymPy`) and numerical (`SciPy` `solve_bvp` / `DOP853` / `RK45` / `brentq`) verification suites for **Cycle I** of **Covariant Bitemporal Geometrodynamics (CBG)** on the 5-dimensional orbifold bundle:

$$\mathcal{M}_{3,2} = \left(\mathbb{R}_{t_1} \times_{\Phi} S^1_{t_2}/\mathbb{Z}_2\right) \times \Sigma_3, \qquad \text{signature } (-,-,+,+,+), \qquad (d_t, d_s, D) = (2, 3, 5).$$

Every equation, theorem, and numerical benchmark across the corpus is classified under a strict four-tier epistemological protocol:
- **`[A]` Foundational Axiom:** The three structural axioms of *Paper 0* (`[A1]` 5D orbifold arena $\mathcal{M}_{3,2}$; `[A2]` spectral-action UV winding finiteness; `[A3]` Killing horizon curvature regularity $|\mathcal{K}_5(r_h)| < \infty$).
- **`[D]` Pure 5D Mathematical Deduction:** Exact rational symbolic proofs (`SymPy`) or zero-residual 5D ODE/BVP integrations (`< 1e-9` ODE-independent 4th-order finite-difference and Hamiltonian constraint residuals) without external ad-hoc functions.
- **`[H]` Constitutive Hypothesis:** Explicitly declared bridge postulates and constitutive effective models (e.g., projective cylinder constraint $\mathcal{L}_{\xi_{(2)}} = 0$, vacuum fiber bridge $\Phi_\infty$, bundle thresholds, Hayward UV core, five-sector cosmological effective model & smooth $S^3$ apogee exhaustion/reflection, $+0.3025^\circ$ Gauss–Bonnet birefringence holonomy).
- **`[N]` Empirical / CODATA Input:** Observed physical constants, cosmological baselines, and empirical scaling benchmarks.

---

## Published Foundational Monographs (Cycle I)

| Treatise | Version | Folder | Verification Status | Zenodo DOI |
| :--- | :---: | :--- | :---: | :--- |
| **Paper 0:** *Principia Geometrica of CBG* | `v4` | [`Paper_0_Principia_Geometrica/`](Paper_0_Principia_Geometrica/) | **24 Theorems, 94 Corollaries PASSED** | [`10.5281/zenodo.23829404`](https://doi.org/10.5281/zenodo.23829404) |
| **Paper I:** *Foundational Principles, Frobenius Black Holes & Galaxies* | `v10` | [`Paper_I_Foundations/`](Paper_I_Foundations/) | **100% PASSED** | [`10.5281/zenodo.23443511`](https://doi.org/10.5281/zenodo.23443511) |
| **Paper II:** *Relativistic Black Hole Mechanics, Interior Continuation & Kasner Inversion* | `v10` | [`Paper_II_Black_Hole_Mechanics/`](Paper_II_Black_Hole_Mechanics/) | **184/184 Checks PASSED (0 Failures)** | [`10.5281/zenodo.23451973`](https://doi.org/10.5281/zenodo.23451973) |
| **Paper III:** *Relativistic Bitemporal Cosmology, $k=+1$ Bounce, Universal BH Hair & $+0.3025^\circ$ Birefringence* | `v12` | [`Paper_III_Cosmology/`](Paper_III_Cosmology/) | **157/157 Checks PASSED (0 Failures)** | [`10.5281/zenodo.23458461`](https://doi.org/10.5281/zenodo.23458461) |

---

## Quick Start (Reproduce All Proofs Locally)

```bash
git clone [https://github.com/CBG-physics/cbg-cycle1-foundations.git](https://github.com/CBG-physics/cbg-cycle1-foundations.git)
cd cbg-cycle1-foundations
pip install -r requirements.txt

# Execute all symbolic & numerical verification suites automatically (Paper 0 -> Paper III):
python run_all_audits.py
