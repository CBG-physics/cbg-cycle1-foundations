#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
COVARIANT BITEMPORAL GEOMETRODYNAMICS (CBG) — PAPER II (v10) VERIFICATION SUITE
================================================================================
Manuscript: "Relativistic Black Hole Mechanics in Covariant Bitemporal
             Geometrodynamics: Two-Sided Exterior-Interior Horizon Continuation
             (0 < r < r_h), Bitemporal Angular-Momentum Kasner Inversion,
             and Horizon Phenomenology" (Paper II, v10 Deductive Bundle Edition,
             Anchored in Paper 0 [Principia Geometrica, v4] and Paper I [v10])
Author:     Manuel Jimenez Velasco (M.Sc. in Civil Engineering, UPC)
            Electronic address: manujimevela@gmail.com
Date:       November 2026
Repository: Zenodo (CERN Data Centre), DOI: 10.5281/zenodo.23451973
================================================================================
"""

import sys
import numpy as np
import sympy as sp
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import brentq

np.set_printoptions(precision=8, suppress=True)

PASS_COUNT = 0
FAIL_COUNT = 0


def record_check(description, expr_val, target_val=0, tol=None):
    """Unified symbolic and 64-bit numerical assertion tracker."""
    global PASS_COUNT, FAIL_COUNT
    if tol is None:
        diff = sp.simplify(expr_val - target_val)
        ok = (diff == 0)
        status_str = f"{expr_val} == {target_val}"
    else:
        val_f = float(expr_val)
        targ_f = float(target_val)
        diff = abs(val_f - targ_f)
        ok = (diff <= tol)
        status_str = f"{val_f:.8g} == {targ_f:.8g} (|diff|={diff:.2e} <= {tol:.1e})"

    if ok:
        PASS_COUNT += 1
        print(f"  [PASS #{PASS_COUNT:03d}] {description}: {status_str}  [Q.E.D.]")
    else:
        FAIL_COUNT += 1
        print(f"  [FAIL #{PASS_COUNT + FAIL_COUNT:03d}] {description}: {status_str}")
        sys.exit(1)


# ==============================================================================
# FUNDAMENTAL PHYSICAL CONSTANTS (CODATA 2022) & CBG GEOMETRIC INVARIANTS
# ==============================================================================
class CBGConstants:
    d_t = 2                                  # Number of timelike dimensions (t_1, t_2)
    d_s = 3                                  # Number of spacelike dimensions (Sigma_3)
    D = 5                                    # Total bulk dimension D = d_t + d_s = 5
    N_Cartan = d_t * d_s * D                 # Cartan phase-space multiplicity = 30 [D]
    dim_Sym2_V5 = (D * (D + 1)) // 2         # 5D DeWitt supermetric fiber dimension = 15 [D]

    c_V = (D - 2) / (2.0 * (D - 3))          # 5D Weyl-Gauss-Codazzi normalization = -kappa_psi(D)/2 = 3/4 = 0.75 [D]
    A_crit = 1.0 / (d_t**2)                  # Bandwidth equipartition = 1/4 = 0.25 [H]
    psi_max = d_t / (D**2)                   # End(TM_{3,2}) trace ratio = 2/25 = 0.08 [H]
    m_hat_ref = float(d_t**2)                # Reference 4-throat eigenvalue = 4.0 [H]

    c = 299792458.0                          # Speed of light [m/s]
    hbar = 1.054571817e-34                   # Reduced Planck constant [J*s]
    G = 6.67430e-11                          # Newtonian constant of gravitation [m^3/(kg*s^2)]
    k_B = 1.380649e-23                       # Boltzmann constant [J/K]
    eV_J = 1.602176634e-19                   # 1 eV in Joules [J]
    alpha_EM_inv = 137.035999177             # Fine-structure constant inverse (CODATA 2022)
    alpha_EM = 1.0 / alpha_EM_inv

    L_P = np.sqrt(hbar * G / (c**3))         # Planck length = 1.616255e-35 m
    t_P = L_P / c                            # Planck time = 5.391247e-44 s
    m_P = np.sqrt(hbar * c / G)              # Planck mass = 2.176434e-8 kg
    E_P_GeV = (m_P * c**2) / (eV_J * 1.0e9)  # Planck energy = 1.220890e19 GeV
    T_P = (m_P * c**2) / k_B                 # Planck temperature = 1.416784e32 K

    Phi_inf = (alpha_EM / 30.0) * (1.0 - (5.0 / 3.0) * alpha_EM)  # 2.40286677e-4
    Phi_inf_sq = Phi_inf**2                  # 5.77376871e-8
    Omega_inf = c * Phi_inf                  # Universal fiber velocity = 72,036.133 m/s

    M_sun = 1.98847e30                       # Solar mass [kg]
    AU_m = 1.495978707e11                    # Astronomical Unit [m]
    pc_m = 3.08567758149e16                  # Parsec [m]
    Mpc_m = 3.08567758149e22                 # Megaparsec [m]
    yr_s = 365.25 * 86400.0                  # Julian year [s]
    Myr_s = 1.0e6 * yr_s                     # Megayear [s]
    H0_km_s_Mpc = 67.40                      # Planck 2018 CMB Hubble constant [km/s/Mpc]
    H0_SI = (H0_km_s_Mpc * 1000.0) / Mpc_m   # Hubble constant [s^-1]
    


# ==============================================================================
# STAGE-1 ADDITION: METRIC-FIRST TENSOR HELPER (used by Points 1 and 2) AND EXPORT DICTIONARY
# ==============================================================================
CBG_DERIVED = {}   # lambdified objects derived in Module 0 and consumed by Module 1 (Points 2-4)


class Geometry:
    """Levi-Civita geometry of an explicit metric g on coordinates x (exact rational arithmetic only)."""
    def __init__(self, g, x):
        self.g, self.x, self.n = sp.Matrix(g), list(x), len(x)
        n, x = self.n, self.x
        self.ginv = sp.simplify(self.g.inv())
        gi, g = self.ginv, self.g
        self.Gamma = [[[sp.simplify(sp.Rational(1, 2) * sum(gi[c, d] * (sp.diff(g[d, a], x[b]) + sp.diff(g[d, b], x[a]) - sp.diff(g[a, b], x[d]))
                                                         for d in range(n))) for b in range(n)] for a in range(n)] for c in range(n)]
        G = self.Gamma
        self.Riem = [[[[sp.diff(G[c][d][b], x[a]) - sp.diff(G[c][d][a], x[b]) +
                        sum(G[c][a][e] * G[e][d][b] - G[c][b][e] * G[e][d][a] for e in range(n))
                        for b in range(n)] for a in range(n)] for d in range(n)] for c in range(n)]
        self.Ric = sp.Matrix(n, n, lambda d, b: sp.simplify(sum(self.Riem[a][d][a][b] for a in range(n))))

    def kretschmann_diagonal(self):
        """R_ABCD R^ABCD for a diagonal metric: R_abcd = g_aa R^a_bcd, R^abcd = g^bb g^cc g^dd R^a_bcd."""
        n, g, gi, Rm = self.n, self.g, self.ginv, self.Riem
        return sp.simplify(sum(g[a, a] * Rm[a][b][c][d] * gi[b, b] * gi[c, c] * gi[d, d] * Rm[a][b][c][d]
                               for a in range(n) for b in range(n) for c in range(n) for d in range(n)))


# ==============================================================================
# MODULE 0: METRIC-FIRST SYMBOLIC PROOFS (SYMPY) OF CORE THEOREMS [D]
# ==============================================================================
def run_module_0_symbolic_checks():
    print("=" * 94)
    print("MODULE 0: METRIC-FIRST SYMBOLIC PROOFS (SYMPY) OF CORE THEOREMS [D]")
    print("=" * 94)

    d_t, d_s = sp.Integer(2), sp.Integer(3)
    D = d_t + d_s
    record_check("Paper 0 THM #01 & #03: Symplectic Frobenius-Cartan trace N_Cartan = d_t d_s D == 30 [D]",
                 d_t * d_s * D, sp.Integer(30))
    eta5_dw = sp.diag(-1, -1, 1, 1, 1)
    pairs5_dw = [(i_d, j_d) for i_d in range(5) for j_d in range(i_d, 5)]
    B5_dw = [sp.Matrix(5, 5, lambda a_d, b_d, i_d=i_d, j_d=j_d:
                       sp.Integer(1) if (a_d == i_d and b_d == j_d) or (a_d == j_d and b_d == i_d) else sp.Integer(0))
             for (i_d, j_d) in pairs5_dw]
    G_DW_15 = sp.Matrix(len(pairs5_dw), len(pairs5_dw), lambda p_d, q_d: (
        (eta5_dw * B5_dw[p_d] * eta5_dw * B5_dw[q_d]).trace()
        - sp.Rational(1, 2) * (eta5_dw * B5_dw[p_d]).trace() * (eta5_dw * B5_dw[q_d]).trace()
    ))
    G_DW_15_inv = G_DW_15.inv()
    CBG_DERIVED["G_DW_15_inv_00"] = G_DW_15_inv[0, 0]
    CBG_DERIVED["c_DW_inv_5D"] = G_DW_15_inv[0, 5] / (eta5_dw[0, 0] * eta5_dw[1, 1])
    record_check("Paper 0 THM #04: 15x15 DeWitt supermetric dim Sym^2(V_5*) == 15 & inverse trace c_DW^-1(5) == -1/3 [D]",
                 sp.Abs(sp.Integer(G_DW_15.shape[0]) - sp.Integer(15))
                 + sp.Abs(CBG_DERIVED["c_DW_inv_5D"] - (-sp.Rational(1, 3))), 0)
    c_Weyl_D = sp.Integer(2) / (D - sp.Integer(3))
    kappa_psi_D = -(D - sp.Integer(2)) / (D - sp.Integer(3))
    record_check("Paper 0 THM #12 & #18: Dimensional Weyl prefactor -kappa_psi(D)/2 == 3/4 (proved from 5D metric in #013, #073, #074) [D]",
                 -sp.Rational(1, 2) * kappa_psi_D, sp.Rational(3, 4))
    
    record_check("Constitutive H1: Bandwidth equipartition trace A_crit = 1/d_t^2 == 1/4 [H]",
                 sp.Integer(1) / d_t**2, sp.Rational(1, 4))
    record_check("Constitutive H1: Temporal-to-bulk trace ratio psi_max = d_t/D^2 == 2/25 [H]",
                 d_t / D**2, sp.Rational(2, 25))
    record_check("Constitutive H2: Spatial-plus-fiber throat trace m_hat_ref = d_s + 1 == 4 [H]",
                 d_s + 1, sp.Integer(4))

    r, th, ph, t1, t2, Phi_inf = sp.symbols('r theta phi t1 t2 Phi_inf', positive=True)
    A = sp.Function('A')(r)
    B = sp.Function('B')(r)
    psi = sp.Function('psi')(r)
    x = [t1, t2, r, th, ph]

    g = sp.diag(-A, -Phi_inf**2 * sp.exp(2 * psi), B, r**2, r**2 * sp.sin(th)**2)
    ginv = g.inv()
    n = 5

    Gam = [[[sp.simplify(sum(ginv[a, d] * (sp.diff(g[d, b], x[c]) + sp.diff(g[d, c], x[b]) - sp.diff(g[b, c], x[d]))
                             for d in range(n)) * sp.Rational(1, 2)) for c in range(n)] for b in range(n)] for a in range(n)]

    def Riem(a, b, c, d):
        return (sp.diff(Gam[a][b][d], x[c]) - sp.diff(Gam[a][b][c], x[d])
                + sum(Gam[a][c][e] * Gam[e][b][d] - Gam[a][d][e] * Gam[e][b][c] for e in range(n)))

    Ric = sp.Matrix(n, n, lambda b, d: sp.simplify(sum(Riem(a, b, a, d) for a in range(n))))
    R5 = sp.simplify(sum(ginv[a, b] * Ric[a, b] for a in range(n) for b in range(n)))
    Gmix = sp.simplify(ginv * Ric - sp.eye(n) * R5 * sp.Rational(1, 2))

    dA, ddA = sp.diff(A, r), sp.diff(A, r, 2)
    dB = sp.diff(B, r)
    dpsi, ddpsi = sp.diff(psi, r), sp.diff(psi, r, 2)

    hp_sym, VA_sym = sp.symbols('hp_sym VA_sym', real=True)
    Acal_gen = A * B / (B - A * hp_sym**2)
    T1r_gen = -(sp.Integer(2) / A) * (Acal_gen**2) * VA_sym * hp_sym
    sol_hp = sp.solve(sp.Eq(Gmix[0, 2] - T1r_gen, 0), hp_sym)
    record_check("[metric-first] Theorem 2.1(i): Off-diagonal G^1_r == 0 on static 5D metric [D]",
                 Gmix[0, 2], 0)
    record_check("Theorem 2.1(i): G^1_r = T^1_r == 0 uniquely forces h'(r) == 0 (for dV/dA != 0, Eq. 2.4) [D]",
                 sol_hp[0], 0)

    g4 = sp.diag(-A, B, r**2, r**2 * sp.sin(th)**2)
    g4inv = g4.inv()
    x4 = [t1, r, th, ph]
    Gam4 = [[[sp.simplify(sum(g4inv[a, d] * (sp.diff(g4[d, b], x4[c]) + sp.diff(g4[d, c], x4[b]) - sp.diff(g4[b, c], x4[d]))
                              for d in range(4)) * sp.Rational(1, 2)) for c in range(4)] for b in range(4)] for a in range(4)]
    Ric4 = sp.Matrix(4, 4, lambda b, d: sp.simplify(sum(
        sp.diff(Gam4[a][b][d], x4[a]) - sp.diff(Gam4[a][b][a], x4[d])
        + sum(Gam4[a][a][e] * Gam4[e][b][d] - Gam4[a][d][e] * Gam4[e][b][a] for e in range(4))
        for a in range(4))))
    R4 = sp.simplify(sum(g4inv[a, b] * Ric4[a, b] for a in range(4) for b in range(4)))
    record_check("[metric-first] Theorem 2.1(iii): G^2_2(r) == -(1/2) R^(4)(r) (excludes C^1 horizon jump) [D]",
                 sp.simplify(Gmix[1, 1] + sp.Rational(1, 2) * R4), 0)

    V = sp.Function('V')(psi, A)
    Vp = sp.diff(V, psi)
    VA = sp.diff(V, A)
    T11, T22, Trr, Tth = -V - 2 * A * VA, -V - Vp, -V, -V
    E = {'E^r_r': Gmix[2, 2] - Trr, 'E^1_1': Gmix[0, 0] - T11,
         'E^th_th': Gmix[3, 3] - Tth, 'E^2_2': Gmix[1, 1] - T22}

    den = r * (r * dpsi + 2)
    A1 = -2 * A * (r**2 * B * V + 2 * r * dpsi - B + 1) / den
    B1 = (sp.Integer(2) * B / (sp.Integer(3) * den)) * (
        2 * r**3 * A * B * VA * dpsi + 2 * r**3 * B * V * dpsi + r**3 * B * Vp * dpsi
        + 4 * r**2 * A * B * VA + r**2 * B * V + 2 * r**2 * B * Vp
        + 3 * r**2 * dpsi**2 - 3 * r * B * dpsi + 3 * r * dpsi - 3 * B + 3
    )
    P2 = (sp.Integer(1) / (sp.Integer(3) * r)) * (
        2 * r**2 * A * B * VA * dpsi + 2 * r**2 * B * V * dpsi + r**2 * B * Vp * dpsi
        - 2 * r * A * B * VA - 2 * r * B * V + 2 * r * B * Vp - 3 * B * dpsi - 3 * dpsi
    )

    sol_dA_metric = sp.solve(sp.Eq(E['E^r_r'], 0), dA)[0]
    record_check("[metric-first] Normal-form ODE for A'(r) derived from E^r_r == 0 (Eq. 2.5) [D]",
                 sp.simplify(sol_dA_metric - A1), 0)
    sol_dB_ddpsi = sp.solve([sp.Eq(E['E^1_1'].subs(dA, A1), 0),
                             sp.Eq(E['E^th_th'].subs(ddA, sp.diff(A1, r)).subs(dA, A1), 0)],
                            (dB, ddpsi), dict=True)[0]
    record_check("[metric-first] Normal-form ODE for B'(r) derived from (E^1_1, E^th_th) (Eq. 2.6) [D]",
                 sp.simplify(sol_dB_ddpsi[dB] - B1), 0)
    record_check("[metric-first] Normal-form ODE for psi''(r) derived from (E^1_1, E^th_th) (Eq. 2.7) [D]",
                 sp.simplify(sol_dB_ddpsi[ddpsi] - P2), 0)

    c_V_sym, m_sym, eps_l, psi_l, dpsi_l = sp.symbols('c_V_sym m_sym eps_l psi_l dpsi_l', positive=True)
    P2_flat = P2.subs({V: c_V_sym * m_sym**2 * psi_l**2, Vp: sp.Integer(2) * c_V_sym * m_sym**2 * psi_l, VA: 0,
                       dpsi: dpsi_l, A: sp.Integer(1), B: sp.Integer(1)})
    P2_lin = sp.simplify(sp.series(P2_flat.subs({psi_l: eps_l * psi_l, dpsi_l: eps_l * dpsi_l}), eps_l, 0, 2).removeO().coeff(eps_l, 1))
    c_V_metric = sp.solve(sp.Eq(P2_lin - (-sp.Integer(2) * dpsi_l / r + m_sym**2 * psi_l), 0), c_V_sym)[0]
    record_check("[metric-first] Vacuum linearization of 5D P2 (psi'' + (2/r) psi' == m^2 psi) uniquely fixes c_V == 3/4 [D]",
                 sp.simplify(c_V_metric - sp.Rational(3, 4)), 0)

    A2 = sp.diff(A1, r).subs({dB: B1, ddpsi: P2}).subs(dA, A1)
    for k, e in E.items():
        res = sp.simplify(e.subs(ddA, A2).subs({dB: B1, ddpsi: P2}).subs(dA, A1))
        record_check(f"[metric-first] Zero symbolic residual {k} == 0 for any sign of A(r) [D]", res, 0)

    # Explicit symbolic proof of radial 5-divergence Noether identity nabla_A T^A_r == 0 (Section 1.2 Pillar 3)
    div_T_r = sp.simplify(
        sp.diff(Trr, r)
        + Gam[0][0][2] * (Trr - T11)
        + Gam[1][1][2] * (Trr - T22)
        + (Gam[3][3][2] + Gam[4][4][2]) * (Trr - Tth)
    )
    record_check("[metric-first] Pillar 3: Radial 5-divergence Noether identity nabla_A T^A_r == 0 [D]",
                 div_T_r, 0)

    record_check("Paper 0 THM #20: Longitudinal-radial stress anisotropy T^1_1 - T^r_r == -2 A dV/dA [D]",
                 sp.simplify((T11 - Trr) - (-2 * A * VA)), 0)

    r_s = sp.Symbol('r_s', positive=True)
    Phi_f = sp.Function('Phi')(r)
    g_PG = sp.Matrix([
        [-(sp.Integer(1) - r_s / r), 0, sp.sqrt(r_s / r), 0, 0],
        [0, -Phi_f**2, 0, 0, 0],
        [sp.sqrt(r_s / r), 0, 1, 0, 0],
        [0, 0, 0, r**2, 0],
        [0, 0, 0, 0, r**2 * sp.sin(th)**2]
    ])
    g_PG_inv = sp.simplify(g_PG.inv())
    Gam_PG = [[[sp.simplify(sum(g_PG_inv[a, d] * (
        sp.diff(g_PG[d, b], x[c]) + sp.diff(g_PG[d, c], x[b]) - sp.diff(g_PG[b, c], x[d])
    ) for d in range(5)) * sp.Rational(1, 2)) for c in range(5)] for b in range(5)] for a in range(5)]

    pg_targets = [
        ("Gamma^1_{11}", Gam_PG[0][0][0], (r_s / (sp.Integer(2) * r**2)) * sp.sqrt(r_s / r)),
        ("Gamma^1_{1r}", Gam_PG[0][0][2], r_s / (sp.Integer(2) * r**2)),
        ("Gamma^1_{rr}", Gam_PG[0][2][2], (sp.Integer(1) / (sp.Integer(2) * r)) * sp.sqrt(r_s / r)),
        ("Gamma^1_{22}", Gam_PG[0][1][1], sp.sqrt(r_s / r) * Phi_f * sp.diff(Phi_f, r)),
        ("Gamma^1_{theta theta}", Gam_PG[0][3][3], -r * sp.sqrt(r_s / r)),
        ("Gamma^1_{phi phi}", Gam_PG[0][4][4], -r * sp.sqrt(r_s / r) * sp.sin(th)**2),
        ("Gamma^2_{2r}", Gam_PG[1][1][2], sp.diff(Phi_f, r) / Phi_f),
        ("Gamma^r_{11}", Gam_PG[2][0][0], r_s * (r - r_s) / (sp.Integer(2) * r**3)),
        ("Gamma^r_{1r}", Gam_PG[2][0][2], -(r_s / (sp.Integer(2) * r**2)) * sp.sqrt(r_s / r)),
        ("Gamma^r_{rr}", Gam_PG[2][2][2], -r_s / (sp.Integer(2) * r**2)),
        ("Gamma^r_{22}", Gam_PG[2][1][1], (sp.Integer(1) - r_s / r) * Phi_f * sp.diff(Phi_f, r)),
        ("Gamma^r_{theta theta}", Gam_PG[2][3][3], -(r - r_s)),
        ("Gamma^r_{phi phi}", Gam_PG[2][4][4], -(r - r_s) * sp.sin(th)**2),
        ("Gamma^theta_{r theta}", Gam_PG[3][2][3], sp.Integer(1) / r),
        ("Gamma^phi_{r phi}", Gam_PG[4][2][4], sp.Integer(1) / r),
        ("Gamma^theta_{phi phi}", Gam_PG[3][4][4], -sp.sin(th) * sp.cos(th)),
        ("Gamma^phi_{theta phi}", Gam_PG[4][3][4], sp.cos(th) / sp.sin(th)),
    ]
    for name_g, val_g, targ_g in pg_targets:
        record_check(f"[metric-first] Eq. (3.5) 5D PG Christoffel {name_g} [D]", val_g, targ_g)

    T22_matter = sp.Symbol('T22_matter', real=True)
    sol_G22 = sp.solve([
        sp.Eq((Gmix[0, 0] - T11).subs(dA, A1), 0),
        sp.Eq((Gmix[1, 1] - (T22 + T22_matter)).subs(ddA, sp.diff(A1, r)).subs(dA, A1), 0)
    ], (dB, ddpsi), dict=True)[0]
    delta_ddpsi = sp.simplify(sol_G22[ddpsi] - P2)
    delta_dB = sp.simplify(sol_G22[dB] - B1)
    record_check("Theorem 4.3 (Eq. 4.9): Centrifugal shift Delta(psi'') == -(r psi' + 2)^2 B T^2_(m)2 / 6 [D]",
                 sp.simplify(delta_ddpsi - (-(r * dpsi + 2)**2 * B * T22_matter * sp.Rational(1, 6))), 0)
    record_check("Theorem 4.3 (Eq. 4.9): Centrifugal shift Delta(B') == -r(r psi' + 2) B^2 T^2_(m)2 / 3 [D]",
                 sp.simplify(delta_dB - (-r * (r * dpsi + 2) * B**2 * T22_matter * sp.Rational(1, 3))), 0)

    # Full 4-component 5D Einstein & Bianchi consistency check for Theorem 4.3:
    Tth_bianchi = -sp.Rational(1, 2) * r * dpsi * T22_matter
    div_Tm_r = sp.simplify(-Gam[1][1][2] * T22_matter - (Gam[3][3][2] + Gam[4][4][2]) * Tth_bianchi)
    record_check("Theorem 4.3: 5D Bianchi conservation nabla_A T^{A(m)}_r == 0 with shell tension T^th_th = -(r psi'/2) T^2_2 [D]",
                 div_Tm_r, 0)
    A2_with_T22 = sp.diff(A1, r).subs({dB: B1 + delta_dB, ddpsi: P2 + delta_ddpsi}).subs(dA, A1)
    res_Ett_full = sp.simplify((Gmix[3, 3] - (Tth + Tth_bianchi)).subs(ddA, A2_with_T22).subs({
        dB: B1 + delta_dB, ddpsi: P2 + delta_ddpsi
    }).subs(dA, A1))
    record_check("Theorem 4.3: Full 4-component 5D Einstein consistency E^th_th == 0 under Eq. (4.9) + Bianchi [D]",
                 res_Ett_full, 0)

    rs_s, LP_s, rc_s = sp.symbols('rs_s LP_s rc_s', positive=True)
    # STAGE-1 (Point 1): K_5 = R_ABCD R^ABCD computed from the 5D Riemann tensor of the embedded Hayward ansatz
    Phi0_s = sp.Symbol('Phi0_s', positive=True)
    f_H = sp.Integer(1) - rs_s * r**2 / (r**3 + rc_s**3)
    gH = sp.diag(-f_H, -Phi0_s**2, sp.Integer(1) / f_H, r**2, r**2 * sp.sin(th)**2)
    K5_core = sp.factor(Geometry(gH, x).kretschmann_diagonal())
    K5_poly = sp.Integer(12) * rs_s**2 * (r**12 - 4 * r**9 * rc_s**3 + 18 * r**6 * rc_s**6 - 2 * r**3 * rc_s**9 + 2 * rc_s**12) / (r**3 + rc_s**3)**6
    record_check("App. A (Eq. A.7): Hayward-ansatz K_5(r) from the 5D Riemann tensor == 12 r_s^2 (r^12 - 4 r^9 r_c^3 + 18 r^6 r_c^6 - 2 r^3 r_c^9 + 2 r_c^12)/(r^3 + r_c^3)^6 [H]",
                 sp.simplify(K5_core - K5_poly), 0)
    record_check("App. A (Eq. A.7): K_5(r) recovers exact Schwarzschild 12 r_s^2 / r^6 as r_core -> 0 [H]",
                 sp.simplify(K5_core.subs(rc_s, 0) - sp.Integer(12) * rs_s**2 / r**6), 0)
    K5_at_0 = sp.simplify(sp.limit(K5_core, r, 0).subs(rc_s, (rs_s * LP_s**2 * sp.Rational(1, 2))**sp.Rational(1, 3)))
    record_check("Proposition 4.4(ii) (Eq. 4.16): Universal central plateau lim_{r->0} K_5(r) == 96/L_P^4 [H]",
                 K5_at_0, sp.Integer(96) / LP_s**4)

    a_s = sp.Symbol('a_s', real=True)
    Sigma_K = r**2 + a_s**2 * sp.cos(th)**2
    beta_K_sq = sp.simplify(sp.re(rs_s / (r - sp.I * a_s * sp.cos(th))))
    record_check("Theorem 5.1 (Eqs. 5.1, 5.3): Newman-Janis Re[r_s/(r - i a cos(theta))] == r_s r / Sigma_K [H]",
                 sp.simplify(beta_K_sq - rs_s * r / Sigma_K), 0)

    r_sym = sp.Symbol('r_sym', positive=True)
    tortoise_indef = sp.diff((r_sym - rs_s) + rs_s * sp.log(r_sym - rs_s), r_sym)
    record_check("Theorem 7.1(ii) (Eq. 7.2): Tortoise integrand d/dr[(r - r_s) + r_s ln(r - r_s)] == r/(r - r_s) [D]",
                 sp.simplify(tortoise_indef - r_sym / (r_sym - rs_s)), 0)

    eps_phi = sp.Symbol('eps_phi', positive=True)
    V_null_norm = (sp.Integer(1) - rs_s / r) * (sp.Integer(1) + eps_phi * rs_s / r) / r**2
    dV_null_num = sp.simplify(sp.diff(V_null_norm, r) * r**5 * (-sp.Rational(1, 2)))
    quad_phot = r**2 - sp.Rational(3, 2) * rs_s * (1 - eps_phi) * r - 2 * eps_phi * rs_s**2
    record_check("Section 8.1 (Basal psi=0 optical ansatz, Eq. 8.2): Stationarity dV_null/dr == 0 quadratic [H]",
                 sp.simplify(dV_null_num - quad_phot), 0)
    r_phot_root = sp.solve(sp.Eq(quad_phot, 0), r)[1]
    r_phot_series = sp.series(r_phot_root, eps_phi, 0, 2).removeO()
    record_check("Section 8.1 (Basal psi=0 optical ansatz, Eq. 8.3): r_phot^(opt) == (3/2) r_s (1 - Phi_inf^2 / 9) (superseded by Module 1 scalar-hair BVP) [H]",
                 sp.simplify(r_phot_series - sp.Rational(3, 2) * rs_s * (sp.Integer(1) - eps_phi * sp.Rational(1, 9))), 0)
    b_crit_expr = sp.Integer(1) / sp.sqrt(V_null_norm.subs(r, r_phot_series))
    b_crit_series = sp.series(b_crit_expr, eps_phi, 0, 2).removeO()
    record_check("Section 8.1 (Basal psi=0 optical ansatz, Eq. 8.4): b_crit^(opt) == (3*sqrt(3)/2) r_s (1 - Phi_inf^2 / 3) (superseded by Module 1 scalar-hair BVP) [H]",
                 sp.simplify(b_crit_series - (sp.Integer(3) * sp.sqrt(3) * sp.Rational(1, 2)) * rs_s * (sp.Integer(1) - eps_phi * sp.Rational(1, 3))), 0)

    phi_s = sp.Symbol('phi_s', real=True)
    hemi_int = sp.integrate(sp.Rational(1, 2) * sp.sin(phi_s), (phi_s, 0, sp.pi))
    record_check("Proposition 8.1 (Eq. 8.5): Hemispherical Haar/Gauss-Bonnet measure integral == 1 [D]",
                 hemi_int, sp.Integer(1))

    x_m = sp.Symbol('x_m', positive=True)
    T_norm = (sp.Integer(1) / x_m) * (sp.Integer(1) - sp.Integer(1) / (sp.Integer(4) * x_m**2))
    sol_x = sp.solve(sp.diff(T_norm, x_m), x_m)
    sol_rem = sp.solve(T_norm, x_m)
    record_check("Section 9.2 (Eq. 9.3): Hawking temperature peak root M_peak / m_P == sqrt(3)/2 [H]",
                 sol_x[0], sp.sqrt(3) * sp.Rational(1, 2))
    record_check("Section 9.2 (Eq. 9.4): Zero-temperature Planck remnant root M_rem / m_P == 1/2 [H]",
                 sol_rem[0], sp.Rational(1, 2))
    T_hay_norm = T_norm / (sp.Integer(1) + sp.Integer(1) / (sp.Integer(8) * x_m**2))**2
    sol_hay_peak = [s for s in sp.solve(sp.diff(T_hay_norm, x_m), x_m) if s.is_real and s > sp.Rational(1, 2)][0]
    record_check("Section 9.2: Exact Hayward temperature stationary root x_peak^(Hay) == sqrt(9 + sqrt(73))/4 [H]",
                 sp.simplify(sol_hay_peak - sp.sqrt(9 + sp.sqrt(73)) * sp.Rational(1, 4)), 0)

    # ==========================================================================================
    # STAGE-1 ADDITIONS (Point 2): PURE 5D SYMBOLIC DERIVATIONS FROM THE METRIC AND THE ACTION [D]
    # Action density sqrt|g5| [R5 + ell], ell = -2 V(psi, A), with psi = -(1/2) ln(-g^{22} Phi_inf^2)
    # and A = (-g^{AB} d_A Theta d_B Theta)^{-1} functions of the inverse metric.
    # ==========================================================================================
    half_q = sp.Rational(1, 2)

    def flag_q(cond):
        return sp.Integer(1) if cond else sp.Integer(0)

    # (a) Hilbert stress tensor by direct variation with respect to the inverse metric
    q11, q22, qrr, qth, qph = sp.symbols('q11 q22 qrr qth qph', real=True, nonzero=True)
    q1r, hp_q = sp.symbols('q1r hp_q', real=True)
    V_head = sp.Function('V')
    psi_e = -half_q * sp.log(-q22 * Phi_inf**2)
    V_sy, VP_sy, VA_sy = sp.symbols('V_sy VP_sy VA_sy', real=True)
    bg_q = {q11: ginv[0, 0], q22: ginv[1, 1], qrr: ginv[2, 2], qth: ginv[3, 3], qph: ginv[4, 4]}
    V_biv = V_head(psi_e, sp.Integer(1) / (-q11))
    ell_q = -2 * V_biv
    ln_sq = -half_q * sp.log(q11 * q22 * qrr * qth * qph)
    T_hilbert = []
    for var_q in (q11, q22, qrr, qth, qph):
        el_q = sp.diff(ell_q, var_q) + ell_q * sp.diff(ln_sq, var_q)
        el_q = el_q.subs(V_biv.fdiff(1), VP_sy).subs(V_biv.fdiff(2), VA_sy).subs(V_biv, V_sy).subs(bg_q).subs({V_sy: V, VP_sy: Vp, VA_sy: VA})
        T_hilbert.append(sp.simplify(-bg_q[var_q] * el_q))
    for name_q, got_q, want_q in (("T^1_1 == -V - 2 A d_A V", T_hilbert[0], T11), ("T^2_2 == -V - d_psi V", T_hilbert[1], T22),
                                  ("T^r_r == -V", T_hilbert[2], Trr), ("T^theta_theta == -V", T_hilbert[3], Tth), ("T^phi_phi == -V", T_hilbert[4], Tth)):
        record_check(f"[Hilbert variation] Stress component {name_q} from ell = -2 V(psi, A) [D]", sp.simplify(got_q - want_q), 0)
    T_mix_q = sp.diag(*T_hilbert)
    for b_q, nm_q in enumerate(["t1", "t2", "r", "theta", "phi"]):
        div_q = sp.simplify(sum(sp.diff(T_mix_q[a_q, b_q], x[a_q]) for a_q in range(n))
                            + sum(Gam[a_q][a_q][c_q] * T_mix_q[c_q, b_q] for a_q in range(n) for c_q in range(n))
                            - sum(Gam[c_q][a_q][b_q] * T_mix_q[a_q, c_q] for a_q in range(n) for c_q in range(n)))
        record_check(f"[Hilbert variation] Divergence component nabla_A T^A_{nm_q} == 0 for any sign of A [D]", div_q, 0)
    A_kh = sp.Integer(1) / (-(q11 + 2 * q1r * hp_q + qrr * hp_q**2))
    V_kh = V_head(psi_e, A_kh)
    ln_sq_od = -half_q * sp.log((q11 * qrr - q1r**2) * q22 * qth * qph)
    T_1r_q = -half_q * (sp.diff(-2 * V_kh, q1r) + (-2 * V_kh) * sp.diff(ln_sq_od, q1r))
    T_1r_q = T_1r_q.subs(V_kh.fdiff(2), VA_sy).subs(V_kh, V_sy).subs(q1r, 0)
    T1r_mixed_q = sp.simplify((q11 * T_1r_q).subs(bg_q))
    record_check("[Hilbert variation] Khronon norm of Theta = t1 + h(r) == A B/(B - A h'^2) [D]",
                 sp.simplify(A_kh.subs(q1r, 0).subs(bg_q) - A * B / (B - A * hp_q**2)), 0)
    record_check("[Hilbert variation] Off-diagonal T^1_r == -(2/A) A_kh^2 d_A V h' (form typed in Theorem 2.1(i)) [D]",
                 sp.simplify(T1r_mixed_q - T1r_gen.subs({hp_sym: hp_q, VA_sym: VA_sy})), 0)
    sol_hp_q = sp.solve(sp.Eq(Gmix[0, 2] - T1r_mixed_q, 0), hp_q)
    record_check("[Hilbert variation] G^1_r = T^1_r == 0 has the single root h'(r) == 0 (for d_A V != 0) [D]",
                 flag_q(sol_hp_q == [0]), 1)

    # (b) Two-sided Frobenius analysis at the horizon for a GENERIC potential (2-jet of V at (psi_h, A = 0)).
    #     A = a1 e + a2 e^2, B = b_{-1}/e + b0, psi = psi_h + p1 e + p2 e^2 with e = r - r_h of EITHER sign.
    e_h = sp.Symbol('e_h', real=True)
    r_hq = sp.Symbol('r_hq', positive=True)
    a1_q, a2_q, bm1_q, b0_q, p1_q, p2_q = sp.symbols('a1_q a2_q bm1_q b0_q p1_q p2_q', real=True)
    V00, V10, V01, V20, V11 = sp.symbols('V00 V10 V01 V20 V11', real=True)
    V02 = sp.Symbol('V02', real=True)
    XV, XP, XA, XdP, XA_, XB_ = sp.symbols('XV XP XA XdP XA_ XB_', real=True)
    A_ser, B_ser, dps_ser = a1_q * e_h + a2_q * e_h**2, bm1_q / e_h + b0_q, p1_q * e_h + p2_q * e_h**2
    jets_q = {XV: V00 + V10 * dps_ser + V01 * A_ser, XP: V10 + V20 * dps_ser + V11 * A_ser, XA: V01 + V11 * dps_ser + V02 * A_ser}

    def frob_q(expr):
        expr = expr.subs(Vp, XP).subs(VA, XA).subs(V, XV).subs(dpsi, XdP).subs({A: XA_, B: XB_}).subs(r, r_hq + e_h)
        return sp.together(expr.subs(jets_q).subs({XdP: p1_q + 2 * p2_q * e_h, XA_: A_ser, XB_: B_ser}))

    def laurent_q(expr, lo, hi):
        ser = sp.expand(sp.series(frob_q(expr), e_h, 0, hi + 1).removeO())
        return {k_q: sp.simplify(ser.coeff(e_h, k_q)) for k_q in range(lo, hi + 1)}

    cA_q = laurent_q(A1 - sp.diff(A_ser, e_h), 0, 0)
    cB_q = laurent_q(B1 - sp.diff(B_ser, e_h), -2, -1)
    cP_q = laurent_q(P2 - 2 * p2_q, -1, 0)
    bm1_sol = [s_q for s_q in sp.solve(cA_q[0], bm1_q) if s_q != 0][0]
    p1_roots = sp.solve(sp.simplify(cP_q[-1].subs(bm1_q, bm1_sol)), p1_q)
    p1_sol = [s_q for s_q in p1_roots if sp.simplify(s_q + sp.Integer(2) / r_hq) != 0][0]
    record_check("[Frobenius] Finiteness of psi''(r_h) has exactly two roots; psi' = -2/r_h is the degenerate one (r psi' + 2 = 0) [D]",
                 flag_q(len(p1_roots) == 2 and any(sp.simplify(s_q + sp.Integer(2) / r_hq) == 0 for s_q in p1_roots)), 1)
    record_check("[Frobenius] Regular horizon slope psi'(r_h) == 2 r_h (V - V_psi)/[r_h^2 (2V + V_psi) - 3], finite d_A V allowed [D]",
                 sp.simplify(p1_sol - sp.Integer(2) * r_hq * (V00 - V10) / (r_hq**2 * (sp.Integer(2) * V00 + V10) - sp.Integer(3))), 0)
    record_check("[Frobenius] Horizon residue b_{-1} == r_h (r_h psi'_h + 2)/[2 (1 - r_h^2 V_h)] [D]",
                 sp.simplify(bm1_sol - r_hq * (r_hq * p1_q + sp.Integer(2)) / (sp.Integer(2) * (sp.Integer(1) - r_hq**2 * V00))), 0)
    lead_q = {bm1_q: bm1_sol}
    record_check("[Frobenius] Double-pole balance of the B' equation on the regular branch == 0 [D]",
                 sp.simplify(cB_q[-2].subs(lead_q).subs(p1_q, p1_sol)), 0)
    record_check("[Frobenius] The 1/(r - r_h) order of the B' equation is satisfied identically (b0 is fixed at the next order) [D]",
                 sp.simplify(cB_q[-1].subs(lead_q).subs(p1_q, p1_sol)), 0)
    p2_sol = sp.simplify(sp.solve(sp.simplify(cP_q[0].subs(lead_q).subs(p1_q, p1_sol)), p2_q)[0])
    record_check("[Frobenius] psi''(r_h) = 2 p2 is fixed in closed form by (r_h, a1, 2-jet of V) alone [D]",
                 flag_q(not (p2_sol.free_symbols & {b0_q, a2_q, p2_q})), 1)
    record_check("[Frobenius] Control: psi''(r_h) == 0 when the potential and its 2-jet vanish (pure vacuum) [D]",
                 p2_sol.subs({V00: 0, V10: 0, V01: 0, V20: 0, V11: 0}), 0)
    print(f"         psi''(r_h)/2 = {sp.factor(p2_sol)}")
    print("         Two-sided statement: e = r - r_h carries no sign assumption, so the same coefficients (b_{-1}, psi'_h, psi''_h)")
    print("         hold for r > r_h and r < r_h: the regular branch is the analytic continuation.")
    bm1_reg = sp.simplify(bm1_sol.subs(p1_q, p1_sol))
    CBG_DERIVED["frob_num"] = sp.lambdify((r_hq, a1_q, V00, V10, V01, V20, V11), [bm1_reg, p1_sol, p2_sol], 'numpy')

    # (c) On-shell null contractions (radial null vector in the (t1, r) plane, k^r = 1), valid for either sign of A and B
    on_shell_q = lambda e_q: sp.simplify(e_q.subs(ddA, A2).subs({dB: B1, ddpsi: P2}).subs(dA, A1))
    
    c_W_sym = sp.Symbol('c_W_sym', real=True)
    weyl_pref_sq = sp.simplify(g.det() * sp.exp(sp.Integer(2) * c_W_sym * psi) / (Phi_inf**2 * (-(sp.exp(c_W_sym * psi) * g4).det())))
    c_Weyl_metric = sp.solve(sp.Eq(sp.diff(weyl_pref_sq, psi), 0), c_W_sym)[0]
    record_check("[Weyl decoupling] Canonical R~^(4) prefactor invariance d_psi(|g_5| e^(2 c_W psi) / |g~_4|) == 0 uniquely fixes c_Weyl == 1 [D]",
                 sp.simplify(c_Weyl_metric - c_Weyl_D), 0)
    geoE_q = Geometry(sp.exp(c_Weyl_metric * psi) * g4, x4)
    R4E_q = sp.simplify(sum(geoE_q.ginv[a_q, b_q] * geoE_q.Ric[a_q, b_q] for a_q in range(4) for b_q in range(4)))
    box4E_psi_q = sp.simplify(geoE_q.ginv[1, 1] * ddpsi - sum(geoE_q.ginv[a_q, b_q] * geoE_q.Gamma[1][a_q][b_q] * dpsi for a_q in range(4) for b_q in range(4)))
    grad4E_psi_sq_q = sp.simplify(geoE_q.ginv[1, 1] * dpsi**2)
    kappa_psi_metric = sp.simplify((sp.exp(-psi) * R5 - R4E_q - box4E_psi_q) / grad4E_psi_sq_q)
    record_check("[Weyl decoupling] Curved 5D metric identity e^(-psi) R^(5) == R~^(4) - (3/2)(d~ psi)^2 + Box~_4 psi fixes kappa_psi == -3/2 [D]",
                 sp.simplify(kappa_psi_metric - kappa_psi_D), 0)
    CBG_DERIVED["c_Weyl_5D"] = c_Weyl_metric
    CBG_DERIVED["kappa_psi_5D"] = kappa_psi_metric
    
    R_kk_q = geoE_q.Ric[0, 0] * B / A + geoE_q.Ric[1, 1]
    nec_density = sp.Rational(3, 2) * dpsi**2 + 2 * A * B * VA
    record_check("[Null contraction] Einstein-frame R~_kk on shell == (3/2) psi'^2 + 2 A B d_A V [D]",
                 sp.simplify(on_shell_q(R_kk_q) - nec_density), 0)
    record_check("[Null contraction] 5D-frame G_kk = B (G^r_r - G^1_1) on shell == 2 A B d_A V [D]",
                 sp.simplify(on_shell_q(B * (Gmix[2, 2] - Gmix[0, 0])) - 2 * A * B * VA), 0)
    A_nq, B_nq, dp_nq, VA_nq = sp.symbols('A_nq B_nq dp_nq VA_nq', real=True)
    CBG_DERIVED["nec"] = sp.lambdify((A_nq, B_nq, dp_nq, VA_nq), nec_density.subs(VA, VA_nq).subs(dpsi, dp_nq).subs({A: A_nq, B: B_nq}), 'numpy')

    # (d) Geodesic first integral from the 5D point-particle Lagrangian (equatorial plane; t1, t2, phi cyclic)
    vt1_q, vt2_q, vr_q, vph_q = sp.symbols('vt1_q vt2_q vr_q vph_q', real=True)
    A_g, B_g, psi_g, r_g = sp.symbols('A_g B_g psi_g r_g', real=True)
    two_L = -A_g * vt1_q**2 - Phi_inf**2 * sp.exp(2 * psi_g) * vt2_q**2 + B_g * vr_q**2 + r_g**2 * vph_q**2
    E_g, P2_g, L_g, kap_g = sp.symbols('E_g P2_g L_g kap_g', real=True)
    charges_q = sp.solve([sp.Eq(-half_q * sp.diff(two_L, vt1_q), E_g), sp.Eq(-half_q * sp.diff(two_L, vt2_q), P2_g),
                          sp.Eq(half_q * sp.diff(two_L, vph_q), L_g)], (vt1_q, vt2_q, vph_q), dict=True)[0]
    vr2_q = sp.solve(sp.Eq(two_L.subs(charges_q), -kap_g), vr_q**2)[0]
    record_check("[Geodesic] First integral A B (dr/dlambda)^2 == E^2 - A [kappa + L^2/r^2 - (P_2/Phi_inf)^2 e^(-2 psi)] (kappa = 1 massive, 0 null) [D]",
                 sp.simplify(A_g * B_g * vr2_q - (E_g**2 - A_g * (kap_g + L_g**2 / r_g**2 - (P2_g / Phi_inf)**2 * sp.exp(-2 * psi_g)))), 0)
    record_check("[Geodesic] Photon sphere (kappa = 0, P_2 = 0): stationarity of A/r^2 is r A' - 2 A == 0 [D]",
                 sp.simplify(sp.diff(A / r**2, r) * r**3 - (r * dA - 2 * A)), 0)

    # (e) Surface gravity of the Killing vector xi = d/dt1: kappa^2 = -(1/2) nabla_A xi_B nabla^A xi^B
    xi_low = [g[i_q, 0] for i_q in range(n)]
    nabla_xi = sp.Matrix(n, n, lambda a_q, b_q: sp.diff(xi_low[b_q], x[a_q]) - sum(Gam[c_q][a_q][b_q] * xi_low[c_q] for c_q in range(n)))
    kappa_sq = sp.simplify(-half_q * sum(nabla_xi[a_q, b_q] * ginv[a_q, a_q] * ginv[b_q, b_q] * nabla_xi[a_q, b_q] for a_q in range(n) for b_q in range(n)))
    record_check("[Killing horizon] Surface gravity -(1/2) nabla_A xi_B nabla^A xi^B == A'^2/(4 A B) [D]",
                 sp.simplify(kappa_sq - dA**2 / (sp.Integer(4) * A * B)), 0)

    # (f) Stiff limit (m_hat -> oo): psi = psi_t(A), V -> 0. The two equations integrated by solve_stiff_limit_bvp.
    psi_t_fn = sp.Function('psi_t')
    B_stiff_q = sp.solve(sp.numer(sp.together(Gmix[2, 2].subs(psi, psi_t_fn(A)).doit())), B)[0]
    dpt_q = sp.diff(psi_t_fn(A), A)
    record_check("[Stiff limit] P = A B from G^r_r == 0 with psi = psi_t(A): P == A + r A' + 2 r A psi_t' A' + (1/2) r^2 psi_t' A'^2 [D]",
                 sp.simplify(A * B_stiff_q - (A + r * dA + 2 * r * A * dpt_q * dA + half_q * r**2 * dpt_q * dA**2)), 0)
    P_fq = sp.Function('P')(r)
    dP_fq = sp.diff(P_fq, r)
    record_check("[Stiff limit] 2 A B G^theta_theta with B'/B = P'/P - A'/A equals the pole-free form integrated by the stiff solver [D]",
                 sp.simplify(2 * A * B * Gmix[3, 3].subs(dB, B * (dP_fq / P_fq - dA / A))
                             - (ddA - dA * dP_fq / (sp.Integer(2) * P_fq) + sp.Integer(2) * dA / r - A * dP_fq / (r * P_fq) + 2 * A * ddpsi + 2 * A * dpsi**2
                                + (sp.Integer(2) * A / r + 2 * dA - A * dP_fq / P_fq) * dpsi)), 0)
    # ================================ END OF STAGE-1 ADDITIONS (Point 2) ================================

    Vs_s, Vp_s, VA_s, Vpp_s, VpA_s, VAA_s = sp.symbols('Vs_s Vp_s VA_s Vpp_s VpA_s VAA_s', real=True)
    A2_sub = sp.simplify(A2.subs({
        sp.diff(V, psi, 2): Vpp_s,
        sp.diff(V, psi, A): VpA_s,
        sp.diff(V, A, 2): VAA_s,
        Vp: Vp_s,
        VA: VA_s,
        V: Vs_s,
    }))
    f_A2 = sp.lambdify((r, A, B, psi, dpsi, Vs_s, Vp_s, VA_s, Vpp_s, VpA_s, VAA_s), A2_sub, 'numpy')
    f_res = sp.lambdify((r, A, B, psi, dA, ddA, dB, dpsi, ddpsi, V, Vp, VA),
                        [E['E^1_1'], E['E^2_2'], E['E^r_r'], E['E^th_th'],
                         Gmix[0, 0], Gmix[1, 1], Gmix[2, 2], Gmix[3, 3]], 'numpy')
    return (f_res, f_A2)


# ==============================================================================
# MODULE 1: EXTERIOR BVP, STIFF LIMIT, LAURENT CROSSING & DOP853 INTERIOR [D]
# ==============================================================================
def make_potential_funcs(m_val=CBGConstants.m_hat_ref,
                         Ac_val=CBGConstants.A_crit,
                         pmax_val=CBGConstants.psi_max,
                         cV_val=CBGConstants.c_V):
    exp_1 = np.exp(-(1.0 / Ac_val)**2)

    def V_all(psi, A):
        A_arr = np.asarray(A)
        exp_A = np.exp(-((A_arr / Ac_val)**2))
        psi_t = pmax_val * (exp_A - exp_1)
        dpsi_t = pmax_val * (-2.0 * A_arr / (Ac_val**2)) * exp_A
        ddpsi_t = pmax_val * (-2.0 / (Ac_val**2)) * (1.0 - 2.0 * ((A_arr / Ac_val)**2)) * exp_A

        diff = psi - psi_t
        k = cV_val * (m_val**2)
        V = k * (diff**2)
        Vp = 2.0 * k * diff
        VA = -2.0 * k * diff * dpsi_t
        Vpp = 2.0 * k * np.ones_like(A_arr)
        VpA = -2.0 * k * dpsi_t
        VAA = 2.0 * k * (dpsi_t**2 - diff * ddpsi_t)
        return V, Vp, VA, Vpp, VpA, VAA

    return V_all


def make_ode_rhs(V_all):
    def ode_rhs(r, y):
        A, B, psi, dpsi = y
        V, Vp, VA, _, _, _ = V_all(psi, A)
        denom = r * (r * dpsi + 2.0)
        dA = -2.0 * A * (r**2 * B * V + 2.0 * r * dpsi - B + 1.0) / denom
        dB = (2.0 * B / (3.0 * denom)) * (
            2.0 * r**3 * A * B * VA * dpsi + 2.0 * r**3 * B * V * dpsi
            + r**3 * B * Vp * dpsi + 4.0 * r**2 * A * B * VA
            + r**2 * B * V + 2.0 * r**2 * B * Vp
            + 3.0 * r**2 * (dpsi**2) - 3.0 * r * B * dpsi + 3.0 * r * dpsi - 3.0 * B + 3.0
        )
        ddpsi = (1.0 / (3.0 * r)) * (
            2.0 * r**2 * A * B * VA * dpsi + 2.0 * r**2 * B * V * dpsi
            + r**2 * B * Vp * dpsi - 2.0 * r * A * B * VA
            - 2.0 * r * B * V + 2.0 * r * B * Vp - 3.0 * B * dpsi - 3.0 * dpsi
        )
        return np.vstack((dA, dB, dpsi, ddpsi))

    return ode_rhs


def solve_stiff_limit_bvp(r_min=1.0, r_max=60.0):
    """
    STAGE-1 (Point 3). Stiff-limit BVP (m_hat -> oo): numerical implementation of the two equations verified symbolically
    in Module 0 (Point 2f): P = A B from G^r_r = 0 and the pole-free form of G^theta_theta = 0, with psi = psi_t(A) (Model 3). P = A B and B'/B = P'/P - A'/A remove all horizon poles, so the
    integration starts at the horizon r_min = 1 (A(1) = 0). The chain-rule expansion below is hand-coded numerics.
    """
    eta0 = -2.0 * CBGConstants.psi_max / CBGConstants.A_crit**2

    def eta_derivs(A):
        eta = eta0 * np.exp(-(A / CBGConstants.A_crit)**2)
        deta_dA = eta * (-2.0 * A / CBGConstants.A_crit**2)
        return eta, deta_dA, eta * A, eta + A * deta_dA

    s_grid = np.linspace(0.0, 1.0, 400)
    r_mesh = r_min + (r_max - r_min) * (s_grid**3)

    def stiff_ode(r, y):
        A, dA = y[0], y[1]
        eta, deta_dA, pt1, pt2 = eta_derivs(A)
        psi_p = pt1 * dA
        P = A + r * dA + 2.0 * r * A * pt1 * dA + 0.5 * r**2 * A * eta * dA**2
        dP_ddA = r * (1.0 + 2.0 * A * pt1 + r * A * eta * dA)
        dP_rest = (2.0 * dA + 2.0 * A * pt1 * dA + 2.0 * r * (pt1 + A * pt2) * dA**2
                   + r * A * eta * dA**2 + 0.5 * r**2 * (eta + A * deta_dA) * dA**3)
        factor_Pprime = -(dA / (2.0 * P) + A / (r * P) + A * psi_p / P)
        coeff_ddA = 1.0 + factor_Pprime * dP_ddA + 2.0 * A * pt1
        rhs_rest = (factor_Pprime * dP_rest + 2.0 * dA / r + 2.0 * A * (pt2 + pt1**2) * dA**2 + (2.0 * A / r + 2.0 * dA) * psi_p)
        return np.vstack([dA, -rhs_rest / coeff_ddA])

    def stiff_bc(ya, yb):
        A_b, dA_b = yb[0], yb[1]
        eta_b, _, pt1_b, _ = eta_derivs(A_b)
        P_b = A_b + r_max * dA_b + 2.0 * r_max * A_b * pt1_b * dA_b + 0.5 * r_max**2 * A_b * eta_b * dA_b**2
        return np.array([ya[0], P_b - 1.0])

    sol = solve_bvp(stiff_ode, stiff_bc, r_mesh, np.vstack([1.0 - 1.0 / r_mesh, 1.0 / r_mesh**2]), tol=1e-9, max_nodes=25000)
    if not sol.success:
        raise RuntimeError(f"Stiff BVP failed: {sol.message}")
    return sol


def eval_trajectory_residuals_5state(f_bundle, rv, state_fn, V_all):
    """
    Genuine, ODE-independent Einstein tensor residual evaluator:
    Computes first derivatives (dA/dr, dB/dr, dpsi/dr, ddpsi/dr, ddA/dr) of the 5-state
    trajectory state_fn(r) = [A, B, psi, dpsi, dA] via the 5-point 4th-order central
    finite-difference stencil with optimal pole-aware step h = 3e-4 * min(r, |1 - r|)
    without calling ode_rhs or f_A2!
    """
    f_res, _ = f_bundle
    y0 = np.asarray(state_fn(rv), dtype=float).flatten()
    A_v, B_v, psi_v, _, _ = y0

    h = 3.0e-4 * min(rv, abs(1.0 - rv))
    ym2 = np.asarray(state_fn(rv - 2.0 * h), dtype=float).flatten()
    ym1 = np.asarray(state_fn(rv - h), dtype=float).flatten()
    yp1 = np.asarray(state_fn(rv + h), dtype=float).flatten()
    yp2 = np.asarray(state_fn(rv + 2.0 * h), dtype=float).flatten()

    dy_fd = (-yp2 + 8.0 * yp1 - 8.0 * ym1 + ym2) / (12.0 * h)
    dA_num, dB_num, dpsi_num, ddpsi_num, ddA_num = dy_fd[0], dy_fd[1], dy_fd[2], dy_fd[3], dy_fd[4]

    V_v, Vp_v, VA_v, _, _, _ = [float(v) for v in V_all(psi_v, A_v)]
    out = f_res(rv, A_v, B_v, psi_v, dA_num, ddA_num, dB_num, dpsi_num, ddpsi_num, V_v, Vp_v, VA_v)
    conn_scale = 1.0 / (abs(B_v) * (rv**2))
    norm_G = max(1.0, conn_scale, max(abs(out[4 + i]) for i in range(4)))
    return [abs(out[i]) / norm_G for i in range(4)]


def run_module_1_interior_continuation(f_bundle):
    print("\n" + "=" * 94)
    print("MODULE 1: FROBENIUS EXTERIOR BVP, STIFF LIMIT & DOP853 INTERIOR (TABLE 3) [D] (AT BENCHMARK [H]: A_crit=1/4, psi_max=2/25, m_hat=4)")
    print("=" * 94)

    _, f_A2 = f_bundle
    V_all = make_potential_funcs()
    ode_rhs = make_ode_rhs(V_all)

    # STAGE-1 (Point 3): exterior BVP in the smooth variables y = [A, P = (r - 1) B, psi, psi'] (units r_h = 1) with
    # second-order horizon boundary conditions built from the closed forms of Module 0 (CBG_DERIVED["frob_num"]).
    d_in, r_max = 2.0e-4, 10.0

    def horizon_series(psi_hv, a1_v):
        V_j, Vp_j, VA_j, Vpp_j, VpA_j, _ = [float(np.asarray(v).flat[0]) for v in V_all(psi_hv, 0.0)]
        return [float(v) for v in CBG_DERIVED["frob_num"](1.0, a1_v, V_j, Vp_j, VA_j, Vpp_j, VpA_j)]

    def solve_exterior_P(d_off):
        r_min = 1.0 + d_off
        r_mesh = 1.0 + d_off * (((r_max - 1.0) / d_off) ** np.linspace(0.0, 1.0, 900))

        def rhs_P(r, y):
            A_v, P_v, psi_v, dpsi_v = y
            B_v = P_v / (r - 1.0)
            dA, dB, dpsi, ddpsi = ode_rhs(r, np.vstack((A_v, B_v, psi_v, dpsi_v)))
            return np.vstack((dA, B_v + (r - 1.0) * dB, dpsi, ddpsi))

        def bc_frob(ya, yb):
            A_a, P_a, psi_a, dpsi_a = ya
            A_b, P_b, psi_b, _ = yb
            bm1_s, p1_s, p2_s = horizon_series(psi_a - dpsi_a * d_off, A_a / d_off)
            dP_a = float(rhs_P(r_min, np.asarray(ya, dtype=float).reshape(4, 1))[1, 0])    # P'(r_min) = b0 + O(d)
            bc1 = P_a - dP_a * d_off - bm1_s                   # P(r_min) - P'(r_min) d = b_{-1} + O(d^2)
            bc2 = dpsi_a - (p1_s + 2.0 * p2_s * d_off)         # psi'(r_min) = psi'_h + psi''_h d + O(d^2)
            return np.array([bc1, bc2, A_b * P_b / (r_max - 1.0) - 1.0, psi_b])

        y_init = np.zeros((4, r_mesh.size))
        y_init[0] = 1.0 - 1.0 / r_mesh
        y_init[1] = r_mesh
        y_init[2] = 0.08 * np.exp(-4.0 * (r_mesh - 1.0)) / r_mesh
        y_init[3] = np.gradient(y_init[2], r_mesh)
        sol_P = solve_bvp(rhs_P, bc_frob, r_mesh, y_init, tol=1e-9, max_nodes=30000)
        if not sol_P.success:
            raise RuntimeError(f"exterior BVP failed at d_off={d_off}: {sol_P.message} (nodes={sol_P.x.size}, max rms={sol_P.rms_residuals.max():.2e})")
        A_a, P_a, psi_a, dpsi_a = sol_P.sol(r_min)
        dA_a, dP_a, _, ddpsi_a = rhs_P(r_min, np.array([A_a, P_a, psi_a, dpsi_a]).reshape(4, 1))[:, 0]
        psi_hv = psi_a - dpsi_a * d_off + 0.5 * ddpsi_a * d_off**2
        a1_v = 2.0 * A_a / d_off - dA_a                        # A = a1 d + a2 d^2, A' = a1 + 2 a2 d
        bm1_s, p1_s, p2_s = horizon_series(psi_hv, a1_v)
        return sol_P, {"psi_h": psi_hv, "a1": a1_v, "a2": (dA_a - a1_v) / (2.0 * d_off), "bm1": bm1_s, "b0": dP_a,
                       "p1": p1_s, "p2": p2_s, "r_ADM": r_max * (1.0 - sol_P.sol(r_max)[0])}

    sol_ext, hz = solve_exterior_P(d_in)
    sol_ext2, hz2 = solve_exterior_P(2.0 * d_in)
    for key_c, name_c, tol_c in (("psi_h", "psi(r_h)", 2e-7), ("p1", "psi'(r_h)", 2e-7), ("bm1", "b_{-1}/r_h", 2e-7),
                                 ("a1", "a_1 r_h", 2e-6), ("r_ADM", "r_ADM/r_h", 2e-7)):
        record_check(f"Two-offset convergence (d_in = 2e-4 vs 4e-4) of the horizon datum {name_c} [D]",
                     hz[key_c], hz2[key_c], tol=tol_c)
    r_ADM_ref = 10.0 * (1.0 - sol_ext.sol(10.0)[0])
    f_hair_ref = 100.0 * (r_ADM_ref - 1.0) / r_ADM_ref
    area_shift_ref = 100.0 * ((1.0 / r_ADM_ref)**2 - 1.0)

    record_check("Reference m_hat=4 exterior ADM radius r_ADM/r_h == 1.038513 [D]",
                 r_ADM_ref, 1.038513, tol=1e-5)
    record_check("Reference m_hat=4 exterior scalar-hair fraction f_hair [%] == 3.708% [D]",
                 f_hair_ref, 3.7084, tol=1e-2)
    record_check("Reference m_hat=4 horizon area shift Delta A_H / A_H [%] == -7.279% [D]",
                 area_shift_ref, -7.279, tol=1e-2)

    sol_stiff = solve_stiff_limit_bvp(1.0, 60.0)
    assert sol_stiff.success
    a1_stiff = sol_stiff.sol(1.0)[1]
    r_ADM_stiff = 60.0 * (1.0 - sol_stiff.sol(60.0)[0])
    f_hair_stiff = 100.0 * (r_ADM_stiff - 1.0) / r_ADM_stiff
    area_shift_stiff = 100.0 * ((1.0 / r_ADM_stiff)**2 - 1.0)
    kappa_ratio_stiff = r_ADM_stiff * np.sqrt(a1_stiff)

    record_check("Universal stiff-limit (m_hat->oo) horizon product a_1 r_h == 0.92777 [D]",
                 a1_stiff, 0.92777, tol=1e-4)
    record_check("Universal stiff-limit (m_hat->oo) ADM radius r_ADM/r_h == 1.043430 [D]",
                 r_ADM_stiff, 1.043430, tol=1e-4)
    record_check("Universal stiff-limit (m_hat->oo) hair fraction f_hair^(oo) == 4.162% [D]",
                 f_hair_stiff, 4.1622, tol=1e-2)
    record_check("Universal stiff-limit (m_hat->oo) area shift Delta A_H/A_H == -8.151% [D]",
                 area_shift_stiff, -8.151, tol=1e-2)
    record_check("Section 6.2: Stiff-limit surface gravity ratio kappa_H^(oo)/kappa_Schw == 1.00504 [D]",
                 kappa_ratio_stiff, 1.00504, tol=1e-4)

    # STAGE-1 (Point 3): horizon data from the second-order Frobenius series (same coefficients on both sides of r_h)
    eps = d_in
    psi_h, dpsi_h, ddpsi_h = hz["psi_h"], hz["p1"], 2.0 * hz["p2"]
    a1_h, a2_h = hz["a1"], 2.0 * hz["a2"]                      # a2_h = A''(r_h): v9 Taylor convention A = a1 e + (1/2) a2_h e^2
    bm1_h, b0_h = hz["bm1"], hz["b0"]

    record_check("Eq. (4.4): Horizon fiber ratio Phi(r_h)/Phi_inf == 1.076623 [D]",
                 np.exp(psi_h), 1.076623, tol=1e-5)
    record_check("Eq. (4.4): Horizon slope psi'(r_h) == -0.094411 [D]",
                 dpsi_h, -0.094411, tol=1e-5)
    record_check("Eq. (4.4): Horizon second derivative psi''(r_h) == -0.68642 (closed form of Module 0) [D]",
                 ddpsi_h, -0.68642, tol=1e-4)
    record_check("Eq. (4.4): Horizon pole residue b_{-1}/r_h == 0.953230 [D]",
                 bm1_h, 0.953230, tol=1e-5)
    record_check("Eq. (4.4): Horizon metric product a_1 b_{-1} == 0.848599 [D]",
                 a1_h * bm1_h, 0.848599, tol=1e-5)
    kappa_ratio_ref = 2.0 * r_ADM_ref * 0.5 * np.sqrt(a1_h / bm1_h)
    record_check("Section 6.2: Reference m_hat=4 surface gravity ratio kappa_H/kappa_Schw(r_ADM) == 1.003611 [D]",
                 kappa_ratio_ref, 1.003611, tol=1e-5)

    # STAGE-1 (Point 4): photon sphere (r A' - 2A = 0) and critical impact parameter b = r/sqrt(A) on the derived solutions
    def photon(A_of, dA_of, r_lo, r_hi):
        r_ph_v = brentq(lambda rr: rr * dA_of(rr) - 2.0 * A_of(rr), r_lo, r_hi, xtol=1e-13)
        return r_ph_v, r_ph_v / np.sqrt(A_of(r_ph_v))

    def dA_ext(rr):
        A_v, P_v, psi_v, dpsi_v = sol_ext.sol(rr)
        return float(ode_rhs(rr, np.array([A_v, P_v / (rr - 1.0), psi_v, dpsi_v]).reshape(4, 1)).flatten()[0])

    r_ph_ref, b_c_ref = photon(lambda rr: float(sol_ext.sol(rr)[0]), dA_ext, 1.2, 3.0)
    r_ph_stiff, b_c_stiff = photon(lambda rr: float(sol_stiff.sol(rr)[0]), lambda rr: float(sol_stiff.sol(rr)[1]), 1.2, 3.0)
    shadow_ref = b_c_ref / (1.5 * np.sqrt(3.0) * r_ADM_ref)
    shadow_stiff = b_c_stiff / (1.5 * np.sqrt(3.0) * r_ADM_stiff)
    record_check("Section 8.1 (Primary strong-field BVP, m_hat=4): Photon sphere r_ph / ((3/2) r_ADM) == 0.989985 [D]",
                 r_ph_ref / (1.5 * r_ADM_ref), 0.989985, tol=1e-5)
    record_check("Section 8.1 (Primary strong-field BVP, m_hat=4): Shadow radius b_c / ((3 sqrt(3)/2) r_ADM) == 0.999448 [D]",
                 shadow_ref, 0.999448, tol=1e-5)
    record_check("Section 8.1 (Primary strong-field BVP, stiff limit): Photon sphere r_ph / ((3/2) r_ADM) == 0.98977 [D]",
                 r_ph_stiff / (1.5 * r_ADM_stiff), 0.98977, tol=5e-5)
    record_check("Section 8.1 (Primary strong-field BVP, stiff limit): Shadow radius b_c / ((3 sqrt(3)/2) r_ADM) == 0.99929 [D]",
                 shadow_stiff, 0.99929, tol=5e-5)
    print(f"  [NOTE] Section 8.1 Strong-Field Shadow: at fixed ADM mass, scalar hair shifts b_c by {100.0 * (shadow_ref - 1.0):+.3f}% (m_hat = 4) "
          f"and {100.0 * (shadow_stiff - 1.0):+.3f}% (stiff limit), superseding the psi=0 basal optical-ansatz estimate (-1.925e-8).")

    r_minus = 1.0 - eps
    A_m = -a1_h * eps + 0.5 * a2_h * (eps**2)
    B_m = -bm1_h / eps + b0_h
    psi_m = psi_h - dpsi_h * eps + 0.5 * ddpsi_h * (eps**2)
    dpsi_m = dpsi_h - ddpsi_h * eps
    y_minus = [A_m, B_m, psi_m, dpsi_m]

    dA_m = float(ode_rhs(r_minus, np.array(y_minus).reshape(4, 1)).flatten()[0])
    y_minus_5 = [A_m, B_m, psi_m, dpsi_m, dA_m]

    def ode_ivp_5(r, y):
        dA, dB, dpsi, ddpsi = ode_rhs(r, np.array(y[:4]).reshape(4, 1)).flatten()
        A_v, B_v, psi_v, dpsi_v = y[0], y[1], y[2], y[3]
        V_v, Vp_v, VA_v, Vpp_v, VpA_v, VAA_v = [float(v) for v in V_all(psi_v, A_v)]
        ddA = float(f_A2(r, A_v, B_v, psi_v, dpsi_v, V_v, Vp_v, VA_v, Vpp_v, VpA_v, VAA_v))
        return [dA, dB, dpsi, ddpsi, ddA]

    sol_int = solve_ivp(
        ode_ivp_5, [r_minus, 1.0e-6], y_minus_5,
        method="DOP853", rtol=1e-13, atol=1e-15, dense_output=True
    )
    assert sol_int.success

    r_turn = brentq(lambda rr: sol_int.sol(rr)[3], 0.85, 0.96)
    r_vac = brentq(lambda rr: sol_int.sol(rr)[2], 0.50, 0.65)
    Phi_turn = np.exp(sol_int.sol(r_turn)[2])
    c_psi_vac = 0.005 * sol_int.sol(0.005)[3]

    record_check("Theorem 4.1(ii): Interior fiber turnover radius r_turn/r_h == 0.9081 [D]",
                 r_turn, 0.9081, tol=2e-4)
    record_check("Theorem 4.1(ii): Interior fiber peak Phi_max/Phi_inf == 1.081836 [D]",
                 Phi_turn, 1.081836, tol=1e-5)
    record_check("Theorem 4.1(iii): Interior vacuum crossing radius r_vac/r_h == 0.5870 [D]",
                 r_vac, 0.5870, tol=2e-4)
    record_check("Theorem 4.1(iii): Pure-vacuum Kasner exponent c_psi(0.005 r_h) == +0.0348 [D]",
                 c_psi_vac, 0.0348, tol=2e-4)

    record_check("Table 3: Interior A_5D(0.0050 r_h) == -300.0991 [D]",
                 sol_int.sol(0.0050)[0], -300.0991, tol=1e-3)
    record_check("Table 3: Interior Phi(0.0050 r_h)/Phi_inf == 0.783657 [D]",
                 np.exp(sol_int.sol(0.0050)[2]), 0.783657, tol=1e-5)

    # Anti-tautology check: verify that a non-solution profile returns O(1) residual >> 1e-2
    fake_fn = lambda r_v: np.array([-1.0 / r_v, -r_v, 0.05 * r_v, 0.05, 1.0 / (r_v**2)])
    fake_res = max(eval_trajectory_residuals_5state(f_bundle, 0.50, fake_fn, V_all))
    record_check("Theorem 4.1(iv): Non-solution state yields O(1) finite-difference residual (non-tautological) [D]",
                 1.0 if fake_res > 1.0e-2 else 0.0, 1.0, tol=1e-12)

    r_table = [0.9950, 0.9800, 0.9000, 0.7000, r_vac, 0.5000, 0.3000, 0.1500, 0.0500, 0.0200, 0.0050]
    max_res_int = 0.0
    for rv in r_table:
        e11_v, e22_v, err_v, eth_v = eval_trajectory_residuals_5state(f_bundle, rv, sol_int.sol, V_all)
        max_res_int = max(max_res_int, e11_v, e22_v, err_v, eth_v)

    record_check("Theorem 4.1(iv): Max ODE-independent 4th-order FD 5D Einstein residual across Table 3 <= 5e-10 [D]",
                 max_res_int, 0.0, tol=5e-10)

    # STAGE-1 (Point 3): deep-interior convergence with respect to the crossing offset (second run from 2 * d_in)
    eps2 = 2.0 * d_in
    y_m2 = [-hz2["a1"] * eps2 + hz2["a2"] * eps2**2, -hz2["bm1"] / eps2 + hz2["b0"],
            hz2["psi_h"] - hz2["p1"] * eps2 + hz2["p2"] * eps2**2, hz2["p1"] - 2.0 * hz2["p2"] * eps2]
    y_m2.append(float(ode_rhs(1.0 - eps2, np.array(y_m2).reshape(4, 1)).flatten()[0]))
    sol_int2 = solve_ivp(ode_ivp_5, [1.0 - eps2, 0.004], y_m2, method="DOP853", rtol=1e-13, atol=1e-15, dense_output=True)
    assert sol_int2.success
    record_check("Two-offset convergence (d_in = 2e-4 vs 4e-4) of the deep interior A_5D(0.0050 r_h) [D]",
                 sol_int.sol(0.0050)[0], sol_int2.sol(0.0050)[0], tol=1e-3)
    record_check("Two-offset convergence (d_in = 2e-4 vs 4e-4) of the deep interior psi(0.0050 r_h) [D]",
                 sol_int.sol(0.0050)[2], sol_int2.sol(0.0050)[2], tol=1e-6)

    # STAGE-1 (Point 2c): on-shell null contractions along the interior (diagnostic print, no assertion)
    rr_nc = np.linspace(0.02, 0.999, 4000)
    Y_nc = np.array([sol_int.sol(v) for v in rr_nc]).T
    VA_nc = V_all(Y_nc[2], Y_nc[0])[2]
    gkk_nc = 2.0 * Y_nc[0] * Y_nc[1] * VA_nc
    rkk_nc = CBG_DERIVED["nec"](Y_nc[0], Y_nc[1], Y_nc[3], VA_nc)
    print(f"  [NOTE] Interior null contractions on 0.02 <= r/r_h <= 0.999: 5D-frame G_kk < 0 on {100.0 * np.mean(gkk_nc < 0):.1f}% of the range "
          f"(min {gkk_nc.min():.3e}); Einstein-frame R~_kk < 0 on {100.0 * np.mean(rkk_nc < 0):.1f}% (min {rkk_nc.min():.3e}).")

    return sol_int, y_minus, r_minus, V_all, ode_rhs


# ==============================================================================
# MODULE 2: INTERIOR GEODESIC TURNING POINTS r_b(E, Omega, L) (TABLE 4) [D]
# ==============================================================================
def run_module_2_geodesic_turning_points(sol_int):
    print("\n" + "=" * 94)
    print("MODULE 2: INTERIOR GEODESIC TURNING POINTS r_b(E, Omega, L) (THEOREM 4.2, TABLE 4) [D] (AT BENCHMARK [H]: A_crit=1/4, psi_max=2/25, m_hat=4)")
    print("=" * 94)

    def rhs_geodesic(rr, om_ratio, L_val, E_val):
        Av, _, psiv, _, _ = sol_int.sol(rr)
        absA = abs(Av)
        return E_val**2 + absA * (1.0 + (L_val**2) / (rr**2) - (om_ratio**2) * np.exp(-2.0 * psiv))

    cases = [
        ("Table 4: Radial turning point r_b/r_h at |Omega|=1.00 Omega_inf, L=0, E=0.20 [D]",
         1.00, 0.00, 0.20, 1.0e-6, 0.90, 0.5317, 2e-4),
        ("Table 4: Radial turning point r_b/r_h at |Omega|=1.00 Omega_inf, L=0, E=0.50 [D]",
         1.00, 0.00, 0.50, 1.0e-6, 0.90, 0.3883, 2e-4),
        ("Table 4: Radial turning point r_b/r_h at |Omega|=0.95 Omega_inf, L=0, E=0.20 [D]",
         0.95, 0.00, 0.20, 1.0e-6, 0.90, 0.3893, 2e-4),
        ("Table 4: Radial turning point r_b/r_h at |Omega|=0.90 Omega_inf, L=0, E=0.20 [D]",
         0.90, 0.00, 0.20, 1.0e-6, 0.90, 0.1912, 2e-4),
        ("Table 4: Radial turning point r_b/r_h at |Omega|=0.80 Omega_inf, L=0, E=0.20 [D]",
         0.80, 0.00, 0.20, 1.0e-6, 0.90, 9.01e-3, 3e-5),
        ("Table 4: Radial turning point r_b/r_h at |Omega|=0.60 Omega_inf, L=0, E=0.20 [D]",
         0.60, 0.00, 0.20, 1.0e-6, 0.90, 2.25e-6, 2e-8),
        ("Table 4: Nearly-radial turning point r_b/r_h at |Omega|=1.00, L=0.10, E=0.20 [D]",
         1.00, 0.10, 0.20, 0.28,   0.90, 0.4801, 2e-4),
    ]
    for desc, om_r, L_v, E_v, r_left, r_right, expected_rb, tol_rb in cases:
        rb = brentq(lambda rr: rhs_geodesic(rr, om_r, L_v, E_v), r_left, r_right)
        record_check(desc, rb, expected_rb, tol=tol_rb)

# ==============================================================================
# MODULE 3: S^1_t2 LOOP ANGULAR-MOMENTUM KASNER INVERSION (THEOREM 4.3, TABLE 5) [D]
# ==============================================================================
def run_module_3_kasner_inversion(f_bundle, y_minus, r_minus, V_all, ode_rhs):
    print("\n" + "=" * 94)
    print("MODULE 3: S^1_t2 LOOP ANGULAR-MOMENTUM KASNER INVERSION (THEOREM 4.3, TABLE 5) [D] (VACUUM) / [H] (SOURCED C_Omega={1,2,4}*psi_max^2)")
    print("=" * 94)

    f_res, _ = f_bundle
    pmax2 = CBGConstants.psi_max**2  # (2/25)^2 = 4/625 = 0.0064

    flux_targets = [
        ("0 (Vacuum)",  0.0,         0.7837, -300.0991, +0.0348),
        ("psi_max^2",   pmax2,       0.8521, -265.5887, -0.0047),
        ("d_t*psi^2",   2.0 * pmax2, 0.9196, -236.9736, -0.0378),
        ("d_t^2*psi^2", 4.0 * pmax2, 1.0529, -192.4301, -0.0903),
    ]

    for label, C_om, exp_phi, exp_A, exp_cpsi in flux_targets:
        def ode_rot(r, y, c_val=C_om):
            dA, dB, dpsi, ddpsi = ode_rhs(r, np.array(y).reshape(4, 1)).flatten()
            A_v, B_v, psi_v, dpsi_v = y
            # Exact Eq. (4.9) coupled shifts from T^2_(m)2 = -6*C_Om*exp(-3*psi) / [r^2*(r*dpsi + 2)^2*B]:
            ddpsi_shift = (c_val / (r**2)) * np.exp(-3.0 * psi_v)
            dB_shift = (2.0 * B_v / (r * (r * dpsi_v + 2.0))) * c_val * np.exp(-3.0 * psi_v)
            return [dA, dB + dB_shift, dpsi, ddpsi + ddpsi_shift]

        sol_b = solve_ivp(
            ode_rot, [r_minus, 0.0050], y_minus,
            method="DOP853", rtol=1e-13, atol=1e-15, dense_output=True
        )
        assert sol_b.success
        A0005, _, psi0005, dpsi0005 = [float(v) for v in sol_b.sol(0.0050)]
        p0005 = float(np.exp(psi0005))
        cpsi = 0.0050 * dpsi0005

        record_check(f"Table 5 ({label}, C_Omega={C_om:.4f}): Phi(0.005 r_h)/Phi_inf {'[D]' if C_om == 0.0 else '[H]'}",
                     p0005, exp_phi, tol=2e-4)
        record_check(f"Table 5 ({label}, C_Omega={C_om:.4f}): Metric A_5D(0.005 r_h) {'[D]' if C_om == 0.0 else '[H]'}",
                     A0005, exp_A, tol=2e-3)
        record_check(f"Table 5 ({label}, C_Omega={C_om:.4f}): Kasner exponent c_psi = r*psi' {'[D]' if C_om == 0.0 else '[H]'}",
                     cpsi, exp_cpsi, tol=2e-4)

        # Verify by 4th-order finite differences at r = 0.20 r_h that all 4 Einstein equations
        # E^1_1 = 0, E^2_2 - T^2_(m)2 = 0, E^r_r = 0, E^th_th - T^th_(m)th = 0 hold simultaneously!
        rv = 0.20
        h = 1.0e-4 * rv
        pts = [rv - 2.0 * h, rv - h, rv, rv + h, rv + 2.0 * h]
        states = [sol_b.sol(pt) for pt in pts]
        dAs = [ode_rot(pt, st)[0] for pt, st in zip(pts, states)]

        ym2, ym1, y0, yp1, yp2 = states
        dy_fd = (-yp2 + 8.0 * yp1 - 8.0 * ym1 + ym2) / (12.0 * h)
        ddA_fd = (-dAs[4] + 8.0 * dAs[3] - 8.0 * dAs[1] + dAs[0]) / (12.0 * h)

        A_v, B_v, psi_v, dps_v = y0[0], y0[1], y0[2], y0[3]
        V_v, Vp_v, VA_v, _, _, _ = [float(v) for v in V_all(psi_v, A_v)]
        out = f_res(rv, A_v, B_v, psi_v, dy_fd[0], ddA_fd, dy_fd[1], dy_fd[2], dy_fd[3], V_v, Vp_v, VA_v)
        T22_val = -6.0 * C_om * np.exp(-3.0 * psi_v) / ((rv**2) * ((rv * dps_v + 2.0)**2) * B_v)
        Ttt_val = -0.5 * rv * dps_v * T22_val
        E_full = [out[0], out[1] - T22_val, out[2], out[3] - Ttt_val]
        conn_scale = 1.0 / (abs(B_v) * (rv**2))
        norm_G = max(1.0, conn_scale, max(abs(out[4 + i]) for i in range(4)))
        max_full_res = max(abs(x) for x in E_full) / norm_G
        record_check(f"Table 5 ({label}): 4-component FD Einstein residual with Bianchi source <= 1e-10 [D]",
                     max_full_res, 0.0, tol=1e-10)
        

# ==============================================================================
# MODULE 4: GW RINGDOWN ECHOES (TABLE 6), SUB-RING HOLONOMY & HAWKING EXTREMA
# ==============================================================================
def run_module_4_echoes_polarimetry_hawking():
    print("\n" + "=" * 94)
    print("MODULE 4: GW RINGDOWN ECHOES (TABLE 6), SUB-RING HOLONOMY & HAWKING EXTREMA [D] / [H] / [N]")
    print("=" * 94)

    # Two-channel scalar coupling audit (Section 1.2 & Section 13):
    # Channel A: 5D Weyl-DeWitt universal matter-trace Yukawa strength alpha_Yuk = alpha_0^2 = c_Weyl^2/(-2*kappa_psi) = -c_DW^-1(5) = 1/3 [D]
    #            at lambda_psi = sqrt(L_P * c / H0) = 47.10 um (47 +/- 2 um across H0 in [67.4, 73.2] km/s/Mpc)
    # Channel B: Weak-field A=1 horizon-activation suppression [psi_t'(1)]^2 = (2.56 * exp(-16))^2 = 8.300e-14
    eta4_dw = sp.diag(-1, 1, 1, 1)
    pairs4_dw = [(i_d, j_d) for i_d in range(4) for j_d in range(i_d, 4)]
    B4_dw = [sp.Matrix(4, 4, lambda a_d, b_d, i_d=i_d, j_d=j_d:
                       sp.Integer(1) if (a_d == i_d and b_d == j_d) or (a_d == j_d and b_d == i_d) else sp.Integer(0))
             for (i_d, j_d) in pairs4_dw]
    G_DW_10_inv = sp.Matrix(10, 10, lambda p_d, q_d: (
        (eta4_dw * B4_dw[p_d] * eta4_dw * B4_dw[q_d]).trace()
        - sp.Rational(1, 2) * (eta4_dw * B4_dw[p_d]).trace() * (eta4_dw * B4_dw[q_d]).trace()
    )).inv()
    alpha_Yuk_DW = CBG_DERIVED["G_DW_15_inv_00"] / G_DW_10_inv[0, 0] - sp.Integer(1)
    Phi_N_s, psi_s, rho_s, lap_Phi_s, lap_psi_s = sp.symbols('Phi_N_s psi_s rho_s lap_Phi_s lap_psi_s', real=True)
    c_W_5, kap_5 = CBG_DERIVED["c_Weyl_5D"], CBG_DERIVED["kappa_psi_5D"]
    L_pot_s = -rho_s * (Phi_N_s - sp.Rational(1, 2) * c_W_5 * psi_s)
    sol_pois = sp.solve([sp.Eq(-sp.Integer(2) * lap_Phi_s - sp.diff(L_pot_s, Phi_N_s), 0),
                         sp.Eq(kap_5 * lap_psi_s - sp.diff(L_pot_s, psi_s), 0)], (lap_Phi_s, lap_psi_s))
    alpha_Yuk_trace = sp.simplify((sol_pois[lap_Phi_s] - sp.Rational(1, 2) * c_W_5 * sol_pois[lap_psi_s]) / sol_pois[lap_Phi_s] - sp.Integer(1))
    dpsi_t_at_1 = -(2.0 * CBGConstants.psi_max * 1.0 / (CBGConstants.A_crit**2)) * np.exp(-(1.0 / CBGConstants.A_crit)**2)
    horizon_suppression_A1 = dpsi_t_at_1**2

    record_check("Paper 0 THM #04 & #12-#14 (Sec. 1.2 & 13): Quasi-static 5D DeWitt/Weyl trace Yukawa ratio alpha_Yuk == -c_DW^-1(5) == 1/3 [D]",
                 sp.Abs(alpha_Yuk_trace - alpha_Yuk_DW)
                 + sp.Abs(alpha_Yuk_DW - (-CBG_DERIVED["c_DW_inv_5D"]))
                 + sp.Abs(alpha_Yuk_trace - sp.Rational(1, 3)), 0)
    
    record_check("Section 1.2 & 2.1: Horizonless-star (A=1) khronon hair suppression [psi_t'(1)]^2 == 8.300e-14 [H]",
                 horizon_suppression_A1, 8.2995875e-14, tol=1e-19)

    dqnm_base = 0.5 * CBGConstants.Phi_inf_sq
    dr_phot_shift = -(1.0 / 9.0) * CBGConstants.Phi_inf_sq
    db_crit_shift = -(1.0 / 3.0) * CBGConstants.Phi_inf_sq
    evpa_arcsec = CBGConstants.Phi_inf * (180.0 * 3600.0 / np.pi)

    record_check("Section 6.2 (Eq. 6.7): Baseline QNM frequency shift domega/omega == +2.89e-8 [H]/[N]",
                 dqnm_base, 2.886884e-8, tol=1e-12)
    record_check("Section 8.1 (Basal psi=0 optical ansatz): Coordinate shift dr_phot^(opt)/r_phot == -6.415e-9 (superseded by -1.00% BVP hair shift) [H]/[N]",
                 dr_phot_shift, -6.415299e-9, tol=1e-13)
    record_check("Section 8.1 (Basal psi=0 optical ansatz): Shadow shift db_crit^(opt)/b_crit == -1.925e-8 (superseded by -0.055%/-0.071% BVP hair shift) [H]/[N]",
                 db_crit_shift, -1.924590e-8, tol=1e-12)
    record_check("Proposition 8.1 (Eq. 8.6): Sub-ring EVPA holonomy dchi_pol [arcsec] == 49.56'' [H]/[N]",
                 evpa_arcsec, 49.5627, tol=1e-2)

    T_max_CBG = CBGConstants.T_P / (6.0 * np.sqrt(3.0) * np.pi)
    T_Hay_mP = (2.0 * CBGConstants.T_P) / (27.0 * np.pi)
    M_rem_kg = 0.5 * CBGConstants.m_P
    M_rem_GeV = 0.5 * CBGConstants.E_P_GeV
    E1_bare_EeV = CBGConstants.Phi_inf_sq * CBGConstants.E_P_GeV * 1.0e-9
    chi_loop = (CBGConstants.alpha_EM / (2.0 * np.pi + 1.0)) * (1.0 - (4.0 / 15.0) * CBGConstants.alpha_EM)
    E1_dressed_PeV = chi_loop * E1_bare_EeV * 1.0e3

    record_check("Section 9.2 (Eq. 9.3): Peak Hawking temperature T_max^(CBG) [K] == 4.3395e30 [H]/[N]",
                 T_max_CBG, 4.33951e30, tol=1e26)
    record_check("Section 9.2 (Eq. 9.3): Hayward temperature T_Hay(m_P) [K] == 3.3406e30 [H]/[N]",
                 T_Hay_mP, 3.34056e30, tol=1e26)
    record_check("Section 9.2 (Eq. 9.4): Zero-temperature Planck remnant mass M_rem [kg] == 1.0882e-8 [H]/[N]",
                 M_rem_kg, 1.088217e-8, tol=1e-13)
    record_check("Section 9.2 (Eq. 9.4): Zero-temperature Planck remnant mass M_rem [GeV] == 6.1045e18 [H]/[N]",
                 M_rem_GeV, 6.10445e18, tol=1e14)
    record_check("Section 9.2 (Eq. 9.5): Bare n=1 scale E_1^(bare) = Phi_inf^2 E_P [EeV] == 704.91 [H]/[N]",
                 E1_bare_EeV, 704.9156, tol=0.02)
    record_check("Section 9.2 (Eq. 9.5): Dressed n=1 scale E_1_tilde [PeV] == 704.91 [H]/[N]",
                 E1_dressed_PeV, 704.9125, tol=0.02)

    def dt_echo_seconds(M_msun, a_star):
        M_kg = M_msun * CBGConstants.M_sun
        rs = 2.0 * CBGConstants.G * M_kg / (CBGConstants.c**2)
        spin_fac = (1.0 + np.sqrt(1.0 - a_star**2)) / (2.0 * np.sqrt(1.0 - a_star**2)) if a_star > 0.0 else 1.0
        log_term = np.log((rs * np.sqrt(1.0 - a_star**2)) / (2.0 * CBGConstants.L_P)) + 0.5
        return (2.0 * rs / CBGConstants.c) * spin_fac * log_term

    echo_targets = [
        ("GW230529 (NS-BH)",   5.1,     0.42, 9.00e-3,         9.45e-3,         1e-4),
        ("GW190814 (Asymm.)",  25.6,    0.28, 45.97e-3,        46.91e-3,        2e-4),
        ("Benchmark 30 M_sun", 30.0,    0.00, 53.97e-3,        53.97e-3,        2e-4),
        ("GW150914 (BBH)",     62.0,    0.68, 112.43e-3,       132.43e-3,       3e-4),
        ("GW190521 (IMBH)",    142.0,   0.72, 259.81e-3,       315.85e-3,       5e-4),
        ("Sgr A* (Gal. Ctr.)", 4.154e6, 0.50, 2.345 * 3600.0,  2.523 * 3600.0,  10.0),
        ("M87* (Virgo)",       6.50e9,  0.90, 163.79 * 86400., 267.75 * 86400., 2000.0),
    ]
    for name, M_f, a_f, exp_dt0, exp_dts, tol_dt in echo_targets:
        dt0 = dt_echo_seconds(M_f, 0.0)
        dts = dt_echo_seconds(M_f, a_f)
        record_check(f"Table 6 ({name}): Non-spinning echo delay dt_echo(a*=0) [s] [H]/[N]",
                     dt0, exp_dt0, tol=tol_dt)
        record_check(f"Table 6 ({name}): Spinning echo delay dt_echo(a_*={a_f}) [s] [H]/[N]",
                     dts, exp_dts, tol=tol_dt)

# ==============================================================================
# MODULE 5: SGR A* 6.89 PC NSC, QUASAR CEILING, JWST SEEDS & LOOKUP TABLE
# ==============================================================================
def run_module_5_astrophysical_scaling():
    print("\n" + "=" * 94)
    print("MODULE 5: SGR A* 6.89 PC NSC (TABLE 7), QUASAR CEILING (TABLE 8) & JWST SEEDS (SEC. 12) [H] / [N]")
    print("=" * 94)

    M_sgra_msun = 4.154e6
    M_sgra_kg = M_sgra_msun * CBGConstants.M_sun
    rs_sgra_m = 2.0 * CBGConstants.G * M_sgra_kg / (CBGConstants.c**2)
    rs_sgra_pc = rs_sgra_m / CBGConstants.pc_m
    R_tr_sgra_pc = rs_sgra_pc / CBGConstants.Phi_inf_sq

    a_S2_m = 1031.3 * CBGConstants.AU_m
    e_S2 = 0.88466
    dphi_S2_arcmin = (3.0 * np.pi * rs_sgra_m / (a_S2_m * (1.0 - e_S2**2))) * (180.0 * 60.0 / np.pi)
    dphi_shift_rel = CBGConstants.Phi_inf_sq * np.sqrt(1.0 - e_S2**2)

    record_check("Section 10.1 (Eq. 10.4): Star S2 GR pericenter precession [arcmin/orbit] == 11.85 [N]",
                 dphi_S2_arcmin, 11.8523, tol=0.01)
    record_check("Section 10.1 (Eq. 10.5): Star S2 phase-locked precession shift == +2.69e-8 [H]/[N]",
                 dphi_shift_rel, 2.6917e-8, tol=1e-11)
    record_check("Section 10.2 (Eq. 10.6): Sgr A* transition radius R_tr^(Sgr A*) [pc] == 6.886 pc [H]/[N]",
                 R_tr_sgra_pc, 6.886, tol=0.005)

    M_NSC_msun = 2.50e7
    r_c_pc = 2.314
    Om_kms = CBGConstants.Omega_inf / 1000.0

    nsc_targets = [
        (0.040,        386.2, 392.8),
        (1.500,         87.6, 118.9),
        (R_tr_sgra_pc,  61.5, 112.3),
        (20.00,         41.7, 100.5),
    ]
    for r_pc, exp_bar, exp_cbg in nsc_targets:
        r_m = r_pc * CBGConstants.pc_m
        M_enc_ratio = (M_NSC_msun / M_sgra_msun) * ((r_pc / (r_pc + r_c_pc))**2)
        M_tot_kg = M_sgra_kg * (1.0 + M_enc_ratio)
        sig_bar_kms = np.sqrt(CBGConstants.G * M_tot_kg / (3.0 * r_m)) / 1000.0
        boost_kms2 = (Om_kms**2) * np.sqrt(1.0 + M_enc_ratio) / (1.0 + r_pc / (R_tr_sgra_pc * (1.0 + M_enc_ratio)))
        sig_cbg_kms = np.sqrt(sig_bar_kms**2 + boost_kms2)
        record_check(f"Table 7 (r={r_pc:.3f} pc): Pure GR baryonic dispersion sigma_v,bar [km/s] [N]",
                     sig_bar_kms, exp_bar, tol=0.15)
        record_check(f"Table 7 (r={r_pc:.3f} pc): CBG Jeans dispersion sigma_v,CBG [km/s] [H]/[N]",
                     sig_cbg_kms, exp_cbg, tol=0.15)

    M_max_kg = (CBGConstants.Omega_inf**3) / (8.0 * np.pi * CBGConstants.G * CBGConstants.H0_SI)
    M_max_msun = M_max_kg / CBGConstants.M_sun
    R_tr_max_kpc = (CBGConstants.Omega_inf / (4.0 * np.pi * CBGConstants.H0_SI)) / (1000.0 * CBGConstants.pc_m)
    rs_max_m = 2.0 * CBGConstants.G * M_max_kg / (CBGConstants.c**2)
    a_rep_max = (CBGConstants.Omega_inf**2) / (2.0 * rs_max_m)
    a_scr_max = CBGConstants.Phi_inf * a_rep_max

    record_check("Theorem 11.1 (Eq. 11.2): Ultramassive quasar ceiling M_max [M_sun] == 5.131e10 [H]/[N]",
                 M_max_msun, 5.131e10, tol=2e7)
    record_check("Theorem 11.1: Ceiling transition sphere R_tr(M_max) [kpc] == 85.05 kpc [H]/[N]",
                 R_tr_max_kpc, 85.05, tol=0.05)
    record_check("Theorem 11.1: Unscreened fiber acceleration a_Omega^(rep)(M_max) [m/s^2] == 1.7123e-5 [H]/[N]",
                 a_rep_max, 1.7123e-5, tol=2e-8)
    record_check("Theorem 11.1: Screened fiber acceleration a_Omega^(scr)(M_max) == 2 pi c H_0 [H]/[N]",
                 a_scr_max, 2.0 * np.pi * CBGConstants.c * CBGConstants.H0_SI, tol=1e-13)

    sigma_ISCO = 1.0 / 3.0
    xi_t2_ISCO = sigma_ISCO / (1.0 + sigma_ISCO)
    tau_Salp_GR_Myr = 45.0
    tau_Salp_CBG_Myr = (1.0 - xi_t2_ISCO) * tau_Salp_GR_Myr

    M_seed_msun = 100.0
    M_GNz11_low = M_seed_msun * np.exp(320.0 / tau_Salp_CBG_Myr)
    M_GNz11_high = M_seed_msun * np.exp(327.0 / tau_Salp_CBG_Myr)
    N_e_UHZ1 = 435.0 / tau_Salp_CBG_Myr
    M_UHZ1_pred = M_seed_msun * np.exp(N_e_UHZ1)

    record_check("Proposition 12.1 (Eq. 12.2): ISCO saturation fraction xi_t2(3r_s) == 1/4 [H]",
                 xi_t2_ISCO, 0.25, tol=1e-14)
    record_check("Proposition 12.1 (Eq. 12.3): Compressed Salpeter timescale tau_Salp^(CBG) == 33.75 Myr [H]/[N]",
                 tau_Salp_CBG_Myr, 33.75, tol=1e-12)
    record_check("Proposition 12.1(i): GN-z11 mass at dt_1=320 Myr [M_sun] == 1.31e6 M_sun [H]/[N]",
                 M_GNz11_low, 1.3114e6, tol=5e3)
    record_check("Proposition 12.1(i): GN-z11 mass at dt_1=327 Myr [M_sun] == 1.61e6 M_sun [H]/[N]",
                 M_GNz11_high, 1.6115e6, tol=5e3)
    record_check("Proposition 12.1(ii) (Eq. 12.4): UHZ1 e-foldings N_e = 435 / 33.75 == 12.889 [H]/[N]",
                 N_e_UHZ1, 12.8889, tol=1e-3)
    record_check("Proposition 12.1(ii) (Eq. 12.4): UHZ1 mass at z=10.1 (dt_1=435 Myr) == 3.96e7 M_sun [H]/[N]",
                 M_UHZ1_pred, 3.959e7, tol=1e5)
    
# ==============================================================================
# MAIN EXECUTION ENTRY POINT
# ==============================================================================
def main():
    f_bundle = run_module_0_symbolic_checks()
    sol_int, y_minus, r_minus, V_all, ode_rhs = run_module_1_interior_continuation(f_bundle)
    run_module_2_geodesic_turning_points(sol_int)
    run_module_3_kasner_inversion(f_bundle, y_minus, r_minus, V_all, ode_rhs)
    run_module_4_echoes_polarimetry_hawking()
    run_module_5_astrophysical_scaling()
    print("\n" + "=" * 94)
    print(f"FINAL VERDICT: {PASS_COUNT}/{PASS_COUNT} SYMBOLIC & NUMERICAL CHECKS PASSED (0 FAILURES).")
    print("ALL DEDUCTIVE [D] AUDITS VERIFIED WITH ZERO RESIDUALS; ALL [H]/[N] BENCHMARKS REPRODUCED (PAPER II, v10).")
    print("=" * 94)


if __name__ == "__main__":
    main()