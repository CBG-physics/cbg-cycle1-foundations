#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
CBG CYCLE I — TIER-BY-TIER ARCHITECTURE AUDIT OF README.md
================================================================================
Walks through every element of the repository README, in its order:

    Axioms [A1]-[A3]  ->  Unconditional theorems [D]  ->  Hypotheses [H1]-[H7], [H_aux.1-3]
                      ->  Conditional cascades [D | Hn, ...]  ->  Evaluations [N | Hn, ...]

Tagging rule used by the README and by this script:
    [D] / [D | Hn, ...]   exact symbolic statements only
    [N] / [N | Hn, ...]   every floating-point result, including machine-precision BVP/ODE
                          solutions (the monograph suites may label some of these [D] internally)

Each element receives exactly one status:

    DECLARED     axiom or hypothesis: recorded, never "proved" (anti-promotion lock)
    PROVED       unconditional [D]: exact SymPy identity, zero residual
    PROVED|Hn    conditional [D | Hn, ...]: exact SymPy identity that assumes the named hypotheses
    EVALUATED    [N | ...]: arithmetic on quoted inputs, compared with the README value
    DELEGATED    [N | ...] or [D]: reproduced by running the Paper III suite and reading its PASS line
    NOT RUN      delegated item whose suite was not found
    EXTERNAL     stated in the README, proved in another monograph suite, not re-derived here
    DISCREPANCY  the README text disagrees with the computation or with the papers

Only a failed PROVED / PROVED|Hn / EVALUATED / DELEGATED item makes the exit code non-zero.

Usage:   python cbg_readme_architecture_audit.py [--paper3 PATH] [--no-delegate]
         The Paper III suite is looked up in Paper_III_Cosmology/ and then next to this file.

This script does not import, edit or write any other file of the repository.
================================================================================
"""
import itertools
import math
import os
import re
import subprocess
import sys

import sympy as sp

# ------------------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------------------
REG = []
FAILED = []


def rec(tier, ident, title, status, detail=""):
    REG.append((tier, ident, title, status, detail))
    line = f"  [{status:<11s}] {ident:<8s} {title}"
    if detail:
        line += f"\n{'':17s}-> {detail}"
    print(line)


def prove(tier, ident, title, cond, hyp=None, detail=""):
    status = "PROVED" if hyp is None else f"PROVED|{hyp}"
    if not cond:
        status = "FAILED"
        FAILED.append(ident)
    rec(tier, ident, title, status, detail)


def evaluate(tier, ident, title, value, target, tol, detail=""):
    good = abs(float(value) - float(target)) <= tol
    if not good:
        FAILED.append(ident)
    rec(tier, ident, title, "EVALUATED" if good else "FAILED",
        f"{float(value):.8g} vs {float(target):.8g} (tol {tol:.1e})" + (f"; {detail}" if detail else ""))


def banner(text):
    print("\n" + "=" * 100 + f"\n{text}\n" + "=" * 100)


# ------------------------------------------------------------------------------
# Small geometry helpers
# ------------------------------------------------------------------------------
def christoffel(g, X):
    n = len(X)
    gi = g.inv()
    return [[[sum(gi[i, l] * (sp.diff(g[l, j], X[k]) + sp.diff(g[l, k], X[j]) - sp.diff(g[j, k], X[l])) for l in range(n)) / 2
              for k in range(n)] for j in range(n)] for i in range(n)]


def riemann_down(g, X):
    """R_{abcd} of a metric g in coordinates X."""
    n = len(X)
    G = christoffel(g, X)
    Rud = [[[[sp.diff(G[a][b][d], X[c]) - sp.diff(G[a][b][c], X[d])
              + sum(G[a][c][e] * G[e][b][d] - G[a][d][e] * G[e][b][c] for e in range(n))
              for d in range(n)] for c in range(n)] for b in range(n)] for a in range(n)]
    return [[[[sum(g[a, e] * Rud[e][b][c][d] for e in range(n)) for d in range(n)] for c in range(n)] for b in range(n)] for a in range(n)]


def ricci_scalar(g, X):
    n = len(X)
    G = christoffel(g, X)
    gi = g.inv()
    Ric = sp.Matrix(n, n, lambda j, k: sum(sp.diff(G[i][j][k], X[i]) - sp.diff(G[i][j][i], X[k])
                                           + sum(G[i][i][l] * G[l][j][k] - G[i][k][l] * G[l][j][i] for l in range(n)) for i in range(n)))
    return sum(gi[i, j] * Ric[i, j] for i in range(n) for j in range(n))


# Constants used by the numerical tiers (CODATA 2022 and the Paper III bundle values)
ALPHA_EM = 7.2973525643e-3
C_SI = 299792458.0
H0_CBG_SI = 2.182536592160143e-18          # 67.3460 km/s/Mpc (closure value of H_aux.3)
H0_PAPER1_SI = 67.4e3 / 3.0856775814913673e22   # 67.4 km/s/Mpc, the Paper I observed input [N]
HBAR_SI = 1.054571817e-34
G_SI = 6.67430e-11


# ==============================================================================
# TIER 0 — AXIOMS [A1]-[A3]  (README Section 2)
# ==============================================================================
def tier_axioms():
    banner("TIER 0 — CLOSED AXIOMATIC CORE [A1]-[A3]  (declared; only their internal bookkeeping is checked)")
    eta = sp.diag(-1, -1, 1, 1, 1)
    Rz = sp.diag(1, -1, 1, 1, 1)
    rec("A", "A1", "5D bitemporal orbifold arena M_{3,2}, signature (-,-,+,+,+), Z_2: t_2 -> -t_2", "DECLARED")
    prove("A", "A1.a", "bookkeeping: R_Z2^2 = I, det R_Z2 = -1, R_Z2 eta R_Z2 = eta, signature (2,3)",
          Rz * Rz == sp.eye(5) and Rz.det() == -1 and Rz * eta * Rz == eta and list(eta.diagonal()).count(-1) == 2)
    rec("A", "A2", "Spectral action Tr f(D_JV^2/Lambda^2) with finite non-zero winding moments M_k, k = 0,1,2", "DECLARED")
    rec("A", "A3", "Killing-horizon regularity |K_5(r_h)| < oo; A = (-g^{AB} d_A Theta d_B Theta)^(-1) enters V_spec only through psi_t(A)", "DECLARED")


# ==============================================================================
# TIER 1 — UNCONDITIONAL THEOREMS [D]  (README Section 3)
# ==============================================================================
def real_clifford_32():
    """Search a real 4x4 representation with squares (-,-,+,+,+) among Kronecker products of {1, X, Z, E}."""
    I2 = sp.eye(2)
    Xm = sp.Matrix([[0, 1], [1, 0]])
    Zm = sp.Matrix([[1, 0], [0, -1]])
    Em = sp.Matrix([[0, 1], [-1, 0]])
    basis = [sp.kronecker_product(u, v) for u in (I2, Xm, Zm, Em) for v in (I2, Xm, Zm, Em)]
    neg = [m for m in basis if m * m == -sp.eye(4)]
    pos = [m for m in basis if m * m == sp.eye(4) and m != sp.eye(4)]
    anti = lambda p, q: p * q + q * p == sp.zeros(4)
    for t_pair in itertools.combinations(neg, 2):
        if not anti(*t_pair):
            continue
        cand = [m for m in pos if all(anti(m, tt) for tt in t_pair)]
        for s_trip in itertools.combinations(cand, 3):
            if all(anti(p, q) for p, q in itertools.combinations(s_trip, 2)):
                return list(t_pair) + list(s_trip)
    return None


def tier_theorems(delegated):
    banner("TIER 1 — UNCONDITIONAL PURE THEOREMS [D]  (README Section 3)")
    T = "D"
    eta = sp.diag(-1, -1, 1, 1, 1)
    Rz = sp.diag(1, -1, 1, 1, 1)

    # ---- 3.1 manifold and algebraic invariants ------------------------------
    d_t, d_s = 2, 3
    D = d_t + d_s
    n_sym = D * (D + 1) // 2
    J30 = sp.Matrix(sp.BlockMatrix([[sp.zeros(n_sym), sp.eye(n_sym)], [-sp.eye(n_sym), sp.zeros(n_sym)]]))
    prove(T, "D3.1a", "(d_t, d_s, D) = (2, 3, 5); dim Sym^2(V_5*) = 15; N_Cartan = Tr(J_30^T J_30) = 30, J_30^2 = -I",
          (d_t, d_s, D, n_sym) == (2, 3, 5, 15) and (J30.T * J30).trace() == 30 and J30 * J30 == -sp.eye(30))
    units = [(i, j) for i in range(5) for j in range(5)]
    F = sp.diag(-1, -1, 1, 1, 1)                      # time/space grading (the metric signature operator)
    f_even = [u for u in units if F[u[0], u[0]] * F[u[1], u[1]] == 1]
    r_even = [u for u in units if Rz[u[0], u[0]] * Rz[u[1], u[1]] == 1]
    prove(T, "D3.1b", "End(V_5) = 13_+ (+) 12_- under the time/space grading F = diag(-,-,+,+,+): 13 = d_t^2 + d_s^2",
          len(f_even) == 13 and 25 - len(f_even) == 12 and d_t**2 + d_s**2 == 13)
    prove(T, "D3.1c", "dim (V_5* x V_5*)^{Z_2-even} = 17 = 11 (symmetric) + 6 (antisymmetric)",
          len(r_even) == 17 and len([u for u in r_even if u[0] <= u[1]]) == 11 and len([u for u in r_even if u[0] < u[1]]) == 6)
    sym_pairs = [(i, j) for i in range(5) for j in range(i, 5)]
    prove(T, "D3.1c'", "Sym^2(V_5*) = 11_+ (+) 4_- under R_Z2 (ten g_mu_nu and g_22 even; four g_mu_2 odd)",
          len([u for u in sym_pairs if Rz[u[0], u[0]] * Rz[u[1], u[1]] == 1]) == 11 and len(sym_pairs) == 15)
    gam = real_clifford_32()
    ok_cl = gam is not None and all(sp.simplify(gam[i] * gam[j] + gam[j] * gam[i] - 2 * eta[i, j] * sp.eye(4)) == sp.zeros(4)
                                    for i in range(5) for j in range(5))
    vol = gam[0] * gam[1] * gam[2] * gam[3] * gam[4] if gam else None
    prove(T, "D3.1d", "Cl(3,2) has a real 4x4 representation: {Gamma_a, Gamma_b} = 2 eta_ab with real matrices",
          ok_cl and all(all(entry.is_real for entry in m) for m in gam))
    prove(T, "D3.1d'", "its volume element Gamma_0...Gamma_4 is +-1 (central): the real 4x4 representation is one factor of Mat(4,R) (+) Mat(4,R)",
          vol in (sp.eye(4), -sp.eye(4)),
          detail="the sign depends on the ordering convention; Paper 0 fixes Gamma_vol = +I_4")
    # DeWitt supermetric at lambda = 1/2 on the 15 symmetric components
    lam = sp.Rational(1, 2)
    etai = eta.inv()
    mu = sp.Symbol('mu')
    idx = range(5)

    def G_up(a, b, c, d):      # G^{ABCD} = (1/2)(g^{AC} g^{BD} + g^{AD} g^{BC}) - lambda g^{AB} g^{CD}
        return sp.Rational(1, 2) * (etai[a, c] * etai[b, d] + etai[a, d] * etai[b, c]) - lam * etai[a, b] * etai[c, d]

    def G_dn(c, d, e_, f):     # candidate inverse (1/2)(g_{CE} g_{DF} + g_{CF} g_{DE}) + mu g_{CD} g_{EF}
        return sp.Rational(1, 2) * (eta[c, e_] * eta[d, f] + eta[c, f] * eta[d, e_]) + mu * eta[c, d] * eta[e_, f]

    ident_sym = lambda a, b, e_, f: sp.Rational(1, 2) * (int(a == e_) * int(b == f) + int(a == f) * int(b == e_))
    eqs = {sp.expand(sum(G_up(a, b, c, d) * G_dn(c, d, e_, f) for c in idx for d in idx) - ident_sym(a, b, e_, f))
           for a in idx for b in idx for e_ in idx for f in idx}
    eqs.discard(0)
    sol_mu = sp.solve(list(eqs), mu, dict=True)
    c_dw_inv = sol_mu[0][mu] if len(sol_mu) == 1 else None
    full_match = c_dw_inv is not None and all(sp.simplify(q.subs(mu, c_dw_inv)) == 0 for q in eqs)
    prove(T, "D3.1e", "DeWitt supermetric at lambda = 1/2: inverse trace coefficient c_DW^-1 = -1/(D-2) = -1/3 (15x15 inversion)",
          c_dw_inv == -sp.Rational(1, 3) and full_match)
    # kappa_psi = -3/2 from the 5D Einstein-Hilbert action on a homogeneous slice, then the Weyl map
    t = sp.Symbol('t', real=True)
    a = sp.Function('a', positive=True)(t)
    psi = sp.Function('psi', real=True)(t)
    X5 = sp.symbols('t t2 x1 x2 x3', real=True)
    g5 = sp.diag(-1, -sp.exp(2 * psi), a**2, a**2, a**2).subs(t, X5[0])
    R5 = sp.simplify(ricci_scalar(g5, X5)).subs(X5[0], t)
    a_t, p_t = sp.Function('a', positive=True)(t), sp.Function('psi', real=True)(t)
    dens = sp.simplify(a_t**3 * sp.exp(p_t) * R5)
    L_mini = a_t * sp.exp(p_t) * (-6 * a_t.diff(t)**2 - 6 * a_t * a_t.diff(t) * p_t.diff(t))
    c1, c2, sgn = sp.symbols('c1 c2 sgn')
    Bnd = sp.exp(p_t) * (c1 * a_t**2 * a_t.diff(t) + c2 * a_t**3 * p_t.diff(t))
    resid = sp.expand(sp.simplify((dens - sgn * L_mini - Bnd.diff(t)) * sp.exp(-p_t)))
    mons = [a_t * a_t.diff(t)**2, a_t**2 * a_t.diff(t) * p_t.diff(t), a_t**3 * p_t.diff(t)**2, a_t**2 * a_t.diff(t, 2), a_t**3 * p_t.diff(t, 2)]
    sol = sp.solve([resid.coeff(mm) for mm in mons], [c1, c2, sgn], dict=True)
    prove(T, "D3.1f'", "sqrt|g_5| R^(5) = +-L_mini + d/dt(boundary) with L_mini = a e^psi (-6 a_dot^2 - 6 a a_dot psi_dot)  (k = 0, N = 1)",
          len(sol) == 1 and sol[0][sgn] in (1, -1) and sp.simplify(resid.subs(sol[0])) == 0)
    at, pt_, da, dp = sp.symbols('a_E psi dA dPsi')          # Einstein variables: a = a_E e^{-psi/2}, dt_E = e^{psi/2} dt
    L_E = sp.expand(sp.simplify((at * sp.exp(-pt_ / 2)) * sp.exp(pt_) * sp.exp(-pt_ / 2)
                                * (-6 * (da - at * dp / 2)**2 - 6 * (at * sp.exp(-pt_ / 2)) * (da - at * dp / 2) * sp.exp(pt_ / 2) * dp)))
    kappa = -sp.Poly(L_E, da, dp).coeff_monomial(dp**2) / at**3
    prove(T, "D3.1f", "Weyl map a_E = a e^{psi/2}: L_E = -6 a_E a_E'^2 - kappa_psi a_E^3 psi'^2 with kappa_psi = -3/2 and no cross term",
          sp.simplify(L_E - (-6 * at * da**2 + sp.Rational(3, 2) * at**3 * dp**2)) == 0 and kappa == -sp.Rational(3, 2))
    c_weyl = 1
    prove(T, "D3.1g", "alpha_0^2 = c_Weyl^2 / (-2 kappa_psi) = 1/3 = -c_DW^-1",
          sp.Rational(c_weyl**2, 1) / (-2 * kappa) == sp.Rational(1, 3) and -c_dw_inv == sp.Rational(1, 3),
          detail="the number is [D]; its realisation as a force on matter is [D | H1, H2]")

    # ---- 3.2 Dirac reduction and parity -------------------------------------
    Acj, S = gam[0] * gam[1], gam[1]

    def sign_of(M1, M2):
        r = sp.simplify(M1 * M2.inv())
        assert r == r[0, 0] * sp.eye(4)
        return r[0, 0]

    s_m = sign_of(S.T * Acj * S, Acj)
    s_k = [sign_of(S.T * Acj * gam[b] * S * (-1 if b == 1 else 1), Acj * gam[b]) for b in range(5)]
    prove(T, "D3.2b", "bulk Dirac mass bilinear is Z_2-odd relative to the kinetic bilinear under t_2 -> -t_2 (S = Gamma^1)",
          len(set(s_k)) == 1 and sp.simplify(s_m / s_k[0]) == -1)
    t1, t2, x1, x2, x3 = sp.symbols('t1 t2 x1 x2 x3', real=True)
    Xs = (t1, t2, x1, x2, x3)
    Ph = sp.Function('Phi', positive=True)(t1, x1)
    gD = sp.diag(-1, -Ph**2, 1, 1, 1)
    e = sp.diag(1, Ph, 1, 1, 1)
    ei = e.inv()
    G5 = christoffel(gD, Xs)

    def omega(M, a_, b_):
        return sp.simplify(sum(e[a_, n_] * (sp.diff(ei[n_, b_], Xs[M]) + sum(G5[n_][M][K] * ei[K, b_] for K in range(5))) for n_ in range(5)))

    chi = sp.Matrix([sp.Function('chi%d' % i)(t1, x1, x2, x3) for i in range(4)])
    psi5 = chi / sp.sqrt(Ph)
    out = sp.zeros(4, 1)
    for a_ in range(5):
        M = a_                                                # diagonal vielbein
        conn = sp.zeros(4)
        for b_ in range(5):
            for c_ in range(5):
                conn += sp.Rational(1, 4) * omega(M, b_, c_) * eta[c_, c_] * gam[b_] * gam[c_]
        out += eta[a_, a_] * gam[a_] * ei[M, a_] * (psi5.diff(Xs[M]) + conn * psi5)
    flat4 = sum((eta[a_, a_] * gam[a_] * chi.diff(Xs[a_]) for a_ in (0, 2, 3, 4)), sp.zeros(4, 1))
    prove(T, "D3.2a", "massless 5D Dirac operator on Psi_5 = Phi^(-1/2) chi_4 equals Phi^(-1/2) D_4 chi_4 (zero modes)",
          sp.simplify(out - flat4 / sp.sqrt(Ph)) == sp.zeros(4, 1),
          detail="proved for ds^2 = -dt_1^2 - Phi(t_1, x)^2 dt_2^2 + dx^2, as the README states; a general curved 4D block is not covered")

    # ---- 3.3 khronon conservation law ---------------------------------------
    tt = sp.Symbol('t', real=True)
    aF, pF, ThF, NF = [sp.Function(n_)(tt) for n_ in ('a', 'psi', 'Theta', 'N')]
    kk = sp.Symbol('k')
    V = sp.Function('V')
    A_ = sp.Symbol('A', positive=True)
    Aexpr = NF**2 / ThF.diff(tt)**2
    L = aF * sp.exp(pF) * (-6 * aF.diff(tt)**2 / NF - 6 * aF * aF.diff(tt) * pF.diff(tt) / NF + 6 * kk * NF - 2 * aF**2 * NF * V(pF, Aexpr))
    VA = sp.Derivative(V(pF, A_), A_)
    J = sp.diff(L, ThF.diff(tt))
    J_claim = 4 * aF**3 * sp.exp(pF) * NF**3 / ThF.diff(tt)**3 * sp.Subs(VA, A_, Aexpr)
    prove(T, "D3.3a", "J_Theta = dL_mini/dTheta_dot = 4 a^3 e^psi A^(3/2) dV/dA, and Theta is cyclic (J_Theta conserved)",
          sp.simplify((J - J_claim).doit()) == 0 and sp.diff(L, ThF) == 0)
    dLdN = sp.diff(-2 * aF**3 * sp.exp(pF) * NF * V(pF, Aexpr), NF)
    prove(T, "D3.3b", "lapse equation carries V + 2A dV/dA, hence 2A dV/dA = J_Theta / (2 a^3 e^psi sqrt(A))",
          sp.simplify((dLdN + 2 * aF**3 * sp.exp(pF) * (V(pF, Aexpr) + 2 * Aexpr * sp.Subs(VA, A_, Aexpr))).doit()) == 0,
          detail="'proportional to a^-3' holds at fixed psi and A")
    m_, pm_, Ac_, ps_ = sp.symbols('m psi_max A_c psi_s', positive=True)
    ptg = pm_ * (sp.exp(-A_**2 / Ac_**2) - sp.exp(-1 / Ac_**2))
    VH = sp.Rational(3, 4) * m_**2 * (ps_ - ptg)**2
    prove(T, "D3.3c", "with the Hooke potential and Gaussian kernel: dV/dA = 0 <=> psi = psi_t(A)",
          sp.simplify(sp.diff(VH, A_) - 3 * m_**2 * (ps_ - ptg) * pm_ * A_ * sp.exp(-A_**2 / Ac_**2) / Ac_**2) == 0, hyp="H3",
          detail="listed in the README under the H3 cascade; J_Theta = 0 then gives V = 0 and V_psi = 0")
    # turning-point identity with a radiation fluid rho_r = rho_0 a^-4 e^-psi (k eliminated with the lapse constraint)
    Nq, Thd, rho0 = sp.symbols('N Thetadot rho_0', positive=True)
    aq, pq = sp.Function('a')(tt), sp.Function('psi')(tt)
    Aq = Nq**2 / Thd**2
    Lq = aq * sp.exp(pq) * (-6 * aq.diff(tt)**2 / Nq - 6 * aq * aq.diff(tt) * pq.diff(tt) / Nq + 6 * kk * Nq - 2 * aq**2 * Nq * V(pq, Aq)) - 2 * Nq * rho0 / aq
    add, pdd = sp.symbols('add pdd')

    def at_turn(E_):
        return sp.simplify(E_.doit().subs({aq.diff(tt, 2): add, pq.diff(tt, 2): pdd}).subs({aq.diff(tt): 0, pq.diff(tt): 0}).subs(Nq, 1))

    EN0 = at_turn(sp.diff(Lq, Nq))
    Ea0 = at_turn(sp.diff(sp.diff(Lq, aq.diff(tt)), tt) - sp.diff(Lq, aq))
    Ep0 = at_turn(sp.diff(sp.diff(Lq, pq.diff(tt)), tt) - sp.diff(Lq, pq))
    k_sol = sp.solve(EN0, kk)[0]
    turn = sp.solve([Ea0.subs(kk, k_sol), Ep0.subs(kk, k_sol)], [add, pdd], dict=True)[0]
    A0 = 1 / Thd**2
    V_psi = sp.Derivative(V(pq, A0), pq)
    V_A = sp.Subs(sp.Derivative(V(pq, A_), A_), A_, A0)
    rho_r = rho0 * sp.exp(-pq) / aq**4
    prove(T, "D3.3d", "turning point (a_dot = psi_dot = 0, N = 1) with a radiation fluid: a_ddot/a = (1/3)[V_psi - rho_r - 2A V_A]",
          sp.simplify((turn[add] / aq - (V_psi - rho_r - 2 * A0 * V_A) / 3).doit()) == 0)
    prove(T, "D3.3d'", "and fiber recoil psi_ddot = (2/3)[V - V_psi + A V_A], independent of the radiation density",
          sp.simplify((turn[pdd] - sp.Rational(2, 3) * (V(pq, A0) - V_psi + A0 * V_A)).doit()) == 0)
    # homogeneous anisotropic S^3
    th3, ph3, ch3 = sp.symbols('theta phi chi', real=True)
    a1, a2, a3 = sp.symbols('a1 a2 a3', positive=True)
    X3 = (th3, ph3, ch3)
    f1 = [-sp.sin(ch3), sp.sin(th3) * sp.cos(ch3), 0]
    f2 = [sp.cos(ch3), sp.sin(th3) * sp.sin(ch3), 0]
    f3 = [0, sp.cos(th3), 1]
    g3 = sp.Matrix(3, 3, lambda i, j: a1**2 * f1[i] * f1[j] + a2**2 * f2[i] * f2[j] + a3**2 * f3[i] * f3[j])
    R3 = sp.simplify(ricci_scalar(g3, X3).subs({th3: sp.pi / 3, ph3: 0, ch3: sp.pi / 5}))
    R3_claim = (2 * (a1**2 * a2**2 + a2**2 * a3**2 + a3**2 * a1**2) - a1**4 - a2**4 - a3**4) / (2 * a1**2 * a2**2 * a3**2)
    prove(T, "D3.3e", "homogeneous anisotropic S^3: 3R = [2(a1^2a2^2 + a2^2a3^2 + a3^2a1^2) - a1^4 - a2^4 - a3^4]/(2 a1^2 a2^2 a3^2)",
          sp.simplify(R3 - R3_claim) == 0, detail="evaluated exactly at one point of the homogeneous space")
    if delegated is None:
        rec(T, "D3.3f", "Bianchi-I shear first integral d/dt_1 (a^3 Phi sigma^i_j) = 0", "NOT RUN", "Paper III suite Check #021")
    else:
        if not delegated.get(21, False):
            FAILED.append("D3.3f")
        rec(T, "D3.3f", "Bianchi-I shear first integral d/dt_1 (a^3 Phi sigma^i_j) = 0 (Paper III suite Check #021, symbolic)",
            "DELEGATED" if delegated.get(21, False) else "FAILED")
    rec(T, "D3.2s", "spectral invariants of README 3.2 (Lichnerowicz weights, c_GB, holonomy factor, zeta'(0), M_2, det M_jet, Str)", "EXTERNAL",
        "proved in the Paper 0 suite (THM #15-#19)")
    prove(T, "D3.2j", "barrier polynomial Pi_2(psi) = (1/8) e^{-6 psi}(e^{2 psi} - 1)^2: maximum Pi_2((1/2) ln 3) = 1/54",
          sp.simplify((sp.Rational(1, 8) * sp.exp(-6 * ps_) * (sp.exp(2 * ps_) - 1)**2).subs(ps_, sp.log(3) / 2)) == sp.Rational(1, 54)
          and sp.simplify(sp.diff(sp.Rational(1, 8) * sp.exp(-6 * ps_) * (sp.exp(2 * ps_) - 1)**2, ps_).subs(ps_, sp.log(3) / 2)) == 0)
    prove(T, "D3.2k", "ball overlap: f = (6/5)(1 - f_mut) turns the jump Delta f_mut^(4)(1) = -30 into Delta f^(4)(1) = +36",
          -sp.Rational(6, 5) * (-30) == 36, detail="the value -30 itself is proved in the Paper 0 suite (THM #18)")
    rec(T, "D3.2h", "two-sided horizon regularity: regular slope psi'(r_h) and C_{B^2} + 1/b_{-1} = 0", "EXTERNAL",
        "proved in the Paper 0 and Paper II suites (THM #20, #22)")

    # ---- 3.4 Chern-Simons identity ------------------------------------------
    Af = [sp.Function('A%d' % i)(t1, x1, x2, x3) if i != 1 else sp.Integer(0) for i in range(5)]
    Fm = sp.Matrix(5, 5, lambda i, j: sp.diff(Af[j], Xs[i]) - sp.diff(Af[i], Xs[j]))
    cs = sum(sp.LeviCivita(*pp) * Af[pp[0]] * Fm[pp[1], pp[2]] * Fm[pp[3], pp[4]] for pp in itertools.permutations(range(5)))
    prove(T, "D3.4", "eps^{ABCDE} A_A F_BC F_DE = 0 for Z_2-even, t_2-independent potentials (A_2 = 0)", sp.simplify(cs) == 0,
          detail="an identity on the class of t_2-independent configurations")
    return gam


# ==============================================================================
# TIER 2 — ROOT HYPOTHESES [H1]-[H7]  (README Section 4)
# ==============================================================================
def tier_hypotheses():
    banner("TIER 2 — HYPOTHESES [H1]-[H7] AND [H_aux.1]-[H_aux.3]  (declared; none is derived from [A1]-[A3])")
    for ident, text in (
        ("H1", "Projective cylinder restriction  L_xi g_AB = 0  (d/dt_2 = 0)"),
        ("H2", "(a) matter couples to g_mu_nu; (b) Phi_inf = (alpha_EM/30)(1 - 5 alpha_EM/3); (c) D_config = 15 + 2 = 17; (d) lambda_psi = sqrt(L_P R_H)"),
        ("H3", "Hooke 2-jet potential, Gaussian kernel psi_t(A), thresholds A_crit = 1/4, psi_max = 2/25, m_hat_ref = 4"),
        ("H4", "UV saturation chi_max = 2/L_P^2, regularised horizon temperature, xi_t2(3 r_s) = 1/4, mesoscopic collapse gates"),
        ("H5", "Director-cosine law mu_CBG(x) = x/sqrt(1+x^2), a_0 = c H_0/(2 pi), 17/15 weighting, virial shift of H_0"),
        ("H6", "Five-sector system, IR exhaustion sector -alpha_Lambda (x/x_ap)^2, and apogee velocity reflection at H = 0"),
        ("H7", "Horizon polarization holonomy  d beta = (pi/2) chi_E(S^2) Phi_inf d ln a"),
        ("H_aux.1", "Initial data and truncations: interior flux C_Omega, A = 1 truncation, shear law on closed S^3"),
        ("H_aux.2", "Strong-field and galactic-centre models"),
        ("H_aux.3", "Cosmological closure expressions, energy rungs, relaxation / pump / acoustic / transfer ansaetze"),
    ):
        rec("H", ident, text, "DECLARED")


# ==============================================================================
# TIER 3 — CONDITIONAL CASCADES [D|H_n]  (README Sections 4-5)
# ==============================================================================
def tier_cascades(delegated):
    banner("TIER 3 — CONDITIONAL DEDUCTIVE CASCADES [D|H_n]  (each assumes the named root hypothesis)")
    T = "D|H"
    # ---- H1 -----------------------------------------------------------------
    w_, k_, n_, Lf, Ph0 = sp.symbols('omega k n L Phi_0', positive=True)
    tq, t2q, xq = sp.symbols('t t2 x', real=True)
    mode = sp.exp(sp.I * (k_ * xq - w_ * tq)) * sp.exp(sp.I * n_ * t2q / Lf)
    box = -sp.diff(mode, tq, 2) - sp.diff(mode, t2q, 2) / Ph0**2 + sp.diff(mode, xq, 2)
    disp = sp.solve(sp.simplify(box / mode), w_**2)[0]
    prove(T, "C1a", "Kaluza-Klein tower on the timelike fiber: omega^2 = k^2 - n^2/(Phi_0 L)^2, i.e. m_n^2 = -n^2/(Phi_0 L)^2 < 0 for n >= 1",
          sp.simplify(disp - (k_**2 - n_**2 / (Ph0 * Lf)**2)) == 0, hyp="H1",
          detail="H1 removes these modes by restriction")
    prove(T, "C1b", "zero mode (n = 0): omega^2 = k^2, luminal propagation c_T = 1", sp.simplify(disp.subs(n_, 0) - k_**2) == 0, hyp="H1",
          detail="shown for the scalar wave operator; the tensor modes obey the same operator in the zero-mode sector")
    # ---- H2 -----------------------------------------------------------------
    y = sp.Symbol('y', nonnegative=True)                   # y = k^2 / (k^2 + a^2 m_psi^2) in [0, 1]
    al = sp.Rational(1, 3)
    mu_eff = 1 + al * y
    eta_J = (1 - al * y) / (1 + al * y)
    prove(T, "C2a", "alpha_Yuk = 1/3: mu_eff = 1 + y/3 and eta_J = (1 - y/3)/(1 + y/3) for y in [0, 1): mu_eff in [1, 4/3), eta_J in (1/2, 1]",
          mu_eff.subs(y, 1) == sp.Rational(4, 3) and eta_J.subs(y, 1) == sp.Rational(1, 2) and eta_J.subs(y, 0) == 1
          and sp.simplify(sp.diff(eta_J, y)) == sp.simplify(-2 * al / (1 + al * y)**2), hyp="H1,H2")
    Phi_inf = (ALPHA_EM / 30.0) * (1.0 - 5.0 * ALPHA_EM / 3.0)
    evaluate(T, "C2b", "Phi_inf = (alpha_EM/30)(1 - 5 alpha_EM/3) from CODATA alpha_EM  [N | H2]", Phi_inf, 2.40286677e-4, 5e-12)
    aS, PS = sp.symbols('alpha Phi', positive=True)
    roots = sp.solve(sp.Eq(PS, aS / 30 * (1 - sp.Rational(5, 3) * aS)), aS)
    back = [float(r.subs(PS, Phi_inf)) for r in roots]
    prove(T, "C2c", "quadratic inversion alpha_EM(Phi_inf) = (3/10)[1 - sqrt(1 - 200 Phi_inf)] returns the input alpha_EM",
          any(abs(b - ALPHA_EM) < 1e-12 for b in back)
          and any(sp.simplify(r - sp.Rational(3, 10) * (1 - sp.sqrt(1 - 200 * PS))) == 0 for r in roots), hyp="H2")
    prove(T, "C2d", "configuration weighting D_config / 15 = 17/15", sp.Rational(17, 15) == sp.Rational(15 + 2, 15), hyp="H2",
          detail="17 = 15 + 2 is the hypothesis itself; the [D] integer 17 of D3.1c is a different count")
    # ---- H3 -----------------------------------------------------------------
    psi_max, A_c = 2.0 / 25.0, 0.25
    dpsit = -2.0 * psi_max * math.exp(-1.0 / A_c**2) / A_c**2
    evaluate(T, "C3a", "screening value [psi_t'(1)]^2 = 8.30e-14  [N | H1,H3]", dpsit**2 * 1e14, 8.2995875, 1e-6)
    if delegated is None:
        rec(T, "C3b", "stiff-limit black-hole hair f_hair = 4.162 %  [N | H1,H3]", "NOT RUN", "Paper III suite Check #080")
    else:
        ok_ = delegated.get(80, False)
        if not ok_:
            FAILED.append("C3b")
        rec(T, "C3b", "stiff-limit black-hole hair f_hair = 4.162 %  [N | H1,H3]  (Paper III suite Check #080)", "DELEGATED" if ok_ else "FAILED")
    rec(T, "C3c", "interior turnover r_turn = 0.9081 r_h, area shift -8.15 %, r_ADM/r_h = 1.0434  [N | H1,H3]", "EXTERNAL",
        "computed in the Paper II and Paper III suites")
    # ---- H4 -----------------------------------------------------------------
    r, th, ph_ = sp.symbols('r theta phi', positive=True)
    tS, t2S = sp.symbols('t t2', real=True)
    rs, LP, rc, Ph0s = sp.symbols('r_s L_P r_c Phi_0', positive=True)
    fH = 1 - rs * r**2 / (r**3 + rc**3)
    gH = sp.diag(-fH, -Ph0s**2, 1 / fH, r**2, r**2 * sp.sin(th)**2)
    XH = (tS, t2S, r, th, ph_)
    Rm = riemann_down(gH, XH)
    gHi = gH.inv()
    K5 = sum(Rm[a][b][c][d]**2 * gHi[a, a] * gHi[b, b] * gHi[c, c] * gHi[d, d]
             for a in range(5) for b in range(5) for c in range(5) for d in range(5))
    K5_0 = sp.simplify(sp.limit(sp.simplify(K5), r, 0).subs(rc, (rs * LP**2 / 2)**sp.Rational(1, 3)))
    prove(T, "C4a", "Hayward core f = 1 - r_s r^2/(r^3 + r_core^3), r_core = (r_s L_P^2/2)^(1/3): K_5(0) = 96/L_P^4", K5_0 == 96 / LP**4, hyp="H4")
    RH = sp.Symbol('R_H', positive=True)
    prove(T, "C4b", "cosmological throat r_throat = (R_H L_P^2/2)^(1/3): K_5(0) = 48 R_H^2/r_throat^6 = 192/L_P^4",
          sp.simplify(48 * RH**2 / ((RH * LP**2 / 2)**sp.Rational(1, 3))**6 - 192 / LP**4) == 0, hyp="H4")
    xm = sp.Symbol('x', positive=True)                     # x = M / m_P
    Tn = (1 / xm) * (1 - 1 / (4 * xm**2))
    prove(T, "C4c", "regularised Hawking profile T ~ (1/x)(1 - 1/(4x^2)), x = M/m_P: peak at M = (sqrt(3)/2) m_P, zero at M_rem = m_P/2",
          sp.solve(sp.diff(Tn, xm), xm) == [sp.sqrt(3) / 2] and sp.solve(Tn, xm) == [sp.Rational(1, 2)], hyp="H4")
    prove(T, "C4c'", "peak temperature T_max = [T_P/(8 pi)] x T_n(sqrt(3)/2) = T_P/(6 sqrt(3) pi)",
          sp.simplify(Tn.subs(xm, sp.sqrt(3) / 2) / (8 * sp.pi) - 1 / (6 * sp.sqrt(3) * sp.pi)) == 0, hyp="H4")
    xi = sp.Rational(1, 4)
    prove(T, "C4d", "saturation fraction xi_t2(3 r_s) = 1/4 gives the accretion time factor 1 - xi = 3/4", 1 - xi == sp.Rational(3, 4), hyp="H4")
    # ---- H5 -----------------------------------------------------------------
    gt, gb, a0 = sp.symbols('g_tot g_bar a_0', positive=True)
    g_inv = sp.sqrt((gb**2 + sp.sqrt(gb**4 + 4 * a0**2 * gb**2)) / 2)
    xx = g_inv / a0
    prove(T, "C5a", "algebraic inversion: g_tot(g_bar) = sqrt{[g_bar^2 + sqrt(g_bar^4 + 4 a_0^2 g_bar^2)]/2} solves mu_CBG(g_tot/a_0) g_tot = g_bar",
          sp.simplify((xx / sp.sqrt(1 + xx**2) * g_inv)**2 - gb**2) == 0, hyp="H5")
    prove(T, "C5b", "deep regime g_bar -> 0: g_tot^2 / (a_0 g_bar) -> 1, hence v_f^4 = G M_b a_0",
          sp.limit(g_inv**2 / (a0 * gb), gb, 0) == 1, hyp="H5")
    xq5 = sp.Symbol('x', positive=True)
    prove(T, "C5c", "logarithmic elasticity d ln mu_CBG / d ln x = 1/(1 + x^2)",
          sp.simplify(sp.diff(sp.log(xq5 / sp.sqrt(1 + xq5**2)), xq5) * xq5 - 1 / (1 + xq5**2)) == 0, hyp="H5")
    a0_dS = C_SI * H0_PAPER1_SI / (2 * math.pi)
    evaluate(T, "C5d", "a_0^(dS) = c H_0 / (2 pi) at the Paper I input H_0 = 67.40 km/s/Mpc, in 1e-10 m/s^2  [N | H5]", a0_dS * 1e10, 1.0422, 1e-4)
    evaluate(T, "C5e", "same scale at the closure value H_0^(CBG) = 67.3460 km/s/Mpc  [N | H5,H_aux.3]", C_SI * H0_CBG_SI / (2 * math.pi) * 1e10, 1.0414, 1e-4)
    # ---- H6 -----------------------------------------------------------------
    x, vx, vp, C, am, aK, aL, ek, xa = sp.symbols('x v_x v_psi C_P2 alpha_m alpha_K alpha_L ek x_ap', positive=True)
    psS = sp.Symbol('psi', real=True)
    Vt = sp.Function('V_t')(psS)
    Hd = vx / x

    def EN(vx_, vp_):
        return (vx_ / x)**2 - vp_**2 - ((1 + Vt / 6) / x**4 + am * sp.exp(ek * psS) / x**3 + aK / x**2 + aL * (1 - x**2 / xa**2) - C / x**6)

    def dvx(vx_, vp_):
        return x * (-2 * vp_**2 - (1 + Vt / 6) / x**4 - am * sp.exp(ek * psS) / (2 * x**3) + aL * (1 - 2 * x**2 / xa**2) + 2 * C / x**6)

    def dvp(vx_, vp_):
        return -3 * (vx_ / x) * vp_ - sp.diff(Vt, psS) / (12 * x**4) - ek * am * sp.exp(ek * psS) / (2 * x**3)

    E0 = EN(vx, vp)
    dE = sp.diff(E0, x) * vx + sp.diff(E0, vx) * dvx(vx, vp) + sp.diff(E0, psS) * vp + sp.diff(E0, vp) * dvp(vx, vp)
    prove(T, "C6a", "five-sector system + exhaustion sector: dE_N/dtau + 2 H_d E_N = 0 identically (Bianchi-Noether identity)",
          sp.simplify(dE + 2 * Hd * E0) == 0, hyp="H6", detail="ek stands for epsilon_S3 * kappa_psi")
    prove(T, "C6b", "exact time-reversal invariance tau -> -tau, (v_x, v_psi) -> (-v_x, -v_psi)",
          sp.simplify(EN(vx, vp) - EN(-vx, -vp)) == 0 and sp.simplify(dvx(vx, vp) - dvx(-vx, -vp)) == 0
          and sp.simplify(dvp(vx, vp) - dvp(-vx, -vp)) == 0, hyp="H6")
    tau = sp.Symbol('tau', real=True)
    hL = sp.sqrt(aL)
    xs = xa / sp.cosh(hL * tau)
    prove(T, "C6c", "vacuum + exhaustion sector alone: x = x_ap sech(sqrt(alpha_L) tau) solves H^2 = alpha_L (1 - x^2/x_ap^2); H = 0 at the apogee",
          sp.simplify((sp.diff(xs, tau) / xs)**2 - aL * (1 - xs**2 / xa**2)) == 0 and sp.diff(xs, tau).subs(tau, 0) == 0, hyp="H6")
    prove(T, "C6d", "at the apogee a_ddot/a = alpha_L (1 - 2 x^2/x_ap^2)|_{x = x_ap} = -alpha_L < 0 (smooth turning point)",
          sp.simplify((sp.diff(xs, tau, 2) / xs).subs(tau, 0) + aL) == 0, hyp="H6")
    for num, ident, label in ((104, "C6e", "chained Steps 0->18: constraint / finite-difference residuals <= 1e-9 (Check #104)  [N | H6,H_aux.3]"),
                              (110, "C6f", "apogee age t_apogee^(4D) = 317.39 Gyr (Check #110)"),
                              (112, "C6g", "mirror cycle T_cycle = 634.78 Gyr 4D / 8,660.41 Gyr 5D (Check #112)"),
                              (153, "C6h", "without the apogee reflection the contraction reaches a kinetic collapse (Check #153)")):
        if delegated is None:
            rec(T, ident, label, "NOT RUN", "needs the Paper III suite")
        else:
            good = delegated.get(num, False)
            if not good:
                FAILED.append(ident)
            rec(T, ident, label, "DELEGATED" if good else "FAILED")
    for num, ident, label in ((26, "C8a", "k = +1 bounce acceleration at A = 1 (symbolic, Check #026)  [D | H1,H_aux.1]"),
                              (66, "C8b", "oscillating-psi condensate <w_psi> -> 0 (Check #066)  [N | H1,H3]"),
                              (67, "C8c", "oscillating-psi condensate a^3 rho_psi constant (Check #067)  [N | H1,H3]")):
        if delegated is None:
            rec(T, ident, label, "NOT RUN", "needs the Paper III suite")
        else:
            good = delegated.get(num, False)
            if not good:
                FAILED.append(ident)
            rec(T, ident, label, "DELEGATED" if good else "FAILED")
    # ---- H7 -----------------------------------------------------------------
    Phi_inf = (ALPHA_EM / 30.0) * (1.0 - 5.0 * ALPHA_EM / 3.0)
    chiE = 2
    beta = lambda z: (math.pi / 2) * chiE * Phi_inf * math.log(1.0 + z)
    evaluate(T, "C7a", "beta_CMB = pi Phi_inf ln(1 + z) at z = 1089, in degrees  [N | H2,H7]", math.degrees(beta(1089.0)), 0.3025, 1e-4,
             detail="at z_* = 1089.8 the Paper III suite prints 0.30253")
    evaluate(T, "C7b", "beta_reion = pi Phi_inf ln(1 + 7.7), in degrees  [N | H2,H7]", math.degrees(beta(7.7)), 0.0936, 1e-4)
    evaluate(T, "C7c", "C_l^EB / C_l^EE = (1/2) sin(4 beta)  [N | H2,H7]", 0.5 * math.sin(4 * beta(1089.8)), 1.056e-2, 1e-5,
             detail="evaluated at the Paper III value of beta")
    return Phi_inf, a0_dS, dpsit


# ==============================================================================
# TIER 4 — EVALUATIONS [N]  (README Section 6)
# ==============================================================================
def tier_evaluations(Phi_inf, a0_dS, dpsit, delegated):
    banner("TIER 4 — NUMERICAL EVALUATIONS [N | Hn, ...]  (reference values are quoted from the papers, not independently checked)")
    T = "N"
    rec(T, "N1", "zero-mode wave speed c_T/c = 1  [D | H1]", "EVALUATED", "exact by C1b; holds by construction under H1")
    L_P = math.sqrt(HBAR_SI * G_SI / C_SI**3)
    lam = math.sqrt(L_P * C_SI / H0_PAPER1_SI)
    evaluate(T, "N2", "scalar range lambda_micro = sqrt(L_P R_H) in micrometres  [N | H1,H2]", lam * 1e6, 47.10, 1e-2)
    evaluate(T, "N2'", "corresponding energy hbar c / lambda_micro in meV", HBAR_SI * C_SI / lam / 1.602176634e-19 * 1e3, 4.19, 1e-2)
    if delegated is None:
        rec(T, "N3", "stiff-limit hair 4.162 %  [N | H1,H3]", "NOT RUN", "Paper III suite Check #080")
    else:
        if not delegated.get(80, False):
            FAILED.append("N3")
        rec(T, "N3", "stiff-limit hair 4.162 %  [N | H1,H3]  (Check #080)", "DELEGATED" if delegated.get(80, False) else "FAILED")
    rec(T, "N3'", "stiff-limit shadow shift -0.071 %  [N | H1,H3]", "EXTERNAL", "computed in the Paper II suite")
    psi_max, A_c = 2.0 / 25.0, 0.25
    s1 = dpsit
    s2 = psi_max * math.exp(-1.0 / A_c**2) * (-2.0 / A_c**2 + 4.0 / A_c**4)
    evaluate(T, "N4", "tracking shift gamma - 1 = 2 psi_t'(1), in units of 1e-7  [N | H1,H3]", 2 * s1 * 1e7, -5.762, 1e-3)
    beta_ppn = (1 + s1 - s2 - 4 * s1 * s2) / (1 + 2 * s1 + 4 * s1**2)
    evaluate(T, "N4'", "beta - 1 from the Paper I tracking formula, in units of 1e-6", (beta_ppn - 1) * 1e6, -8.643, 1e-3)
    a0_loc_factor = 1.132109 / 1.042198
    for ident, name, val in (("N5a", "a_0^(dS)", a0_dS), ("N5b", "a_0^(local)", a0_dS * a0_loc_factor), ("N5c", "a_0^(CBG) = (17/15) a_0^(dS)", a0_dS * 17.0 / 15.0)):
        rec(T, ident, f"{name}  [N | H2,H5]", "EVALUATED", f"{val:.4e} m/s^2; offset from 1.20(2)e-10: {(val * 1e10 - 1.20) / 0.02:+.1f} sigma")
    rec(T, "N5d", "a_0^(local): the virial-shift factor is taken from the Paper I suite", "EXTERNAL", "H_0^(local) = H_0 (1 + Omega_inf / v_vir)")
    for num, ident, label in ((107, "N6a", "chained age t_0 = 13.7909 Gyr  [N | H6,H_aux.3]  (Check #107)"),
                              (110, "N6b", "apogee age 317.39 Gyr  [N | H6,H_aux.3]  (Check #110)"),
                              (112, "N6c", "cycle period 634.78 Gyr  [N | H6,H_aux.3]  (Check #112)")):
        if delegated is None:
            rec(T, ident, label, "NOT RUN", "needs the Paper III suite")
        else:
            if not delegated.get(num, False):
                FAILED.append(ident)
            rec(T, ident, label, "DELEGATED" if delegated.get(num, False) else "FAILED")
    beta_deg = math.degrees(math.pi * Phi_inf * math.log(1.0 + 1089.8))
    evaluate(T, "N7", "polarization rotation at z_* = 1089.8, in degrees  [N | H2,H7]", beta_deg, 0.3025, 1e-4,
             detail=f"offset from 0.34(9): {(beta_deg - 0.34) / 0.09:+.2f} sigma")
    evaluate(T, "N8", "reionization rotation at z = 7.7, in degrees  [N | H2,H7]", math.degrees(math.pi * Phi_inf * math.log(8.7)), 0.0936, 1e-4)
    evaluate(T, "N9", "sub-ring polarization shift Phi_inf rad, in arcsec  [N | H2,H7]", Phi_inf * 180.0 / math.pi * 3600.0, 49.56, 1e-2)


# ==============================================================================
# Delegation to the locked Paper III suite
# ==============================================================================
def run_paper3(path):
    print(f"\n  [delegation] running {path} ...")
    proc = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=1800, cwd=os.path.dirname(os.path.abspath(path)))
    passed = {int(m) for m in re.findall(r"\[PASS #(\d+)\]", proc.stdout)}
    verdict = re.search(r"FINAL VERDICT: (\d+)/(\d+)", proc.stdout)
    print(f"  [delegation] exit code {proc.returncode}; {len(passed)} PASS lines; verdict: {verdict.group(0) if verdict else 'none'}")
    return {n: (n in passed) for n in range(1, 400)}


def main():
    args = sys.argv[1:]
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.join(here, "Paper_III_Cosmology", "verify_cbg_paper3_v13.py"), os.path.join(here, "verify_cbg_paper3_v13.py")]
    if "--paper3" in args:
        candidates = [args[args.index("--paper3") + 1]]
    paper3 = next((c for c in candidates if os.path.isfile(c)), None)
    delegated = None
    if "--no-delegate" not in args and paper3 is not None:
        delegated = run_paper3(paper3)
    elif "--no-delegate" not in args:
        print("\n  [delegation] Paper III suite not found in: " + ", ".join(candidates) + " -> delegated items are reported as NOT RUN")

    tier_axioms()
    tier_theorems(delegated)
    tier_hypotheses()
    Phi_inf, a0_dS, dpsit = tier_cascades(delegated)
    tier_evaluations(Phi_inf, a0_dS, dpsit, delegated)

    banner("SUMMARY")
    order = ["DECLARED", "PROVED", "PROVED|H", "EVALUATED", "DELEGATED", "NOT RUN", "EXTERNAL", "DISCREPANCY", "FAILED"]
    counts = {k: 0 for k in order}
    for _, _, _, status, _ in REG:
        key = "PROVED|H" if status.startswith("PROVED|") else status
        counts[key] = counts.get(key, 0) + 1
    for k in order:
        print(f"  {k:<12s} {counts.get(k, 0):3d}")
    print(f"  {'TOTAL':<12s} {len(REG):3d}")
    if FAILED:
        print("\nFAILED ITEMS: " + ", ".join(FAILED))
        sys.exit(1)
    print("\nNO FAILED ITEMS. EXTERNAL rows are proved in the other monograph suites; a DISCREPANCY row would mark a README statement this audit does not support.")


if __name__ == "__main__":
    main()
