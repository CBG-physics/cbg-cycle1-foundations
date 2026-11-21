#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================================
 CBG PRINCIPIA GEOMETRICA — COLAB CELL 1: CORE ENGINE + MODULE 1 (THM #01 .. THM #07)
========================================================================================
 Self-contained Colab cell:
   1. Defines Section 0 (Core Deductive DAG `DeductiveState`, normalizers & `Geometry`).
   2. Defines & executes `run_module_1(master_state)` ([THM #01]..[THM #07], [COR #001]..[COR #040]).
   3. Saves `master_state` and `_COLAB_CHECKPOINT_M1` in global notebook memory so Cell 2
      (Module 2) can resume instantaneously without re-running Module 1.
========================================================================================
"""

import itertools
import time
import copy
import sympy as sp
from mpmath import mp


# ========================================================================================
# SECTION 0: CORE DEDUCTIVE DAG STATE, SYMBOLIC NORMALIZERS & PSEUDO-RIEMANNIAN GEOMETRY
# ========================================================================================
def banner(title: str) -> None:
    print("\n" + "=" * 88)
    print(f" {title}")
    print("=" * 88)


def _normalize_exp_powers_global(expr):
    """Rewrites integer-multiple exponentials exp(k*x) -> exp(x)**k for exact rational algebra."""
    e = sp.expand_power_exp(sp.powdenest(sp.sympify(expr), force=True))
    rep = {}
    for atom in list(e.atoms(sp.exp)):
        arg = sp.expand(atom.args[0])
        c_num, rest = arg.as_coeff_Mul()
        if c_num.is_Rational and c_num.q == 1 and c_num != 1 and rest != 1:
            rep[atom] = sp.exp(rest) ** int(c_num.p)
    return e.xreplace(rep) if rep else e


def exact_rat(expr):
    """
    Canonical rational normalizer over Q:
      1. Normalizes integer exponential powers.
      2. Fast-path zero check via expanded syntactic numerator (O(monomials), zero GCD hang).
      3. Exact polynomial factor cancellation via sp.cancel when Subs-free and non-zero.
    """
    if isinstance(expr, sp.MatrixBase):
        return expr.applyfunc(exact_rat)
    e = _normalize_exp_powers_global(sp.sympify(expr).doit())
    if e == 0:
        return sp.Integer(0)
    tog = sp.together(sp.expand_mul(e))
    num, den = sp.fraction(tog)
    num_exp = _normalize_exp_powers_global(sp.expand(num))
    if num_exp == 0:
        return sp.Integer(0)
    tog2 = sp.together(num_exp / den)
    num2, den2 = sp.fraction(tog2)
    num2_exp = sp.expand(num2)
    if num2_exp == 0:
        return sp.Integer(0)
    res = num2_exp / den2
    if not res.atoms(sp.Subs):
        return sp.cancel(res)
    return res


def _is_equal_sym(lhs, rhs) -> bool:
    """Strict exact symbolic equality verifier (zero floating-point tolerances)."""
    if isinstance(lhs, (tuple, list)) and isinstance(rhs, (tuple, list)):
        return len(lhs) == len(rhs) and all(_is_equal_sym(a, b) for a, b in zip(lhs, rhs))
    if isinstance(lhs, bool) or isinstance(rhs, bool):
        return bool(lhs) is bool(rhs)
    if isinstance(lhs, sp.MatrixBase) and isinstance(rhs, sp.MatrixBase):
        if lhs.shape != rhs.shape:
            return False
        diff_m = lhs - rhs
        return all(_is_equal_sym(diff_m[i, j], 0) for i in range(lhs.rows) for j in range(lhs.cols))
    diff = sp.sympify(lhs) - sp.sympify(rhs)
    if diff == 0:
        return True
    r = exact_rat(diff)
    if r == 0:
        return True
    return sp.simplify(r) == 0


def pfaffian_exact(M: sp.Matrix) -> sp.Expr:
    """Exact recursive Pfaffian over any commutative ring for 2n x 2n skew-symmetric matrices."""
    n = M.rows
    if n == 0:
        return sp.Integer(1)
    if n % 2 == 1:
        return sp.Integer(0)
    if n == 2:
        return M[0, 1]
    res = sp.Integer(0)
    for j in range(1, n):
        if M[0, j] != 0:
            idx = [k for k in range(1, n) if k != j]
            sub = M.extract(idx, idx)
            res += ((-1) ** (j + 1)) * M[0, j] * pfaffian_exact(sub)
    return sp.expand(res)


class DeductiveState:
    """Strict Acyclic Deductive Graph container enforcing constructive proof dependencies."""

    def __init__(self):
        self.axioms = {}
        self.theorems = {}
        self.corollaries = {}
        self.thm_count = 0
        self.cor_count = 0

    def inject_axiom(self, name: str, payload: dict) -> None:
        if name in self.axioms:
            raise RuntimeError(f"[AXIOM DUPLICATION] Axiom `{name}` already injected.")
        self.axioms[name] = payload
        print(f"  [AXIOM INJECTED] {name}: {payload}")

    def deduce_theorem(self, key: str, value, description: str, check: bool = True) -> None:
        if not bool(check):
            raise AssertionError(f"[THEOREM DEDUCTION FAILED] {key}: {description}")
        if key in self.theorems:
            raise RuntimeError(f"[DAG CYCLE DETECTED] Theorem `{key}` already registered.")
        self.thm_count += 1
        self.theorems[key] = value
        preview = str(value)
        if len(preview) > 85:
            preview = preview[:82] + "..."
        print(f"  [THM #{self.thm_count:02d}] {description}")
        print(f"            ==> `{key}` = {preview}")

    def deduce_corollary(self, parent_thm: str, subkey: str, lhs, rhs, description: str) -> None:
        if parent_thm not in self.theorems:
            raise KeyError(f"[DAG VIOLATION] Corollary `{subkey}` references unknown theorem `{parent_thm}`")
        if not _is_equal_sym(lhs, rhs):
            raise AssertionError(f"[COROLLARY FAILED] ({parent_thm} -> {subkey}) {description}: {lhs} != {rhs}")
        self.cor_count += 1
        self.corollaries[(parent_thm, subkey)] = lhs
        print(f"    [COR #{self.cor_count:03d} <- {parent_thm}] {description}  [Q.E.D.]")


class Geometry:
    """Exact symbolic Koszul / Levi-Civita differential geometry engine on arbitrary metrics."""

    def __init__(self, g: sp.Matrix, coords: list, sqrt_abs_det_g: sp.Expr, compute_curvature: bool = True):
        self.g = sp.Matrix(g)
        self.x = list(coords)
        self.n = len(coords)
        self.sqrtg = sp.sympify(sqrt_abs_det_g)
        self.ginv = self.g.inv().applyfunc(exact_rat)

        # Koszul formula for Levi-Civita connection Gamma^c_{ab}:
        self.Gamma = [[[sp.Integer(0) for _ in range(self.n)] for _ in range(self.n)] for _ in range(self.n)]
        dg = [[[sp.diff(self.g[a, b], self.x[c]) for c in range(self.n)] for b in range(self.n)] for a in range(self.n)]
        for c in range(self.n):
            for a in range(self.n):
                for b in range(a, self.n):
                    val = sp.Integer(0)
                    for d in range(self.n):
                        if self.ginv[c, d] != 0:
                            term = dg[d][b][a] + dg[a][d][b] - dg[a][b][d]
                            if term != 0:
                                val += sp.Rational(1, 2) * self.ginv[c, d] * term
                    val = exact_rat(val)
                    self.Gamma[c][a][b] = val
                    self.Gamma[c][b][a] = val

        self.Riem = None
        self.Ric = None
        self.R = None
        self.Ein = None
        if compute_curvature:
            self._build_curvature()

    def _build_curvature(self) -> None:
        n, x, G = self.n, self.x, self.Gamma
        R_up = [[[[sp.Integer(0) for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    for d in range(c + 1, n):
                        val = sp.diff(G[a][d][b], x[c]) - sp.diff(G[a][c][b], x[d])
                        for e in range(n):
                            if G[a][c][e] != 0 and G[e][d][b] != 0:
                                val += G[a][c][e] * G[e][d][b]
                            if G[a][d][e] != 0 and G[e][c][b] != 0:
                                val -= G[a][d][e] * G[e][c][b]
                        val = exact_rat(val)
                        R_up[a][b][c][d] = val
                        R_up[a][b][d][c] = -val
        self.Riem_up = R_up

        self.Riem = [[[[sp.Integer(0) for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(n):
                    for d in range(c + 1, n):
                        val = sp.Integer(0)
                        for e in range(n):
                            if self.g[a, e] != 0 and R_up[e][b][c][d] != 0:
                                val += self.g[a, e] * R_up[e][b][c][d]
                        val = exact_rat(val)
                        self.Riem[a][b][c][d] = val
                        self.Riem[b][a][c][d] = -val
                        self.Riem[a][b][d][c] = -val
                        self.Riem[b][a][d][c] = val

        self.Ric = sp.zeros(n, n)
        for b in range(n):
            for d in range(b, n):
                val = sum(R_up[a][b][a][d] for a in range(n))
                val = exact_rat(val)
                self.Ric[b, d] = val
                self.Ric[d, b] = val

        self.R = exact_rat(sum(self.ginv[a, b] * self.Ric[a, b] for a in range(n) for b in range(n)))
        self.Ein = sp.Matrix(n, n, lambda a, b: exact_rat(self.Ric[a, b] - sp.Rational(1, 2) * self.g[a, b] * self.R))

    def mixed(self, T_cov: sp.Matrix) -> sp.Matrix:
        return (self.ginv * T_cov).applyfunc(exact_rat)

    def grad_sq(self, f: sp.Expr) -> sp.Expr:
        df = [sp.diff(f, c) for c in self.x]
        return exact_rat(sum(self.ginv[a, b] * df[a] * df[b] for a in range(self.n) for b in range(self.n)))

    def box(self, f: sp.Expr) -> sp.Expr:
        df = [sp.diff(f, c) for c in self.x]
        flux = [self.sqrtg * sum(self.ginv[a, b] * df[b] for b in range(self.n)) for a in range(self.n)]
        return exact_rat(sum(sp.diff(flux[a], self.x[a]) for a in range(self.n)) / self.sqrtg)

    def hessian_cov(self, f: sp.Expr) -> sp.Matrix:
        df = [sp.diff(f, c) for c in self.x]
        H = sp.zeros(self.n, self.n)
        for a in range(self.n):
            for b in range(a, self.n):
                val = sp.diff(df[b], self.x[a]) - sum(self.Gamma[c][a][b] * df[c] for c in range(self.n))
                val = exact_rat(val)
                H[a, b] = val
                H[b, a] = val
        return H

    def cov_div_mixed(self, T_mix: sp.Matrix) -> sp.Matrix:
        out = []
        for b in range(self.n):
            val = sum(sp.diff(T_mix[a, b], self.x[a]) for a in range(self.n))
            for a in range(self.n):
                for c in range(self.n):
                    if self.Gamma[a][a][c] != 0 and T_mix[c, b] != 0:
                        val += self.Gamma[a][a][c] * T_mix[c, b]
                    if self.Gamma[c][a][b] != 0 and T_mix[a, c] != 0:
                        val -= self.Gamma[c][a][b] * T_mix[a, c]
            out.append(exact_rat(val))
        return sp.Matrix(out)


# ========================================================================================
# MODULE 1 (FROM AXIOM 1): POINTWISE ALGEBRA, DEWITT SUPERMETRIC, KLEIN QUADRIC & RKHS
# ========================================================================================
def run_module_1(state: DeductiveState) -> None:
    banner("MODULE 1 (FROM AXIOM 1): CANONICAL ALGEBRA, DEWITT SUPERMETRIC, Cl(3,2) & RKHS")

    state.inject_axiom("AXIOM_1", {
        "signature": (-1, -1, 1, 1, 1),
        "fiber_index": 1,
        "orbifold_parity": -1,
    })
    sig = state.axioms["AXIOM_1"]["signature"]
    f_idx = state.axioms["AXIOM_1"]["fiber_index"]
    eta5 = sp.diag(*sig)

    # =========================================================================
    # THM 1: Dimension-Count Equality d_t d_s D == D(D+1) & Spatial Hodge Self-Duality (consistency with AXIOM_1)
    # =========================================================================
    print("  [LOG] Module 1 -> Computing THM 1 (Dimension-Count Equality & Spatial Hodge Self-Duality)...")
    dt_s, ds_s = sp.symbols('d_t d_s', complex=True)
    D_s = dt_s + ds_s
    ind_extr = sp.factor(dt_s * ds_s * D_s - D_s * (D_s + 1))
    expected_factored_ind = D_s * ((dt_s - 1) * (ds_s - 1) - 2)
    hodge_lie_eq = sp.Rational(1, 2) * ds_s * (ds_s - 1) - ds_s

    roots_ds = [r for r in sp.solve(sp.Eq(hodge_lie_eq, 0), ds_s) if r != 0]
    ds_ded = int(roots_ds[0])
    roots_dt = [r for r in sp.solve(sp.Eq(ind_extr.subs(ds_s, ds_ded), 0), dt_s) if (r + ds_ded) != 0]
    dt_ded = int(roots_dt[0])
    D_ded = dt_ded + ds_ded

    dt_ax = sum(1 for s in sig if s < 0)
    ds_ax = sum(1 for s in sig if s > 0)

    state.deduce_theorem(
        "dimensions",
        (dt_ded, ds_ded, D_ded),
        "Unique positive integer root (d_t, d_s, D) = (2, 3, 5) of d_t d_s D == D(D+1) & dim Lambda^2(R^{d_s}) == d_s (consistent with AXIOM_1)",
        (dt_ded, ds_ded, D_ded) == (2, 3, 5) and (dt_ax, ds_ax) == (dt_ded, ds_ded)
    )
    state.deduce_corollary("dimensions", "index_factorization", sp.expand(ind_extr - expected_factored_ind), 0,
                           "Exact polynomial factorization d_t d_s D - D(D+1) == D[(d_t - 1)(d_s - 1) - 2]")
    state.deduce_corollary("dimensions", "spatial_hodge_self_duality", roots_ds, [3],
                           "Unique non-zero spatial Hodge-Lie root dim Lambda^2(R^{d_s}) == d_s <=> d_s == 3")
    state.deduce_corollary("dimensions", "dim_4d_extrinsic_deficit", ind_extr.subs({dt_s: 1, ds_s: 3}), -8,
                           "4D (1,3,4) dimension-count difference d_t d_s D - D(D+1) equals -8")
    state.deduce_corollary("dimensions", "dim_5d_bijection", ind_extr.subs({dt_s: 2, ds_s: 3}), 0,
                           "5D (2,3,5) dimension-count difference vanishes (30 == 30)")
    state.deduce_corollary("dimensions", "dim_6d_surplus", ind_extr.subs({dt_s: 3, ds_s: 3}), 12,
                           "6D (3,3,6) dimension-count difference equals +12")

    # =========================================================================
    # THM 2: Orthogonal Tangent Projectors, Homogeneous Adjoint Grading on End(TM)
    #        & Euclidean Director-Cosine Quadrature (ZERO heterogeneous trace mixing!)
    # =========================================================================
    print("  [LOG] Module 1 -> Computing THM 2 (Homogeneous Projectors, Adjoint End(TM) Grading & Quadrature)...")
    P_T2 = sp.diag(*[1 if s < 0 else 0 for s in sig])
    P_S3 = sp.diag(*[1 if s > 0 else 0 for s in sig])
    F_inv = P_S3 - P_T2

    Ad_F_25 = sp.kronecker_product(F_inv, F_inv)
    P_End_even = sp.Rational(1, 2) * (sp.eye(D_ded**2) + Ad_F_25)
    P_End_odd = sp.Rational(1, 2) * (sp.eye(D_ded**2) - Ad_F_25)
    dim_End_even = int(sp.trace(P_End_even))
    dim_End_odd = int(sp.trace(P_End_odd))

    w_T2 = sp.Rational(int(sp.trace(P_T2)), D_ded)
    w_S3 = sp.Rational(int(sp.trace(P_S3)), D_ded)

    a0_s, a1_s, x_s, abar_s = sp.symbols('a_0 a_1 x_r a_bar', positive=True)
    mu_dir = sp.simplify((a1_s / sp.sqrt(a0_s**2 + a1_s**2)).subs(a1_s, x_s * a0_s))
    eps2_slope = sp.simplify(x_s * sp.diff(sp.log(mu_dir), x_s))
    u_var = sp.Symbol('u_var', positive=True)
    quad_poly = sp.Eq(u_var**2 - abar_s**2 * (a0_s**2 + u_var), 0)
    pos_roots_u = [r for r in sp.solve(quad_poly, u_var) if r.subs({a0_s: 1, abar_s: 1}) > 0]
    u_pos_root = pos_roots_u[0]
    sqrt_denom_exact = sp.Rational(1, 2) * (abar_s + sp.sqrt(4 * a0_s**2 + abar_s**2))
    denest_sq_diff = sp.expand(sqrt_denom_exact**2 - (a0_s**2 + u_pos_root))
    quad_check = exact_rat(u_pos_root / sqrt_denom_exact - abar_s)

    state.deduce_theorem(
        "projector_invariants",
        {"P_T2": P_T2, "P_S3": P_S3, "F_inv": F_inv, "w_T2": w_T2, "w_S3": w_S3,
         "dim_End_even": dim_End_even, "dim_End_odd": dim_End_odd, "eps2_slope": eps2_slope},
        "Orthogonal tangent projectors, homogeneous End(TM) = 13_(+) (+) 12_(-) grading & director quadrature",
        dim_End_even == 13 and dim_End_odd == 12 and denest_sq_diff == 0 and quad_check == 0
    )
    state.deduce_corollary("projector_invariants", "completeness", P_T2 + P_S3, sp.eye(5),
                           "Tangent projector completeness P_{T2} + P_{Sigma3} == I_5")
    state.deduce_corollary("projector_invariants", "idempotence_T2", P_T2**2, P_T2,
                           "Temporal projector idempotence P_{T2}^2 == P_{T2}")
    state.deduce_corollary("projector_invariants", "idempotence_S3", P_S3**2, P_S3,
                           "Spatial projector idempotence P_{Sigma3}^2 == P_{Sigma3}")
    state.deduce_corollary("projector_invariants", "orthogonality", P_T2 * P_S3, sp.zeros(5, 5),
                           "Projector orthogonality P_{T2} P_{Sigma3} == 0_5")
    state.deduce_corollary("projector_invariants", "involution_F", F_inv**2, sp.eye(5),
                           "Almost-product structure F^2 == (P_{Sigma3} - P_{T2})^2 == I_5")
    state.deduce_corollary("projector_invariants", "trace_F", sp.trace(F_inv), ds_ded - dt_ded,
                           "Almost-product trace Tr(F) == d_s - d_t == +1")
    state.deduce_corollary("projector_invariants", "homogeneous_end_idempotence",
                           (P_End_even**2 - P_End_even, P_End_odd**2 - P_End_odd, P_End_even * P_End_odd),
                           (sp.zeros(25, 25), sp.zeros(25, 25), sp.zeros(25, 25)),
                           "Homogeneous adjoint projectors on End(V_5) satisfy P_+^2==P_+, P_-^2==P_-, P_+ P_-==0_25")
    state.deduce_corollary("projector_invariants", "homogeneous_end_dimensions",
                           (dim_End_even, dim_End_odd, dim_End_even + dim_End_odd, int(sp.trace(Ad_F_25))),
                           (dt_ded**2 + ds_ded**2, 2 * dt_ded * ds_ded, D_ded**2, (ds_ded - dt_ded)**2),
                           "Homogeneous End(V_5) partition 25 == 13_(+) (End T_2 (+) End Sigma_3) + 12_(-) (Hom) & Tr(Ad_F)==1")
    state.deduce_corollary("projector_invariants", "homogeneous_vector_weights", (w_T2, w_S3, w_T2 + w_S3),
                           (sp.Rational(2, 5), sp.Rational(3, 5), 1),
                           "Homogeneous vector-bundle trace weights w_{T2}=2/5, w_{Sigma3}=3/5, sum == 1")
    state.deduce_corollary("projector_invariants", "homogeneous_sym2_partition_ratios",
                           (sp.Rational(dt_ded * (dt_ded + 1) // 2, D_ded * (D_ded + 1) // 2),
                            sp.Rational(ds_ded * (ds_ded + 1) // 2, D_ded * (D_ded + 1) // 2),
                            sp.Rational(dt_ded * ds_ded, D_ded * (D_ded + 1) // 2)),
                           (sp.Rational(1, 5), sp.Rational(2, 5), sp.Rational(2, 5)),
                           "Homogeneous Sym^2(V_5^*) block weights: 3/15==1/5 (temporal), 6/15==2/5 (spatial), 6/15==2/5 (mixed)")
    state.deduce_corollary("projector_invariants", "director_cosine", mu_dir, x_s / sp.sqrt(1 + x_s**2),
                           "Euclidean temporal director cosine mu(x) == x / sqrt(1 + x^2)")
    state.deduce_corollary("projector_invariants", "director_log_slope", eps2_slope, 1 / (1 + x_s**2),
                           "Logarithmic director elasticity d ln mu / d ln x == 1 / (1 + x^2)")
    state.deduce_corollary("projector_invariants", "director_quadrature_inversion",
                           (denest_sq_diff, quad_check), (0, 0),
                           "Exact algebraic root inversion of mu(a_1/a_0) a_1 == a_bar")

    # =========================================================================
    # THM 3: Canonical Cotangent Symplectic Form J_30 on T*Sym^2(V_5^*) & explicit frame-dependent linear bijection Phi_30
    # =========================================================================
    print("  [LOG] Module 1 -> Computing THM 3 (Cotangent Darboux Symplectic Form J_30 & frame-dependent linear bijection Phi_30)...")
    N_sym = D_ded * (D_ded + 1) // 2  # 15
    J_30 = sp.zeros(2 * N_sym, 2 * N_sym)
    for k in range(N_sym):
        J_30[2 * k, 2 * k + 1] = -1
        J_30[2 * k + 1, 2 * k] = 1
    pf_J30 = pfaffian_exact(J_30)

    sym_pairs = [(a, b) for a in range(D_ded) for b in range(a, D_ded)]
    pair_to_idx = {p: idx for idx, p in enumerate(sym_pairs)}
    t_pairs = [(0, 0), (0, 1), (1, 1)]

    def _eps3(i, j, k):
        return (i - j) * (j - k) * (k - i) // 2 if len({i, j, k}) == 3 else 0

    Phi_30 = sp.zeros(30, 30)
    col = 0
    for alpha in range(2):
        offset = 15 * alpha
        for i_sp in range(3):
            s_coord = 2 + i_sp
            for A_idx in range(5):
                if A_idx == s_coord:
                    Phi_30[offset + pair_to_idx[(s_coord, s_coord)], col] = 1
                elif A_idx in (2, 3, 4):
                    j_sp = A_idx - 2
                    p_sym = (min(s_coord, A_idx), max(s_coord, A_idx))
                    Phi_30[offset + pair_to_idx[p_sym], col] = sp.Rational(1, 2)
                    for k_sp in range(3):
                        eps_val = _eps3(i_sp, j_sp, k_sp)
                        if eps_val != 0:
                            Phi_30[offset + pair_to_idx[t_pairs[k_sp]], col] = sp.Rational(1, 2) * eps_val
                elif A_idx == alpha:
                    Phi_30[offset + pair_to_idx[(0, s_coord)], col] = 1
                else:
                    Phi_30[offset + pair_to_idx[(1, s_coord)], col] = 1
                col += 1

    det_Phi_30 = Phi_30.det()
    rk_Phi_30 = Phi_30.rank()
    Darboux_dual = (Phi_30.inv()).T
    J_pullback = Phi_30.T * J_30 * Phi_30

    state.deduce_theorem(
        "symplectic_darboux_isomorphism",
        {"J_30": J_30, "Pf_J30": pf_J30, "det_Phi_30": det_Phi_30, "rank_Phi_30": rk_Phi_30},
        "Canonical Darboux form J_30 (Pf=-1, det=+1) & explicit invertible 30x30 frame-dependent linear map Phi_30 (example of the 30 == 30 count)",
        pf_J30 == -1 and sp.Abs(det_Phi_30) == sp.Rational(1, 64) and rk_Phi_30 == 30
    )
    state.deduce_corollary("symplectic_darboux_isomorphism", "complex_symplectic_identities",
                           (J_30.T + J_30, J_30**2 + sp.eye(30), J_30.T * J_30 - sp.eye(30)),
                           (sp.zeros(30, 30), sp.zeros(30, 30), sp.zeros(30, 30)),
                           "Skew-symmetry J_30^T == -J_30, complex structure J_30^2 == -I_30 & orthogonality")
    state.deduce_corollary("symplectic_darboux_isomorphism", "pfaffian_and_trace",
                           (sp.trace(J_30), J_30.det(), pf_J30, sp.trace(J_30.T * J_30)),
                           (0, 1, -1, 30),
                           "Tr(J_30)==0, det(J_30)==+1, Pf(J_30)==-1 & Frobenius norm Tr(J_30^T J_30)==30")
    state.deduce_corollary("symplectic_darboux_isomorphism", "darboux_biorthogonality",
                           (sp.Abs(det_Phi_30), Darboux_dual.T * Phi_30, (Phi_30.inv() * J_30 * Phi_30)**2),
                           (sp.Rational(1, 64), sp.eye(30), -sp.eye(30)),
                           "|det(Phi_30)| == 1/64, primal/dual biorthogonality <E^*_I, E_J> == I_30 & J_pull^2 == -I_30")
    state.deduce_corollary("symplectic_darboux_isomorphism", "pullback_skew_nondegenerate",
                           (J_pullback.T + J_pullback, J_pullback.det() != 0),
                           (sp.zeros(30, 30), True),
                           "Pulled-back symplectic form Phi_30^T J_30 Phi_30 is skew-symmetric and non-degenerate on R^30")

    # =========================================================================
    # THM 4: DeWitt-type trace-inversion operator G_lambda at lambda = 1/2 on Sym^2(R^{3,2}) (inverse trace weight -1/(D-2)),
    #        Molien-Gorenstein Ring & Painleve-Gullstrand Schur Complement
    # Trace-inversion operator G_lambda = 1/2(g^{AC}g^{BD} + g^{AD}g^{BC}) - lambda g^{AB}g^{CD}
    # =========================================================================
    print("  [LOG] Module 1 -> Computing THM 4 (15x15 trace-inversion operator G_lambda at lambda = 1/2, inverse trace weight, Molien & Schur)...")
    R_Z2 = sp.diag(*[-1 if idx == f_idx else 1 for idx in range(D_ded)])

    sym_basis_15 = []
    for a, b in sym_pairs:
        M_ab = sp.zeros(D_ded, D_ded)
        if a == b:
            M_ab[a, a] = 1
        else:
            M_ab[a, b] = sp.Rational(1, 2)
            M_ab[b, a] = sp.Rational(1, 2)
        sym_basis_15.append(M_ab)

    dewitt_parameter_lambda = sp.Rational(1, 2)  # Trace-inversion parameter lambda = 1/2

    G_DeWitt_15 = sp.zeros(15, 15)
    Frob_eta_15 = sp.zeros(15, 15)
    for I_idx, H1 in enumerate(sym_basis_15):
        for J_idx, H2 in enumerate(sym_basis_15):
            H1_up = eta5 * H1 * eta5
            tr_12 = sp.trace(H1_up * H2)
            tr_1 = sp.trace(eta5 * H1)
            tr_2 = sp.trace(eta5 * H2)
            Frob_eta_15[I_idx, J_idx] = tr_12
            G_DeWitt_15[I_idx, J_idx] = tr_12 - dewitt_parameter_lambda * tr_1 * tr_2

    G_DeWitt_inv_15 = G_DeWitt_15.inv()
    Frob_inv_15 = Frob_eta_15.inv()
    v_tr_15 = sp.Matrix(15, 1, lambda idx, _: sp.trace(eta5 * sym_basis_15[idx]))
    u_eta_15 = Frob_inv_15 * v_tr_15

    P_DW_inv_15 = G_DeWitt_inv_15 * Frob_eta_15
    rank1_proj_15 = P_DW_inv_15 - sp.eye(15)
    outer_uv_15 = u_eta_15 * v_tr_15.T
    c_DW_inv_ded = sp.simplify(rank1_proj_15[0, 0] / outer_uv_15[0, 0])
    dw_inv_check = rank1_proj_15 - c_DW_inv_ded * outer_uv_15

    t_mol = sp.Symbol('t_mol')
    P_molien_exact = sp.cancel(
        sp.Rational(1, 2) * (1 / (sp.eye(5) - t_mol * sp.eye(5)).det() + 1 / (sp.eye(5) - t_mol * R_Z2).det())
    )
    stanley_ratio = sp.cancel(P_molien_exact.subs(t_mol, 1 / t_mol) / P_molien_exact)

    odd_indices_15 = [idx for idx, (a, b) in enumerate(sym_pairs) if (a == f_idx) ^ (b == f_idx)]
    even_indices_15 = [idx for idx, (a, b) in enumerate(sym_pairs) if idx not in odd_indices_15]
    G_DW_odd_4x4 = G_DeWitt_15.extract(odd_indices_15, odd_indices_15)
    odd_eigs = sorted([sp.Rational(ev) for ev, mult in G_DW_odd_4x4.eigenvals().items() for _ in range(mult)])

    Phi_s = sp.Symbol('Phi', positive=True)
    g4_sym = sp.Matrix(4, 4, lambda i, j: sp.Symbol(f'g4_{min(i,j)}{max(i,j)}', real=True))
    macro_slots = [0, 2, 3, 4]
    g5_proj = sp.zeros(5, 5)
    g5_proj[f_idx, f_idx] = -Phi_s**2
    for m_a, idx_a in enumerate(macro_slots):
        for m_b, idx_b in enumerate(macro_slots):
            g5_proj[idx_a, idx_b] = g4_sym[m_a, m_b]
    det_fact_diff = sp.expand(g5_proj.det() + Phi_s**2 * g4_sym.det())

    v1, v2, v3 = sp.symbols('v1 v2 v3', real=True)
    gam_3 = sp.diag(sp.Symbol('gam1', positive=True), sp.Symbol('gam2', positive=True), sp.Symbol('gam3', positive=True))
    v_vec = sp.Matrix([v1, v2, v3])
    v_sq = (v_vec.T * gam_3.inv() * v_vec)[0, 0]
    g5_pg = sp.zeros(5, 5)
    g5_pg[0, 0] = -(1 - v_sq)
    g5_pg[1, 1] = -Phi_s**2
    for j_i in range(3):
        g5_pg[0, 2 + j_i] = v_vec[j_i]
        g5_pg[2 + j_i, 0] = v_vec[j_i]
        g5_pg[2 + j_i, 2 + j_i] = gam_3[j_i, j_i]
    pg_schur_diff = exact_rat(g5_pg.det() - Phi_s**2 * gam_3.det())

    state.deduce_theorem(
        "molien_dewitt_sym2",
        {"R_Z2": R_Z2, "c_DW_inv_trace": c_DW_inv_ded, "det_G_DeWitt_15": G_DeWitt_15.det(),
         "P_molien": P_molien_exact, "even_dim": len(even_indices_15), "odd_dim": len(odd_indices_15)},
        "15x15 trace-inversion operator G_lambda at lambda = 1/2 (inverse trace weight lambda/(1 - lambda D) = -1/(D-2) = -1/3), Molien series & PG Schur",
        c_DW_inv_ded == dewitt_parameter_lambda / (1 - dewitt_parameter_lambda * D_ded)
        and c_DW_inv_ded == -sp.Rational(1, D_ded - 2) and c_DW_inv_ded == -sp.Rational(1, 3)
        and dw_inv_check == sp.zeros(15, 15) and len(even_indices_15) == 11 and pg_schur_diff == 0
    )
    state.deduce_corollary("molien_dewitt_sym2", "sym2_sector_partition",
                           (len(even_indices_15) - 1, len(odd_indices_15), 1, len(sym_pairs)),
                           (10, 4, 1, 15),
                           "Disjoint Sym^2(T*_p M) partition: 10 (g_{mu nu}) + 4 (Z_2-odd g_{mu 2}) + 1 (g_{22}) == 15")
    state.deduce_corollary("molien_dewitt_sym2", "dewitt_inverse_trace_minus_one_third",
                           (c_DW_inv_ded, dw_inv_check, G_DeWitt_inv_15 * G_DeWitt_15),
                           (-sp.Rational(1, 3), sp.zeros(15, 15), sp.eye(15)),
                           "Inverse of G_lambda at lambda = 1/2 has trace weight lambda/(1 - lambda D) == -1/(D-2) == -1/3")
    state.deduce_corollary("molien_dewitt_sym2", "molien_gorenstein_self_duality",
                           (sp.cancel(P_molien_exact - 1 / ((1 - t_mol)**4 * (1 - t_mol**2))), stanley_ratio),
                           (0, -t_mol**6),
                           "Exact Molien series P_even(t) == 1/[(1-t)^4(1-t^2)] & Gorenstein self-duality ratio -t^6")
    state.deduce_corollary("molien_dewitt_sym2", "odd_sector_lorentzian_signature",
                           (len([e for e in odd_eigs if e > 0]), len([e for e in odd_eigs if e < 0])),
                           (1, 3),
                           "DeWitt pairing restricted to Z_2-odd graviphoton sector carries 4D Lorentzian signature (+,-,-,-)")
    state.deduce_corollary("molien_dewitt_sym2", "orbifold_reflection_operator",
                           (R_Z2**2, R_Z2.det(), sp.trace(R_Z2)), (sp.eye(5), -1, 3),
                           "Orbifold reflection operator R_{Z_2}^2 == I_5, det(R_{Z_2}) == -1, Tr(R_{Z_2}) == +3")
    state.deduce_corollary("molien_dewitt_sym2", "determinant_and_fiber_inverse",
                           (det_fact_diff, (sp.diag(-1, -Phi_s**2, 1, 1, 1).inv())[1, 1]),
                           (0, -1 / Phi_s**2),
                           "Exact determinant factorization det(g_proj) == -Phi^2 det(g_4) & g^{22}_{proj} == -Phi^{-2}")
    state.deduce_corollary("molien_dewitt_sym2", "painleve_gullstrand_schur", pg_schur_diff, 0,
                           "5D Painleve-Gullstrand Schur complement identity det(g_proj^(PG)) == +Phi^2 det(gamma_ij)")

    # =========================================================================
    # THM 5: Constructive Discovery of Real Cl(3,2) = Mat(4, R), Sp(4, R) & Connes KO-dim = 1
    # =========================================================================
    print("  [LOG] Module 1 -> Computing THM 5 (Constructive Discovery of Real Cl(3,2), Sp(4,R) & KO-dim=1)...")
    I2 = sp.eye(2)
    eps2 = sp.Matrix([[0, -1], [1, 0]])
    sig1 = sp.Matrix([[0, 1], [1, 0]])
    sig3 = sp.Matrix([[1, 0], [0, -1]])
    pauli_real_pool = []
    for A_2 in [I2, eps2, sig1, sig3]:
        for B_2 in [I2, eps2, sig1, sig3]:
            M4 = sp.kronecker_product(A_2, B_2)
            if M4 != sp.eye(4):
                pauli_real_pool.append(M4)

    neg_pool = [M for M in pauli_real_pool if M**2 == -sp.eye(4)]
    pos_pool = [M for M in pauli_real_pool if M**2 == sp.eye(4) and sp.trace(M) == 0]

    Gamma_hat = None
    for g0, g1 in itertools.combinations(neg_pool, 2):
        if g0 * g1 + g1 * g0 != sp.zeros(4, 4):
            continue
        compat_pos = [M for M in pos_pool if M * g0 + g0 * M == sp.zeros(4, 4) and M * g1 + g1 * M == sp.zeros(4, 4)]
        for g2, g3, g4 in itertools.combinations(compat_pos, 3):
            if (g2 * g3 + g3 * g2 == sp.zeros(4, 4)
                    and g2 * g4 + g4 * g2 == sp.zeros(4, 4)
                    and g3 * g4 + g4 * g3 == sp.zeros(4, 4)):
                cand_vol = g0 * g1 * g2 * g3 * g4
                if cand_vol == sp.eye(4):
                    Gamma_hat = [g0, g1, g2, g3, g4]
                    break
            if Gamma_hat is not None:
                break
        if Gamma_hat is not None:
            break

    assert Gamma_hat is not None, "Real Cl(3,2) representation must exist in Mat(4, R)"
    Gamma_vol = Gamma_hat[0] * Gamma_hat[1] * Gamma_hat[2] * Gamma_hat[3] * Gamma_hat[4]

    Sigma_gen = []
    for a in range(5):
        for b in range(a + 1, 5):
            Sigma_gen.append(sp.Rational(1, 4) * (Gamma_hat[a] * Gamma_hat[b] - Gamma_hat[b] * Gamma_hat[a]))

    Omega_4 = None
    for cand in neg_pool:
        if cand.T == -cand and all(S.T * cand + cand * S == sp.zeros(4, 4) for S in Sigma_gen):
            Omega_4 = cand
            break

    ko_dim_mod8 = (ds_ded - dt_ded) % 8
    spatial_slots_cliff = [idx for idx, s_val in enumerate(sig) if s_val > 0]
    M_gen_cliff = [sp.Rational(1, 2) * Gamma_hat[f_idx] * Gamma_hat[idx] for idx in spatial_slots_cliff]
    M_gen_props_ok = all((2 * M_g)**2 == sp.eye(4) and sp.trace(M_g) == 0 for M_g in M_gen_cliff)
    Gram_M_gen = sp.Matrix(len(M_gen_cliff), len(M_gen_cliff),
                           lambda i, j: sp.Rational(1, 4) * sp.trace((2 * M_gen_cliff[i]).T * (2 * M_gen_cliff[j])))
    N_gen_cliff = Gram_M_gen.rank()

    cliff_checks = [
        Gamma_hat[a] * Gamma_hat[b] + Gamma_hat[b] * Gamma_hat[a] - 2 * eta5[a, b] * sp.eye(4)
        for a in range(5) for b in range(5)
    ]

    state.deduce_theorem(
        "clifford_sp4_spinor",
        {"Gamma_hat": Gamma_hat, "Omega_4": Omega_4, "KO_dim": ko_dim_mod8, "N_gen": N_gen_cliff},
        "Constructive real Cl(3,2) in Mat(4,R), Spin_0(3,2) ~ Sp(4,R) & KO-dimension d_s - d_t == 1 (mod 8)",
        all(C == sp.zeros(4, 4) for C in cliff_checks) and Omega_4 is not None
        and ko_dim_mod8 == 1 and M_gen_props_ok and Gram_M_gen == sp.eye(3) and N_gen_cliff == 3
    )
    state.deduce_corollary("clifford_sp4_spinor", "clifford_anticommutators_and_volume",
                           (all(C == sp.zeros(4, 4) for C in cliff_checks), Gamma_vol,
                            [Gamma_vol * G - G * Gamma_vol for G in Gamma_hat]),
                           (True, sp.eye(4), [sp.zeros(4, 4)] * 5),
                           "All 25 real Cl(3,2) anticommutators, Gamma_vol == +I_4 & odd-D centrality verified")
    state.deduce_corollary("clifford_sp4_spinor", "symplectic_spinor_lie_algebra",
                           (Omega_4.T + Omega_4, Omega_4**2 + sp.eye(4), Omega_4.det(), len(Sigma_gen)),
                           (sp.zeros(4, 4), sp.zeros(4, 4), 1, 10),
                           "Omega_4^T == -Omega_4, Omega_4^2 == -I_4, det(Omega_4) == +1 & all 10 Sigma^{ab} in sp(4,R)")
    state.deduce_corollary("clifford_sp4_spinor", "ko_dimension_one",
                           (ko_dim_mod8, all(G.conjugate() == G for G in Gamma_hat)), (1, True),
                           "Real Majorana representation (Gamma^* == Gamma) in signature (3,2) establishes KO-dim == 1 (mod 8)")
    state.deduce_corollary("clifford_sp4_spinor", "fiber_lorentz_chirality",
                           ([Gamma_hat[1] * Gamma_hat[m] + Gamma_hat[m] * Gamma_hat[1] for m in [0, 2, 3, 4]],
                            M_gen_props_ok, Gram_M_gen, N_gen_cliff),
                           ([sp.zeros(4, 4)] * 4, True, sp.eye(3), 3),
                           "Fiber Gamma^2 anticommutes with all Gamma^mu & Kahler-Dirac {M_g} has Frobenius Gram == I_3 (rank 3 [D])")

    # =========================================================================
    # THM 6: 6D Klein Pfaffian Quadric on Lambda^2(R^4) -> Emergence of R^{3,2} on Lambda^2_0(S)
    #        & Odd-D Dual Grassmannian Pfaffian Kernel
    # =========================================================================
    print("  [LOG] Module 1 -> Computing THM 6 (6D Klein Pfaffian Quadric -> R^{3,2} on Traceless Sp(4,R) Bivectors)...")
    X0, X1, X2, X3, X4, X5 = sp.symbols('X0 X1 X2 X3 X4 X5', real=True)
    skew_basis_6 = [Omega_4] + [Omega_4 * Gamma_hat[a] for a in range(5)]

    coords_6 = [X0, X1, X2, X3, X4, X5]
    M_klein_4x4 = sum((c_i * B_i for c_i, B_i in zip(coords_6, skew_basis_6)), sp.zeros(4, 4))
    Pf_Klein_6D = pfaffian_exact(M_klein_4x4)
    pf_sq_diff = sp.expand(Pf_Klein_6D**2 - M_klein_4x4.det())

    Pf_5D_traceless = sp.expand(Pf_Klein_6D.subs(X0, 0))
    coords_5 = [X1, X2, X3, X4, X5]
    H_klein_5x5 = sp.Matrix(5, 5, lambda i, j: sp.Rational(1, 2) * sp.diff(Pf_5D_traceless, coords_5[i], coords_5[j]))
    eigs_5d_klein = sorted([int(ev) for ev, mult in H_klein_5x5.eigenvals().items() for _ in range(mult)])
    n_pos_k = sum(1 for e in eigs_5d_klein if e > 0)
    n_neg_k = sum(1 for e in eigs_5d_klein if e < 0)
    klein_gives_3_2 = sorted([n_pos_k, n_neg_k]) == [2, 3]

    W_5 = sp.zeros(5, 5)
    for i in range(5):
        for j in range(i + 1, 5):
            s_ij = sp.Symbol(f'w_{i}{j}', real=True)
            W_5[i, j] = s_ij
            W_5[j, i] = -s_ij
    pf_vec_5 = sp.Matrix(5, 1, lambda k, _: ((-1) ** k) * pfaffian_exact(
        W_5.extract([i for i in range(5) if i != k], [i for i in range(5) if i != k])
    ))
    grassmann_ker_check = (W_5 * pf_vec_5).applyfunc(sp.expand)

    state.deduce_theorem(
        "klein_plucker_pfaffian_duality",
        {"Pf_Klein_6D": Pf_Klein_6D, "Pf_5D_traceless": Pf_5D_traceless, "eigs_5d": eigs_5d_klein},
        "6D Klein Pfaffian quadric on Lambda^2(R^4), canonical (3,2) signature on Lambda^2_0(R^4) & 5D Pfaffian kernel",
        pf_sq_diff == 0 and klein_gives_3_2 and W_5.det() == 0 and grassmann_ker_check == sp.zeros(5, 1)
    )
    state.deduce_corollary("klein_plucker_pfaffian_duality", "frobenius_orthonormality_and_cayley",
                           (sp.Matrix(6, 6, lambda i, j: sp.Rational(1, 4) * sp.trace(skew_basis_6[i].T * skew_basis_6[j])), pf_sq_diff),
                           (sp.eye(6), 0),
                           "Exact 6x6 Frobenius orthonormality on Lambda^2(R^4) & Cayley identity Pf(M)^2 == det(M)")
    state.deduce_corollary("klein_plucker_pfaffian_duality", "traceless_bivector_signature_3_2",
                           (sp.Matrix(5, 1, lambda k, _: sp.trace(Omega_4.T * skew_basis_6[k + 1])), klein_gives_3_2, abs(H_klein_5x5.det())),
                           (sp.zeros(5, 1), True, 1),
                           "Omega_4-traceless bivector space Lambda^2_0(R^4) has unimodular Klein signature {2, 3} ~ R^{3,2}")
    state.deduce_corollary("klein_plucker_pfaffian_duality", "odd_skew_pfaffian_kernel",
                           (W_5.det(), grassmann_ker_check), (0, sp.zeros(5, 1)),
                           "5x5 skew determinant det(W_5) == 0 & exact Plucker-Pfaffian kernel W_5 * pf_vec(W_5) == 0_5")

    # =========================================================================
    # THM 7: Projective Polar Self-Duality (Q^vee = Q <=> Q = 2/3) & Laplace-Cauchy RKHS
    # =========================================================================
    print("  [LOG] Module 1 -> Computing THM 7 (Projective Polar Self-Duality Q=2/3 & Laplace-Cauchy RKHS Positivity)...")
    J_dem = sp.ones(3, 3)
    P_1 = sp.Rational(1, 3) * J_dem
    P_2 = sp.eye(3) - P_1

    Q_s = sp.Symbol('Q_s', real=True)
    A_Q = P_2 + (1 - 3 * Q_s) * P_1
    A_Q_inv_c1 = 1 / (1 - 3 * Q_s)
    Q_dual_mobius = sp.cancel(sp.solve(sp.Eq(1 - 3 * sp.Symbol('Qv'), A_Q_inv_c1), sp.Symbol('Qv'))[0])
    mobius_involution_check = sp.cancel(Q_dual_mobius.subs(Q_s, Q_dual_mobius) - Q_s)
    self_dual_roots = [r for r in sp.solve(sp.Eq(Q_dual_mobius, Q_s), Q_s) if r != 0]
    Q_self_dual = self_dual_roots[0]
    A_two_thirds = A_Q.subs(Q_s, Q_self_dual)

    U_S3 = sp.Matrix([
        [1 / sp.sqrt(3), 1 / sp.sqrt(2), 1 / sp.sqrt(6)],
        [1 / sp.sqrt(3), -1 / sp.sqrt(2), 1 / sp.sqrt(6)],
        [1 / sp.sqrt(3), 0, -2 / sp.sqrt(6)],
    ])
    U_S3_polar_dual = A_two_thirds * U_S3

    mu1, mu2, mu3, s_lap = sp.symbols('mu1 mu2 mu3 s_lap', positive=True)
    rkhs_pair = sp.simplify(sp.integrate(sp.sqrt(2 * mu1) * sp.exp(-mu1 * s_lap) * sp.sqrt(2 * mu2) * sp.exp(-mu2 * s_lap), (s_lap, 0, sp.oo)))
    mus = [mu1, mu2, mu3]
    C_cauchy_3 = sp.Matrix(3, 3, lambda i, j: 2 * sp.sqrt(mus[i] * mus[j]) / (mus[i] + mus[j]))
    minors_cauchy = [
        C_cauchy_3[0, 0],
        sp.factor(sp.simplify(C_cauchy_3[:2, :2].det())),
        sp.factor(sp.simplify(C_cauchy_3.det())),
    ]
    expected_delta3_vandermonde = (
        (mu1 - mu2)**2 * (mu1 - mu3)**2 * (mu2 - mu3)**2
        / ((mu1 + mu2)**2 * (mu1 + mu3)**2 * (mu2 + mu3)**2)
    )

    state.deduce_theorem(
        "s3_self_dual_conic_and_cauchy",
        {"Q_dual_mobius": Q_dual_mobius, "Q_self_dual": Q_self_dual, "A_two_thirds": A_two_thirds,
         "cauchy_minors": minors_cauchy},
        "Projective polar duality Q^vee = Q/(3Q-1) => unique self-dual conic Q = 2/3 & Laplace-Cauchy RKHS positivity",
        Q_self_dual == sp.Rational(2, 3) and mobius_involution_check == 0 and A_two_thirds**2 == sp.eye(3)
        and exact_rat(minors_cauchy[2] - expected_delta3_vandermonde) == 0
    )
    state.deduce_corollary("s3_self_dual_conic_and_cauchy", "s3_central_idempotents",
                           (P_1**2 - P_1, P_2**2 - P_2, P_1 * P_2, sp.trace(P_1), sp.trace(P_2)),
                           (sp.zeros(3, 3), sp.zeros(3, 3), sp.zeros(3, 3), 1, 2),
                           "S_3 central idempotents P_1^2==P_1, P_2^2==P_2, P_1 P_2==0_3 & traces (1, 2) for 1_0 (+) 2_E")
    state.deduce_corollary("s3_self_dual_conic_and_cauchy", "mobius_polar_involution_and_cone",
                           (mobius_involution_check, Q_self_dual, A_two_thirds**2, A_two_thirds.det(), A_two_thirds - (P_2 - P_1)),
                           (0, sp.Rational(2, 3), sp.eye(3), -1, sp.zeros(3, 3)),
                           "Involution (Q^vee)^vee == Q, unique self-dual Q==2/3, A_{2/3}^2 == I_3 & ||P_2 z||^2 == ||P_1 z||^2")
    state.deduce_corollary("s3_self_dual_conic_and_cauchy", "s3_polar_biorthogonality",
                           (sp.simplify(U_S3.T * U_S3), sp.simplify(U_S3_polar_dual.T * A_two_thirds * U_S3)),
                           (sp.eye(3), sp.eye(3)),
                           "Orthonormality U_{S_3}^T U_{S_3} == I_3 & polar primal/dual biorthogonality U^{*T} A_{2/3} U == I_3")
    state.deduce_corollary("s3_self_dual_conic_and_cauchy", "rkhs_vandermonde_total_positivity",
                           (sp.simplify(rkhs_pair - C_cauchy_3[0, 1]),
                            minors_cauchy[0],
                            exact_rat(minors_cauchy[1] - (mu1 - mu2)**2 / (mu1 + mu2)**2),
                            exact_rat(minors_cauchy[2] - expected_delta3_vandermonde)),
                           (0, 1, 0, 0),
                           "Laplace RKHS integral == 2 sqrt(mu_i mu_j)/(mu_i+mu_j) & exact Cauchy double-alternant Vandermonde minors")


# ========================================================================================
# COLAB CELL 1 EXECUTION & MEMORY CHECKPOINT
# ========================================================================================
if __name__ == "__main__":
    t0_m1 = time.time()
    master_state = DeductiveState()
    run_module_1(master_state)
    globals()["_COLAB_CHECKPOINT_M1"] = copy.deepcopy(master_state)
    print(f"\n  --> [COLAB CELL 1 READY] Module 1 completed in {time.time() - t0_m1:.2f}s "
          f"(Cumulative: {master_state.thm_count} THMs, {master_state.cor_count} CORs saved in `_COLAB_CHECKPOINT_M1`)")


# ========================================================================================
# CBG PRINCIPIA GEOMETRICA — COLAB CELL 2: MODULE 2 (THM #08 .. THM #11)
# ========================================================================================
# Resumes automatically from `_COLAB_CHECKPOINT_M1` in notebook memory and saves
# `_COLAB_CHECKPOINT_M2` upon completion.
# ========================================================================================

# ========================================================================================
# MODULE 2: LIE DERIVATIVES, MARSDEN-WEINSTEIN/BRST, O'NEILL, PG WAVES & CONFORMAL MAXWELL
# ========================================================================================
def _build_kk_mode_operators(sig, f_idx, level_zero: bool = False):
    """
    Constructs the exact linearised 5D Einstein operator E, gauge map P, and Bianchi
    divergence map Dv on M_4 x [0, pi L_0] with Z_2 orbifold parity from AXIOM_1.
    """
    D = len(sig)
    macro = [i for i in range(D) if i != f_idx]
    space = [i for i in macro if sig[i] > 0]
    time1 = [i for i in macro if sig[i] < 0][0]

    X = sp.symbols(f'X0:{D}', real=True)
    y = X[f_idx]
    w, q, Phi0 = sp.symbols('omega q Phi0', positive=True)
    kv = sp.symbols('k1:4', real=True)
    eta = sp.diag(*[(-Phi0**2 if i == f_idx else sp.Integer(s)) for i, s in enumerate(sig)])
    eta_inv = eta.inv()

    phase = sp.exp(sp.I * (sum(kv[j] * X[space[j]] for j in range(3)) - w * X[time1]))
    C, S = (sp.Integer(1), sp.Integer(0)) if level_zero else (sp.cos(q * y), sp.sin(q * y))

    a = {}
    for i, m in enumerate(macro):
        for n in macro[i:]:
            a[(m, n)] = sp.Symbol(f'a_{m}{n}')
    b = {m: sp.Symbol(f'b_{m}') for m in macro}
    c = sp.Symbol('c_yy')
    amps = [a[k_] for k_ in sorted(a)] + ([] if level_zero else [b[m] for m in macro]) + [c]

    h = sp.zeros(D, D)
    for (m, n), s_sym in a.items():
        h[m, n] = h[n, m] = s_sym * C * phase
    if not level_zero:
        for m in macro:
            h[m, f_idx] = h[f_idx, m] = b[m] * S * phase
    h[f_idx, f_idx] = c * C * phase

    dh = [[[sp.diff(h[A, B], X[Cc]) for Cc in range(D)] for B in range(D)] for A in range(D)]
    G1 = [[[sp.Rational(1, 2) * eta_inv[Cc, Cc] * (dh[Cc][B][A] + dh[Cc][A][B] - dh[A][B][Cc])
            for B in range(D)] for A in range(D)] for Cc in range(D)]
    R1 = sp.zeros(D, D)
    for A in range(D):
        for B in range(D):
            R1[A, B] = (
                sum(sp.diff(G1[Cc][A][B], X[Cc]) for Cc in range(D))
                - sp.diff(sum(G1[Cc][A][Cc] for Cc in range(D)), X[B])
            )
    Rs = sum(eta_inv[A, A] * R1[A, A] for A in range(D))
    dG = (R1 - sp.Rational(1, 2) * eta * Rs).applyfunc(sp.expand)

    slots = [(m, n) for (m, n) in sorted(a)] + ([] if level_zero else [(m, f_idx) for m in macro]) + [(f_idx, f_idx)]

    def _strip(expr, odd: bool):
        prof = (S if odd else C) * phase
        val = sp.simplify(expr / prof)
        assert not val.has(y) and not any(val.has(x_) for x_ in X), f"Profile mismatch: {val}"
        return sp.expand(val)

    rows = [_strip(dG[A, B], odd=((A == f_idx) ^ (B == f_idx))) for (A, B) in slots]
    E = sp.Matrix([[sp.diff(r_expr, s_sym) for s_sym in amps] for r_expr in rows])

    e_par = {m: sp.Symbol(f'e_{m}') for m in macro}
    f_par = sp.Symbol('f_y')
    gpar = [e_par[m] for m in macro] + ([] if level_zero else [f_par])
    xi = [sp.Integer(0)] * D
    for m in macro:
        xi[m] = e_par[m] * C * phase
    if not level_zero:
        xi[f_idx] = f_par * S * phase
    dh_g = sp.Matrix(D, D, lambda A, B: sp.diff(xi[B], X[A]) + sp.diff(xi[A], X[B]))
    grow = [_strip(dh_g[A, B], odd=((A == f_idx) ^ (B == f_idx))) for (A, B) in slots]
    P = sp.Matrix([[sp.diff(r_expr, s_sym) for s_sym in gpar] for r_expr in grow])

    T_mat = sp.zeros(D, D)
    tsym = {}
    for (A, B) in slots:
        tsym[(A, B)] = sp.Symbol(f't_{A}{B}')
        odd = (A == f_idx) ^ (B == f_idx)
        T_mat[A, B] = T_mat[B, A] = tsym[(A, B)] * (S if odd else C) * phase
    out_slots = macro + ([] if level_zero else [f_idx])
    drow = [_strip(sum(eta_inv[A, A] * sp.diff(T_mat[A, B], X[A]) for A in range(D)), odd=(B == f_idx)) for B in out_slots]
    Dv = sp.Matrix([[sp.diff(r_expr, tsym[s_slot]) for s_slot in slots] for r_expr in drow])

    p_sq = -w**2 + sum(k_**2 for k_ in kv) - (0 if level_zero else q**2 / Phi0**2)
    info = dict(w=w, q=q, Phi0=Phi0, k=kv, amps=amps, slots=slots, macro=macro, space=space,
                time1=time1, f_idx=f_idx, p_sq=p_sq, n_a=len(a), eta=eta, a=a)
    return E, P, Dv, info


def _analyse_kk_spectrum(sig, f_idx):
    """Deduces exact off-shell/on-shell KK cohomologies and n=0 state counts from AXIOM_1."""
    E, P, Dv, s = _build_kk_mode_operators(sig, f_idx, level_zero=False)
    w, q, Phi0, kv = s['w'], s['q'], s['Phi0'], s['k']
    n_a = s['n_a']

    gauge_inv = (E * P).applyfunc(sp.expand)
    bianchi_id = (Dv * E).applyfunc(sp.expand)
    det_stueck_op = sp.factor(P[n_a:, :].det())

    E_U = E[:, :n_a]
    FP = E_U[:n_a, :]
    u_inv = sp.Symbol('u_inv', positive=True)
    det_poly = FP.subs(Phi0, sp.Integer(1) / sp.sqrt(u_inv)).applyfunc(sp.expand).to_DM().det().as_expr()
    det_FP = sp.factor(det_poly).subs(u_inv, sp.Integer(1) / Phi0**2)

    off_pt = {w: 3, kv[0]: 2, kv[1]: 5, kv[2]: 7, q: 4, Phi0: 2}
    on_real_pt = {w: 28, kv[0]: 10, kv[1]: 15, kv[2]: 30, q: 42, Phi0: 2}
    on_imag_pt = {w: 3 * sp.I, kv[0]: 0, kv[1]: 0, kv[2]: 4, q: 10, Phi0: 2}

    counts = {}
    for name, pt in (('off_shell', off_pt), ('on_shell_real', on_real_pt), ('on_shell_imag', on_imag_pt)):
        En, Pn = E.subs(pt), P.subs(pt)
        dim_ker = 15 - En.rank()
        rk_P = Pn.rank()
        counts[name] = (dim_ker, rk_P, dim_ker - rk_P)
        if name.startswith('on'):
            ker = E_U.subs(pt).nullspace()
            kcov = {s['time1']: -pt[w]}
            kcov.update({s['space'][j]: pt[kv[j]] for j in range(3)})
            eta_pt = s['eta'].subs(pt)
            tt_ok = True
            for v in ker:
                A4 = sp.zeros(5, 5)
                for idx, key in enumerate(sorted(s['a'])):
                    A4[key[0], key[1]] = A4[key[1], key[0]] = v[idx]
                tr_val = sum(A4[m, m] / eta_pt[m, m] for m in s['macro'])
                div_val = [sum(kcov[m] / eta_pt[m, m] * A4[m, n] for m in s['macro']) for n in s['macro']]
                tt_ok = tt_ok and (sp.simplify(tr_val) == 0) and all(sp.simplify(d_) == 0 for d_ in div_val)
            counts[name + '_unitary'] = (len(ker), tt_ok)

    omega_sq_ded = sp.solve(sp.Eq(s['p_sq'], 0), w**2)[0]

    E0, P0, _, s0 = _build_kk_mode_operators(sig, f_idx, level_zero=True)
    pt0 = {s0['w']: 35, s0['k'][0]: 10, s0['k'][1]: 15, s0['k'][2]: 30, s0['Phi0']: 2}
    k0 = 11 - E0.subs(pt0).rank()
    rk_P0 = P0.subs(pt0).rank()
    zero_mode_info = ((E0 * P0).applyfunc(sp.expand), k0, rk_P0, k0 - rk_P0)

    return {
        "gauge_invariance": gauge_inv,
        "bianchi": bianchi_id,
        "det_stueck_op": det_stueck_op,
        "P_stueck_block": P[n_a:, :],
        "det_FP": det_FP,
        "counts": counts,
        "omega_sq": omega_sq_ded,
        "zero_mode": zero_mode_info,
        "symbols": s,
    }


def _derive_fiber_mode_equations(sig, f_idx):
    """
    Scalar fibre-mode equations deduced from the 5D wave operator Delta_5 built on AXIOM_1:
      (a) plane-wave dispersion on the flat background with constant fibre lapse Phi_0;
      (b) indicial (Frobenius) equation of a static n >= 1 mode on the power-law throat Phi = (Phi_inf / r_0) r,
          its Liouville normal form, and the square-integrability exponents of both solutions as r -> 0.
    """
    D = len(sig)
    macro = [i for i in range(D) if i != f_idx]
    space = [i for i in macro if sig[i] > 0]
    time1 = [i for i in macro if sig[i] < 0][0]
    q, Phi0, Phi_inf, r0, w, k = sp.symbols('q Phi0 Phi_inf r0 omega k_w', positive=True)

    # (a) flat background, Cartesian chart: Delta_5 Psi = sum_A g^{AA} d_A^2 Psi
    X = sp.symbols(f'Xw0:{D}', real=True)
    eta_inv = [(sig[i] / Phi0**2 if i == f_idx else sp.Integer(sig[i])) for i in range(D)]
    Psi_pw = sp.exp(sp.I * (k * X[space[0]] - w * X[time1])) * sp.cos(q * X[f_idx])
    disp = sp.simplify(sum(eta_inv[i] * sp.diff(Psi_pw, X[i], 2) for i in range(D)) / Psi_pw)
    omega_sq = sp.solve(sp.Eq(disp, 0), w**2)[0]
    omega_zero = sp.sqrt(omega_sq.subs(q, 0))

    # (b) static spherical chart with Phi(r) = (Phi_inf / r_0) r
    t_m, y_m, ph_m = sp.symbols('t_m y_m ph_m', real=True)
    r_m, th_m = sp.symbols('r_m th_m', positive=True)
    Phi_r = Phi_inf * r_m / r0
    coords, g_diag = [None] * D, [None] * D
    coords[time1], g_diag[time1] = t_m, sp.Integer(sig[time1])
    coords[f_idx], g_diag[f_idx] = y_m, sig[f_idx] * Phi_r**2
    coords[space[0]], g_diag[space[0]] = r_m, sp.Integer(1)
    coords[space[1]], g_diag[space[1]] = th_m, r_m**2
    coords[space[2]], g_diag[space[2]] = ph_m, r_m**2 * sp.sin(th_m)**2
    sqrtg = Phi_r * r_m**2 * sp.sin(th_m)
    det_g = sp.Integer(1)
    for g_ii in g_diag:
        det_g *= g_ii
    assert sp.simplify(sqrtg**2 - sp.Abs(det_g)) == 0

    alpha = sp.Symbol('alpha_fuchs')
    Psi_st = r_m**alpha * sp.cos(q * y_m)
    box = sum(sp.diff(sqrtg / g_diag[i] * sp.diff(Psi_st, coords[i]), coords[i]) for i in range(D)) / sqrtg
    indicial = sp.expand(sp.simplify(box / (r_m**(alpha - 2) * sp.cos(q * y_m))))
    assert not indicial.has(r_m) and not indicial.has(y_m)

    g_s = sp.Symbol('gamma_n', positive=True)               # gamma_n = q r_0 / Phi_inf
    gamma_n_expr = q * r0 / Phi_inf
    indicial_g = sp.expand(indicial.subs(q, g_s * Phi_inf / r0))

    # Liouville normal form Psi = r^{-s} u: shift alpha = beta - s so that the linear coefficient is -1 (beta (beta - 1) = W)
    beta, s_sh = sp.symbols('beta_fuchs s_shift')
    shifted = sp.expand(indicial_g.subs(alpha, beta - s_sh))
    s_val = sp.solve(sp.Eq(shifted.coeff(beta, 1), -1), s_sh)[0]
    normal = sp.expand(shifted.subs(s_sh, s_val))            # beta^2 - beta - W
    W_normal = sp.simplify(-normal.coeff(beta, 0))
    roots = sp.solve(sp.Eq(normal, 0), beta)

    # Square integrability of u ~ r^beta at r -> 0 requires 2 Re(beta) + 1 > 0.
    # Real branch 0 < gamma_n <= 1: for the smaller root, (2 beta_- + 1)(1 + S) == 2 gamma_n^2 with S = sqrt(1 - gamma_n^2) >= 0.
    S_root = sp.sqrt(1 - g_s**2)
    beta_minus = min(roots, key=lambda r_: sp.N(r_.subs(g_s, sp.Rational(1, 2))))
    integ_real_identity = sp.simplify(sp.expand((2 * beta_minus + 1) * (1 + S_root)) - 2 * g_s**2)
    # Complex branch gamma_n > 1: gamma_n = sqrt(1 + h^2), h > 0
    h_s = sp.Symbol('h_fuchs', positive=True)
    integ_complex = [sp.simplify(2 * sp.re(sp.simplify(r_.subs(g_s, sp.sqrt(1 + h_s**2)))) + 1) for r_ in roots]

    return dict(disp=disp, omega_sq=omega_sq, omega_zero=omega_zero, k=k, w=w, q=q, Phi0=Phi0,
                alpha=alpha, beta=beta, g_s=g_s, gamma_n_expr=gamma_n_expr, indicial_g=indicial_g,
                s_shift=s_val, normal=normal, W_normal=W_normal, roots=roots,
                integ_real_identity=integ_real_identity, integ_complex=integ_complex)


def run_module_2(state: DeductiveState) -> None:
    banner("MODULE 2: LIE DERIVATIVES, MARSDEN-WEINSTEIN/BRST, O'NEILL, PG & MAXWELL")

    dt_ded, ds_ded, D_ded = state.theorems["dimensions"]
    R_Z2 = state.theorems["molien_dewitt_sym2"]["R_Z2"]
    sig_ax = state.axioms["AXIOM_1"]["signature"]
    f_idx_ax = state.axioms["AXIOM_1"]["fiber_index"]

    # =========================================================================
    # THM 8: Full 5D Non-Diagonal Lie Derivative Blocks, Stueckelberg Bijection,
    #        Linearised 5D Einstein KK Cohomology (Off-Shell 0, On-Shell 5 TT) & n=0 DoF == 3
    # =========================================================================
    print("  [LOG] Module 2 -> Computing THM 8 (5D Lie Blocks, Stueckelberg Jacobian & Linearised 5D Einstein KK Cohomology)...")
    x0, x2, x1, x3, x4 = sp.symbols('x0 x2 x1 x3 x4', real=True)
    coords5 = [x0, x2, x1, x3, x4]
    m_slots = [0, 2, 3, 4]

    h00 = sp.Function('h00')(x0, x1, x2)
    h01 = sp.Function('h01')(x0, x1, x2)
    h11 = sp.Function('h11')(x0, x1, x2)
    h33 = sp.Function('h33')(x0, x1, x2)
    h44 = sp.Function('h44')(x0, x1, x2)
    Phi_f = sp.Function('Phi_f')(x0, x1, x2)

    g5_lie = sp.Matrix([
        [h00, 0, h01, 0, 0],
        [0, -Phi_f**2, 0, 0, 0],
        [h01, 0, h11, 0, 0],
        [0, 0, 0, h33, 0],
        [0, 0, 0, 0, h44],
    ])
    g4_sub = g5_lie.extract(m_slots, m_slots)
    coords4 = [x0, x1, x3, x4]
    G4_lie = Geometry(g4_sub, coords4, sp.Symbol('sqrtg4_dummy', positive=True), compute_curvature=False)

    xi0 = sp.Function('xi0')(x0, x1, x2)
    xi2 = sp.Function('xi2')(x0, x1, x2)
    xi1 = sp.Function('xi1')(x0, x1, x2)
    xi_up = [xi0, xi2, xi1, sp.Integer(0), sp.Integer(0)]

    def _lie_metric(g_mat, v_up, x_list):
        n_d = len(x_list)
        L = sp.zeros(n_d, n_d)
        for A_i in range(n_d):
            for B_i in range(n_d):
                val = sum(v_up[C_i] * sp.diff(g_mat[A_i, B_i], x_list[C_i]) for C_i in range(n_d))
                val += sum(g_mat[C_i, B_i] * sp.diff(v_up[C_i], x_list[A_i]) for C_i in range(n_d))
                val += sum(g_mat[A_i, C_i] * sp.diff(v_up[C_i], x_list[B_i]) for C_i in range(n_d))
                L[A_i, B_i] = exact_rat(val)
        return L

    L_d2_raw = _lie_metric(g5_lie, [0, 1, 0, 0, 0], coords5)
    zero_mode_subs = {
        sp.diff(fn, x2): 0 for fn in [h00, h01, h11, h33, h44, Phi_f]
    }
    L_d2_ker = L_d2_raw.subs(zero_mode_subs)

    L_xi_full = _lie_metric(g5_lie, xi_up, coords5)

    xi4_up = [xi0, xi1, sp.Integer(0), sp.Integer(0)]
    xi4_cov = G4_lie.g * sp.Matrix(xi4_up)
    nabla_sym_xi4 = sp.zeros(4, 4)
    for m_a in range(4):
        for m_b in range(4):
            nab_ab = sp.diff(xi4_cov[m_b], coords4[m_a]) - sum(G4_lie.Gamma[m_c][m_a][m_b] * xi4_cov[m_c] for m_c in range(4))
            nab_ba = sp.diff(xi4_cov[m_a], coords4[m_b]) - sum(G4_lie.Gamma[m_c][m_b][m_a] * xi4_cov[m_c] for m_c in range(4))
            nabla_sym_xi4[m_a, m_b] = exact_rat(nab_ab + nab_ba)

    lie_macro_diff = sp.Matrix(4, 4, lambda a_i, b_i: exact_rat(
        L_xi_full[m_slots[a_i], m_slots[b_i]] - (xi2 * sp.diff(g4_sub[a_i, b_i], x2) + nabla_sym_xi4[a_i, b_i])
    ))
    lie_odd_diff = sp.Matrix(4, 1, lambda a_i, _: exact_rat(
        L_xi_full[m_slots[a_i], 1]
        - (sum(g4_sub[a_i, b_i] * sp.diff(xi4_up[b_i], x2) for b_i in range(4)) - Phi_f**2 * sp.diff(xi2, coords4[a_i]))
    ))
    lie_22_diff = exact_rat(
        L_xi_full[1, 1]
        - (-2 * Phi_f * (sum(xi4_up[a_i] * sp.diff(Phi_f, coords4[a_i]) for a_i in range(4)) + sp.diff(Phi_f * xi2, x2)))
    )

    # Deductive 5D Linearised Einstein Operator Cohomology (replaces hand-written Q_brst & arithmetic DoF):
    kk_res = _analyse_kk_spectrum(sig_ax, f_idx_ax)
    s_kk = kk_res["symbols"]
    q_kk, Phi0_kk, kv_kk = s_kk["q"], s_kk["Phi0"], s_kk["k"]
    k_sq_kk = sum(k_**2 for k_ in kv_kk)
    c_kk = kk_res["counts"]

    # Stueckelberg Jacobian for contravariant parameters (xi^mu, xi^y) -> (h_{mu y}, h_{yy}), deduced from the gauge map P
    # of the linearised operator by lowering the parameter index with the background metric (Eq. 41):
    eta_kk = s_kk["eta"]
    gauge_slots_kk = list(s_kk["macro"]) + [s_kk["f_idx"]]
    M_Stueck = kk_res["P_stueck_block"] * sp.diag(*[eta_kk[i_s, i_s] for i_s in gauge_slots_kk])
    det_M_Stueck = sp.factor(M_Stueck.det())
    stueck_momentum_dep = [sp.simplify(sp.diff(det_M_Stueck, m_sym)) for m_sym in (s_kk["w"],) + tuple(kv_kk)]

    dim_H_off_shell = c_kk["off_shell"][2]          # 0 (off-shell: ker E == im P, dim 5)
    dim_H_on_shell_tt = c_kk["on_shell_real"][2]    # 5 (on-shell p^2 == 0: 5 transverse-traceless states in h_{mu nu}^{(n)})
    n_dof_physical = kk_res["zero_mode"][3]         # 3 (n = 0 level: dim ker E_0 - rank P_0 == 7 - 4 == 3)

    n_cfg_even = int(state.theorems["molien_dewitt_sym2"]["even_dim"])  # 11
    n_diffeo_4d = D_ded - 1  # 4
    n_mw_arith = (2 * n_cfg_even - 4 * n_diffeo_4d) // 2  # 3

    det_E_U_munu_check = sp.simplify(
        kk_res["det_FP"] - sp.Rational(3, 1024) * q_kk**10 * s_kk["p_sq"]**5 / Phi0_kk**10
    )
    disp_kk_check = sp.expand(kk_res["omega_sq"] - (k_sq_kk - q_kk**2 / Phi0_kk**2))

    state.deduce_theorem(
        "lie_stueckelberg_brst",
        {"det_M_Stueck": det_M_Stueck, "dim_H_off_shell": dim_H_off_shell,
         "dim_H_on_shell_tt": dim_H_on_shell_tt, "n_dof_physical": n_dof_physical,
         "omega_sq": kk_res["omega_sq"], "det_E_U_munu": kk_res["det_FP"]},
        "Full 5D Lie blocks, Stueckelberg Jacobian, linearised 5D Einstein KK cohomology (off-shell 0, on-shell 5 TT) & n=0 DoF == 3",
        L_d2_ker == sp.zeros(5, 5) and lie_macro_diff == sp.zeros(4, 4)
        and lie_odd_diff == sp.zeros(4, 1) and lie_22_diff == 0
        and det_M_Stueck != 0 and kk_res["gauge_invariance"] == sp.zeros(15, 5)
        and kk_res["bianchi"] == sp.zeros(5, 15) and det_E_U_munu_check == 0
        and disp_kk_check == 0 and dim_H_off_shell == 0 and dim_H_on_shell_tt == 5
        and n_dof_physical == 3 and n_dof_physical == n_mw_arith
    )
    state.deduce_corollary("lie_stueckelberg_brst", "zero_mode_killing_kernel", L_d2_ker, sp.zeros(5, 5),
                           "Lie derivative L_{d_2} g_proj == 0_5 identically on zero-mode subspace Ker(d_2)")
    state.deduce_corollary("lie_stueckelberg_brst", "macroscopic_covariant_lie_block", lie_macro_diff, sp.zeros(4, 4),
                           "Non-diagonal 4D Lie block (L_xi g_proj)_{mu nu} == xi^2 d_2 g_{mu nu} + nabla_mu xi_nu + nabla_nu xi_mu")
    state.deduce_corollary("lie_stueckelberg_brst", "odd_graviphoton_lie_block", lie_odd_diff, sp.zeros(4, 1),
                           "Exact Z_2-odd Lie block (L_xi g_proj)_{mu 2} == g_{mu alpha} d_2 xi^alpha - Phi^2 d_mu xi^2")
    state.deduce_corollary("lie_stueckelberg_brst", "even_fiber_scalar_lie_block", lie_22_diff, 0,
                           "Exact Z_2-even fiber Lie component (L_xi g_proj)_{22} == -2 Phi [xi^alpha d_alpha Phi + d_2(Phi xi^2)]")
    state.deduce_corollary("lie_stueckelberg_brst", "invertible_stueckelberg_jacobian",
                           (sp.simplify(sp.Abs(det_M_Stueck) - 2 * Phi0_kk**2 * q_kk**5),
                            sp.simplify(sp.Abs(kk_res["det_stueck_op"]) - 2 * q_kk**5),
                            stueck_momentum_dep),
                           (0, 0, [0, 0, 0, 0]),
                           "Momentum-independent Stueckelberg Jacobian |det(M_Stueck^(n))| == 2 Phi_0^2 q^5 != 0 (q = n/L_0) for all n >= 1")
    state.deduce_corollary("lie_stueckelberg_brst", "brst_cohomology_and_marsden_weinstein_dof",
                           (kk_res["gauge_invariance"], kk_res["bianchi"], det_E_U_munu_check, disp_kk_check,
                            c_kk["off_shell"], c_kk["on_shell_real"], c_kk["on_shell_imag"],
                            c_kk["on_shell_real_unitary"], c_kk["on_shell_imag_unitary"], kk_res["zero_mode"][1:]),
                           (sp.zeros(15, 5), sp.zeros(5, 15), 0, 0,
                            (5, 5, 0), (10, 5, 5), (10, 5, 5),
                            (5, True), (5, True), (7, 4, 3)),
                           "Linearised 5D Einstein E P==0, Dv E==0, off-shell DoF==0, on-shell TT DoF==5 & zero-mode DoF == 7 - 4 == 3")

    # =========================================================================
    # THM 9: Non-Diagonal Curved O'Neill Submersion & Gauss-Codazzi Reduction
    # =========================================================================
    print("  [LOG] Module 2 -> Computing THM 9 (Non-Diagonal Curved O'Neill Submersion & Gauss-Codazzi)...")
    t_c, r_c, th_c, ph_c = sp.symbols('t_c r_c th_c ph_c', real=True)
    N0 = sp.Function('N0')(t_c, r_c)
    vr = sp.Function('vr')(t_c, r_c)
    B1 = sp.Function('B1')(t_c, r_c)
    C2 = sp.Function('C2')(t_c, r_c)
    Phi_tr = sp.Function('Phi_tr')(t_c, r_c)

    g4_nd = sp.Matrix([
        [-(N0**2 - B1**2 * vr**2), B1**2 * vr, 0, 0],
        [B1**2 * vr, B1**2, 0, 0],
        [0, 0, C2**2, 0],
        [0, 0, 0, C2**2 * sp.sin(th_c)**2],
    ])
    sqrt_g4_nd = N0 * B1 * C2**2 * sp.sin(th_c)
    G4_nd = Geometry(g4_nd, [t_c, r_c, th_c, ph_c], sqrt_g4_nd)

    g5_nd = sp.zeros(5, 5)
    g5_nd[1, 1] = -Phi_tr**2
    m_idx = [0, 2, 3, 4]
    for a_i in range(4):
        for b_i in range(4):
            g5_nd[m_idx[a_i], m_idx[b_i]] = g4_nd[a_i, b_i]
    sqrt_g5_nd = Phi_tr * sqrt_g4_nd
    G5_nd = Geometry(g5_nd, [t_c, x2, r_c, th_c, ph_c], sqrt_g5_nd)

    horiz_gamma_diffs = [
        exact_rat(G5_nd.Gamma[m_idx[c_i]][m_idx[a_i]][m_idx[b_i]] - G4_nd.Gamma[c_i][a_i][b_i])
        for c_i in range(4) for a_i in range(4) for b_i in range(4)
    ]
    odd_gamma_1 = [G5_nd.Gamma[1][m_idx[a_i]][m_idx[b_i]] for a_i in range(4) for b_i in range(4)]
    odd_gamma_2 = [G5_nd.Gamma[m_idx[c_i]][1][m_idx[a_i]] for c_i in range(4) for a_i in range(4)]
    odd_gamma_3 = [G5_nd.Gamma[1][1][1]]

    coords4_nd = [t_c, r_c, th_c, ph_c]
    bend_2_2mu = [exact_rat(G5_nd.Gamma[1][1][m_idx[a_i]] - sp.diff(Phi_tr, coords4_nd[a_i]) / Phi_tr) for a_i in range(4)]
    grad_Phi_up = G4_nd.ginv * sp.Matrix([sp.diff(Phi_tr, c_v) for c_v in coords4_nd])
    bend_mu_22 = [exact_rat(G5_nd.Gamma[m_idx[a_i]][1][1] - Phi_tr * grad_Phi_up[a_i]) for a_i in range(4)]

    tr_gamma_mu = [
        exact_rat(sum(G5_nd.Gamma[B_i][m_idx[a_i]][B_i] for B_i in range(5)) - sp.diff(sp.log(sqrt_g5_nd), coords4_nd[a_i]))
        for a_i in range(4)
    ]
    tr_gamma_2 = sum(G5_nd.Gamma[B_i][1][B_i] for B_i in range(5))

    H_Phi_4 = G4_nd.hessian_cov(Phi_tr)
    riem_horiz_ok = all(
        exact_rat(G5_nd.Riem[m_idx[a_i]][m_idx[b_i]][m_idx[c_i]][m_idx[d_i]] - G4_nd.Riem[a_i][b_i][c_i][d_i]) == 0
        for a_i in range(4) for b_i in range(4) for c_i in range(4) for d_i in range(4)
    )
    riem_odd_ok = all(
        G5_nd.Riem[m_idx[a_i]][m_idx[b_i]][m_idx[c_i]][1] == 0
        for a_i in range(4) for b_i in range(4) for c_i in range(4)
    )
    riem_tidal_ok = all(
        exact_rat(G5_nd.Riem[m_idx[a_i]][1][m_idx[b_i]][1] - Phi_tr * H_Phi_4[a_i, b_i]) == 0
        for a_i in range(4) for b_i in range(4)
    )

    z2_ric_diff = (R_Z2 * G5_nd.Ric * R_Z2 - G5_nd.Ric).applyfunc(exact_rat)
    ric_macro_ok = all(
        exact_rat(G5_nd.Ric[m_idx[a_i], m_idx[b_i]] - (G4_nd.Ric[a_i, b_i] - H_Phi_4[a_i, b_i] / Phi_tr)) == 0
        for a_i in range(4) for b_i in range(4)
    )
    box4_Phi = G4_nd.box(Phi_tr)
    ric_22_diff = exact_rat(G5_nd.Ric[1, 1] - Phi_tr * box4_Phi)

    c_GC_ded = exact_rat((G5_nd.R - G4_nd.R) / (box4_Phi / Phi_tr))
    G5_mix_nd = G5_nd.mixed(G5_nd.Ein)
    G2_2_over_R4 = exact_rat(G5_mix_nd[1, 1] / G4_nd.R)

    div_Phi_4 = sum(
        sp.diff(sqrt_g4_nd * sum(G4_nd.ginv[a_i, b_i] * sp.diff(Phi_tr, coords4_nd[b_i]) for b_i in range(4)), coords4_nd[a_i])
        for a_i in range(4)
    )
    eh_density_diff = exact_rat(sqrt_g5_nd * G5_nd.R - (Phi_tr * sqrt_g4_nd * G4_nd.R + c_GC_ded * div_Phi_4))

    state.deduce_theorem(
        "oneill_submersion_and_gauss_codazzi",
        {"c_GC": c_GC_ded, "G2_2_over_R4": G2_2_over_R4},
        "Non-diagonal curved O'Neill submersion: discovered R^(5) = R^(4) - 2(Delta_4 Phi/Phi) & G^2_2 = -(1/2)R^(4)",
        c_GC_ded == -2 and G2_2_over_R4 == -sp.Rational(1, 2) and eh_density_diff == 0
    )
    state.deduce_corollary("oneill_submersion_and_gauss_codazzi", "oneill_integrability_A_zero",
                           (all(v == 0 for v in horiz_gamma_diffs), all(v == 0 for v in odd_gamma_1 + odd_gamma_2 + odd_gamma_3)),
                           (True, True),
                           "O'Neill integrability A == 0: Gamma^(5)lambda_{mu nu} == ^(4)Gamma^lambda_{mu nu} & all Z_2-odd Gamma == 0")
    state.deduce_corollary("oneill_submersion_and_gauss_codazzi", "oneill_fiber_bending_T",
                           (bend_2_2mu, bend_mu_22), ([0] * 4, [0] * 4),
                           "O'Neill fiber bending T != 0: Gamma^2_{2 mu} == d_mu ln Phi & Gamma^mu_{22} == +Phi g^{mu nu} d_nu Phi")
    state.deduce_corollary("oneill_submersion_and_gauss_codazzi", "contracted_christoffel_traces",
                           (tr_gamma_mu, tr_gamma_2), ([0] * 4, 0),
                           "Contracted 5D traces Gamma^B_{mu B} == d_mu ln(Phi sqrt(-g_4)) & Gamma^B_{2 B} == 0")
    state.deduce_corollary("oneill_submersion_and_gauss_codazzi", "riemann_256_64_16_blocks",
                           (riem_horiz_ok, riem_odd_ok, riem_tidal_ok), (True, True, True),
                           "All 256 horizontal R^(5)_{mu nu al be}, 64 odd R^(5)_{mu nu al 2}==0 & 16 tidal R^(5)_{mu 2 nu 2}==Phi nabla_nu d_mu Phi")
    state.deduce_corollary("oneill_submersion_and_gauss_codazzi", "ricci_blocks_and_z2_parity",
                           (z2_ric_diff, ric_macro_ok, ric_22_diff), (sp.zeros(5, 5), True, 0),
                           "R_{Z_2} Ric^(5) R_{Z_2} == Ric^(5), R^(5)_{mu nu} == R^(4)_{mu nu} - H_Phi/Phi & R^(5)_{22} == Phi Delta_4 Phi")
    state.deduce_corollary("oneill_submersion_and_gauss_codazzi", "gauss_codazzi_density_and_constraint",
                           (eh_density_diff, sp.diff(G2_2_over_R4, Phi_tr), G2_2_over_R4),
                           (0, 0, -sp.Rational(1, 2)),
                           "Exact density sqrt|g_5| R^(5) == Phi sqrt(-g_4) R^(4) - 2 d_mu(sqrt(-g_4) d^mu Phi) & Phi-free G^2_2/R^(4) == -1/2")

    # =========================================================================
    # THM 10: Time-Dependent 5D Painleve-Gullstrand Connections, 5D Wave Operator & Indicial Equation from Delta_5
    # =========================================================================
    print("  [LOG] Module 2 -> Computing THM 10 (5D Painleve-Gullstrand Connections, Wave Decomposition & Indicial Equation from Delta_5)...")
    v_pg = sp.Function('v_pg')(t_c, r_c)
    Phi_pg = sp.Function('Phi_pg')(t_c, r_c)
    g5_pg_sph = sp.Matrix([
        [-(1 - v_pg**2), 0, v_pg, 0, 0],
        [0, -Phi_pg**2, 0, 0, 0],
        [v_pg, 0, 1, 0, 0],
        [0, 0, 0, r_c**2, 0],
        [0, 0, 0, 0, r_c**2 * sp.sin(th_c)**2],
    ])
    G5_pg = Geometry(g5_pg_sph, [t_c, x2, r_c, th_c, ph_c], Phi_pg * r_c**2 * sp.sin(th_c), compute_curvature=False)

    dv_dt, dv_dr = sp.diff(v_pg, t_c), sp.diff(v_pg, r_c)
    dP_dt, dP_dr = sp.diff(Phi_pg, t_c), sp.diff(Phi_pg, r_c)
    pg_gamma_checks = [
        exact_rat(G5_pg.Gamma[0][0][0] - (-v_pg**2 * dv_dr)),
        exact_rat(G5_pg.Gamma[0][0][2] - (-v_pg * dv_dr)),
        exact_rat(G5_pg.Gamma[0][2][2] - (-dv_dr)),
        exact_rat(G5_pg.Gamma[2][0][0] - (dv_dt - v_pg * (1 - v_pg**2) * dv_dr)),
        exact_rat(G5_pg.Gamma[1][1][0] - dP_dt / Phi_pg),
        exact_rat(G5_pg.Gamma[1][1][2] - dP_dr / Phi_pg),
        exact_rat(G5_pg.Gamma[0][1][1] - Phi_pg * (-dP_dt + v_pg * dP_dr)),
        exact_rat(G5_pg.Gamma[2][1][1] - Phi_pg * (v_pg * dP_dt + (1 - v_pg**2) * dP_dr)),
    ]

    Psi_5 = sp.Function('Psi_5')(t_c, x2, r_c)
    box5_Psi = G5_nd.box(Psi_5)
    df4_Psi = [sp.diff(Psi_5, c_v) for c_v in coords4_nd]
    box4_Psi = exact_rat(
        sum(sp.diff(sqrt_g4_nd * sum(G4_nd.ginv[a_i, b_i] * df4_Psi[b_i] for b_i in range(4)), coords4_nd[a_i]) for a_i in range(4))
        / sqrt_g4_nd
    )
    grad_dot = sum(G4_nd.ginv[a_i, b_i] * sp.diff(Phi_tr, coords4_nd[a_i]) * df4_Psi[b_i] for a_i in range(4) for b_i in range(4))
    wave_5d_diff = exact_rat(box5_Psi - (box4_Psi + grad_dot / Phi_tr - sp.diff(Psi_5, x2, 2) / Phi_tr**2))

    # Dispersion and indicial equation deduced from Delta_5 (no typed polynomial):
    fm = _derive_fiber_mode_equations(sig_ax, f_idx_ax)
    k_w = fm['k']
    v_phase_0 = sp.simplify(fm['omega_zero'] / k_w)
    v_group_0 = sp.simplify(sp.diff(fm['omega_zero'], k_w))
    disp_scalar_vs_tensor = sp.simplify(
        fm['omega_sq'] - kk_res["omega_sq"].subs({kv_kk[0]: k_w, kv_kk[1]: 0, kv_kk[2]: 0})
    )

    al_fuchs, gam_n = fm['alpha'], fm['g_s']
    indicial_diff = sp.expand(fm['indicial_g'] - (al_fuchs**2 + 2 * al_fuchs + gam_n**2))
    fuchs_roots = fm['roots']
    fuchs_sum = sp.simplify(fuchs_roots[0] + fuchs_roots[1])
    fuchs_prod_diff = sp.simplify(fuchs_roots[0] * fuchs_roots[1] - (gam_n**2 - sp.Rational(3, 4)))
    W_normal_diff = sp.simplify(fm['W_normal'] - (sp.Rational(3, 4) - gam_n**2))

    state.deduce_theorem(
        "painleve_gullstrand_and_5d_waves",
        {"v_phase_0": v_phase_0, "v_group_0": v_group_0, "omega_sq": fm['omega_sq'],
         "indicial": fm['indicial_g'], "fuchs_roots": fuchs_roots, "W_normal": fm['W_normal']},
        "Time-dependent 5D PG connections, curved 5D wave split, n=0 luminality & indicial equation alpha^2 + 2 alpha + gamma_n^2 == 0 from Delta_5",
        all(c == 0 for c in pg_gamma_checks) and wave_5d_diff == 0 and v_phase_0 == 1 and v_group_0 == 1
        and disp_scalar_vs_tensor == 0 and indicial_diff == 0 and fuchs_sum == 1 and fuchs_prod_diff == 0
        and W_normal_diff == 0 and fm['integ_real_identity'] == 0 and fm['integ_complex'] == [2, 2]
    )
    state.deduce_corollary("painleve_gullstrand_and_5d_waves", "pg_christoffel_identities",
                           pg_gamma_checks, [0] * len(pg_gamma_checks),
                           "All time-dependent 5D Painleve-Gullstrand Christoffel components verified identically")
    state.deduce_corollary("painleve_gullstrand_and_5d_waves", "curved_5d_wave_decomposition", wave_5d_diff, 0,
                           "Exact non-diagonal wave split Delta_5 Psi == Delta_4 Psi + (1/Phi) g^{mu nu} d_mu Phi d_nu Psi - Phi^{-2} d_2^2 Psi")
    state.deduce_corollary("painleve_gullstrand_and_5d_waves", "luminality_and_fuchs_duality",
                           (v_phase_0, v_group_0, disp_scalar_vs_tensor, indicial_diff, fm['s_shift'],
                            fuchs_sum, fuchs_prod_diff, W_normal_diff, fm['integ_real_identity'], fm['integ_complex']),
                           (1, 1, 0, 0, sp.Rational(3, 2), 1, 0, 0, 0, [2, 2]),
                           "n=0 luminality; omega^2 == k^2 - q^2/Phi_0^2; indicial alpha^2 + 2 alpha + gamma_n^2 == 0; normal form beta(beta-1) == 3/4 - gamma_n^2 "
                           "(beta_- + beta_+ == 1); both solutions square-integrable at r -> 0 for every n >= 1 (limit-circle)")

    # =========================================================================
    # THM 11: Exterior Nilpotency d^2 A == 0 & Unique d_macro == 4 Curved Weyl Invariance of Maxwell
    # =========================================================================
    print("  [LOG] Module 2 -> Computing THM 11 (Exterior Nilpotency d^2 A == 0 & Conformal Maxwell Uniqueness)...")
    A_pot = [sp.Function(f'A_pot_{i}')(*coords4_nd) for i in range(4)]
    F_cov = sp.Matrix(4, 4, lambda m, n: sp.diff(A_pot[n], coords4_nd[m]) - sp.diff(A_pot[m], coords4_nd[n]))
    bianchi_em = [
        sp.expand(sp.diff(F_cov[b_i, c_i], coords4_nd[a_i]) + sp.diff(F_cov[c_i, a_i], coords4_nd[b_i]) + sp.diff(F_cov[a_i, b_i], coords4_nd[c_i]))
        for a_i, b_i, c_i in itertools.combinations(range(4), 3)
    ]

    # Weyl weight of sqrt|g| g^{mu alpha} g^{nu beta} under g = e^{-c psi} g~, computed from d x d metrics for d = 2..8:
    psi_conf = sp.Symbol('psi_conf', real=True)
    c_conf = sp.Symbol('c_conf', positive=True)
    maxwell_weyl_exponents = {}
    for d_try in range(2, 9):
        g_try = sp.diag(*[sp.exp(-c_conf * psi_conf)] * d_try)
        dens_try = sp.powdenest(sp.sqrt(g_try.det()) * g_try.inv()[0, 0]**2, force=True)
        maxwell_weyl_exponents[d_try] = sp.simplify(sp.diff(sp.expand_log(sp.log(dens_try), force=True), psi_conf))
    conformal_dims = [d_try for d_try, e_try in maxwell_weyl_exponents.items() if e_try == 0]
    d_macro_conformal = sp.Integer(conformal_dims[0])
    maxwell_exponent_linear = [sp.simplify(e_try - c_conf * sp.Rational(4 - d_try, 2)) for d_try, e_try in maxwell_weyl_exponents.items()]

    psi_nd = sp.Function('psi_nd')(t_c, r_c)
    g4_jord_em = sp.exp(-psi_nd) * g4_nd
    sqrt_jord_em = sp.exp(-2 * psi_nd) * sqrt_g4_nd
    ginv_jord_em = sp.exp(psi_nd) * G4_nd.ginv

    F_test = sp.zeros(4, 4)
    F_test[0, 1] = sp.Function('F01')(t_c, r_c)
    F_test[1, 0] = -F_test[0, 1]
    F_test[1, 2] = sp.Function('F12')(t_c, r_c)
    F_test[2, 1] = -F_test[1, 2]

    L_Max_jord = exact_rat(
        sqrt_jord_em * sum(
            ginv_jord_em[m, a_i] * ginv_jord_em[n, b_i] * F_test[m, n] * F_test[a_i, b_i]
            for m in range(4) for n in range(4) for a_i in range(4) for b_i in range(4)
        )
    )
    dL_Max_dpsi = exact_rat(sp.diff(L_Max_jord, psi_nd))

    state.deduce_theorem(
        "maxwell_curved_weyl_invariance",
        {"d_macro_conformal": d_macro_conformal, "dL_Max_dpsi": dL_Max_dpsi},
        "Exterior nilpotency d^2 A == 0 & unique d_macro == D - 1 == 4 curved non-diagonal Weyl invariance of Maxwell",
        all(b == 0 for b in bianchi_em) and conformal_dims == [D_ded - 1] and d_macro_conformal == D_ded - 1
        and all(e_try == 0 for e_try in maxwell_exponent_linear) and dL_Max_dpsi == 0
    )
    state.deduce_corollary("maxwell_curved_weyl_invariance", "exterior_bianchi_nilpotency", bianchi_em, [0] * 4,
                           "All 4 cyclic exterior components (dF)_{lambda mu nu} == (d^2 A)_{lambda mu nu} == 0 verified")
    state.deduce_corollary("maxwell_curved_weyl_invariance", "conformal_dimension_four_uniqueness",
                           (d_macro_conformal, d_macro_conformal - (D_ded - 1), dL_Max_dpsi), (4, 0, 0),
                           "Unique conformal Maxwell dimension d_macro == 4 == D - 1 & non-diagonal d L_Max / d psi == 0")


# ========================================================================================
# COLAB CELL 2 EXECUTION & MEMORY CHECKPOINT
# ========================================================================================
if __name__ == "__main__":
    if "_COLAB_CHECKPOINT_M1" not in globals():
        raise RuntimeError("Please run Colab Cell 1 (Module 1) first so `_COLAB_CHECKPOINT_M1` is in memory!")
    t0_m2 = time.time()
    master_state = copy.deepcopy(globals()["_COLAB_CHECKPOINT_M1"])
    print("  [INSTANT RESUME] Loaded Module 1 state (7 THMs, 40 CORs) from `_COLAB_CHECKPOINT_M1`!")
    run_module_2(master_state)
    globals()["_COLAB_CHECKPOINT_M2"] = copy.deepcopy(master_state)
    print(f"\n  --> [COLAB CELL 2 READY] Module 2 completed in {time.time() - t0_m2:.2f}s "
          f"(Cumulative: {master_state.thm_count} THMs, {master_state.cor_count} CORs saved in `_COLAB_CHECKPOINT_M2`)")


# ========================================================================================
# CBG PRINCIPIA GEOMETRICA — COLAB CELL 3: MODULE 3 (THM #12 .. THM #14)
# ========================================================================================
# Resumes automatically from `_COLAB_CHECKPOINT_M2` in notebook memory (still using ONLY
# AXIOM_1!) and saves `_COLAB_CHECKPOINT_M3` upon completion.
# ========================================================================================

# ========================================================================================
# MODULE 3: 5D JORDAN VACUUM, NON-DIAGONAL WEYL DISCOVERY, GEODESICS & YUKAWA KERNEL
# ========================================================================================
def run_module_3(state: DeductiveState) -> None:
    banner("MODULE 3: 5D JORDAN VACUUM, CURVED WEYL DISCOVERY, 5D GEODESICS & YUKAWA")

    _, _, D_ded = state.theorems["dimensions"]
    d_macro_ded = state.theorems["maxwell_curved_weyl_invariance"]["d_macro_conformal"]  # 4 from THM 11
    c_DW_inv_ded = state.theorems["molien_dewitt_sym2"]["c_DW_inv_trace"]                # -1/3 from THM 4!

    # =========================================================================
    # THM 12: 5D Jordan Vacuum Equations & Constructive Non-Diagonal Weyl Discovery
    #         (c_Weyl = 1, c_div = 1, kappa_psi = -3/2)
    # =========================================================================
    print("  [LOG] Module 3 -> Computing THM 12 (5D Jordan Vacuum & Non-Diagonal Curved Weyl Discovery)...")
    t_w, r_w, th_w, ph_w, x2 = sp.symbols('t_w r_w th_w ph_w x2', real=True)
    coords4 = [t_w, r_w, th_w, ph_w]

    N_e = sp.Function('N_e')(t_w, r_w)
    v_e = sp.Function('v_e')(t_w, r_w)
    B_e = sp.Function('B_e')(t_w, r_w)
    psi_w = sp.Function('psi_w')(t_w, r_w)
    Phi_inf = sp.Symbol('Phi_inf', positive=True)

    gt4 = sp.Matrix([
        [-(N_e**2 - B_e**2 * v_e**2), B_e**2 * v_e, 0, 0],
        [B_e**2 * v_e, B_e**2, 0, 0],
        [0, 0, r_w**2, 0],
        [0, 0, 0, r_w**2 * sp.sin(th_w)**2],
    ])
    sqrt_gt4 = N_e * B_e * r_w**2 * sp.sin(th_w)
    Gt4 = Geometry(gt4, coords4, sqrt_gt4)

    # Deduce c_Weyl from the curved non-diagonal reduction: with g_{mu nu} = exp(-c psi) g~_{mu nu}, the multiplier of R~^(4)
    # in (Phi / Phi_inf) sqrt(-g_4) R^(4)[g] is read off from a second derivative that enters only through R~ and must not
    # depend on psi. (The divergence term of THM 9 contains no R~.) The volume ratio is taken from the determinants.
    c_w_sym = sp.Symbol('c_w_sym', real=True)
    g4_c = sp.exp(-c_w_sym * psi_w) * gt4
    vol_ratio_c = sp.powdenest(sp.sqrt(sp.simplify(g4_c.det() / gt4.det())), force=True)
    G4_c = Geometry(g4_c, coords4, vol_ratio_c * sqrt_gt4)
    eh_probe = sp.diff(N_e, r_w, 2)
    Phi_j = Phi_inf * sp.exp(psi_w)
    eh_conformal_weight = sp.simplify(
        sp.diff((Phi_j / Phi_inf) * G4_c.sqrtg * G4_c.R, eh_probe) / sp.diff(sqrt_gt4 * Gt4.R, eh_probe)
    )
    eh_psi_exponent = sp.simplify(sp.diff(sp.expand_log(sp.log(eh_conformal_weight), force=True), psi_w))
    c_w_ded = sp.solve(sp.Eq(eh_psi_exponent, 0), c_w_sym)[0]

    g4_j = sp.exp(-c_w_ded * psi_w) * gt4
    sqrt_g4_j = vol_ratio_c.subs(c_w_sym, c_w_ded) * sqrt_gt4
    G4_j = Geometry(g4_j, coords4, sqrt_g4_j)

    g5_j = sp.zeros(5, 5)
    g5_j[1, 1] = -Phi_j**2
    m_idx = [0, 2, 3, 4]
    for a_i in range(4):
        for b_i in range(4):
            g5_j[m_idx[a_i], m_idx[b_i]] = g4_j[a_i, b_i]
    G5_j = Geometry(g5_j, [t_w, x2, r_w, th_w, ph_w], Phi_j * sqrt_g4_j)

    H_Phi_j = G4_j.hessian_cov(Phi_j)
    box_Phi_j = G4_j.box(Phi_j)
    vac_macro_ok = all(
        exact_rat(G5_j.Ein[m_idx[a_i], m_idx[b_i]] - (G4_j.Ein[a_i, b_i] - (H_Phi_j[a_i, b_i] - g4_j[a_i, b_i] * box_Phi_j) / Phi_j)) == 0
        for a_i in range(4) for b_i in range(4)
    )
    vac_odd_ok = all(G5_j.Ein[m_idx[a_i], 1] == 0 for a_i in range(4))
    vac_22_diff = exact_rat(G5_j.Ein[1, 1] - sp.Rational(1, 2) * Phi_j**2 * G4_j.R)

    g22_inv_sym = sp.Symbol('g22_inv', negative=True)
    psi_from_g22 = sp.Rational(1, 2) * sp.log(-1 / (Phi_inf**2 * g22_inv_sym))
    dpsi_dg22 = exact_rat(sp.diff(psi_from_g22, g22_inv_sym) + 1 / (2 * g22_inv_sym))

    dpsi_4 = [sp.diff(psi_w, c_v) for c_v in coords4]
    dpsi_up = Gt4.ginv * sp.Matrix(dpsi_4)
    weyl_gamma_ok = all(
        exact_rat(
            G4_j.Gamma[c_i][a_i][b_i] - Gt4.Gamma[c_i][a_i][b_i]
            - (-sp.Rational(1, 2) * (
                (1 if c_i == a_i else 0) * dpsi_4[b_i]
                + (1 if c_i == b_i else 0) * dpsi_4[a_i]
                - gt4[a_i, b_i] * dpsi_up[c_i]
            ))
        ) == 0
        for c_i in range(4) for a_i in range(4) for b_i in range(4)
    )
    weyl_gamma_trace = [
        exact_rat(sum(G4_j.Gamma[b_i][a_i][b_i] - Gt4.Gamma[b_i][a_i][b_i] for b_i in range(4)) - (-2 * dpsi_4[a_i]))
        for a_i in range(4)
    ]

    div_gt_psi = sum(
        sp.diff(sqrt_gt4 * sum(Gt4.ginv[a_i, b_i] * dpsi_4[b_i] for b_i in range(4)), coords4[a_i])
        for a_i in range(4)
    )
    grad_sq_tilde = Gt4.grad_sq(psi_w)
    excess_5d = exact_rat((Phi_j * sqrt_g4_j * G5_j.R) / Phi_inf - sqrt_gt4 * Gt4.R)
    ddpsi_rr = sp.diff(psi_w, r_w, 2)
    c_div_ded = exact_rat(sp.diff(excess_5d, ddpsi_rr) / sp.diff(div_gt_psi, ddpsi_rr))
    rem_kin = exact_rat(excess_5d - c_div_ded * div_gt_psi)
    kappa_psi_ded = exact_rat(rem_kin / (sqrt_gt4 * grad_sq_tilde))

    R4_weyl_diff = exact_rat(G4_j.R - sp.exp(psi_w) * (Gt4.R + 3 * Gt4.box(psi_w) - sp.Rational(3, 2) * grad_sq_tilde))

    div_Ein_tilde = Gt4.cov_div_mixed(Gt4.mixed(Gt4.Ein))
    dpsi_col = sp.Matrix(dpsi_4)
    T_psi_cov = (-kappa_psi_ded) * (dpsi_col * dpsi_col.T - sp.Rational(1, 2) * gt4 * grad_sq_tilde)
    div_T_psi = Gt4.cov_div_mixed(Gt4.mixed(T_psi_cov))
    div_T_psi_diff = sp.Matrix(4, 1, lambda a_i, _: exact_rat(div_T_psi[a_i] - (-kappa_psi_ded) * Gt4.box(psi_w) * dpsi_4[a_i]))

    state.deduce_theorem(
        "jordan_vacuum_and_curved_weyl_discovery",
        {"c_Weyl": c_w_ded, "c_div": c_div_ded, "kappa_psi": kappa_psi_ded},
        "Non-diagonal curved 4D Weyl decoupling solved constructively: c_Weyl=1, c_div=1, kappa_psi=-3/2",
        c_w_ded == 1 and kappa_psi_ded == -sp.Rational(3, 2) and vac_macro_ok and vac_odd_ok and vac_22_diff == 0
    )
    state.deduce_corollary("jordan_vacuum_and_curved_weyl_discovery", "jordan_5d_einstein_blocks",
                           (vac_macro_ok, vac_odd_ok, vac_22_diff, dpsi_dg22),
                           (True, True, 0, 0),
                           "5D Jordan blocks G^(5)_{mu nu}, G^(5)_{mu 2}==0, G^(5)_{22}==(1/2)Phi^2 R^(4) & dpsi/dg^{22}==-1/(2g^{22})")
    state.deduce_corollary("jordan_vacuum_and_curved_weyl_discovery", "non_diagonal_weyl_connections_and_ricci",
                           (weyl_gamma_ok, weyl_gamma_trace, R4_weyl_diff),
                           (True, [0] * 4, 0),
                           "All 64 non-diagonal Weyl connection shifts, -2 d_mu psi trace & curved R^(4) transformation verified")
    state.deduce_corollary("jordan_vacuum_and_curved_weyl_discovery", "curved_bianchi_and_scalar_divergence",
                           (div_Ein_tilde, div_T_psi_diff, kappa_psi_ded),
                           (sp.zeros(4, 1), sp.zeros(4, 1), -sp.Rational(3, 2)),
                           "Non-diagonal curved nabla~_mu G~^mu_nu == 0, nabla~_mu T^(psi)mu_nu == (3/2)(Delta~_4 psi) d_nu psi & kappa_psi==-3/2")

    # =========================================================================
    # THM 13: 5D Bitemporal Geodesics, Null-Projection Threshold & Routhian Mass Slope
    # =========================================================================
    print("  [LOG] Module 3 -> Computing THM 13 (5D Bitemporal Geodesics, Null-Projection Threshold & Routhian Mass)...")
    lam_af = sp.Symbol('lam_af', real=True)
    t_l = sp.Function('t_l')(lam_af)
    x2_l = sp.Function('x2_l')(lam_af)
    r_l = sp.Function('r_l')(lam_af)
    th_l = sp.Function('th_l')(lam_af)
    Phi_wl = sp.Function('Phi_wl')(t_l, r_l, th_l)
    g00_wl = sp.Function('g00_wl')(t_l, r_l, th_l)
    g0r_wl = sp.Function('g0r_wl')(t_l, r_l, th_l)
    grr_wl = sp.Function('grr_wl')(t_l, r_l, th_l)

    L_geo_5d = sp.Rational(1, 2) * (
        g00_wl * sp.diff(t_l, lam_af)**2
        + 2 * g0r_wl * sp.diff(t_l, lam_af) * sp.diff(r_l, lam_af)
        + grr_wl * sp.diff(r_l, lam_af)**2
        - Phi_wl**2 * sp.diff(x2_l, lam_af)**2
    )
    P2_conj = sp.diff(L_geo_5d, sp.diff(x2_l, lam_af))
    dL_dx2 = sp.diff(L_geo_5d, x2_l)

    P2_s, eps_5, m0_s = sp.symbols('P_2 eps_5 m_0', positive=True)
    dx2_sol = sp.solve(sp.Eq(P2_conj, P2_s), sp.diff(x2_l, lam_af))[0]
    F_fiber_r = exact_rat(sp.diff(L_geo_5d, r_l).subs(sp.diff(x2_l, lam_af), dx2_sol) - sp.diff(L_geo_5d.subs(Phi_wl, 0), r_l))
    expected_F_fiber_r = -(P2_s**2 / Phi_wl**3) * sp.diff(Phi_wl, r_l)

    Phi_pos = sp.Symbol('Phi_pos', positive=True)
    # First integral 2 L == -eps_5^2 with the conserved fibre momentum: macroscopic norm g_{mu nu} x'^mu x'^nu on shell,
    # and the value of Phi at which the projected 4-velocity becomes null (condition for a causal projection, Eq. 80):
    fiber_part_geo = 2 * L_geo_5d - 2 * L_geo_5d.subs(sp.diff(x2_l, lam_af), 0)
    macro_norm_on_shell = sp.simplify(
        (-eps_5**2 - fiber_part_geo).subs(sp.diff(x2_l, lam_af), dx2_sol).subs(Phi_wl, Phi_pos)
    )
    Phi_null_threshold = sp.solve(sp.Eq(macro_norm_on_shell, 0), Phi_pos)[0]

    psi_s = sp.Symbol('psi_s', real=True)
    # Extract inverse fiber metric -g^{22}(psi) = -G5_j.ginv[1, 1] directly from G5_j of THM 12:
    inv_g22_abs = (-G5_j.ginv[1, 1]).subs(psi_w, psi_s)
    weyl_ds_factor = sp.sqrt((G4_j.g[2, 2] / Gt4.g[2, 2]).subs(psi_w, psi_s))
    mu_geom = weyl_ds_factor * sp.sqrt(m0_s**2 + P2_s**2 * inv_g22_abs)
    beta_slope_all = exact_rat(-sp.diff(sp.log(mu_geom), psi_s))
    beta_zero_mode = sp.simplify(beta_slope_all.subs(P2_s, 0))
    beta_chrono_dom = sp.simplify(sp.limit(beta_slope_all, P2_s, sp.oo))

    state.deduce_theorem(
        "bitemporal_5d_geodesics",
        {"P2_conjugate": P2_conj, "Phi_null_threshold": Phi_null_threshold,
         "beta_zero_mode": beta_zero_mode, "beta_chrono_dom": beta_chrono_dom},
        "5D cyclic fiber conservation P_2, null-projection threshold Phi=|P_2|/eps_5 & Routhian slopes (+1/2, +3/2)",
        dL_dx2 == 0 and exact_rat(F_fiber_r - expected_F_fiber_r) == 0
        and beta_zero_mode == sp.Rational(1, 2) and beta_chrono_dom == sp.Rational(3, 2)
    )
    state.deduce_corollary("bitemporal_5d_geodesics", "cyclic_momentum_and_chrono_force",
                           (dL_dx2, exact_rat(F_fiber_r - expected_F_fiber_r)), (0, 0),
                           "Cyclic conservation d_lam P_2 == 0 & exact fiber force -(P_2^2 / Phi^3) d_mu Phi")
    state.deduce_corollary("bitemporal_5d_geodesics", "causal_null_threshold",
                           (sp.simplify(macro_norm_on_shell - (P2_s**2 / Phi_pos**2 - eps_5**2)), Phi_null_threshold), (0, P2_s / eps_5),
                           "On shell g_{mu nu} x'x' == P_2^2/Phi^2 - eps_5^2: the projected 4-velocity is causal iff Phi >= |P_2| / eps_5 (null at equality)")
    state.deduce_corollary("bitemporal_5d_geodesics", "routhian_conformal_slopes",
                           (beta_zero_mode, beta_chrono_dom,
                            exact_rat(beta_slope_all - (sp.Rational(1, 2) + P2_s**2 / (m0_s**2 * Phi_inf**2 * sp.exp(2 * psi_s) + P2_s**2)))),
                           (sp.Rational(1, 2), sp.Rational(3, 2), 0),
                           "Exact Routhian slope -d ln mu_geom / dpsi for all P_2: zero-mode +1/2 & chrono-dominated +3/2")

    # =========================================================================
    # THM 14: Frechet Linearization on Expanding FLRW a(t), DeWitt-Weyl Unification
    #         of alpha_0^2 == 1/3 == -c_DW_inv & Scale-Dependent Yukawa Slip
    # =========================================================================
    print("  [LOG] Module 3 -> Computing THM 14 (Frechet Linearization on a(t), alpha_0^2 == 1/3 == -c_DW_inv & Yukawa)...")
    tt, xx, yy, zz = sp.symbols('t_f x_f y_f z_f', real=True)
    eps_f = sp.Symbol('eps_f', real=True)
    a_t = sp.Function('a_t')(tt)
    Phi_N = sp.Function('Phi_N')(tt, xx, yy, zz)
    Psi_N = sp.Function('Psi_N')(tt, xx, yy, zz)

    gt_pert = sp.diag(
        -(1 + 2 * eps_f * Psi_N),
        a_t**2 * (1 - 2 * eps_f * Phi_N),
        a_t**2 * (1 - 2 * eps_f * Phi_N),
        a_t**2 * (1 - 2 * eps_f * Phi_N),
    )
    sqrt_pert = a_t**3 * sp.sqrt(1 + 2 * eps_f * Psi_N) * (1 - 2 * eps_f * Phi_N)**sp.Rational(3, 2)
    Gt_pert = Geometry(gt_pert, [tt, xx, yy, zz], sqrt_pert)

    dEin_xy = exact_rat(sp.diff(Gt_pert.Ein[1, 2], eps_f).subs(eps_f, 0))
    aniso_diff = exact_rat(dEin_xy - sp.diff(Phi_N - Psi_N, xx, yy))
    lap3_PhiN = sp.diff(Phi_N, xx, 2) + sp.diff(Phi_N, yy, 2) + sp.diff(Phi_N, zz, 2)
    H_t = sp.diff(a_t, tt) / a_t
    Gmix_pert_00 = exact_rat(Gt_pert.ginv[0, 0] * Gt_pert.Ein[0, 0])
    dGmix_00 = exact_rat(sp.diff(Gmix_pert_00, eps_f).subs(eps_f, 0).subs(Psi_N, Phi_N).doit())
    expected_dGmix_00 = -(2 / a_t**2) * lap3_PhiN + 6 * H_t * (sp.diff(Phi_N, tt) + H_t * Phi_N)
    poisson_flrw_diff = exact_rat(dGmix_00 - expected_dGmix_00)

    # Quasi-static linear response (time derivatives of the background and of the amplitudes dropped; scale factor a_stat):
    rho_m, k_f, m_eff, a_stat, Vpp_0 = sp.symbols('rho_m k_f m_eff a_stat Vpp_0', positive=True)
    amp_N, amp_psi = sp.symbols('amp_N amp_psi')
    plane_k = sp.exp(sp.I * k_f * xx)

    # (a) Einstein-frame potential from the linearised operator delta G~^0_0 computed above (with Psi_N == Phi_N),
    #     sourced by delta T^0_0 == -rho_m (rho_m is defined as the energy density of the source):
    dG00_plane = dGmix_00.subs(Phi_N, amp_N * plane_k).doit().subs(sp.diff(a_t, tt), 0).subs(a_t, a_stat)
    poisson_amp = sp.simplify(dG00_plane / plane_k)
    Psi_N_k = sp.solve(sp.Eq(poisson_amp, -rho_m), amp_N)[0]

    # (b) Scalar equation: Euler-Lagrange equation of kappa_psi g~^{mu nu} d psi d psi - 2 V(psi) - 2 n mu_geom(psi) (Eq. 86),
    #     with mu_geom at P_2 = 0 from THM 13 and V expanded to second order about a stationary point (V'(0) = 0, V''(0) = Vpp_0):
    psi_fld = sp.Function('psi_fld')(xx, yy, zz)
    n_src = sp.Function('n_src')(xx, yy, zz)
    mu_zero_mode = mu_geom.subs(P2_s, 0)
    grad_sq_stat = sum(sp.diff(psi_fld, c_v)**2 for c_v in (xx, yy, zz)) / a_stat**2
    ell_psi = (kappa_psi_ded * grad_sq_stat - 2 * (sp.Rational(1, 2) * Vpp_0 * psi_fld**2)
               - 2 * n_src * mu_zero_mode.subs(psi_s, psi_fld))
    EL_psi_stat = sp.diff(ell_psi, psi_fld) - sum(
        sp.diff(sp.diff(ell_psi, sp.diff(psi_fld, c_v)), c_v) for c_v in (xx, yy, zz)
    )
    EL_psi_lin = sp.diff(
        EL_psi_stat.subs({psi_fld: eps_f * amp_psi * plane_k, n_src: eps_f * (rho_m / m0_s) * plane_k}).doit(), eps_f
    ).subs(eps_f, 0)
    delta_psi_k = sp.solve(sp.Eq(sp.simplify(EL_psi_lin / plane_k), 0), amp_psi)[0]
    m_eff_sq_ded = Vpp_0 / (-kappa_psi_ded)                       # mass of the linearised operator
    c_trace_source = exact_rat(delta_psi_k * (k_f**2 / a_stat**2 + m_eff_sq_ded) / rho_m)
    delta_psi_k = delta_psi_k.subs(Vpp_0, -kappa_psi_ded * m_eff**2)

    # (c) Jordan-frame potentials from the Weyl map g = exp(-c_Weyl psi) g~ of THM 12 at first order:
    Psi_E, Phi_E, dpsi_amp = sp.symbols('Psi_E Phi_E dpsi_amp', real=True)
    weyl_lin = sp.exp(-c_w_ded * eps_f * dpsi_amp)
    #     g_tt = -(1 + 2 eps Psi_J) and g_xx / a^2 = 1 - 2 eps Phi_J define the Jordan-frame potentials.
    Psi_J_lin = sp.Rational(1, 2) * sp.diff(-(weyl_lin * (-(1 + 2 * eps_f * Psi_E))), eps_f).subs(eps_f, 0)
    Phi_J_lin = -sp.Rational(1, 2) * sp.diff(weyl_lin * (1 - 2 * eps_f * Phi_E), eps_f).subs(eps_f, 0)
    Psi_J_k = Psi_J_lin.subs({Psi_E: Psi_N_k, dpsi_amp: delta_psi_k})
    Phi_J_k = Phi_J_lin.subs({Phi_E: Psi_N_k, dpsi_amp: delta_psi_k})
    mu_eff_ak = exact_rat(Psi_J_k / Psi_N_k)
    alpha_0_sq = exact_rat(sp.limit(mu_eff_ak, k_f, sp.oo) - 1)
    eta_slip_ak = exact_rat(Phi_J_k / Psi_J_k)

    state.deduce_theorem(
        "frechet_perturbation_and_yukawa_discovery",
        {"c_trace_source": c_trace_source, "alpha_0_sq": alpha_0_sq,
         "mu_eff_ak": mu_eff_ak, "eta_slip_ak": eta_slip_ak},
        "Frechet a(t) linearization & quasi-static Yukawa kernel mu_eff(a,k) with alpha_0^2 == 1/3 (equal to -c_DW_inv of THM 4 at lambda = 1/2, D = 5)",
        aniso_diff == 0 and poisson_flrw_diff == 0 and alpha_0_sq == sp.Rational(1, 3) and alpha_0_sq == -c_DW_inv_ded
    )
    state.deduce_corollary("frechet_perturbation_and_yukawa_discovery", "frechet_anisotropic_and_poisson",
                           (aniso_diff, poisson_flrw_diff), (0, 0),
                           "Exact a(t) Frechet G~_xy => Psi_N==Phi_N & delta G~^0_0 == -(2/a^2) nabla^2 Phi_N + 6 H(Phi_N_dot + H Phi_N)")
    state.deduce_corollary("frechet_perturbation_and_yukawa_discovery", "yukawa_kernel_and_dewitt_unification",
                           (c_trace_source, alpha_0_sq, alpha_0_sq + c_DW_inv_ded,
                            exact_rat(mu_eff_ak - (1 + sp.Rational(1, 3) * k_f**2 / (k_f**2 + a_stat**2 * m_eff**2))),
                            sp.limit(mu_eff_ak, k_f, 0), sp.limit(mu_eff_ak, k_f, sp.oo),
                            mu_eff_ak.subs(k_f, a_stat * m_eff)),
                           (sp.Rational(1, 3), sp.Rational(1, 3), 0, 0, 1, sp.Rational(4, 3), sp.Rational(7, 6)),
                           "Exact Yukawa kernel mu_eff(a,k) == 1 + (1/3)k^2/(k^2+a^2 m^2), alpha_0^2 == -c_DW_inv == 1/3, IR=1, UV=4/3, mid=7/6")
    state.deduce_corollary("frechet_perturbation_and_yukawa_discovery", "jordan_gravitational_slip",
                           (sp.limit(eta_slip_ak, k_f, 0), sp.limit(eta_slip_ak, k_f, sp.oo)),
                           (1, sp.Rational(1, 2)),
                           "Scale-dependent Jordan gravitational slip eta_J(a,k) == Phi_J/Psi_J: screened GR limit 1 (IR) & BD limit 1/2 (UV)")


# ========================================================================================
# COLAB CELL 3 EXECUTION & MEMORY CHECKPOINT
# ========================================================================================
if __name__ == "__main__":
    if "_COLAB_CHECKPOINT_M2" not in globals():
        raise RuntimeError("Please run Colab Cell 2 (Module 2) first so `_COLAB_CHECKPOINT_M2` is in memory!")
    t0_m3 = time.time()
    master_state = copy.deepcopy(globals()["_COLAB_CHECKPOINT_M2"])
    print("  [INSTANT RESUME] Loaded Modules 1-2 state (11 THMs, 57 CORs) from `_COLAB_CHECKPOINT_M2`!")
    run_module_3(master_state)
    globals()["_COLAB_CHECKPOINT_M3"] = copy.deepcopy(master_state)
    print(f"\n  --> [COLAB CELL 3 READY] Module 3 completed in {time.time() - t0_m3:.2f}s "
          f"(Cumulative: {master_state.thm_count} THMs, {master_state.cor_count} CORs saved in `_COLAB_CHECKPOINT_M3`)")


# ========================================================================================
# DEDUCTIVE HELPERS FOR THM #17 (WINDING MOMENTS) & THM #19 (ORBIFOLD DE RHAM COHOMOLOGY)
# ========================================================================================
def _derive_interval_cohomology():
    """
    Betti numbers and spectrum of S^1/Z_2 = [0, pi L_0], derived from the Z_2-invariant
    forms on the covering circle (replaces b0 = int(cos(0)), b1 = int(sin(0))).
    """
    y, L0, mu = sp.symbols('y L0 mu', positive=True)
    n = sp.Symbol('n', integer=True, positive=True)
    C1, C2 = sp.symbols('C1 C2')
    f = sp.Function('f')

    def constraint_rows(sol, form_degree):
        """Linear conditions on (C1, C2) for a Z_2-invariant, 2 pi L_0-periodic solution on the covering circle.
        0-form f(y): invariant iff f(-y) = f(y). 1-form g(y) dy: invariant iff g(-y) = -g(y), because dy -> -dy."""
        parity_sign = 1 if form_degree == 0 else -1
        parity = sol - parity_sign * sol.subs(y, -y)
        seam_val = sol.subs(y, sp.pi * L0) - sol.subs(y, -sp.pi * L0)
        dsol = sp.diff(sol, y)
        seam_der = dsol.subs(y, sp.pi * L0) - dsol.subs(y, -sp.pi * L0)
        return sp.Matrix([[sp.simplify(sp.diff(c_, C1)), sp.simplify(sp.diff(c_, C2))] for c_ in (parity, seam_val, seam_der)])

    # zero modes: -f'' = 0
    sol0 = sp.dsolve(sp.Eq(-f(y).diff(y, 2), 0), f(y)).rhs
    M0_form0 = constraint_rows(sol0, 0)
    M0_form1 = constraint_rows(sol0, 1)
    b0 = 2 - M0_form0.rank()
    b1 = 2 - M0_form1.rank()

    # excited modes: -f'' = mu^2 f
    solm = sp.dsolve(sp.Eq(-f(y).diff(y, 2), mu**2 * f(y)), f(y)).rhs
    out = {}
    for deg in (0, 1):
        M = constraint_rows(solm, deg)
        minors = [sp.simplify(M.extract(list(rows), [0, 1]).det()) for rows in itertools.combinations(range(3), 2)]
        generic_nullity = 2 - M.rank()
        quantised = M.subs(mu, n / L0).applyfunc(sp.simplify)
        out[deg] = dict(
            minors=minors,
            minors_vanish_on_shell=all(sp.simplify(m_.subs(mu, n / L0)) == 0 for m_ in minors),
            secular_ratio_free_of_zeros=[sp.simplify(m_ / sp.sin(sp.pi * L0 * mu)) for m_ in minors],
            generic_nullity=generic_nullity,
            quantised_nullity=2 - quantised.rank(),
        )

    return dict(b0=b0, b1=b1, chi=b0 - b1, rank0=(M0_form0.rank(), M0_form1.rank()), excited=out,
                mu_n=n / L0, symbols=dict(y=y, L0=L0, mu=mu, n=n))


def _derive_winding_moments(D, L0_sym, Phi_sym):
    """
    Per-winding heat kernel from the Poisson/Gaussian transform, its Z_2 character for the
    periodic and antiperiodic sectors, the Mellin moments M_k^(w), their winding exponents p_k,
    and the winding sums evaluated as series (replaces w^-p -> zeta(p) and w^-1 -> log 2).
    """
    s = sp.Symbol('s_m', positive=True)
    w = sp.Symbol('w_m', integer=True, positive=True)
    v = sp.Symbol('v', real=True)
    R = L0_sym * Phi_sym

    K_cos = sp.simplify(sp.integrate(sp.exp(-s * v**2 / R**2) * sp.cos(2 * sp.pi * w * v), (v, -sp.oo, sp.oo)))
    K_sin = sp.integrate(sp.exp(-s * v**2 / R**2) * sp.sin(2 * sp.pi * w * v), (v, -sp.oo, sp.oo))
    char = {delta: sp.simplify(sp.exp(2 * sp.pi * sp.I * w * delta)) for delta in (sp.Integer(0), sp.Rational(1, 2))}

    d_macro = D - 1
    M_single, p_exp, coeff = [], [], []
    for k in range(3):
        Mk = sp.simplify(sp.integrate(s**(k - 1) / (4 * sp.pi * s)**sp.Rational(d_macro, 2) * K_cos, (s, 0, sp.oo)))
        pk = sp.simplify(-w * sp.diff(sp.log(Mk), w))
        ck = sp.simplify(Mk * w**pk)
        assert not ck.has(w)
        M_single.append(Mk)
        p_exp.append(pk)
        coeff.append(ck)

    untwisted_sum = [sp.summation(w**(-p), (w, 1, sp.oo)) for p in p_exp]
    twisted_sum = []  # sum_w char_{1/2}(w) w^-p = sum_w (-1)^w w^-p
    for p in p_exp:
        if p > 1:
            total = sp.summation(w**(-p), (w, 1, sp.oo))
            even = sp.summation((2 * w)**(-p), (w, 1, sp.oo))
            twisted_sum.append(sp.simplify(-(total - 2 * even)))
        else:
            twisted_sum.append(sp.summation(char[sp.Rational(1, 2)] * w**(-p), (w, 1, sp.oo)))

    M_untwisted = [coeff[k] * untwisted_sum[k] for k in range(3)]
    M_twisted = [coeff[k] * twisted_sum[k] for k in range(3)]  # natural sign, weights (-1)^w
    return dict(
        K_w=K_cos, K_sin=K_sin, char_periodic=char[sp.Integer(0)], char_antiperiodic=char[sp.Rational(1, 2)],
        M_single=M_single, p_exponents=p_exp, coeff=coeff,
        untwisted_sum=untwisted_sum, twisted_sum=twisted_sum,
        M_untwisted=M_untwisted, M_twisted=M_twisted,
        symbols=dict(s=s, L0=L0_sym, Phi=Phi_sym, w=w),
    )

    
# ========================================================================================
# MODULE 4 (FROM AXIOM 2): LICHNEROWICZ, LOVELOCK, PLANCHEREL & MCKEAN-SINGER DE RHAM
# ========================================================================================
def run_module_4(state: DeductiveState) -> None:
    banner("MODULE 4 (FROM AXIOM 2): LICHNEROWICZ, LOVELOCK, PLANCHEREL & MCKEAN-SINGER DE RHAM")

    state.inject_axiom("AXIOM_2", {
        "spectral_principle": "Effective action generated by Wick-rotated Tr f(D_JV^2 / Lambda_cut^2) with f_{4-2k} != 0 (k in {0,1,2})",
        "uv_finiteness": "Untwisted Sym^2_+ at k in {0,1} and twisted Sym^2_- at k=2 yield 0 < |M_k| < oo spanning W_3 = Span{e^{-2psi}, e^{-4psi}, e^{-6psi}}",
    })

    dt_ded, ds_ded, D_ded = state.theorems["dimensions"]
    Gamma_hat = state.theorems["clifford_sp4_spinor"]["Gamma_hat"]
    c_Weyl_ded = state.theorems["jordan_vacuum_and_curved_weyl_discovery"]["c_Weyl"]       # 1 from THM 12
    kappa_psi_ded = state.theorems["jordan_vacuum_and_curved_weyl_discovery"]["kappa_psi"]  # -3/2 from THM 12
    eta5 = sp.diag(-1, -1, 1, 1, 1)

    def _require_all(tag: str, named_conditions: list) -> bool:
        """Granular diagnostic verifier: reports the exact sub-condition and value on any failure."""
        for cond_name, cond_ok, debug_val in named_conditions:
            if not bool(cond_ok):
                print(f"\n  [!!! DIAGNOSTIC FAILURE IN {tag} !!!]")
                print(f"    -> Failed condition : {cond_name}")
                print(f"    -> Actual value     : {debug_val}")
                raise AssertionError(f"[DIAGNOSTIC FAIL in {tag}] `{cond_name}` failed; actual = {debug_val}")
        return True

    def _zero_rat_fast(expr):
        """
        GCD-free exact rational normalizer: places expr over a syntactic common denominator
        and expands the numerator. Proves exact zero identities in O(monomials) milliseconds.
        """
        if isinstance(expr, sp.MatrixBase):
            return expr.applyfunc(_zero_rat_fast)
        e = _normalize_exp_powers_global(sp.sympify(expr).doit())
        if e == 0:
            return sp.Integer(0)
        tog = sp.together(sp.expand_mul(e))
        num, den = sp.fraction(tog)
        num_exp = _normalize_exp_powers_global(sp.expand(num))
        if num_exp == 0:
            return sp.Integer(0)
        return num_exp / den

    # =========================================================================
    # THM 15: Non-Diagonal Curved 5-Bein Spinorial Tetrad Postulate, Lichnerowicz
    #         Weitzenbock Square (c_R5 = 1/4, c_Pauli = 1/2) & Gilkey Traces
    # =========================================================================
    t_sub = time.time()
    print("  [LOG] Module 4 -> Computing THM 15 (Non-Diagonal 5-Bein Lichnerowicz Square & Gilkey-DeWitt Traces)...")
    y0, y2, y1, y3, y4 = sp.symbols('y0 y2 y1 y3 y4', real=True)
    y_coords = [y0, y2, y1, y3, y4]

    alpha_e = sp.Function('alpha_e')(y0, y1)
    beta_e = sp.Function('beta_e')(y0, y1)
    phi_e = sp.Function('phi_e')(y0, y1)
    b1_e = sp.Function('b1_e')(y0, y1)

    e_frame = sp.Matrix([
        [alpha_e, 0, beta_e, 0, 0],
        [0, phi_e, 0, 0, 0],
        [0, 0, b1_e, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ])
    e_inv = e_frame.inv().applyfunc(exact_rat)
    g5_spin = (e_frame.T * eta5 * e_frame).applyfunc(exact_rat)
    sqrt_g5_spin = alpha_e * phi_e * b1_e
    G5_spin = Geometry(g5_spin, y_coords, sqrt_g5_spin)
    print(f"    [SUB-LOG THM 15.1] Curved non-diagonal 5-bein geometry G5_spin built in {time.time() - t_sub:.2f}s")

    t_sub = time.time()
    Omega_mat = []
    skew_omega_ok = True
    for M_i in range(5):
        mat_M = sp.zeros(4, 4)
        for a_f in range(5):
            for b_f in range(a_f + 1, 5):
                omega_mixed_ab = sum(
                    e_frame[a_f, N_i] * (
                        sp.diff(e_inv[N_i, b_f], y_coords[M_i])
                        + sum(G5_spin.Gamma[N_i][M_i][L_i] * e_inv[L_i, b_f] for L_i in range(5))
                    )
                    for N_i in range(5)
                )
                omega_mixed_ba = sum(
                    e_frame[b_f, N_i] * (
                        sp.diff(e_inv[N_i, a_f], y_coords[M_i])
                        + sum(G5_spin.Gamma[N_i][M_i][L_i] * e_inv[L_i, a_f] for L_i in range(5))
                    )
                    for N_i in range(5)
                )
                omega_cov_ab = exact_rat(eta5[a_f, a_f] * omega_mixed_ab)
                omega_cov_ba = exact_rat(eta5[b_f, b_f] * omega_mixed_ba)
                if exact_rat(omega_cov_ab + omega_cov_ba) != 0:
                    skew_omega_ok = False
                if omega_cov_ab != 0:
                    Sigma_ab = sp.Rational(1, 4) * (Gamma_hat[a_f] * Gamma_hat[b_f] - Gamma_hat[b_f] * Gamma_hat[a_f])
                    mat_M += omega_cov_ab * Sigma_ab
        Omega_mat.append(mat_M.applyfunc(exact_rat))

    Gamma_curved = [
        sum((e_inv[M_i, a_f] * Gamma_hat[a_f] for a_f in range(5)), sp.zeros(4, 4)).applyfunc(exact_rat)
        for M_i in range(5)
    ]

    tetrad_postulates_ok = True
    failed_tetrad_pair = None
    for M_i in range(5):
        for N_i in range(5):
            cov_G = (
                Gamma_curved[N_i].applyfunc(lambda ex: sp.diff(ex, y_coords[M_i]))
                + sum((G5_spin.Gamma[N_i][M_i][L_i] * Gamma_curved[L_i] for L_i in range(5)), sp.zeros(4, 4))
                + (Omega_mat[M_i] * Gamma_curved[N_i] - Gamma_curved[N_i] * Omega_mat[M_i])
            ).applyfunc(_zero_rat_fast)
            if cov_G != sp.zeros(4, 4):
                tetrad_postulates_ok = False
                failed_tetrad_pair = (M_i, N_i, cov_G)
                break
        if not tetrad_postulates_ok:
            break
    print(f"    [SUB-LOG THM 15.2] 25 curved spinorial tetrad postulates checked in {time.time() - t_sub:.2f}s "
          f"(skew_ok={skew_omega_ok}, tetrad_ok={tetrad_postulates_ok})")

    t_sub = time.time()
    q_e = sp.Symbol('q_e', real=True)
    A0_g = sp.Function('A0_g')(y0, y1)
    A1_g = sp.Function('A1_g')(y0, y1)
    A_gauge = [A0_g, sp.Integer(0), A1_g, sp.Integer(0), sp.Integer(0)]

    def _cov_spin_mat(M4, M_i):
        return (
            M4.applyfunc(lambda ex: sp.diff(ex, y_coords[M_i]))
            + (Omega_mat[M_i] - sp.I * q_e * A_gauge[M_i] * sp.eye(4)) * M4
        )

    def _dirac_mat(M4):
        res = sp.zeros(4, 4)
        for M_i in range(5):
            res += sp.I * Gamma_curved[M_i] * _cov_spin_mat(M4, M_i)
        return res

    f_sp = sp.Function('f_sp')(y0, y1)
    F_test_4 = f_sp * sp.eye(4)
    D2_on_F = _dirac_mat(_dirac_mat(F_test_4))
    DN_on_F = [_cov_spin_mat(F_test_4, N_i) for N_i in range(5)]
    bochner_on_F = sp.zeros(4, 4)
    for M_i in range(5):
        for N_i in range(5):
            if G5_spin.ginv[M_i, N_i] != 0:
                sec_MN_F = _cov_spin_mat(DN_on_F[N_i], M_i) - sum(
                    (G5_spin.Gamma[L_i][M_i][N_i] * DN_on_F[L_i] for L_i in range(5)), sp.zeros(4, 4)
                )
                bochner_on_F -= G5_spin.ginv[M_i, N_i] * sec_MN_F

    diff_op_full = (D2_on_F - bochner_on_F).applyfunc(_zero_rat_fast)
    E_endo = diff_op_full.subs({
        sp.diff(f_sp, y0, 2): 0, sp.diff(f_sp, y0, y1): 0, sp.diff(f_sp, y1, 2): 0,
        sp.diff(f_sp, y0): 0, sp.diff(f_sp, y1): 0, f_sp: 1
    }).applyfunc(exact_rat)
    zero_order_residual = (diff_op_full - f_sp * E_endo).applyfunc(_zero_rat_fast)

    F_02 = sp.diff(A1_g, y0) - sp.diff(A0_g, y1)
    Gamma_02 = sp.Rational(1, 2) * (Gamma_curved[0] * Gamma_curved[2] - Gamma_curved[2] * Gamma_curved[0])
    P_em_sum = 2 * sp.I * q_e * F_02 * Gamma_02
    E_grav_part = E_endo.subs(q_e, 0).applyfunc(exact_rat)

    d2phi_y1 = sp.diff(phi_e, y1, 2)
    dA1_y0 = sp.diff(A1_g, y0)
    c_R5_ded = exact_rat(sp.diff(sp.trace(E_grav_part), d2phi_y1) / sp.diff(4 * G5_spin.R, d2phi_y1))
    c_Pauli_ded = exact_rat(sp.diff(sp.trace((E_endo - E_grav_part) * P_em_sum), dA1_y0, 2) / sp.diff(sp.trace(P_em_sum * P_em_sum), dA1_y0, 2))
    lichnerowicz_check = (E_endo - (c_R5_ded * G5_spin.R * sp.eye(4) + c_Pauli_ded * P_em_sum)).applyfunc(_zero_rat_fast)
    print(f"    [SUB-LOG THM 15.3] Lichnerowicz operator square & Weitzenbock endomorphism verified in {time.time() - t_sub:.2f}s "
          f"(c_R5={c_R5_ded}, c_Pauli={c_Pauli_ded})")

    pE_syms = sp.symbols('pE0 pE1 pE2 pE3 pE4', real=True)
    Gamma_E = [sp.I * Gamma_hat[0], sp.I * Gamma_hat[1], Gamma_hat[2], Gamma_hat[3], Gamma_hat[4]]
    D_wick_sym = sum((pE_syms[a] * Gamma_E[a] for a in range(5)), sp.zeros(4, 4))
    wick_sq_diff = (D_wick_sym**2 - sum(p**2 for p in pE_syms) * sp.eye(4)).applyfunc(sp.expand)

    rk_S = Gamma_hat[0].rows  # 4 deduced from Cl(3,2) spinor rank!
    a0_tr = sp.trace(sp.eye(rk_S))
    a1_over_R5 = sp.simplify(sp.trace((sp.Rational(1, 6) - c_R5_ded) * sp.eye(rk_S)))

    # Constructive U(1) 2-form F_{23} = -F_{32} = B0_em => F_{AB} F^{AB} = 2 * B0_em^2 (Eq. 168):
    B0_em = sp.Symbol('B0_em', nonzero=True)
    F_MN_23 = 2 * B0_em * (sp.Rational(1, 2) * (Gamma_E[2] * Gamma_E[3] - Gamma_E[3] * Gamma_E[2]))
    E_F_mat = -c_Pauli_ded * sp.I * q_e * F_MN_23
    Omega_F_sq_sum = 2 * ((-sp.I * q_e * B0_em * sp.eye(rk_S)) ** 2)
    F_sq_scalar = 2 * B0_em**2
    a2_gauge_coeff = sp.simplify(
        sp.trace(sp.Rational(1, 2) * E_F_mat**2 + sp.Rational(1, 12) * Omega_F_sq_sum) / (q_e**2 * F_sq_scalar)
    )
    eh_to_maxwell_ratio = sp.simplify(sp.Abs(a1_over_R5) / a2_gauge_coeff)

    # Pure Cl(3,2) bivector trace deduction of the spin-curvature trace coefficient (1/2)*Tr(Sigma^{ab} Sigma_{ab}) == -1/2:
    bivector_traces = []
    for a_f in range(5):
        for b_f in range(a_f + 1, 5):
            Sigma_up = sp.Rational(1, 4) * (Gamma_hat[a_f] * Gamma_hat[b_f] - Gamma_hat[b_f] * Gamma_hat[a_f])
            Sigma_dn = eta5[a_f, a_f] * eta5[b_f, b_f] * Sigma_up
            bivector_traces.append(sp.Rational(1, 2) * sp.trace(Sigma_up * Sigma_dn))
    c_spin_curv_ded = bivector_traces[0]

    c_K5_gilkey = rk_S * sp.Rational(1, 180) + sp.Rational(1, 12) * c_spin_curv_ded
    c_R2_gilkey = -rk_S * sp.Rational(1, 180)
    c_Rsq_gilkey = rk_S * (sp.Rational(1, 72) - sp.Rational(1, 6) * c_R5_ded + sp.Rational(1, 2) * c_R5_ded**2)

    thm15_ok = _require_all("THM 15", [
        ("skew_omega_ok (omega_{M ab} == -omega_{M ba})", skew_omega_ok, skew_omega_ok),
        ("tetrad_postulates_ok", tetrad_postulates_ok, failed_tetrad_pair),
        ("zero_order_residual == 0_4", zero_order_residual == sp.zeros(4, 4), zero_order_residual),
        ("lichnerowicz_check == 0_4", lichnerowicz_check == sp.zeros(4, 4), lichnerowicz_check),
        ("c_R5_ded == 1/4", c_R5_ded == sp.Rational(1, 4), c_R5_ded),
        ("c_Pauli_ded == 1/2", c_Pauli_ded == sp.Rational(1, 2), c_Pauli_ded),
        ("all 10 Cl(3,2) bivector traces == -1/2", all(c_p == -sp.Rational(1, 2) for c_p in bivector_traces), bivector_traces),
        ("eh_to_maxwell_ratio == 1/2", eh_to_maxwell_ratio == sp.Rational(1, 2), eh_to_maxwell_ratio),
    ])

    state.deduce_theorem(
        "curved_non_diagonal_lichnerowicz_and_gilkey",
        {"c_R5": c_R5_ded, "c_Pauli": c_Pauli_ded, "c_spin_curv": c_spin_curv_ded,
         "a0": a0_tr, "a1_over_R5": a1_over_R5, "a2_gauge_coeff": a2_gauge_coeff,
         "eh_to_maxwell_ratio": eh_to_maxwell_ratio,
         "gilkey_grav_weights": (c_K5_gilkey, c_R2_gilkey, c_Rsq_gilkey)},
        "Non-diagonal curved 5-bein Lichnerowicz square (c_R5=1/4, c_Pauli=1/2), Wick symbol & Cl(3,2) Gilkey traces",
        thm15_ok
    )
    state.deduce_corollary("curved_non_diagonal_lichnerowicz_and_gilkey", "non_diagonal_tetrad_postulates",
                           (tetrad_postulates_ok, zero_order_residual), (True, sp.zeros(4, 4)),
                           "All 25 non-diagonal curved spinorial tetrad postulates nabla_M^{spin} Gamma^N == 0_4 verified")
    state.deduce_corollary("curved_non_diagonal_lichnerowicz_and_gilkey", "lichnerowicz_pauli_and_wick_ellipticity",
                           (lichnerowicz_check, wick_sq_diff, c_R5_ded, c_Pauli_ded),
                           (sp.zeros(4, 4), sp.zeros(4, 4), sp.Rational(1, 4), sp.Rational(1, 2)),
                           "Exact Lichnerowicz-Weitzenbock endomorphism +(1/4)R^(5) I_4 + (i q_e/2) F_{AB} Gamma^{AB} & elliptic Wick ||p||^2 I_4")
    state.deduce_corollary("curved_non_diagonal_lichnerowicz_and_gilkey", "gilkey_dewitt_spectral_weights",
                           (a0_tr, a1_over_R5, a2_gauge_coeff, eh_to_maxwell_ratio, c_spin_curv_ded,
                            (c_K5_gilkey, c_R2_gilkey, c_Rsq_gilkey)),
                           (4, -sp.Rational(1, 3), sp.Rational(2, 3), sp.Rational(1, 2), -sp.Rational(1, 2),
                            (-sp.Rational(7, 360), -sp.Rational(1, 45), sp.Rational(1, 72))),
                           "Universal EH-to-Maxwell trace ratio 1/2, Cl(3,2) bivector trace -1/2 & Gilkey weights (-7/360 K_5, -1/45 R_2, +1/72 R^2)")

    # =========================================================================
    # THM 16: Non-Diagonal Quadratic Curvature Splits (K_5, R_2^(5)), Ostrogradsky-Free
    #         Lovelock-Gauss-Bonnet Divergence (c_GB = 8), GHY Trace & Throat Holonomy
    # =========================================================================
    t_sub = time.time()
    print("  [LOG] Module 4 -> Computing THM 16 (Non-Diagonal K_5, R_2^(5), Lovelock G_5 == 8 div(G dPhi) & GHY)...")
    t_q, r_q, th_q, ph_q, x2_q = sp.symbols('t_q r_q th_q ph_q x2_q', real=True)
    N_q = sp.Function('N_q')(t_q, r_q)
    v_q = sp.Function('v_q')(t_q, r_q)
    B_q = sp.Function('B_q')(t_q, r_q)
    Phi_q = sp.Function('Phi_q')(t_q, r_q)

    g4_q = sp.Matrix([
        [-(N_q**2 - B_q**2 * v_q**2), B_q**2 * v_q, 0, 0],
        [B_q**2 * v_q, B_q**2, 0, 0],
        [0, 0, r_q**2, 0],
        [0, 0, 0, r_q**2 * sp.sin(th_q)**2],
    ])
    sqrt_g4_q = N_q * B_q * r_q**2 * sp.sin(th_q)
    G4_q = Geometry(g4_q, [t_q, r_q, th_q, ph_q], sqrt_g4_q)

    g5_q = sp.zeros(5, 5)
    g5_q[1, 1] = -Phi_q**2
    m_idx = [0, 2, 3, 4]
    for a_i in range(4):
        for b_i in range(4):
            g5_q[m_idx[a_i], m_idx[b_i]] = g4_q[a_i, b_i]
    sqrt_g5_q = Phi_q * sqrt_g4_q
    G5_q = Geometry(g5_q, [t_q, x2_q, r_q, th_q, ph_q], sqrt_g5_q)

    def _kretschmann_fast(geo):
        n_d = geo.n
        pairs = [(a_i, b_i) for a_i in range(n_d) for b_i in range(a_i + 1, n_d)]
        nz_riem = {
            (a_i, b_i, c_i, d_i): geo.Riem[a_i][b_i][c_i][d_i]
            for a_i, b_i in pairs for c_i, d_i in pairs
            if geo.Riem[a_i][b_i][c_i][d_i] != 0
        }
        val = sp.Integer(0)
        for (a_i, b_i, c_i, d_i), r_abcd in nz_riem.items():
            r_up = sp.Integer(0)
            for (e_i, f_i, g_i, h_i), r_efgh in nz_riem.items():
                g_ab_ef = geo.ginv[a_i, e_i] * geo.ginv[b_i, f_i] - geo.ginv[a_i, f_i] * geo.ginv[b_i, e_i]
                if g_ab_ef != 0:
                    g_cd_gh = geo.ginv[c_i, g_i] * geo.ginv[d_i, h_i] - geo.ginv[c_i, h_i] * geo.ginv[d_i, g_i]
                    if g_cd_gh != 0:
                        r_up += g_ab_ef * g_cd_gh * r_efgh
            val += 4 * r_abcd * r_up
        return _zero_rat_fast(val)

    def _ricci_sq_fast(geo):
        Ric_mix = geo.ginv * geo.Ric
        return _zero_rat_fast(sp.trace(Ric_mix * Ric_mix))

    K4_q = _kretschmann_fast(G4_q)
    K5_q = _kretschmann_fast(G5_q)
    R2_4_q = _ricci_sq_fast(G4_q)
    R2_5_q = _ricci_sq_fast(G5_q)

    H_Phi_q = G4_q.hessian_cov(Phi_q)
    H_mix_q = G4_q.ginv * H_Phi_q
    H_sq_q = _zero_rat_fast(sp.trace(H_mix_q * H_mix_q))
    box_Phi_q = G4_q.box(Phi_q)
    Ric_dot_H_q = _zero_rat_fast(sp.trace((G4_q.ginv * G4_q.Ric) * H_mix_q))

    ddPhi_rr = sp.diff(Phi_q, r_q, 2)
    K5_diff_raw = _zero_rat_fast(K5_q - K4_q)
    c_K5_hess = exact_rat(sp.diff(K5_diff_raw, ddPhi_rr, 2) / sp.diff(H_sq_q / Phi_q**2, ddPhi_rr, 2))
    K5_split_res = _zero_rat_fast(K5_diff_raw - c_K5_hess * (H_sq_q / Phi_q**2))
    R2_5_diff = _zero_rat_fast(R2_5_q - (R2_4_q - (2 / Phi_q) * Ric_dot_H_q + (H_sq_q + box_Phi_q**2) / Phi_q**2))

    GB4_q = K4_q - 4 * R2_4_q + G4_q.R**2
    GB5_q = K5_q - 4 * R2_5_q + G5_q.R**2
    coords4_q = [t_q, r_q, th_q, ph_q]
    Ein4_up = G4_q.ginv * G4_q.Ein * G4_q.ginv
    div_Ein_dPhi = sum(
        sp.diff(sqrt_g4_q * sum(Ein4_up[a_i, b_i] * sp.diff(Phi_q, coords4_q[b_i]) for b_i in range(4)), coords4_q[a_i])
        for a_i in range(4)
    )
    gb_excess = _zero_rat_fast(sqrt_g5_q * GB5_q - Phi_q * sqrt_g4_q * GB4_q)
    c_GB_ded = exact_rat(sp.diff(gb_excess, ddPhi_rr) / sp.diff(div_Ein_dPhi, ddPhi_rr))
    lovelock_diff = _zero_rat_fast(gb_excess - c_GB_ded * div_Ein_dPhi)

    # Extrinsic traces K = nabla_A n^A of the r = const hypersurfaces, from the covariant divergence of the unit normal
    # n^A = g^{A r} / sqrt(g^{rr}) on the non-diagonal node metrics G4_q and G5_q:
    Phi_inf, Phi_s = sp.symbols('Phi_inf Phi', positive=True)
    coords5_q = [t_q, x2_q, r_q, th_q, ph_q]
    r_idx4, r_idx5 = coords4_q.index(r_q), coords5_q.index(r_q)
    n4_up = [G4_q.ginv[a_i, r_idx4] / sp.sqrt(G4_q.ginv[r_idx4, r_idx4]) for a_i in range(4)]
    n5_up = [G5_q.ginv[A_i, r_idx5] / sp.sqrt(G5_q.ginv[r_idx5, r_idx5]) for A_i in range(5)]
    n4_norm = sp.simplify(sum(G4_q.g[a_i, b_i] * n4_up[a_i] * n4_up[b_i] for a_i in range(4) for b_i in range(4)))
    n5_norm = sp.simplify(sum(G5_q.g[A_i, B_i] * n5_up[A_i] * n5_up[B_i] for A_i in range(5) for B_i in range(5)))
    K4_ghy = sum(sp.diff(sqrt_g4_q * n4_up[a_i], coords4_q[a_i]) for a_i in range(4)) / sqrt_g4_q
    K5_ghy = sum(sp.diff(sqrt_g5_q * n5_up[A_i], coords5_q[A_i]) for A_i in range(5)) / sqrt_g5_q
    n_dot_dlnPhi = sum(n4_up[a_i] * sp.diff(Phi_q, coords4_q[a_i]) for a_i in range(4)) / Phi_q
    ghy_split_diff = sp.simplify(K5_ghy - (K4_ghy + n_dot_dlnPhi))

    # Holonomy: by the split above, 2 (K^(5) - K^(4)) = 2 d ln(Phi) / d rho along the unit normal (proper distance rho);
    # integrated over n reflections between a core boundary (Phi = 1) and the asymptotic scale (Phi = Phi_inf):
    n_refl = sp.Symbol('n_refl', integer=True, positive=True)
    rho_h, rho_core, rho_vac = sp.symbols('rho_h rho_core rho_vac', positive=True)
    Phi_prof = sp.Function('Phi_prof')
    holonomy_integrand = 2 * n_refl * sp.diff(sp.log(Phi_prof(rho_h)), rho_h)
    holonomy_antiderivative = 2 * n_refl * sp.log(Phi_prof(rho_h))
    holonomy_ftc_check = sp.simplify(sp.diff(holonomy_antiderivative, rho_h) - holonomy_integrand)
    delta_ell_n = (holonomy_antiderivative.subs(rho_h, rho_vac) - holonomy_antiderivative.subs(rho_h, rho_core)).subs(
        {Phi_prof(rho_core): 1, Phi_prof(rho_vac): Phi_inf})
    holonomy_n = sp.simplify(sp.exp(delta_ell_n))
    print(f"    [SUB-LOG THM 16.1] Non-diagonal K_5, R_2^(5) & Lovelock c_GB={c_GB_ded} checked in {time.time() - t_sub:.2f}s")

    thm16_ok = _require_all("THM 16", [
        ("c_K5_hess == 4", c_K5_hess == 4, c_K5_hess),
        ("K5_split_res == 0", K5_split_res == 0, K5_split_res),
        ("R2_5_diff == 0", R2_5_diff == 0, R2_5_diff),
        ("c_GB_ded == 8", c_GB_ded == 8, c_GB_ded),
        ("lovelock_diff == 0", lovelock_diff == 0, lovelock_diff),
        ("unit normals & ghy_split_diff == 0", (n4_norm, n5_norm, ghy_split_diff, holonomy_ftc_check) == (1, 1, 0, 0), (n4_norm, n5_norm, ghy_split_diff, holonomy_ftc_check)),
    ])

    state.deduce_theorem(
        "curved_quadratic_invariants_lovelock_and_ghy",
        {"c_K5_hess": c_K5_hess, "c_GB": c_GB_ded, "holonomy_n": holonomy_n},
        "Non-diagonal curved K_5, R_2^(5), Ostrogradsky-free Lovelock cancellation (c_GB=8), GHY & holonomy",
        thm16_ok
    )
    state.deduce_corollary("curved_quadratic_invariants_lovelock_and_ghy", "quadratic_kretschmann_and_ricci_splits",
                           (c_K5_hess, K5_split_res, R2_5_diff), (4, 0, 0),
                           "Exact non-diagonal K_5 == K_4 + (4/Phi^2) ||nabla^2 Phi||^2 & R_2^(5) Hessian-d'Alembertian split")
    state.deduce_corollary("curved_quadratic_invariants_lovelock_and_ghy", "lovelock_ostrogradsky_cancellation",
                           (c_GB_ded, lovelock_diff), (8, 0),
                           "Ostrogradsky-free Lovelock identity sqrt|g_5| G_5 == Phi sqrt(-g_4) G_4 + 8 d_mu(sqrt(-g_4) G^(4)mu_nu d_nu Phi)")
    state.deduce_corollary("curved_quadratic_invariants_lovelock_and_ghy", "ghy_extrinsic_split_and_holonomy",
                           (n4_norm, n5_norm, ghy_split_diff, sp.simplify(holonomy_n - Phi_inf**(2 * n_refl))), (1, 1, 0, 0),
                           "K = nabla_A n^A on the node metrics: K^(5) == K^(4) + n^mu d_mu ln(Phi) & n-reflection holonomy Phi_inf^{2n}")

    # =========================================================================
    # THM 17: 50-Digit Poisson-Jacobi Theta Duality, Deductive Riemann Functional
    #         (Euler-Maclaurin continuation of zeta) Ray-Singer Torsion & AXIOM 2 Mellin UV Selection of M_2^{(-)}
    # =========================================================================
    t_sub = time.time()
    print("  [LOG] Module 4 -> Computing THM 17 (50-Digit Poisson-Jacobi Duality, Deductive Ray-Singer & Mellin UV Filter)...")
    mp.dps = 55
    s_val = mp.mpf('0.17')
    L0_val = mp.mpf('1.3')
    Phi_val = mp.mpf('1.1')

    kk_untwisted = 1 + 2 * mp.nsum(lambda n: mp.exp(-s_val * (n / (L0_val * Phi_val))**2), [1, mp.inf])
    wind_untwisted = (mp.sqrt(mp.pi) * L0_val * Phi_val / mp.sqrt(s_val)) * (
        1 + 2 * mp.nsum(lambda w: mp.exp(-((mp.pi * L0_val * Phi_val * w)**2) / s_val), [1, mp.inf])
    )
    kk_twisted = 2 * mp.nsum(lambda n: mp.exp(-s_val * ((n - mp.mpf('0.5')) / (L0_val * Phi_val))**2), [1, mp.inf])
    wind_twisted = (mp.sqrt(mp.pi) * L0_val * Phi_val / mp.sqrt(s_val)) * (
        1 + 2 * mp.nsum(lambda w: mp.cos(mp.pi * w) * mp.exp(-((mp.pi * L0_val * Phi_val * w)**2) / s_val), [1, mp.inf])
    )
    err_untwisted = abs(kk_untwisted - wind_untwisted)
    err_twisted = abs(kk_twisted - wind_twisted)
    jacobi_50digit_ok = (err_untwisted < mp.mpf('1e-48')) and (err_twisted < mp.mpf('1e-48'))

    L0_s, z_rs = sp.symbols('L0 z_rs', positive=True)

    # zeta(0) and zeta'(0) from the Euler-Maclaurin representation
    #     zeta(s) = 1/(s - 1) + 1/2 - s J(s),    J(s) = int_1^oo ({x} - 1/2) x^{-s-1} dx.
    # (i)  For Re s > 1 it is deduced from the Dirichlet series, interval by interval [n, n+1].
    # (ii) J(s) converges for Re s > -1, so the right-hand side is the analytic continuation of zeta there.
    #      (Uniqueness of analytic continuation is the single analytic input of this block.)
    n_em = sp.Symbol('n_em', integer=True, positive=True)
    N_em = sp.Symbol('N_em', integer=True, positive=True)
    x_em, sig_em = sp.symbols('x_em sigma_em', positive=True)
    s_em = 1 + sig_em                                                    # Re s > 1
    F_em = (x_em - n_em - sp.Rational(1, 2)) * x_em**(-s_em) / (-s_em) - x_em**(1 - s_em) / ((-s_em) * (1 - s_em))
    G_em = x_em**(1 - s_em) / (1 - s_em)
    em_antiderivative_checks = (
        sp.simplify(sp.diff(F_em, x_em) - (x_em - n_em - sp.Rational(1, 2)) * x_em**(-s_em - 1)),
        sp.simplify(sp.diff(G_em, x_em) - x_em**(-s_em)),
    )
    j_n_em = F_em.subs(x_em, n_em + 1) - F_em.subs(x_em, n_em)
    em_interval_identity = sp.cancel(sp.together(
        -s_em * j_n_em - (sp.Rational(1, 2) * (n_em + 1)**(-s_em) + sp.Rational(1, 2) * n_em**(-s_em)
                          - (G_em.subs(x_em, n_em + 1) - G_em.subs(x_em, n_em)))
    ))
    em_half_sum = sp.expand_func(
        sp.Rational(1, 2) * sp.summation((n_em + 1)**(-s_em), (n_em, 1, sp.oo))
        + sp.Rational(1, 2) * sp.summation(n_em**(-s_em), (n_em, 1, sp.oo))
    )
    em_tail_integral = sp.limit(G_em, x_em, sp.oo) - G_em.subs(x_em, 1)
    Z_em, J_em_sym = sp.symbols('Z_em J_em_sym')
    zeta_rep_Re_gt_1 = sp.solve(sp.Eq(-s_em * J_em_sym, em_half_sum.subs(sp.zeta(s_em), Z_em) - em_tail_integral), Z_em)[0]
    s_cont = sp.Symbol('s_cont')
    J_em_fn = sp.Function('J_em')
    zeta_rep = 1 / (s_cont - 1) + sp.Rational(1, 2) - s_cont * J_em_fn(s_cont)
    em_representation_diff = sp.simplify(zeta_rep_Re_gt_1 - zeta_rep.subs(J_em_fn(s_cont), J_em_sym).subs(s_cont, s_em))

    # J(0) = sum_n int_n^{n+1} (x - n - 1/2) / x dx: closed partial sums I_N proved by induction, then N -> oo
    F0_em = x_em - (n_em + sp.Rational(1, 2)) * sp.log(x_em)
    em_antiderivative_0_check = sp.simplify(sp.diff(F0_em, x_em) - (x_em - n_em - sp.Rational(1, 2)) / x_em)
    j_n_0 = F0_em.subs(x_em, n_em + 1) - F0_em.subs(x_em, n_em)
    I_N_em = (N_em - 1) - (N_em + sp.Rational(1, 2)) * sp.log(N_em) + sp.log(sp.factorial(N_em))
    em_induction_step = sp.simplify(
        sp.expand_log(sp.expand_func(sp.combsimp(I_N_em.subs(N_em, N_em + 1) - I_N_em)), force=True)
        - sp.expand_log(j_n_0.subs(n_em, N_em), force=True)
    )
    em_induction_base = sp.simplify(I_N_em.subs(N_em, 1))
    J_em_at_0 = sp.limit((N_em - 1) - (N_em + sp.Rational(1, 2)) * sp.log(N_em) + sp.loggamma(N_em + 1), N_em, sp.oo)

    zeta_0_ded = zeta_rep.subs(s_cont, 0)
    zeta_prime_0_ded = sp.simplify(sp.diff(zeta_rep, s_cont).subs(s_cont, 0).doit().subs(J_em_fn(0), J_em_at_0))

    ray_singer_raw = -(sp.diff((L0_s * Phi_s)**(2 * z_rs), z_rs).subs(z_rs, 0) * zeta_0_ded + 2 * zeta_prime_0_ded)
    ray_singer_orb = sp.logcombine(sp.expand(ray_singer_raw), force=True)
    ray_singer_diff = sp.simplify(sp.exp(ray_singer_orb) - 2 * sp.pi * L0_s * Phi_s)

    # Deductive Poisson/Gaussian winding kernel, Z_2 characters, Mellin integrals & series sums (Eqs. 103-107):
    wm = _derive_winding_moments(D_ded, L0_s, Phi_s)
    s_m = wm['symbols']['s']
    w_m = wm['symbols']['w']
    u_mel = sp.Symbol('u_mel', positive=True)
    mellin_single_w = [
        sp.simplify(sp.integrate(u_mel**(k_idx - 1 - sp.Rational(D_ded, 2)) * sp.exp(-w_m**2 / u_mel), (u_mel, 0, sp.oo)))
        for k_idx in range(3)
    ]

    p_exponents = wm['p_exponents']
    untwisted_k2 = wm['untwisted_sum'][2]
    untwisted_k2_diverges = (untwisted_k2 in (sp.oo, sp.zoo))

    M0_untwisted = sp.simplify(wm['M_untwisted'][0])
    M1_untwisted = sp.simplify(wm['M_untwisted'][1])
    V_wind_5d = sp.simplify(-sp.Rational(1, 2) * M0_untwisted)

    # Eq. (107) defines M_k^{(-)} with an explicit minus sign [-Theta~_wind^{(-)}] = sum_{w>=1} (-1)^{w-1} w^{-p_k}:
    twisted_sums = [-ts for ts in wm['twisted_sum']]
    M2_twisted = sp.simplify(-wm['M_twisted'][2])

    m0_diff = sp.simplify(M0_untwisted - 3 * sp.zeta(5) / (64 * sp.pi**6 * (L0_s * Phi_s)**4))
    m1_diff = sp.simplify(M1_untwisted - sp.zeta(3) / (32 * sp.pi**4 * (L0_s * Phi_s)**2))
    vwind_diff = sp.simplify(V_wind_5d - (-3 * sp.zeta(5) / (128 * sp.pi**6 * (L0_s * Phi_s)**4)))
    m2_diff = sp.simplify(M2_twisted - sp.log(2) / (16 * sp.pi**2))
    print(f"    [SUB-LOG THM 17.1] Jacobi 50-digit err=({float(err_untwisted):.1e}, {float(err_twisted):.1e}), "
          f"Ray-Singer={ray_singer_orb} in {time.time() - t_sub:.2f}s")

    thm17_ok = _require_all("THM 17", [
        ("jacobi_50digit_ok", jacobi_50digit_ok, (err_untwisted, err_twisted)),
        ("Euler-Maclaurin representation of zeta deduced from the Dirichlet series (Re s > 1)",
         (em_antiderivative_checks, em_interval_identity, em_representation_diff) == ((0, 0), 0, 0),
         (em_antiderivative_checks, em_interval_identity, em_representation_diff)),
        ("J(0) partial sums by induction", (em_antiderivative_0_check, em_induction_step, em_induction_base) == (0, 0, 0),
         (em_antiderivative_0_check, em_induction_step, em_induction_base)),
        ("zeta'(0) == -(1/2) ln(2 pi)", sp.simplify(zeta_prime_0_ded + sp.log(2 * sp.pi) / 2) == 0, zeta_prime_0_ded),
        ("zeta_0_ded == -1/2", zeta_0_ded == -sp.Rational(1, 2), zeta_0_ded),
        ("ray_singer_diff == 0", ray_singer_diff == 0, ray_singer_orb),
        ("K_sin == 0 & Z_2 characters (1, (-1)^w)",
         (wm['K_sin'], wm['char_periodic'], sp.simplify(wm['char_antiperiodic'] - (-1)**w_m)) == (0, 1, 0),
         (wm['K_sin'], wm['char_periodic'], wm['char_antiperiodic'])),
        ("p_exponents == [5, 3, 1]", p_exponents == [5, 3, 1], p_exponents),
        ("untwisted_k2_diverges", untwisted_k2_diverges, untwisted_k2),
        ("(m0_diff, m1_diff, vwind_diff, m2_diff) == (0,0,0,0)",
         (m0_diff, m1_diff, vwind_diff, m2_diff) == (0, 0, 0, 0),
         (m0_diff, m1_diff, vwind_diff, m2_diff)),
    ])

    state.deduce_theorem(
        "poisson_jacobi_ray_singer_and_uv_twist",
        {"zeta_0": zeta_0_ded, "zeta_prime_0": zeta_prime_0_ded, "ray_singer": ray_singer_orb,
         "p_exponents": tuple(p_exponents), "M0_untwisted": M0_untwisted, "M1_untwisted": M1_untwisted,
         "V_wind_5d": V_wind_5d, "twisted_moments": tuple(twisted_sums), "M2_twisted": M2_twisted},
        "50-digit Jacobi duality, Ray-Singer ln(2 pi L_0 Phi) from the Euler-Maclaurin continuation of zeta & AXIOM 2 selection of M_2^{(-)} = ln(2)/(16 pi^2) (sign by the convention of Eq. 107)",
        thm17_ok
    )
    state.deduce_corollary("poisson_jacobi_ray_singer_and_uv_twist", "jacobi_duality_and_ray_singer",
                           (jacobi_50digit_ok, em_interval_identity, em_representation_diff, em_induction_step, zeta_0_ded,
                            sp.simplify(sp.exp(-2 * zeta_prime_0_ded) - 2 * sp.pi),
                            ray_singer_diff),
                           (True, 0, 0, 0, -sp.Rational(1, 2), 0, 0),
                           "50-digit Jacobi duality; zeta(s) == 1/(s-1) + 1/2 - s J(s) from the Dirichlet series; zeta(0)==-1/2, zeta'(0)==-(1/2)ln(2 pi) & -zeta_orb'(0)==ln(2 pi L_0 Phi)")
    state.deduce_corollary("poisson_jacobi_ray_singer_and_uv_twist", "mellin_winding_filter_axiom2",
                           (mellin_single_w[2] * w_m, untwisted_k2_diverges,
                            (m0_diff, m1_diff, vwind_diff, m2_diff),
                            sp.simplify(twisted_sums[0] - sp.Rational(15, 16) * sp.zeta(5)),
                            sp.simplify(twisted_sums[1] - sp.Rational(3, 4) * sp.zeta(3)),
                            sp.simplify(twisted_sums[2] - sp.log(2))),
                           (sp.sqrt(sp.pi), True, (0, 0, 0, 0), 0, 0, 0),
                           "5D Mellin integrals M_0, M_1, V_wind (Eqs. 103, 106), untwisted k=2 divergence & twisted M_2^{(-)} == ln(2)/(16 pi^2)")

    # =========================================================================
    # THM 18: Canonical (W_3 <-> W_3^*) Hermite-Jet Bases (det M_jet^raw = -16,
    #         det M_jet = -32/3, Eq. 108), Cauchy-Bernstein Semigroup Character
    #         & Radon-Plancherel 3D Ball Convolution (Delta f_mut^(4)=-30, Delta f^(4)=+36)
    # =========================================================================
    t_sub = time.time()
    print("  [LOG] Module 4 -> Computing THM 18 (W_3/W_3^* Bases, Eq. 108, Cauchy-Bernstein Character & Plancherel Convolution)...")
    psi = sp.Symbol('psi', real=True)
    # Pure deduction of W_3 basis exponents (-2, -4, -6) from THM 17 Mellin exponents p_exponents = [5, 3, 1]
    # and THM 12 4D Weyl volume Jacobian exp(-(D - 1)/2 * c_Weyl * psi) = exp(-2 * psi):
    weyl_vol_exp = -sp.Rational(D_ded - 1, 2) * c_Weyl_ded
    phi_raw = [sp.exp((1 - p_k + weyl_vol_exp) * psi) for p_k in reversed(p_exponents)]

    M_jet_raw = sp.Matrix(3, 3, lambda a_i, b_i: sp.diff(phi_raw[b_i], (psi, a_i)).subs(psi, 0))
    det_M_jet_raw = M_jet_raw.det()  # Exact Vandermonde det on (-2, -4, -6) == -16
    M_jet_raw_inv = M_jet_raw.inv()

    # Scaled Taylor-jet evaluation matrix j_0^2[V_spec] = (V_spec(0), V_spec'(0), (-1/kappa_psi) V_spec''(0))^T (Eq. 108):
    c_mass_scale = sp.Integer(1) / (-kappa_psi_ded)  # 2/3 deduced from kappa_psi = -3/2 (THM 12)
    M_jet_taylor = sp.diag(1, 1, c_mass_scale) * M_jet_raw
    det_M_jet_taylor = M_jet_taylor.det()  # -32/3

    Pi_primal = [sp.factor(sum(phi_raw[c_i] * M_jet_raw_inv[c_i, b_i] for c_i in range(3))) for b_i in range(3)]
    biorth_primal_check = sp.Matrix(3, 3, lambda a_i, b_i: sp.simplify(sp.diff(Pi_primal[b_i], (psi, a_i)).subs(psi, 0)))

    V0_s, m_psi_s, psi_t_s = sp.symbols('V0 m_psi psi_t', real=True)
    # Unique algebraic solution of M_jet_taylor * (c0, c1, c2)^T = (3*V0, 0, m_psi^2)^T (Eq. 108):
    c_vec_sol = M_jet_taylor.inv() * sp.Matrix([3 * V0_s, 0, m_psi_s**2])
    expected_c_vec = sp.Matrix([
        9 * V0_s + sp.Rational(3, 16) * m_psi_s**2,
        -9 * V0_s - sp.Rational(3, 8) * m_psi_s**2,
        3 * V0_s + sp.Rational(3, 16) * m_psi_s**2,
    ])
    eq108_coeff_diff = (c_vec_sol - expected_c_vec).applyfunc(sp.simplify)

    mu0_c, mu1_c, mu2_c = sp.symbols('mu0_c mu1_c mu2_c', real=True)
    M_jet_scaled = M_jet_raw * sp.diag(mu0_c, mu1_c, mu2_c)
    Pi_scaled = [
        sp.simplify(sum((phi_raw[c_i] * [mu0_c, mu1_c, mu2_c][c_i]) * (M_jet_scaled.inv())[c_i, b_i] for c_i in range(3)))
        for b_i in range(3)
    ]
    cutoff_indep_diff = [sp.simplify(Pi_scaled[b_i] - Pi_primal[b_i]) for b_i in range(3)]

    dpsi_shift = psi - psi_t_s
    V_spec_full = 3 * V0_s * Pi_primal[0].subs(psi, dpsi_shift) + (-kappa_psi_ded) * m_psi_s**2 * Pi_primal[2].subs(psi, dpsi_shift)
    V_spec_from_c = sum(c_vec_sol[c_i] * phi_raw[c_i].subs(psi, dpsi_shift) for c_i in range(3))
    vspec_eq108_match = exact_rat(V_spec_full - V_spec_from_c)

    jet_at_min = (
        sp.simplify(V_spec_full.subs(psi, psi_t_s)),
        sp.simplify(sp.diff(V_spec_full, psi).subs(psi, psi_t_s)),
        sp.simplify(sp.diff(V_spec_full, (psi, 2)).subs(psi, psi_t_s)),
        sp.simplify((-sp.Integer(1) / kappa_psi_ded) * sp.diff(V_spec_full, (psi, 2)).subs(psi, psi_t_s)),
    )

    u_w, v_w = sp.symbols('u_w v_w', nonnegative=True)
    lam_A = sp.Symbol('lam_A', positive=True)
    F_w = sp.Function('F_whitney')
    cauchy_semigroup_eq = F_w(u_w + v_w) - F_w(u_w) * F_w(v_w)
    cauchy_ode_expr = sp.diff(cauchy_semigroup_eq, v_w).subs(v_w, 0).subs(
        sp.Subs(sp.Derivative(F_w(v_w), v_w), v_w, 0), -lam_A
    ).doit()
    F_bernstein = sp.dsolve(sp.Eq(cauchy_ode_expr, 0), F_w(u_w), ics={F_w(0): 1}).rhs
    A_sym = sp.Symbol('A_sym', real=True)
    psi0_amp = sp.Symbol('psi0_amp', real=True)
    psi_t_canonical = psi0_amp * F_bernstein.subs(u_w, A_sym**2)

    whitney_parity_diff = sp.simplify(psi_t_canonical.subs(A_sym, -A_sym) - psi_t_canonical)
    horizon_slope_zero = sp.simplify(sp.diff(psi_t_canonical, A_sym).subs(A_sym, 0))
    bernstein_cm_checks = [
        sp.simplify(((-1)**k_d * sp.diff(F_bernstein, (u_w, k_d))) / (lam_A**k_d * F_bernstein)) - 1
        for k_d in range(1, 4)
    ]

    # Constructive Radon slice projection R[rho_B](z) of a unit-mass 3D ball of radius R=1:
    k_p, r_p = sp.symbols('k_p r_p', positive=True)
    z_rad, s_cyl = sp.symbols('z_rad s_cyl', real=True)
    vol_unit_ball = sp.integrate(4 * sp.pi * r_p**2, (r_p, 0, 1))
    rho_0_ball = sp.Integer(1) / vol_unit_ball
    radon_ball = sp.simplify(sp.integrate(rho_0_ball * 2 * sp.pi * s_cyl, (s_cyl, 0, sp.sqrt(1 - z_rad**2))))
    rho_hat_k = sp.simplify(sp.integrate(rho_0_ball * 4 * sp.pi * r_p**2 * sp.sin(k_p * r_p) / (k_p * r_p), (r_p, 0, 1)))
    radon_ft_diff = sp.simplify(sp.integrate(radon_ball * sp.cos(k_p * z_rad), (z_rad, -1, 1)) - rho_hat_k)

    u_sep, d_sep, lam_b = sp.symbols('u_sep d_sep lam_b', positive=True)
    C_rad_in = sp.expand(sp.integrate(radon_ball * radon_ball.subs(z_rad, z_rad + u_sep), (z_rad, -1, 1 - u_sep)))
    W0_self = sp.simplify(2 * C_rad_in.subs(u_sep, 0))
    M_in_d = sp.expand(sp.integrate(2 * C_rad_in, (u_sep, 0, d_sep)))
    M_out_const = sp.simplify(M_in_d.subs(d_sep, 2))
    W_in_exact = sp.expand(M_in_d / d_sep)
    W_out_exact = M_out_const / d_sep

    f_in_planch = sp.expand(W_in_exact.subs(d_sep, 2 * lam_b) / W0_self)
    f_out_planch = sp.simplify(W_out_exact.subs(d_sep, 2 * lam_b) / W0_self)
    f_comp_in = sp.expand(W0_self * (1 - f_in_planch))
    f_comp_out = sp.simplify(W0_self * (1 - f_out_planch))

    expected_f_in = 1 - sp.Rational(5, 3) * lam_b**2 + sp.Rational(5, 4) * lam_b**3 - sp.Rational(1, 6) * lam_b**5
    c3_contact_checks = [
        sp.simplify(sp.diff(f_in_planch, (lam_b, k_d)).subs(lam_b, 1) - sp.diff(f_out_planch, (lam_b, k_d)).subs(lam_b, 1))
        for k_d in range(4)
    ]
    c4_jump = sp.simplify(sp.diff(f_in_planch, (lam_b, 4)).subs(lam_b, 1) - sp.diff(f_out_planch, (lam_b, 4)).subs(lam_b, 1))
    c4_jump_comp = sp.simplify(sp.diff(f_comp_in, (lam_b, 4)).subs(lam_b, 1) - sp.diff(f_comp_out, (lam_b, 4)).subs(lam_b, 1))
    print(f"    [SUB-LOG THM 18.1] W_3 bases (det_raw={det_M_jet_raw}, det_scaled={det_M_jet_taylor}) & Radon-Plancherel C^3 ball "
          f"convolution checked in {time.time() - t_sub:.2f}s")

    thm18_ok = _require_all("THM 18", [
        ("phi_raw == [e^{-2 psi}, e^{-4 psi}, e^{-6 psi}]", phi_raw == [sp.exp(-2 * psi), sp.exp(-4 * psi), sp.exp(-6 * psi)], phi_raw),
        ("det_M_jet_raw == -16 & det_M_jet_taylor == -32/3", (det_M_jet_raw, det_M_jet_taylor) == (-16, -sp.Rational(32, 3)), (det_M_jet_raw, det_M_jet_taylor)),
        ("eq108_coeff_diff == 0_3 & vspec_eq108_match == 0", eq108_coeff_diff == sp.zeros(3, 1) and vspec_eq108_match == 0, (eq108_coeff_diff, vspec_eq108_match)),
        ("biorth_primal_check == I_3", biorth_primal_check == sp.eye(3), biorth_primal_check),
        ("W0_self == 6/5", W0_self == sp.Rational(6, 5), W0_self),
        ("f_in_planch == expected_f_in", sp.expand(f_in_planch - expected_f_in) == 0, f_in_planch),
        ("f_out_planch == 5/(12 lam_b)", sp.simplify(f_out_planch - 5 / (12 * lam_b)) == 0, f_out_planch),
        ("c3_contact_checks == [0,0,0,0], c4_jump == -30, c4_jump_comp == +36",
         (c3_contact_checks, c4_jump, c4_jump_comp) == ([0, 0, 0, 0], -30, 36),
         (c3_contact_checks, c4_jump, c4_jump_comp)),
    ])

    state.deduce_theorem(
        "W3_primal_dual_bases_and_3d_ball_overlap",
        {"phi_raw": phi_raw, "det_M_jet": det_M_jet_raw, "det_M_jet_taylor": det_M_jet_taylor,
         "M_jet_inv": M_jet_raw_inv, "Pi_primal": Pi_primal, "c_vec_eq108": c_vec_sol,
         "V_spec_full": V_spec_full, "psi_t_canonical": psi_t_canonical,
         "W0_self": W0_self, "f_in": f_in_planch, "f_out": f_out_planch, "c4_jump": c4_jump},
        "Deduced W_3 <-> W_3^* bases (det_raw=-16, det_scaled=-32/3, Eq. 108), Cauchy-Bernstein e^{-lam A^2} & C^3 ball convolution",
        thm18_ok
    )
    state.deduce_corollary("W3_primal_dual_bases_and_3d_ball_overlap", "hermite_jet_biorthogonality_and_cutoff_invariance",
                           (det_M_jet_raw, det_M_jet_taylor, eq108_coeff_diff, vspec_eq108_match, biorth_primal_check, cutoff_indep_diff),
                           (-16, -sp.Rational(32, 3), sp.zeros(3, 1), 0, sp.eye(3), [0, 0, 0]),
                           "Vandermonde det_raw==-16, scaled det==-32/3, Eq. (108) (c_0,c_1,c_2), <J^a, Pi_b>==I_3 & UV-cutoff invariance")
    state.deduce_corollary("W3_primal_dual_bases_and_3d_ball_overlap", "whitney_bernstein_character_and_shifted_jet",
                           (jet_at_min, whitney_parity_diff, horizon_slope_zero, bernstein_cm_checks),
                           ((3 * V0_s, 0, sp.Rational(3, 2) * m_psi_s**2, m_psi_s**2), 0, 0, [0, 0, 0]),
                           "Shifted 2-jet (3 V_0, 0, (3/2)m_psi^2) & unique Cauchy-Bernstein completely monotone character exp(-lam_A A^2)")
    state.deduce_corollary("W3_primal_dual_bases_and_3d_ball_overlap", "radon_plancherel_ball_convolution_and_newton_shell",
                           (radon_ft_diff, W0_self, M_out_const,
                            sp.expand(f_in_planch - expected_f_in), sp.simplify(f_out_planch - 5 / (12 * lam_b)),
                            c3_contact_checks, c4_jump, c4_jump_comp),
                           (0, sp.Rational(6, 5), 1, 0, 0, [0, 0, 0, 0], -30, 36),
                           "Radon-Plancherel convolution proves W(0)==6/5, Newton M(d>=2)==1, quintic f_in, C^3 contact & jumps (-30, +36)")

    # =========================================================================
    # THM 19: McKean-Singer De Rham Cohomological Supertrace Str(e^{-s Delta}) == chi(S_{t2}) == +1
    #         & Deductive Boundary Defects (+/- 1/2) from Orbifold Interval Cohomology
    # =========================================================================
    t_sub = time.time()
    print("  [LOG] Module 4 -> Computing THM 19 (McKean-Singer De Rham Supertrace Str(e^{-s Delta}) == chi == +1)...")
    x2_dr = sp.Symbol('x2_dr', real=True)
    n_pos = sp.Symbol('n_pos', integer=True, positive=True)
    phi_n_0form = sp.cos(n_pos * x2_dr / L0_s)
    d_phi_n_coeff = sp.diff(phi_n_0form, x2_dr)
    bc_dirichlet_0 = sp.simplify(d_phi_n_coeff.subs(x2_dr, 0))
    bc_dirichlet_pi = sp.simplify(d_phi_n_coeff.subs(x2_dr, sp.pi * L0_s))

    lam_n_kk = (n_pos / (L0_s * Phi_s))**2
    eig_0_diff = sp.simplify(-Phi_s**(-2) * sp.diff(phi_n_0form, x2_dr, 2) - lam_n_kk * phi_n_0form)
    eig_1_diff = sp.simplify(-Phi_s**(-2) * sp.diff(d_phi_n_coeff, x2_dr, 2) - lam_n_kk * d_phi_n_coeff)

    # Deductive Z_2-invariant De Rham cohomology on the covering circle (replaces int(cos(0)), int(sin(0))):
    icoh = _derive_interval_cohomology()
    b0_betti = icoh['b0']
    b1_betti = icoh['b1']
    chi_euler_I = icoh['chi']
    excited_0_ok = (
        icoh['excited'][0]['minors_vanish_on_shell']
        and icoh['excited'][0]['generic_nullity'] == 0
        and icoh['excited'][0]['quantised_nullity'] == 1
    )
    excited_1_ok = (
        icoh['excited'][1]['minors_vanish_on_shell']
        and icoh['excited'][1]['generic_nullity'] == 0
        and icoh['excited'][1]['quantised_nullity'] == 1
    )

    S_kk_pos = sp.Symbol('S_kk_pos', positive=True)
    W_weyl_s = wm['K_w'].subs(w_m, 0) / 2   # w = 0 term of the Poisson-resummed half-line trace (THM 17 kernel)
    S_wind_sym = sp.Symbol('S_wind_pos', real=True)
    Theta_S1_jacobi = 2 * W_weyl_s * (1 + 2 * S_wind_sym)
    S_kk_pos_solved = sp.solve(sp.Eq(1 + 2 * S_kk_pos, Theta_S1_jacobi), S_kk_pos)[0]

    Theta_Neumann_s = sp.expand(b0_betti + S_kk_pos_solved)
    Theta_Dirichlet_s = sp.expand(b1_betti + S_kk_pos_solved)

    Str_DeRham = sp.simplify(Theta_Neumann_s - Theta_Dirichlet_s)
    a_half_N = sp.limit(Theta_Neumann_s.subs(S_wind_sym, 0) - W_weyl_s, s_m, 0, dir='+')
    a_half_D = sp.limit(Theta_Dirichlet_s.subs(S_wind_sym, 0) - W_weyl_s, s_m, 0, dir='+')
    card_boundary_I = sp.simplify(2 * (a_half_N - a_half_D))

    th_N_num = 1 + mp.nsum(lambda n: mp.exp(-s_val * (n / (L0_val * Phi_val))**2), [1, mp.inf])
    th_D_num = mp.nsum(lambda n: mp.exp(-s_val * (n / (L0_val * Phi_val))**2), [1, mp.inf])
    weyl_wind_num = mp.mpf('0.5') * wind_untwisted
    mckean_50digit_ok = (
        abs((th_N_num - th_D_num) - 1) < mp.mpf('1e-48')
        and abs((th_N_num - weyl_wind_num) - mp.mpf('0.5')) < mp.mpf('1e-48')
        and abs((th_D_num - weyl_wind_num) - (-mp.mpf('0.5'))) < mp.mpf('1e-48')
    )

    S_wind_alt = sp.Symbol('S_wind_alt', real=True)
    Theta_SS_twisted_s = W_weyl_s * (1 + 2 * S_wind_alt)
    graded_uv_diff = sp.simplify(Theta_Neumann_s - Theta_SS_twisted_s)
    uv_weyl_cancelled = sp.limit(sp.sqrt(s_m) * graded_uv_diff.subs({S_wind_sym: 0, S_wind_alt: 0}), s_m, 0, dir='+')
    print(f"    [SUB-LOG THM 19.1] McKean-Singer De Rham supertrace Str={Str_DeRham} & deduced boundary defects=({a_half_N}, {a_half_D}) "
          f"checked in {time.time() - t_sub:.2f}s")

    thm19_ok = _require_all("THM 19", [
        ("Str_DeRham == 1", Str_DeRham == 1, Str_DeRham),
        ("Str_DeRham == chi_euler_I", Str_DeRham == chi_euler_I, (Str_DeRham, chi_euler_I)),
        ("b0_betti == 1 & b1_betti == 0", (b0_betti, b1_betti) == (1, 0), (b0_betti, b1_betti)),
        ("excited_0_ok & excited_1_ok (mu_n = n/L0 quantisation & isospectrality)",
         excited_0_ok and excited_1_ok, (excited_0_ok, excited_1_ok)),
        ("a_half_N == 1/2", a_half_N == sp.Rational(1, 2), a_half_N),
        ("a_half_D == -1/2", a_half_D == -sp.Rational(1, 2), a_half_D),
        ("card_boundary_I == 2", card_boundary_I == 2, card_boundary_I),
        ("mckean_50digit_ok", mckean_50digit_ok, mckean_50digit_ok),
    ])

    state.deduce_theorem(
        "mckean_singer_derham_cohomology",
        {"Str_DeRham": Str_DeRham, "b0": b0_betti, "b1": b1_betti, "chi_orbifold": chi_euler_I,
         "a_half_N": a_half_N, "a_half_D": a_half_D, "card_boundary": card_boundary_I},
        "McKean-Singer De Rham supertrace Str(e^{-s Delta}) == chi(S_{t2}) == +1 (b_0=1, b_1=0) & boundary defects (+/- 1/2)",
        thm19_ok
    )
    state.deduce_corollary("mckean_singer_derham_cohomology", "exterior_pairing_and_euler_characteristic",
                           (bc_dirichlet_0, bc_dirichlet_pi, eig_0_diff, eig_1_diff,
                            excited_0_ok and excited_1_ok, Str_DeRham, sp.diff(Str_DeRham, s_m)),
                           (0, 0, 0, 0, True, 1, 0),
                           "Exact De Rham d: Omega^0 -> Omega^1 pairing for all n>=1 & s-independent Str(e^{-s Delta}) == chi([0,pi L_0]) == +1")
    state.deduce_corollary("mckean_singer_derham_cohomology", "boundary_defects_and_scherk_schwarz_weyl_cancellation",
                           (a_half_N, a_half_D, card_boundary_I, mckean_50digit_ok, uv_weyl_cancelled),
                           (sp.Rational(1, 2), -sp.Rational(1, 2), 2, True, 0),
                           "Deduced boundary heat defects (+1/2, -1/2), |partial S_{t2}|==2, 50-digit check & UV Weyl s^{-1/2} cancellation")


# ========================================================================================
# COLAB CELL 4 EXECUTION & MEMORY CHECKPOINT
# ========================================================================================
if __name__ == "__main__":
    if "_COLAB_CHECKPOINT_M3" not in globals():
        raise RuntimeError("Please run Colab Cell 3 (Module 3) first so `_COLAB_CHECKPOINT_M3` is in memory!")
    t0_m4 = time.time()
    master_state = copy.deepcopy(globals()["_COLAB_CHECKPOINT_M3"])
    print("  [INSTANT RESUME] Loaded Modules 1-3 state (14 THMs, 66 CORs) from `_COLAB_CHECKPOINT_M3`!")
    run_module_4(master_state)
    globals()["_COLAB_CHECKPOINT_M4"] = copy.deepcopy(master_state)
    print(f"\n  --> [COLAB CELL 4 READY] Module 4 completed in {time.time() - t0_m4:.2f}s "
          f"(Cumulative: {master_state.thm_count} THMs, {master_state.cor_count} CORs saved in `_COLAB_CHECKPOINT_M4`)")



# ========================================================================================
# CBG PRINCIPIA GEOMETRICA — COLAB CELL 5: MODULE 5 (THM #20 .. THM #24) & FINAL SUMMARY
# ========================================================================================
# Module 5:
#   - Resumes automatically from `_COLAB_CHECKPOINT_M4` (or initializes standalone M1-M4
#     state if run in isolation), injects AXIOM_3, and derives every ODE, Frobenius residue
#     (b_{-1}, psi'(r_h), C_{B^2}), stiff-limit equation, Kasner exponent (p_t, p_r), and
#     5D curved-S^2 Kretschmann polynomial directly from the explicit 4D/5D metric tensors
#     via Koszul connections and Riemann contractions.
# ========================================================================================

import copy
import time
import sympy as sp

# ========================================================================================
# MODULE 5 (FROM AXIOM 3): HORIZON REGULARITY, FIELD EQUATIONS & CAUCHY-GRAM CLOSURE
# ========================================================================================
def _derive_nec_from_action(kappa_psi, c_Weyl):
    """Variational Hilbert derivation of Eq. (85), Eq. (88), and Eq. (149) from the Master Action (Eq. 86)."""
    r, th = sp.symbols('r theta', positive=True)
    psi_s, dpsi = sp.symbols('psi dpsi_dr', real=True)
    F, H = sp.symbols('F H', real=True, nonzero=True)
    V_s, V_A = sp.symbols('V V_A', real=True)
    kr = sp.Symbol('k_r', positive=True)

    gtt, grr, gthth, gphph = sp.symbols('gtt_inv grr_inv gthth_inv gphph_inv', real=True, nonzero=True)
    Vf = sp.Function('V')

    A_of_g = 1 / (-sp.exp(c_Weyl * psi_s) * gtt)                       # Eq. (85) with d_t Theta = 1
    ell = kappa_psi * grr * dpsi**2 - 2 * Vf(psi_s, A_of_g)            # matter part of Eq. (86)
    sqrtg = 1 / sp.sqrt(-gtt * grr * gthth * gphph)                    # sqrt(-g~) as a function of g~^{mu nu}
    L = sqrtg * ell

    def hilbert(var):
        t = sp.diff(L, var) / sqrtg
        t = t.replace(lambda e_: isinstance(e_, sp.Subs), lambda e_: V_A)
        t = t.subs(Vf(psi_s, A_of_g), V_s)
        return sp.simplify(-t)

    on_metric = {gtt: -1 / F, grr: 1 / H, gthth: 1 / r**2, gphph: 1 / (r**2 * sp.sin(th)**2)}
    T_tt = sp.simplify(hilbert(gtt).subs(on_metric))
    T_rr = sp.simplify(hilbert(grr).subs(on_metric))
    A_val = sp.simplify(A_of_g.subs(on_metric))                        # A = e^{-c psi} F

    T_t_t = sp.simplify((-1 / F) * T_tt)
    T_r_r = sp.simplify((1 / H) * T_rr)

    kt_sq = sp.solve(sp.Eq(-F * sp.Symbol('kt2') + H * kr**2, 0), sp.Symbol('kt2'))[0]
    T_kk = sp.simplify(T_tt * kt_sq + T_rr * kr**2)

    A_sym = sp.Symbol('A_norm', real=True, nonzero=True)
    T_kk_in_A = sp.simplify(T_kk.subs(F, sp.solve(sp.Eq(A_sym, A_val), F)[0]))

    return dict(r=r, psi=psi_s, dpsi=dpsi, F=F, H=H, V=V_s, V_A=V_A, kr=kr, A_sym=A_sym,
                T_tt=T_tt, T_rr=T_rr, T_t_t=T_t_t, T_r_r=T_r_r, A_val=A_val, kt_sq=kt_sq,
                T_kk=T_kk, T_kk_in_A=T_kk_in_A)

def run_module_5(state: DeductiveState) -> None:
    banner("MODULE 5 (FROM AXIOM 3): HORIZON REGULARITY, FIELD EQUATIONS & CAUCHY-GRAM CLOSURE")

    state.inject_axiom("AXIOM_3", {
        "horizon_regularity": "Static geometries with psi(r) != const across a Killing horizon r_h > 0 must satisfy |K_5(r_h)| < oo",
        "equilibrium_root_coupling": "Primary temporal foliation norm A enters V_spec(psi, A) through its equilibrium root psi_t(A)",
    })

    dt_ded, ds_ded, D_ded = state.theorems["dimensions"]
    kappa_psi_ded = state.theorems["jordan_vacuum_and_curved_weyl_discovery"]["kappa_psi"]  # -3/2 from THM 12
    c_Weyl_ded = state.theorems["jordan_vacuum_and_curved_weyl_discovery"]["c_Weyl"]        # 1 from THM 12
    beta_chrono_ded = state.theorems["bitemporal_5d_geodesics"]["beta_chrono_dom"]          # 3/2 from THM 13
    c_GC_ded = state.theorems["oneill_submersion_and_gauss_codazzi"]["c_GC"]                # -2 from THM 09
    c_K5_hess_ded = state.theorems["curved_quadratic_invariants_lovelock_and_ghy"]["c_K5_hess"]  # 4 from THM 16
    Pi_primal = state.theorems["W3_primal_dual_bases_and_3d_ball_overlap"]["Pi_primal"]     # Dual basis from THM 18
    cauchy_minors_thm7 = state.theorems["s3_self_dual_conic_and_cauchy"]["cauchy_minors"]   # Cauchy minors from THM 07
    phi_raw_ded = state.theorems["W3_primal_dual_bases_and_3d_ball_overlap"]["phi_raw"]      # W_3 basis from THM 18
    V_spec_full_ded = state.theorems["W3_primal_dual_bases_and_3d_ball_overlap"]["V_spec_full"]  # V_spec(psi) from THM 18

    def _require_all(tag: str, named_conditions: list) -> bool:
        """Granular diagnostic verifier: reports the exact sub-condition and value on any failure."""
        for cond_name, cond_ok, debug_val in named_conditions:
            if not bool(cond_ok):
                print(f"\n  [!!! DIAGNOSTIC FAILURE IN {tag} !!!]")
                print(f"    -> Failed condition : {cond_name}")
                print(f"    -> Actual value     : {debug_val}")
                raise AssertionError(f"[DIAGNOSTIC FAIL in {tag}] `{cond_name}` failed; actual = {debug_val}")
        return True

    def _zero_rat_fast(expr):
        """
        GCD-free exact rational normalizer: places expr over a syntactic common denominator
        and expands the numerator. Proves exact zero identities in O(monomials) milliseconds.
        """
        if isinstance(expr, sp.MatrixBase):
            return expr.applyfunc(_zero_rat_fast)
        e = _normalize_exp_powers_global(sp.sympify(expr).doit())
        if e == 0:
            return sp.Integer(0)
        tog = sp.together(sp.expand_mul(e))
        num, den = sp.fraction(tog)
        num_exp = _normalize_exp_powers_global(sp.expand(num))
        if num_exp == 0:
            return sp.Integer(0)
        return num_exp / den

    class _FastDiagGeometry:
        """
        Ultra-fast Koszul/Levi-Civita geometry engine for diagonal metrics using _zero_rat_fast
        and sp.Integer(1)/g_ii (zero floats, zero multivariate polynomial GCD hangs).
        """
        def __init__(self, g_diag: list, coords: list, sqrt_abs_det_g: sp.Expr):
            self.n = len(coords)
            self.x = list(coords)
            g_sym = [sp.sympify(g_ii) for g_ii in g_diag]
            self.g = sp.diag(*g_sym)
            self.ginv = sp.diag(*[sp.Integer(1) / g_ii for g_ii in g_sym])
            self.sqrtg = sp.sympify(sqrt_abs_det_g)

            n, x = self.n, self.x
            self.Gamma = [[[sp.Integer(0) for _ in range(n)] for _ in range(n)] for _ in range(n)]
            for c in range(n):
                for a in range(n):
                    for b in range(a, n):
                        val = sp.Rational(1, 2) * self.ginv[c, c] * (
                            sp.diff(self.g[c, b], x[a]) + sp.diff(self.g[a, c], x[b]) - sp.diff(self.g[a, b], x[c])
                        )
                        val = _zero_rat_fast(val)
                        self.Gamma[c][a][b] = val
                        self.Gamma[c][b][a] = val

            G = self.Gamma
            self.Ric = sp.zeros(n, n)
            for b in range(n):
                for d in range(b, n):
                    val = sp.Integer(0)
                    for a in range(n):
                        val += sp.diff(G[a][d][b], x[a]) - sp.diff(G[a][a][b], x[d])
                        for e in range(n):
                            if G[a][a][e] != 0 and G[e][d][b] != 0:
                                val += G[a][a][e] * G[e][d][b]
                            if G[a][d][e] != 0 and G[e][a][b] != 0:
                                val -= G[a][d][e] * G[e][a][b]
                    val = _zero_rat_fast(val)
                    self.Ric[b, d] = val
                    self.Ric[d, b] = val

            self.R = _zero_rat_fast(sum(self.ginv[a, a] * self.Ric[a, a] for a in range(n)))
            self.Ein = sp.Matrix(n, n, lambda a, b: _zero_rat_fast(self.Ric[a, b] - sp.Rational(1, 2) * self.g[a, b] * self.R))

        def kretschmann_diag(self) -> sp.Expr:
            """Computes exact Kretschmann scalar K = R_{abcd} R^{abcd} on a diagonal metric from Koszul Gamma."""
            n, x, G = self.n, self.x, self.Gamma
            K_sum = sp.Integer(0)
            for a in range(n):
                for b in range(a + 1, n):
                    for c in range(n):
                        for d in range(c + 1, n):
                            R_up_abcd = sp.diff(G[a][d][b], x[c]) - sp.diff(G[a][c][b], x[d])
                            for e in range(n):
                                if G[a][c][e] != 0 and G[e][d][b] != 0:
                                    R_up_abcd += G[a][c][e] * G[e][d][b]
                                if G[a][d][e] != 0 and G[e][c][b] != 0:
                                    R_up_abcd -= G[a][d][e] * G[e][c][b]
                            R_up_abcd = _zero_rat_fast(R_up_abcd)
                            if R_up_abcd != 0:
                                R_cov_abcd = self.g[a, a] * R_up_abcd
                                R_con_abcd = self.ginv[a, a] * self.ginv[b, b] * self.ginv[c, c] * self.ginv[d, d] * R_cov_abcd
                                K_sum += 4 * R_cov_abcd * R_con_abcd
            return _zero_rat_fast(K_sum)

        def cov_div_mixed(self, T_mixed: sp.Matrix) -> list:
            """Computes nabla_mu T^mu_nu for a mixed tensor T^mu_nu."""
            n, x, G = self.n, self.x, self.Gamma
            res = []
            for nu in range(n):
                val = sum(sp.diff(self.sqrtg * T_mixed[mu, nu], x[mu]) for mu in range(n)) / self.sqrtg
                for mu in range(n):
                    for lam in range(n):
                        if G[lam][mu][nu] != 0 and T_mixed[mu, lam] != 0:
                            val -= G[lam][mu][nu] * T_mixed[mu, lam]
                res.append(_zero_rat_fast(val))
            return res

    # =========================================================================
    # THM 20: Full-Rank (dim = 6) Non-Separable 2-Jet Space J^2(R^2), Hilbert Stress
    #         Decomposition (c_V = -2, c_kh = -2), Bivariate Noether & Khronon Locking
    # =========================================================================
    t_sub = time.time()
    print("  [LOG] Module 5 -> Computing THM 20 (Full-Rank 6D 2-Jet Space, Hilbert Stress, Bivariate Noether & Khronon Locking)...")
    t_k, r_k, th_k, ph_k = sp.symbols('t_k r_k th_k ph_k', real=True)
    coords_k = [t_k, r_k, th_k, ph_k]

    N_k = sp.Function('N_k')(t_k, r_k)
    B_k = sp.Function('B_k')(t_k, r_k)
    P_k = sp.Function('P_k')(t_k, r_k)
    Theta_k = sp.Function('Theta_k')(t_k, r_k)

    G4_kh = _FastDiagGeometry([-N_k**2, B_k**2, r_k**2, r_k**2 * sp.sin(th_k)**2], coords_k, N_k * B_k * r_k**2 * sp.sin(th_k))

    jet2_vars = [
        sp.diff(N_k, t_k, 2), sp.diff(N_k, t_k, r_k), sp.diff(N_k, r_k, 2),
        sp.diff(B_k, t_k, 2), sp.diff(B_k, t_k, r_k), sp.diff(B_k, r_k, 2),
    ]
    jet2_probes = [
        sp.diff(N_k, r_k, 2), sp.diff(B_k, t_k, 2),
        sp.diff(N_k * B_k, t_k, 2), sp.diff(N_k * B_k, t_k, r_k),
        sp.diff(N_k / B_k, t_k, r_k), sp.diff(N_k * B_k, r_k, 2),
    ]
    J_jet6 = sp.Matrix(6, 6, lambda i_r, j_c: _zero_rat_fast(sp.diff(jet2_probes[i_r], jet2_vars[j_c])))
    rank_jet6 = J_jet6.rank()

    # Hilbert weights from the variation of Eq. (86) with respect to g~^{mu nu} (same derivation as THM 23):
    nec_h = _derive_nec_from_action(kappa_psi_ded, c_Weyl_ded)
    c_V_ded = sp.simplify(2 * sp.diff(nec_h['T_t_t'], nec_h['V']))                                        # T^t_t = (1/2) c_V V + ...
    c_kh_ded = sp.simplify(sp.diff(nec_h['T_t_t'], nec_h['V_A']) / nec_h['A_val'])                        # T^t_t = ... + c_kh A dV/dA
    c_kh_cov = sp.simplify(
        sp.diff(nec_h['T_tt'], nec_h['V_A']) / (nec_h['A_val']**2 * sp.exp(c_Weyl_ded * nec_h['psi']))    # T_tt = c_kh_cov A^2 dV/dA e^psi (d_t Theta)^2 + ...
    )

    v0_c, vpq_c = sp.symbols('v0_c vpq_c', real=True)
    dTheta_0, dTheta_1 = sp.diff(Theta_k, t_k), sp.diff(Theta_k, r_k)
    X_kh = (dTheta_0**2) / (N_k**2) - (dTheta_1**2) / (B_k**2)
    A_kh = sp.Integer(1) / (X_kh * P_k)

    # General potential V_gen(P, A) with P = e^psi: partial derivatives in the same form SymPy's chain rule produces
    V_gen_fn = sp.Function('V_gen')
    V_biv = V_gen_fn(P_k, A_kh)
    dV_dA = V_biv.fdiff(2)
    dV_dpsi = P_k * V_biv.fdiff(1)                      # d/dpsi at fixed A, with psi = ln P

    dpsi_vec = [sp.diff(P_k, t_k) / P_k, sp.diff(P_k, r_k) / P_k, sp.Integer(0), sp.Integer(0)]
    dTheta_vec = [dTheta_0, dTheta_1, sp.Integer(0), sp.Integer(0)]
    g_dpsi_sq = sum(G4_kh.ginv[a_i, a_i] * dpsi_vec[a_i]**2 for a_i in range(4))

    T_mixed = sp.zeros(4, 4)
    for mu in range(4):
        for nu in range(4):
            val = (-2 * kappa_psi_ded) * G4_kh.ginv[mu, mu] * dpsi_vec[mu] * dpsi_vec[nu]
            val += c_kh_cov * (A_kh**2) * dV_dA * P_k * G4_kh.ginv[mu, mu] * dTheta_vec[mu] * dTheta_vec[nu]
            if mu == nu:
                val += sp.Rational(1, 2) * (2 * kappa_psi_ded * g_dpsi_sq + c_V_ded * V_biv)
            T_mixed[mu, nu] = _zero_rat_fast(val)

    div_T_vec = G4_kh.cov_div_mixed(T_mixed)
    Bian_div_Ein = G4_kh.cov_div_mixed(G4_kh.ginv * G4_kh.Ein)

    sqrt_g_kh = G4_kh.sqrtg
    E_psi = (
        -2 * kappa_psi_ded * sum(sp.diff(sqrt_g_kh * G4_kh.ginv[a_i, a_i] * dpsi_vec[a_i], coords_k[a_i]) for a_i in range(2)) / sqrt_g_kh
        - dV_dpsi + A_kh * dV_dA
    )
    J_kh_up = [sqrt_g_kh * (A_kh**2) * dV_dA * P_k * G4_kh.ginv[a_i, a_i] * dTheta_vec[a_i] for a_i in range(2)]
    E_Theta = 2 * (sp.diff(J_kh_up[0], t_k) + sp.diff(J_kh_up[1], r_k)) / sqrt_g_kh

    noether_t_res = _zero_rat_fast(div_T_vec[0] - (E_psi * dpsi_vec[0] + E_Theta * dTheta_vec[0]))
    noether_r_res = _zero_rat_fast(div_T_vec[1] - (E_psi * dpsi_vec[1] + E_Theta * dTheta_vec[1]))

    # Deductive proof of stationary foliation uniqueness h'(r) == 0 from G^1_r == T^1_r on static metric (Eq. 87 / COR #082):
    h_r_fn = sp.Function('h_r_fn')(r_k)
    G1_r_static = (G4_kh.ginv[0, 0] * G4_kh.Ein[0, 1]).subs({sp.diff(B_k, t_k): 0, sp.diff(N_k, t_k): 0})
    A_stat_s, V_A_stat_s = sp.symbols('A_stat_s V_A_stat_s', nonzero=True)
    # Theta = t + h(r): T_{t r} = c_kh_cov A^2 dV/dA e^psi d_t Theta d_r Theta (covariant khronon stress), index raised with g~^{tt}:
    T_tr_stationary = c_kh_cov * A_stat_s**2 * V_A_stat_s * P_k * sp.Integer(1) * sp.diff(h_r_fn, r_k)
    T1_r_stationary = G4_kh.ginv[0, 0] * T_tr_stationary
    h_prime_locked = sp.solve(sp.Eq(G1_r_static - T1_r_stationary, 0), sp.diff(h_r_fn, r_k))[0]
    print(f"    [SUB-LOG THM 20.1] Rank-6 jet space (rank={rank_jet6}), Hilbert weights=({c_V_ded},{c_kh_ded}) & bivariate Noether "
          f"checked in {time.time() - t_sub:.2f}s")

    thm20_ok = _require_all("THM 20", [
        ("rank_jet6 == 6", rank_jet6 == 6, rank_jet6),
        ("c_V_ded == -2", c_V_ded == -2, c_V_ded),
        ("c_kh_ded == -2 & c_kh_cov == +2", (c_kh_ded, c_kh_cov) == (-2, 2), (c_kh_ded, c_kh_cov)),
        ("Bian_div_Ein == [0,0,0,0]", Bian_div_Ein == [0, 0, 0, 0], Bian_div_Ein),
        ("noether_t_res == 0", noether_t_res == 0, noether_t_res),
        ("noether_r_res == 0", noether_r_res == 0, noether_r_res),
        ("h_prime_locked == 0", h_prime_locked == 0, h_prime_locked),
    ])

    state.deduce_theorem(
        "khronon_2jet_hilbert_and_noether",
        {"rank_jet6": rank_jet6, "c_V": c_V_ded, "c_kh": c_kh_ded, "h_prime_locked": h_prime_locked},
        "Full-rank (dim=6) 2-jet space J^2(R^2), Hilbert stress (c_V=-2, c_kh=-2), bivariate Noether & Khronon locking",
        thm20_ok
    )
    state.deduce_corollary("khronon_2jet_hilbert_and_noether", "full_rank_6d_bivariate_jet_space",
                           (rank_jet6, Bian_div_Ein), (6, [0, 0, 0, 0]),
                           "Full rank 6 on J^2(R^2) (ddN_tt, ddN_tr, ddN_rr, ddB_tt, ddB_tr, ddB_rr) & exact contracted Bianchi div(G)==0")
    state.deduce_corollary("khronon_2jet_hilbert_and_noether", "hilbert_stress_weights_and_bivariate_noether",
                           (c_V_ded, c_kh_ded, noether_t_res, noether_r_res), (-2, -2, 0, 0),
                           "Hilbert stress coefficients (c_V=-2, c_kh=-2) from the variation of Eq. (86) & bivariate (t,r) Noether identity for a general potential V(psi, A)")
    state.deduce_corollary("khronon_2jet_hilbert_and_noether", "stationary_khronon_flux_locking",
                           h_prime_locked, 0,
                           "Stationary off-diagonal Einstein equation G^1_r == T^1_r == 0 uniquely locks h'(r) == 0")

    # =========================================================================
    # THM 21: Curved-k_c FLRW, Nullspace Raychaudhuri Extraction, Bianchi-I Shear
    #         a^{-6}, 5D Jordan & 4D Einstein Minisuperspace Noether + Constraint
    #         Preservation (Eq. 122), and Wheeler-DeWitt Ordering p = 1 (Eqs. 123-124)
    # =========================================================================
    t_sub = time.time()
    print("  [LOG] Module 5 -> Computing THM 21 (Curved-k_c FLRW, Nullspace Raychaudhuri, Bianchi-I Shear, Eq. 122 & WDW p = 1)...")
    t_c, r_c, th_c, ph_c = sp.symbols('t_c r_c th_c ph_c', real=True)
    k_c = sp.Symbol('k_c', real=True)
    a_t = sp.Function('a_t')(t_c)
    psi_c = sp.Function('psi_c')(t_c)
    V_flrw = sp.Function('V_flrw')(psi_c)

    G4_flrw = _FastDiagGeometry(
        [sp.Integer(-1), a_t**2 / (1 - k_c * r_c**2), a_t**2 * r_c**2, a_t**2 * r_c**2 * sp.sin(th_c)**2],
        [t_c, r_c, th_c, ph_c],
        a_t**3 * r_c**2 * sp.sin(th_c) / sp.sqrt(1 - k_c * r_c**2)
    )
    G_00_flrw = _zero_rat_fast(-G4_flrw.ginv[0, 0] * G4_flrw.Ein[0, 0])   # 3*(da^2 + k_c)/a^2
    G_11_flrw = _zero_rat_fast(-G4_flrw.ginv[1, 1] * G4_flrw.Ein[1, 1])   # 2*dda/a + (da^2 + k_c)/a^2

    # Hilbert stress of the scalar sector of Eq. (86) on a homogeneous background, from the variation with respect to g~^{mu nu}
    # (V_flrw is V_spec(psi); psi_c is psi):
    gtt_h, g11_h, g22_h, g33_h = sp.symbols('gtt_h g11_h g22_h g33_h', real=True, nonzero=True)
    dpsi_h, V_h = sp.symbols('dpsi_h V_h', real=True)
    ell_h = kappa_psi_ded * gtt_h * dpsi_h**2 - 2 * V_h
    sqrtg_h = 1 / sp.sqrt(-gtt_h * g11_h * g22_h * g33_h)
    T_tt_h = sp.simplify(-sp.diff(sqrtg_h * ell_h, gtt_h) / sqrtg_h)
    T_11_h = sp.simplify(-sp.diff(sqrtg_h * ell_h, g11_h) / sqrtg_h)
    on_flrw_h = {gtt_h: G4_flrw.ginv[0, 0], dpsi_h: sp.diff(psi_c, t_c), V_h: V_flrw}
    rho_psi = sp.simplify(T_tt_h.subs(on_flrw_h))                    # T_tt with u^t = 1
    p_psi = sp.simplify((g11_h * T_11_h).subs(on_flrw_h))            # T^1_1

    M_curv_col = sp.Matrix([sp.diff(G_00_flrw, sp.diff(a_t, t_c), 2), sp.diff(G_11_flrw, sp.diff(a_t, t_c), 2)])
    null_vec_ray = M_curv_col.T.nullspace()[0]
    ray_lhs = _zero_rat_fast(null_vec_ray[0] * G_00_flrw + null_vec_ray[1] * G_11_flrw)
    ray_rhs = _zero_rat_fast(null_vec_ray[0] * rho_psi + null_vec_ray[1] * (-p_psi))
    coeff_dda = sp.diff(ray_lhs, sp.diff(a_t, t_c, 2)) * a_t
    dda_over_a_solved = sp.expand(ray_rhs / coeff_dda)
    c_kin_dir = sp.Rational(1, 2) * sp.diff(dda_over_a_solved, sp.diff(psi_c, t_c), 2)   # -1/2
    c_pot_dir = sp.diff(dda_over_a_solved, V_flrw)                                       # +1/3
    ray_ratio_dir = sp.simplify(c_kin_dir / c_pot_dir)                                   # kappa_psi
    # Convention of Eq. (121): V_Lag = 2 V_spec and psi_hat_dot^2 = (1/2) psi_dot^2
    psi_hat_dot, V_Lag = sp.symbols('psi_hat_dot V_Lag', real=True)
    dda_rescaled = sp.expand(dda_over_a_solved.subs({sp.diff(psi_c, t_c): sp.sqrt(2) * psi_hat_dot, V_flrw: V_Lag / 2}))
    c_kin_ray = sp.Rational(1, 2) * sp.diff(dda_rescaled, psi_hat_dot, 2)                # -1
    c_pot_ray = sp.diff(dda_rescaled, V_Lag)                                             # +1/6
    ray_kin_pot_ratio = sp.simplify(c_kin_ray / c_pot_ray)                               # 4 kappa_psi

    x_b, y_b, z_b = sp.symbols('x_b y_b z_b', real=True)
    W1_t = sp.Function('W1_t')(t_c)
    W2_t = sp.Function('W2_t')(t_c)
    G4_bianchi = _FastDiagGeometry(
        [sp.Integer(-1), a_t**2 * W1_t**2, a_t**2 * W2_t**2, a_t**2 / (W1_t**2 * W2_t**2)],
        [t_c, x_b, y_b, z_b],
        a_t**3
    )
    db1_t, db2_t = sp.Symbol('db1_t', real=True), sp.Symbol('db2_t', real=True)
    ddb1_t, ddb2_t = sp.Symbol('ddb1_t', real=True), sp.Symbol('ddb2_t', real=True)
    sub_b = {
        sp.diff(W1_t, t_c, 2): W1_t * (ddb1_t + db1_t**2),
        sp.diff(W2_t, t_c, 2): W2_t * (ddb2_t + db2_t**2),
        sp.diff(W1_t, t_c): W1_t * db1_t,
        sp.diff(W2_t, t_c): W2_t * db2_t,
    }
    aniso_12 = _zero_rat_fast((G4_bianchi.ginv[1, 1] * G4_bianchi.Ric[1, 1] - G4_bianchi.ginv[2, 2] * G4_bianchi.Ric[2, 2]).subs(sub_b))
    c_hubble_friction = sp.simplify(sp.diff(aniso_12, db1_t) / (sp.diff(a_t, t_c) / a_t))  # 3
    shear_energy_exp = -2 * c_hubble_friction                                               # -6

    # Reduced (minisuperspace) Lagrangians deduced from the actions on FLRW metrics with lapse N(t):
    N_m = sp.Function('N_m')(t_c)
    spatial_measure = r_c**2 * sp.sin(th_c) / sp.sqrt(1 - k_c * r_c**2)

    def _reduce_to_first_order(L_full, q_list):
        """Removes the total derivative d/dt( sum_q (dL/d q'') q' ) from a Lagrangian linear in second derivatives."""
        boundary = sum(sp.diff(L_full, sp.diff(q_f, t_c, 2)) * sp.diff(q_f, t_c) for q_f in q_list)
        return sp.simplify(L_full - sp.diff(boundary, t_c))

    # 4D Einstein frame: (1/2) sqrt(-g~) [R~ + kappa_psi g~^{mu nu} d psi d psi - 2 V_spec] / spatial measure  (Eq. 86)
    G4_lapse = _FastDiagGeometry(
        [-N_m**2, a_t**2 / (1 - k_c * r_c**2), a_t**2 * r_c**2, a_t**2 * r_c**2 * sp.sin(th_c)**2],
        [t_c, r_c, th_c, ph_c],
        N_m * a_t**3 * spatial_measure
    )
    lapse_vol_4 = sp.simplify(G4_lapse.sqrtg / spatial_measure)
    R4_lapse = sp.simplify(G4_lapse.R)
    L_E_full = sp.Rational(1, 2) * lapse_vol_4 * (R4_lapse + kappa_psi_ded * G4_lapse.ginv[0, 0] * sp.diff(psi_c, t_c)**2 - 2 * V_flrw)
    L_mini = _reduce_to_first_order(L_E_full, [a_t])
    L_mini_second_order_free = [sp.simplify(sp.diff(L_mini, d_q)) for d_q in (sp.diff(a_t, t_c, 2), sp.diff(N_m, t_c))]
    L_mini_form_diff = sp.simplify(
        L_mini - ((a_t**3 / N_m) * (-3 * (sp.diff(a_t, t_c) / a_t)**2 - sp.Rational(1, 2) * kappa_psi_ded * sp.diff(psi_c, t_c)**2)
                  + 3 * k_c * N_m * a_t - N_m * a_t**3 * V_flrw)
    )
    EL_N = _zero_rat_fast(sp.diff(L_mini, N_m))
    EL_a = _zero_rat_fast(sp.diff(L_mini, a_t) - sp.diff(sp.diff(L_mini, sp.diff(a_t, t_c)), t_c))
    EL_psi = _zero_rat_fast(sp.diff(L_mini, psi_c) - sp.diff(sp.diff(L_mini, sp.diff(psi_c, t_c)), t_c))
    mini_noether_res = _zero_rat_fast(-N_m * sp.diff(EL_N, t_c) + EL_a * sp.diff(a_t, t_c) + EL_psi * sp.diff(psi_c, t_c))

    # 5D Jordan frame: sqrt|g_5| [R^(5) - 2 V_J(psi, A)] / spatial measure on the FLRW x fibre metric with lapse (Phi_inf = 1):
    x2_m = sp.Symbol('x2_m', real=True)
    G5_lapse = _FastDiagGeometry(
        [-N_m**2, -sp.exp(2 * psi_c), a_t**2 / (1 - k_c * r_c**2), a_t**2 * r_c**2, a_t**2 * r_c**2 * sp.sin(th_c)**2],
        [t_c, x2_m, r_c, th_c, ph_c],
        N_m * sp.exp(psi_c) * a_t**3 * spatial_measure
    )
    lapse_vol_5 = sp.simplify(G5_lapse.sqrtg / spatial_measure)
    L_J_geom = _reduce_to_first_order(lapse_vol_5 * sp.simplify(G5_lapse.R), [a_t, psi_c])
    L_J_second_order_free = [sp.simplify(sp.diff(L_J_geom, d_q))
                             for d_q in (sp.diff(a_t, t_c, 2), sp.diff(psi_c, t_c, 2), sp.diff(N_m, t_c))]
    Theta_m = sp.Function('Theta_m')(t_c)
    A_mini_j = N_m**2 / sp.diff(Theta_m, t_c)**2
    V_mini_j = sp.Function('V_mini_j')(psi_c, A_mini_j)
    L_mini_J = L_J_geom - 2 * lapse_vol_5 * V_mini_j
    L_mini_J_form_diff = sp.simplify(
        L_mini_J - a_t * sp.exp(psi_c) * (
            -6 * sp.diff(a_t, t_c)**2 / N_m - 6 * a_t * sp.diff(a_t, t_c) * sp.diff(psi_c, t_c) / N_m
            + 6 * k_c * N_m - 2 * a_t**2 * N_m * V_mini_j)
    )
    EL_N_J = _zero_rat_fast(sp.diff(L_mini_J, N_m))
    EL_a_J = _zero_rat_fast(sp.diff(L_mini_J, a_t) - sp.diff(sp.diff(L_mini_J, sp.diff(a_t, t_c)), t_c))
    EL_psi_J = _zero_rat_fast(sp.diff(L_mini_J, psi_c) - sp.diff(sp.diff(L_mini_J, sp.diff(psi_c, t_c)), t_c))
    EL_Th_J = _zero_rat_fast(-sp.diff(sp.diff(L_mini_J, sp.diff(Theta_m, t_c)), t_c))
    mini_noether_J_res = _zero_rat_fast(
        sp.diff(Theta_m, t_c) * EL_Th_J - N_m * sp.diff(EL_N_J, t_c) + sp.diff(a_t, t_c) * EL_a_J + sp.diff(psi_c, t_c) * EL_psi_J
    )

    # On-shell Friedmann constraint evolution dC_Fried/dt = -(3H + dpsi/dt) C_Fried (Eq. 122).
    # C_Fried is the lapse equation of L_mini_J normalised to unit weight of the potential; rho_r is a declared test fluid
    # (radiation, conformally coupled) added to the source with its own continuity equation:
    rho_r = sp.Function('rho_r')(t_c)
    V_j_only = sp.Function('V_j_only')(psi_c)
    H_j = sp.diff(a_t, t_c) / a_t
    dpsi_t = sp.diff(psi_c, t_c)
    L_J_lapse = L_mini_J.subs(V_mini_j, V_j_only)
    EL_N_vac = sp.diff(L_J_lapse, N_m)
    C_Fried = sp.simplify((EL_N_vac / (-sp.diff(EL_N_vac, V_j_only))).subs(N_m, 1).doit()) - rho_r
    C_Fried_form_diff = sp.simplify(C_Fried - (3 * (H_j**2 + k_c / a_t**2 + H_j * dpsi_t) - (rho_r + V_j_only)))
    L_J_vac = L_J_lapse.subs(N_m, 1).doit()
    EL_a_vac = _zero_rat_fast(sp.diff(L_J_vac, a_t) - sp.diff(sp.diff(L_J_vac, sp.diff(a_t, t_c)), t_c) + 2 * a_t**2 * sp.exp(psi_c) * rho_r)
    EL_psi_vac = _zero_rat_fast(sp.diff(L_J_vac, psi_c) - sp.diff(sp.diff(L_J_vac, sp.diff(psi_c, t_c)), t_c))
    sol_dda_ddpsi = sp.solve([sp.Eq(EL_a_vac, 0), sp.Eq(EL_psi_vac, 0)], [sp.diff(a_t, t_c, 2), sp.diff(psi_c, t_c, 2)])
    dC_Fried_dt = _zero_rat_fast(
        sp.diff(C_Fried, t_c).subs(sol_dda_ddpsi).subs(sp.diff(rho_r, t_c), -(4 * H_j + dpsi_t) * rho_r)
        + (3 * H_j + dpsi_t) * C_Fried
    )

    d_mini_sym = sp.Symbol('d_mini_sym', real=True)
    xi_yamabe = (d_mini_sym - 2) / (4 * (d_mini_sym - 1))
    xi_yamabe_2d = sp.simplify(xi_yamabe.subs(d_mini_sym, dt_ded))

    a_s, psi_s = sp.symbols('a_s psi_s', positive=True)
    H_kin_mat = sp.Matrix(2, 2, lambda i_r, j_c: sp.Rational(1, 2) * sp.diff(
        L_mini.subs(N_m, 1), [sp.diff(a_t, t_c), sp.diff(psi_c, t_c)][i_r], [sp.diff(a_t, t_c), sp.diff(psi_c, t_c)][j_c]
    )).subs({a_t: a_s, psi_c: psi_s})
    sqrt_det_G = sp.sqrt(sp.Abs(H_kin_mat.det()))
    G_mini_inv = H_kin_mat.inv()
    Psi_wdw = sp.Function('Psi_wdw')(a_s, psi_s)
    LB_a_term = (sp.Integer(1) / sqrt_det_G) * sp.diff(sqrt_det_G * G_mini_inv[0, 0] * sp.diff(Psi_wdw, a_s), a_s)
    p_wdw_ded = sp.simplify(
        a_s * sp.diff(LB_a_term, sp.diff(Psi_wdw, a_s)) / sp.diff(LB_a_term, sp.diff(Psi_wdw, a_s, 2))
    )
    print(f"    [SUB-LOG THM 21.1] Curved-k_c FLRW (ratio={ray_kin_pot_ratio}), Bianchi-I shear a^{shear_energy_exp}, "
          f"Eq. 122 Noether=({mini_noether_res},{mini_noether_J_res},{dC_Fried_dt}) & WDW p={p_wdw_ded} checked in {time.time() - t_sub:.2f}s")

    thm21_ok = _require_all("THM 21", [
        ("Hilbert stress (rho, p) == (-kappa/2 psi_dot^2 + V, -kappa/2 psi_dot^2 - V)",
         (sp.simplify(rho_psi - (-kappa_psi_ded / 2 * sp.diff(psi_c, t_c)**2 + V_flrw)),
          sp.simplify(p_psi - (-kappa_psi_ded / 2 * sp.diff(psi_c, t_c)**2 - V_flrw))) == (0, 0), (rho_psi, p_psi)),
        ("direct Raychaudhuri (c_kin, c_pot, ratio) == (-1/2, 1/3, kappa_psi)",
         (c_kin_dir, c_pot_dir, ray_ratio_dir) == (-sp.Rational(1, 2), sp.Rational(1, 3), kappa_psi_ded), (c_kin_dir, c_pot_dir, ray_ratio_dir)),
        ("ray_kin_pot_ratio == 4*kappa_psi_ded == -6", ray_kin_pot_ratio == 4 * kappa_psi_ded, ray_kin_pot_ratio),
        ("(c_kin_ray, c_pot_ray) == (-1, 1/6)", (c_kin_ray, c_pot_ray) == (-1, sp.Rational(1, 6)), (c_kin_ray, c_pot_ray)),
        ("reduced Lagrangians are first order and match Eq. (122) forms",
         (L_mini_second_order_free, L_mini_form_diff, L_J_second_order_free, L_mini_J_form_diff, C_Fried_form_diff) == ([0, 0], 0, [0, 0, 0], 0, 0),
         (L_mini_second_order_free, L_mini_form_diff, L_J_second_order_free, L_mini_J_form_diff, C_Fried_form_diff)),
        ("c_hubble_friction == 3 & shear_energy_exp == -6", (c_hubble_friction, shear_energy_exp) == (3, -6), (c_hubble_friction, shear_energy_exp)),
        ("mini_noether_res == 0 & mini_noether_J_res == 0 & dC_Fried_dt == 0",
         (mini_noether_res, mini_noether_J_res, dC_Fried_dt) == (0, 0, 0),
         (mini_noether_res, mini_noether_J_res, dC_Fried_dt)),
        ("xi_yamabe_2d == 0 & p_wdw_ded == 1", (xi_yamabe_2d, p_wdw_ded) == (0, 1), (xi_yamabe_2d, p_wdw_ded)),
    ])

    state.deduce_theorem(
        "flrw_bianchi_minisuperspace_and_wdw",
        {"ray_kin_pot_ratio": ray_kin_pot_ratio, "c_kin_ray": c_kin_ray, "c_pot_ray": c_pot_ray,
         "shear_energy_exp": shear_energy_exp, "mini_noether_res": mini_noether_res,
         "mini_noether_J_res": mini_noether_J_res, "dC_Fried_dt": dC_Fried_dt, "p_wdw": p_wdw_ded},
        "Curved-k_c FLRW, nullspace Raychaudhuri (dda/a = -dpsi^2 + V/6), Bianchi-I shear a^{-6}, Eq. (122) Noether & WDW p=1",
        thm21_ok
    )
    state.deduce_corollary("flrw_bianchi_minisuperspace_and_wdw", "nullspace_raychaudhuri_and_curved_flrw",
                           (ray_kin_pot_ratio, c_kin_ray, c_pot_ray), (-6, -1, sp.Rational(1, 6)),
                           "Nullspace Raychaudhuri: dda/a == -(1/2) psi_dot^2 + (1/3) V_spec (ratio kappa_psi); in the convention V_Lag = 2 V_spec, psi_hat_dot^2 = psi_dot^2/2: -psi_hat_dot^2 + V_Lag/6 (ratio 4 kappa_psi == -6); independent of k_c")
    state.deduce_corollary("flrw_bianchi_minisuperspace_and_wdw", "bianchi_i_anisotropic_shear_dilution",
                           (c_hubble_friction, shear_energy_exp), (3, -6),
                           "Bianchi-I traceless Ricci d_t(a^3 db_i) == 0 proves shear dilution db_i propto a^{-3} & rho_sigma propto a^{-6}")
    state.deduce_corollary("flrw_bianchi_minisuperspace_and_wdw", "minisuperspace_noether_and_wdw_ordering",
                           (mini_noether_res, mini_noether_J_res, dC_Fried_dt, xi_yamabe_2d, p_wdw_ded),
                           (0, 0, 0, 0, 1),
                           "Reduced 4D/5D Lagrangians deduced from the actions, reparametrisation Noether identities (Eq. 122), dC_Fried/dt == -(3H+dpsi)C_Fried (with a declared radiation test fluid), Yamabe xi_2D==0 & WDW p == 1")

    # =========================================================================
    # THM 22: Universal Bekenstein Star-Shaped Domain (-oo, +1/2 ln 3) on Pi_2(psi),
    #         AXIOM 3 Horizon Regularity, Normal-Form 5D ODEs Solved from G^A_B = T^A_B
    #         (Eqs. 139-142), Frobenius Pole Cancellation C_{B^2} + 1/b_{-1} == 0 (Eq. 143),
    #         Pole-Free Stiff-Limit Reduction (Eq. 144) & Kottler-RN-dS (Eqs. 146-147)
    # =========================================================================
    t_sub = time.time()
    print("  [LOG] Module 5 -> Computing THM 22 (Bekenstein Basin, Deductive Normal-Form ODEs, Frobenius Pole Cancellation & Kottler)...")
    psi = sp.Symbol('psi', real=True)
    Pi2_psi = Pi_primal[2]
    dPi2_dpsi = sp.factor(sp.diff(Pi2_psi, psi))
    dPi2_factor_diff = sp.simplify(
        dPi2_dpsi - (-sp.Rational(1, 4) * sp.exp(-6 * psi) * (sp.exp(2 * psi) - 1) * (sp.exp(2 * psi) - 3))
    )
    u_exp2 = sp.Symbol('u_exp2', positive=True)
    dPi2_u = sp.factor(dPi2_dpsi.subs(sp.exp(psi), sp.sqrt(u_exp2)).subs(sp.exp(-6 * psi), u_exp2**(-3)))
    u_roots = sorted(sp.solve(sp.Eq(dPi2_u, 0), u_exp2))  # [1, 3]
    psi_barrier_max = sp.simplify(sp.Rational(1, 2) * sp.log(u_roots[1]))  # +1/2 ln 3
    Pi2_barrier_height = sp.simplify(Pi2_psi.subs(psi, psi_barrier_max))   # 1/54

    bekenstein_neg_ok = bool(sp.simplify(dPi2_dpsi.subs(psi, -sp.log(2))) < 0)
    bekenstein_pos_ok = bool(sp.simplify(dPi2_dpsi.subs(psi, sp.Rational(1, 4) * sp.log(3))) > 0)

    # Exact 5D Jordan-frame static geometry & deductive solution of G^A_B == T^A_B for (A', B', psi'') (Eqs. 139-142):
    t_j, x2_j, r_j, th_j, ph_j = sp.symbols('t_j x2_j r_j th_j ph_j', real=True)
    A_r = sp.Function('A_r')(r_j)
    B_r = sp.Function('B_r')(r_j)
    psi_r = sp.Function('psi_r')(r_j)
    V_j_pot = sp.Function('V_j_pot')(psi_r, A_r)

    G5_stat = _FastDiagGeometry(
        [-A_r, -sp.exp(2 * psi_r), B_r, r_j**2, r_j**2 * sp.sin(th_j)**2],
        [t_j, x2_j, r_j, th_j, ph_j],
        sp.sqrt(A_r * B_r) * sp.exp(psi_r) * r_j**2 * sp.sin(th_j)
    )
    G5_stat_mix = G5_stat.ginv * G5_stat.Ein

    dV_dpsi_j = sp.diff(V_j_pot, psi_r)
    dV_dA_j = sp.diff(V_j_pot, A_r)
    T1_1_j = -V_j_pot - 2 * A_r * dV_dA_j
    T2_2_j = -V_j_pot - dV_dpsi_j
    Tr_r_j = -V_j_pot
    Tth_th_j = -V_j_pot

    # Radial 5-divergence identity nabla_A T^A_r == 0 from G5_stat.cov_div_mixed (Eq. 139):
    T5_stat_mix = sp.diag(T1_1_j, T2_2_j, Tr_r_j, Tth_th_j, Tth_th_j)
    div5_Tr = G5_stat.cov_div_mixed(T5_stat_mix)[2]

    # Deductive algebraic solve of G^r_r = T^r_r, G^1_1 = T^1_1, and G^2_2 - G^theta_theta = T^2_2 - T^theta_theta
    # for (A'(r), B'(r), psi''(r)) (Eqs. 140-142):
    dA_1, dA_2 = sp.diff(A_r, r_j), sp.diff(A_r, r_j, 2)
    dB_1 = sp.diff(B_r, r_j)
    dpsi_1, dpsi_2 = sp.diff(psi_r, r_j), sp.diff(psi_r, r_j, 2)

    A_prime_ded = sp.solve(sp.Eq(G5_stat_mix[2, 2] - Tr_r_j, 0), dA_1)[0]
    eq_diff_2_th = _zero_rat_fast((G5_stat_mix[1, 1] - G5_stat_mix[3, 3]) - (T2_2_j - Tth_th_j))
    sol_B1_psi2 = sp.solve(
        [sp.Eq((G5_stat_mix[0, 0] - T1_1_j).subs(dA_1, A_prime_ded), 0),
         sp.Eq(eq_diff_2_th.subs(dA_1, A_prime_ded), 0)],
        [dB_1, dpsi_2]
    )
    B_prime_ded = _zero_rat_fast(sol_B1_psi2[dB_1])
    psi_2_ded = _zero_rat_fast(sol_B1_psi2[dpsi_2])

    A_2_ded = sp.diff(A_prime_ded, r_j).subs({dA_1: A_prime_ded, dB_1: B_prime_ded, dpsi_2: psi_2_ded})
    nf_subs = {dA_2: A_2_ded, dpsi_2: psi_2_ded, dA_1: A_prime_ded, dB_1: B_prime_ded}
    E1_1_nf = _zero_rat_fast((G5_stat_mix[0, 0] - T1_1_j).subs(nf_subs))
    E2_2_nf = _zero_rat_fast((G5_stat_mix[1, 1] - T2_2_j).subs(nf_subs))
    Er_r_nf = _zero_rat_fast((G5_stat_mix[2, 2] - Tr_r_j).subs(nf_subs))
    Eth_th_nf = _zero_rat_fast((G5_stat_mix[3, 3] - Tth_th_j).subs(nf_subs))

    # Deductive extraction of Frobenius near-horizon expansion from the solved Normal-Form ODEs (Eq. 143):
    rh_s, eps_h, a1_h, bm1_h, Vh_s, dVh_s, psih_p = sp.symbols('r_h eps_h a1_h bm1_h V_h dV_h psih_p', positive=True)
    frob_subs = {
        dV_dA_j: 0, V_j_pot: Vh_s, dV_dpsi_j: dVh_s,
        A_r: a1_h * eps_h, B_r: bm1_h / eps_h, dpsi_1: psih_p, r_j: rh_s + eps_h,
    }
    # 1. Deduce b_{-1} by solving lim_{eps->0} (A'(r_h+eps)/a_1 - 1) == 0:
    eq_bm1 = sp.limit(sp.cancel(A_prime_ded.subs(frob_subs) / a1_h - 1), eps_h, 0)
    b_minus1_ded = sp.solve(sp.Eq(eq_bm1, 0), bm1_h)[0]

    # 2. Deduce psi'(r_h) by solving the vanishing of the O(eps^{-1}) pole residue of psi''(r_h+eps):
    psi2_pole_res = sp.limit(sp.cancel(eps_h * psi_2_ded.subs(frob_subs).subs(bm1_h, b_minus1_ded)), eps_h, 0)
    psi_prime_rh_ded = sp.solve(sp.Eq(psi2_pole_res, 0), psih_p)[0]

    # When V_spec = e^{-psi} V has (d_psi V_spec)_h == 0 <=> dVh_s == Vh_s, psi'(r_h) vanishes identically:
    dpsi_rh_regular = sp.simplify(psi_prime_rh_ded.subs(dVh_s, Vh_s))

    # 3. Deduce C_{B^2} = lim_{eps->0} [eps^2 B'(r_h+eps) / b_{-1}^2] and prove C_{B^2} + 1/b_{-1} == 0:
    C_B2_ded = sp.limit(sp.cancel(eps_h**2 * B_prime_ded.subs(frob_subs) / bm1_h**2), eps_h, 0)
    frobenius_pole_cancel = _zero_rat_fast((C_B2_ded + 1 / b_minus1_ded).subs(psih_p, psi_prime_rh_ded))

    # Deductive Einstein-frame (Eq. 135) near-horizon covariant Hessian pole nabla_r d_r psi ~ -(1/2) psi'(r_h)/(r-r_h):
    A_E_fn = sp.Function('A_E_fn')(r_j)
    B_E_fn = sp.Function('B_E_fn')(r_j)
    G4_ein_st = _FastDiagGeometry([-A_E_fn**2, B_E_fn**2, r_j**2, r_j**2 * sp.sin(th_j)**2], [t_j, r_j, th_j, ph_j], A_E_fn * B_E_fn * r_j**2 * sp.sin(th_j))
    box4_ein_psi = sp.diff(G4_ein_st.sqrtg * G4_ein_st.ginv[1, 1] * dpsi_1, r_j) / G4_ein_st.sqrtg
    psi_2_ein_on_shell = sp.solve(sp.Eq(box4_ein_psi, 0), dpsi_2)[0]
    nabla_r_dr_psi_ein = (psi_2_ein_on_shell - G4_ein_st.Gamma[1][1][1] * dpsi_1).subs({
        sp.diff(A_E_fn, r_j): a1_h / (2 * sp.sqrt(eps_h)),
        sp.diff(B_E_fn, r_j): -bm1_h / (2 * eps_h**sp.Rational(3, 2)),
        A_E_fn: a1_h * sp.sqrt(eps_h),
        B_E_fn: bm1_h / sp.sqrt(eps_h),
        dpsi_1: psih_p,
        r_j: rh_s + eps_h,
    })
    horizon_hess_residue = sp.limit(sp.cancel(eps_h * nabla_r_dr_psi_ein / psih_p), eps_h, 0)

    # Pole-Free Stiff-Limit Reduction via B'/B = P'/P - A'/A in 2 A B G^theta_theta == 0 (Eq. 144):
    P_stiff = sp.Function('P_stiff')(r_j)
    stiff_comb = _zero_rat_fast(
        (2 * A_r * B_r * G5_stat_mix[3, 3]).subs(
            sp.diff(B_r, r_j), B_r * (sp.diff(P_stiff, r_j) / P_stiff - sp.diff(A_r, r_j) / A_r)
        )
    )
    expected_eq144 = (
        sp.diff(A_r, r_j, 2) - sp.diff(A_r, r_j) * sp.diff(P_stiff, r_j) / (2 * P_stiff)
        + 2 * sp.diff(A_r, r_j) / r_j - A_r * sp.diff(P_stiff, r_j) / (r_j * P_stiff)
        + 2 * A_r * sp.diff(psi_r, r_j, 2) + 2 * A_r * dpsi_1**2
        + (2 * A_r / r_j + 2 * sp.diff(A_r, r_j) - A_r * sp.diff(P_stiff, r_j) / P_stiff) * dpsi_1
    )
    stiff_eq144_diff = _zero_rat_fast(stiff_comb - expected_eq144)

    # Kottler-RN-dS from the static Einstein-frame field equations at the vacuum equilibrium (psi' = 0, d_A V = 0),
    # with the potential value of THM 18 at its minimum and the Maxwell sector -1/4 F_{mu nu} F^{mu nu} of Eq. (86):
    V_vac_k = sp.simplify(V_spec_full_ded.subs(psi, sp.Symbol('psi_t', real=True)))          # 3 V_0
    V0_k = sp.Symbol('V0', real=True)
    rh_m, Qe_m, H_inf = sp.symbols('rh_m Qe_m H_inf', positive=True)
    dA_E, dB_E = sp.diff(A_E_fn, r_j), sp.diff(B_E_fn, r_j)

    # (a) Hilbert stress of -2 V - (1/4) F F for a radial electric field F_{t r}, from the variation with respect to g~^{mu nu}:
    gi_k = sp.symbols('gtt_k grr_k gthth_k gphph_k', real=True, nonzero=True)
    F_tr_s, V_k = sp.symbols('F_tr_s V_k', real=True)
    F_cov_k = sp.zeros(4, 4)
    F_cov_k[0, 1], F_cov_k[1, 0] = F_tr_s, -F_tr_s
    ell_k = -2 * V_k - sp.Rational(1, 4) * sum(gi_k[m_i] * gi_k[n_i] * F_cov_k[m_i, n_i]**2 for m_i in range(4) for n_i in range(4))
    sqrtg_k = 1 / sp.sqrt(-gi_k[0] * gi_k[1] * gi_k[2] * gi_k[3])
    on_static_k = {gi_k[i_k]: G4_ein_st.ginv[i_k, i_k] for i_k in range(4)}
    T_mix_k = [sp.simplify((gi_k[i_k] * (-sp.diff(sqrtg_k * ell_k, gi_k[i_k]) / sqrtg_k)).subs(on_static_k)) for i_k in range(3)]

    # (b) Maxwell equation: A_t is cyclic, so the momentum conjugate to F_{t r} = -A_t' is conserved; Q_e is its value per unit sin(theta):
    maxwell_momentum = sp.simplify((sp.diff(sqrtg_k * ell_k, F_tr_s) / sqrtg_k).subs(on_static_k))     # = -F^{t r} on the static metric
    maxwell_flux = sp.simplify(G4_ein_st.sqrtg * maxwell_momentum / sp.sin(th_j))
    F_tr_sol = sp.solve(sp.Eq(maxwell_flux, Qe_m), F_tr_s)[0]
    T_mix_on = [sp.simplify(T_c.subs(F_tr_s, F_tr_sol)) for T_c in T_mix_k]                  # (T^t_t, T^r_r, T^theta_theta)
    maxwell_stress_coeff = [sp.simplify(r_j**4 * (T_c + V_k)) for T_c in T_mix_on]

    # (c) G^t_t - G^r_r == T^t_t - T^r_r == 0  =>  (A_E B)' == 0
    G4_ein_mix = G4_ein_st.ginv * G4_ein_st.Ein
    focus_eq = sp.simplify((G4_ein_mix[0, 0] - G4_ein_mix[1, 1]) - (T_mix_on[0] - T_mix_on[1]))
    dB_focus = sp.solve(sp.Eq(focus_eq, 0), dB_E)[0]
    AB_const_check = sp.simplify(sp.diff(A_E_fn * B_E_fn, r_j).subs(dB_E, dB_focus))

    # (d) With the constant A_E B normalised to 1 (choice of the time unit), A_E^2 = 1/B^2 = f(r); solve G^t_t == T^t_t for f:
    f_k = sp.Function('f_k')(r_j)
    G4_f = _FastDiagGeometry([-f_k, 1 / f_k, r_j**2, r_j**2 * sp.sin(th_j)**2], [t_j, r_j, th_j, ph_j], r_j**2 * sp.sin(th_j))
    G4_f_mix = G4_f.ginv * G4_f.Ein
    T_mix_vac = [T_c.subs(V_k, V_vac_k) for T_c in T_mix_on]
    T_free_of_metric = all(not T_c.has(A_E_fn) and not T_c.has(B_E_fn) for T_c in T_mix_vac)
    f_general = sp.dsolve(sp.Eq(sp.simplify(G4_f_mix[0, 0] - T_mix_vac[0]), 0), f_k).rhs
    C_int = [s_c for s_c in f_general.free_symbols if s_c not in (r_j, Qe_m, V0_k)]
    mass_coeff = sp.simplify(sp.diff(sp.expand(r_j * f_general), C_int[0]))                  # coefficient of the integration constant in r f
    f_kottler_sol = sp.expand(f_general.subs(C_int[0], -rh_m / mass_coeff))                  # r_h defined by f = 1 - r_h / r + ...
    field_eq_residuals = [
        sp.simplify((G4_f_mix[i_k, i_k] - T_mix_vac[i_k]).subs(f_k, f_kottler_sol).doit()) for i_k in range(3)
    ]
    kottler_diff = sp.simplify(f_kottler_sol - (1 - rh_m / r_j + Qe_m**2 / (4 * r_j**2) - V0_k * r_j**2))

    # (e) Horizons at V_0 = 0 and surface gravity of the de Sitter horizon from the Killing vector xi = d_t:
    r_pm_roots = sp.solve(sp.Eq(sp.expand(r_j**2 * f_kottler_sol.subs(V0_k, 0)), 0), r_j)
    r_roots_sum_prod = (sp.simplify(r_pm_roots[0] + r_pm_roots[1] - rh_m),
                        sp.simplify(r_pm_roots[0] * r_pm_roots[1] - Qe_m**2 / 4))
    xi_cov = [G4_f.g[0, n_i] for n_i in range(4)]
    coords_f = [t_j, r_j, th_j, ph_j]
    nabla_xi = [[sp.diff(xi_cov[n_i], coords_f[m_i]) - sum(G4_f.Gamma[l_i][m_i][n_i] * xi_cov[l_i] for l_i in range(4))
                 for n_i in range(4)] for m_i in range(4)]
    kappa_sq_general = sp.simplify(-sp.Rational(1, 2) * sum(
        G4_f.ginv[m_i, m_i] * G4_f.ginv[n_i, n_i] * nabla_xi[m_i][n_i]**2 for m_i in range(4) for n_i in range(4)))
    kappa_sq_formula_diff = sp.simplify(kappa_sq_general - sp.diff(f_k, r_j)**2 / 4)
    f_dS = f_kottler_sol.subs({rh_m: 0, Qe_m: 0, V0_k: H_inf**2})
    r_dS = [r_c_ for r_c_ in sp.solve(sp.Eq(f_dS, 0), r_j) if r_c_.is_positive][0]
    kappa_dS = sp.simplify(sp.sqrt(kappa_sq_general.subs(f_k, f_dS).doit().subs(r_j, r_dS)))
    print(f"    [SUB-LOG THM 22.1] Bekenstein domain (-oo, {psi_barrier_max}), Eqs. 139-144 Frobenius pole cancel={frobenius_pole_cancel}, "
          f"stiff={stiff_eq144_diff} & Kottler-RN-dS checked in {time.time() - t_sub:.2f}s")

    thm22_ok = _require_all("THM 22", [
        ("u_roots == [1, 3]", u_roots == [1, 3], u_roots),
        ("psi_barrier_max == log(3)/2", sp.simplify(psi_barrier_max - sp.Rational(1, 2) * sp.log(3)) == 0, psi_barrier_max),
        ("Pi2_barrier_height == 1/54", Pi2_barrier_height == sp.Rational(1, 54), Pi2_barrier_height),
        ("bekenstein_neg_ok & bekenstein_pos_ok", bekenstein_neg_ok and bekenstein_pos_ok, (bekenstein_neg_ok, bekenstein_pos_ok)),
        ("dpsi_rh_regular == 0 & horizon_hess_residue == -1/2",
         (dpsi_rh_regular, horizon_hess_residue) == (0, -sp.Rational(1, 2)),
         (dpsi_rh_regular, horizon_hess_residue)),
        ("div5_Tr == 0 & normal-form residuals == (0,0,0,0)",
         (div5_Tr, E1_1_nf, E2_2_nf, Er_r_nf, Eth_th_nf) == (0, 0, 0, 0, 0),
         (div5_Tr, E1_1_nf, E2_2_nf, Er_r_nf, Eth_th_nf)),
        ("frobenius_pole_cancel == 0 & stiff_eq144_diff == 0",
         (frobenius_pole_cancel, stiff_eq144_diff) == (0, 0),
         (frobenius_pole_cancel, stiff_eq144_diff)),
        ("Maxwell stress Q_e^2/(4 r^4) diag(-1,-1,+1), (A_E B)' == 0 & source independent of the metric functions",
         (maxwell_stress_coeff, AB_const_check, T_free_of_metric) == ([-Qe_m**2 / 4, -Qe_m**2 / 4, Qe_m**2 / 4], 0, True),
         (maxwell_stress_coeff, AB_const_check, T_free_of_metric)),
        ("all static field equations satisfied by the dsolve solution", field_eq_residuals == [0, 0, 0], field_eq_residuals),
        ("kottler_diff == 0, r_roots_sum_prod == (0,0), kappa^2 == f'^2/4 & kappa_dS == H_inf",
         (kottler_diff, r_roots_sum_prod, kappa_sq_formula_diff, kappa_dS) == (0, (0, 0), 0, H_inf),
         (kottler_diff, r_roots_sum_prod, kappa_sq_formula_diff, kappa_dS)),
    ])

    state.deduce_theorem(
        "bekenstein_basin_horizon_regularity_and_kottler",
        {"psi_barrier_max": psi_barrier_max, "Pi2_barrier_height": Pi2_barrier_height,
         "dpsi_rh_regular": dpsi_rh_regular, "psi_prime_rh": psi_prime_rh_ded,
         "frobenius_pole_cancel": frobenius_pole_cancel, "f_kottler": f_kottler_sol, "kappa_dS": kappa_dS},
        "Bekenstein domain (-oo, +1/2 ln 3), AXIOM 3 horizon regularity, Frobenius C_{B^2}+1/b_{-1}==0 (Eq. 143), stiff ODE & Kottler-RN-dS from the field equations",
        thm22_ok
    )
    state.deduce_corollary("bekenstein_basin_horizon_regularity_and_kottler", "bekenstein_star_shaped_domain_and_barrier",
                           (dPi2_factor_diff, u_roots, sp.simplify(psi_barrier_max - sp.Rational(1, 2) * sp.log(3)), Pi2_barrier_height,
                            bekenstein_neg_ok and bekenstein_pos_ok),
                           (0, [1, 3], 0, sp.Rational(1, 54), True),
                           "Exact factorization Pi_2'(psi) == -(1/4)e^{-6psi}(e^{2psi}-1)(e^{2psi}-3) proves Bekenstein domain (-oo, +1/2 ln 3) & height 1/54")
    state.deduce_corollary("bekenstein_basin_horizon_regularity_and_kottler", "axiom3_frobenius_horizon_regularity",
                           (horizon_hess_residue, dpsi_rh_regular,
                            div5_Tr, (E1_1_nf, E2_2_nf, Er_r_nf, Eth_th_nf), frobenius_pole_cancel, stiff_eq144_diff),
                           (-sp.Rational(1, 2), 0, 0, (0, 0, 0, 0), 0, 0),
                           "AXIOM 3 psi'(r_h)|_{d_A V=0}==0, normal-form ODEs (Eqs. 139-142), Frobenius C_{B^2}+1/b_{-1}==0 (Eq. 143) & stiff ODE (Eq. 144)")
    state.deduce_corollary("bekenstein_basin_horizon_regularity_and_kottler", "kottler_reissner_nordstrom_de_sitter",
                           (maxwell_stress_coeff, AB_const_check, field_eq_residuals, kottler_diff, r_roots_sum_prod, kappa_sq_formula_diff, kappa_dS),
                           ([-Qe_m**2 / 4, -Qe_m**2 / 4, Qe_m**2 / 4], 0, [0, 0, 0], 0, (0, 0), 0, H_inf),
                           "From Eq. (86): T(Maxwell) == Q_e^2/(4 r^4) diag(-1,-1,+1,+1), A_E B == const, f == 1 - r_h/r + Q_e^2/(4 r^2) - V_0 r^2; "
                           "r_+ + r_- == r_h, r_+ r_- == Q_e^2/4 & kappa_dS == H_inf (V_0 = H_inf^2)")

    # =========================================================================
    # THM 23: Non-Static Interior r < r_h (A_Killing < 0), Deductive 5D Curved-S^2
    #         Kasner Kretschmann Polynomial from Full Riemann Tensor (Theorem 5.6(i)),
    #         Variational NEC Contraction (Eqs. 85, 88, 149), Routhian e^{-3 psi} & O(4) S^3 Flux
    # =========================================================================
    t_sub = time.time()
    print("  [LOG] Module 5 -> Computing THM 23 (Deductive 5D Curved-S^2 Kasner Riemann Contraction, NEC Violation, e^{-3 psi} & O(4) S^3 Flux)...")

    # 1. Variational Hilbert derivation of A(g~), mixed stress components (Eq. 88) & null contraction (Eq. 149):
    nec_d = _derive_nec_from_action(kappa_psi_ded, c_Weyl_ded)
    F_nec, H_nec, psi_nec, dpsi_nec, V_nec, V_A_nec, kr_nec, A_sym_nec = (
        nec_d[k] for k in ('F', 'H', 'psi', 'dpsi', 'V', 'V_A', 'kr', 'A_sym')
    )
    A_out_val = nec_d['A_val'].subs(F_nec, a1_h * eps_h)
    A_in_val = nec_d['A_val'].subs(F_nec, -a1_h * eps_h)
    A_exterior_sign = sp.sign(A_out_val)
    A_interior_sign = sp.sign(A_in_val)

    nu_int, lam_int = sp.symbols('nu_int lam_int', real=True)
    interior_sub = {F_nec: -sp.exp(2 * nu_int), H_nec: -sp.exp(2 * lam_int)}
    A_interior_Formula = sp.simplify(nec_d['A_val'].subs(interior_sub))
    A_interior_formula_diff = sp.simplify(A_interior_Formula + sp.exp(-c_Weyl_ded * psi_nec + 2 * nu_int))

    T_kk_null = sp.simplify(nec_d['T_kk'].subs(interior_sub))
    expected_eq149 = (-kappa_psi_ded * dpsi_nec**2 - 2 * A_sym_nec * V_A_nec * sp.exp(2 * lam_int)) * kr_nec**2
    nec_null_diff = sp.simplify(T_kk_null - expected_eq149.subs(A_sym_nec, A_interior_Formula))
    nec_general_diff = sp.simplify(nec_d['T_kk_in_A'] - (-kappa_psi_ded * dpsi_nec**2 + 2 * A_sym_nec * V_A_nec * H_nec) * kr_nec**2)
    eq88_tt_diff = sp.simplify(nec_d['T_t_t'].subs(F_nec, A_sym_nec * sp.exp(c_Weyl_ded * psi_nec))
                               - (-V_nec - 2 * A_sym_nec * V_A_nec + kappa_psi_ded * dpsi_nec**2 / (2 * H_nec)))
    eq88_rr_diff = sp.simplify(nec_d['T_r_r'] - (-V_nec - kappa_psi_ded * dpsi_nec**2 / (2 * H_nec)))

    # 2. Deduce interior Kasner exponents (p_t, p_r) directly from the 4D Einstein-frame interior metric:
    t_in, r_in, th_in, ph_in, x2_in = sp.symbols('t_in r_in th_in ph_in x2_in', positive=True)
    c_psi, pt_s, pr_s = sp.symbols('c_psi pt_s pr_s', real=True)
    G4_kas = _FastDiagGeometry(
        [-r_in**(2 * pr_s), r_in**(2 * pt_s), r_in**2, r_in**2 * sp.sin(th_in)**2],
        [r_in, t_in, th_in, ph_in],
        r_in**(pr_s + pt_s + 2) * sp.sin(th_in)
    )
    psi_kas = c_psi * sp.log(r_in)
    wave_kas_coeff = sp.simplify(
        r_in**(2 * pr_s + 2) * sp.diff(G4_kas.sqrtg * G4_kas.ginv[0, 0] * sp.diff(psi_kas, r_in), r_in) / (G4_kas.sqrtg * c_psi)
    )
    G4_kas_mix = G4_kas.ginv * G4_kas.Ein
    ein_diff_kas = sp.limit(
        sp.expand(r_in**2 * r_in**(2 * pr_s) * (G4_kas_mix[1, 1] - G4_kas_mix[0, 0] - (-kappa_psi_ded) * r_in**(-2 * pr_s) * sp.diff(psi_kas, r_in)**2)),
        r_in, 0
    )
    kasner_sol = sp.solve([sp.Eq(wave_kas_coeff, 0), sp.Eq(ein_diff_kas, 0)], [pt_s, pr_s])
    pt_kas = kasner_sol[pt_s]  # -1/2 + 3/8 c_psi^2
    pr_kas = kasner_sol[pr_s]  # +1/2 + 3/8 c_psi^2

    # 3. Construct the full 5D Jordan-frame interior metric g_{AB} = e^{-psi} g~_{mu nu} (+) (-Phi^2 dt_2^2)
    #    WITH CURVED S^2 and contract the full 5D Riemann tensor K_5(r) = R_{ABCD} R^{ABCD}:
    G5_kas = _FastDiagGeometry(
        [-r_in**(2 * pr_kas - c_psi), -r_in**(2 * c_psi), r_in**(2 * pt_kas - c_psi),
         r_in**(2 - c_psi), r_in**(2 - c_psi) * sp.sin(th_in)**2],
        [r_in, x2_in, t_in, th_in, ph_in],
        r_in**(pr_kas + pt_kas + 2 - c_psi) * sp.sin(th_in)
    )
    K5_kas_full = G5_kas.kretschmann_diag()
    kasner_exp_power = 4 * pr_kas + 4 - 2 * c_psi  # 6 + 3/2 c_psi^2 - 2 c_psi
    K5_rescaled = sp.expand(K5_kas_full * r_in**kasner_exp_power)
    K5_kasner_coeff = sp.factor(K5_rescaled.subs(r_in, 0))
    K5_at_cpsi_2 = sp.simplify(K5_kas_full.subs(c_psi, 2))  # K_5 = K_{S^2_1} = 4 at c_psi = 2
    A_kasner_consistency_sign = sp.sign(1 / (-G5_kas.ginv[2, 2]))

    expected_K5_kasner_coeff = sp.Rational(3, 16) * (c_psi - 2)**2 * (9 * c_psi**4 - 12 * c_psi**3 + 40 * c_psi**2 + 16 * c_psi + 16)
    kasner_K5_diff = sp.expand(K5_kasner_coeff - expected_K5_kasner_coeff)
    quartic_sos_diff = sp.expand(
        (9 * c_psi**4 - 12 * c_psi**3 + 40 * c_psi**2 + 16 * c_psi + 16)
        - ((3 * c_psi**2 - 2 * c_psi)**2 + 36 * (c_psi + sp.Rational(2, 9))**2 + sp.Rational(128, 9))
    )
    exp_sos_diff = sp.expand(
        kasner_exp_power - (sp.Rational(3, 2) * (c_psi - sp.Rational(2, 3))**2 + sp.Rational(16, 3))
    )
    # Kasner window: the potential term e^{2 lambda} d_psi V_spec of the radial scalar equation, with the steepest W_3 basis
    # element of THM 18 evaluated on psi = c_psi ln r, against the curvature scale g~^{theta theta} of the interior metric:
    steepest_basis = min(phi_raw_ded, key=lambda e_: sp.simplify(sp.diff(sp.expand_log(sp.log(e_), force=True), psi)))
    kasner_pot_term = (-G4_kas.g[0, 0]).subs(pr_s, pr_kas) * steepest_basis.subs(psi, psi_kas)
    kasner_pot_exp = sp.simplify(r_in * sp.diff(sp.expand_log(sp.log(kasner_pot_term), force=True), r_in))
    kasner_curv_exp = sp.simplify(r_in * sp.diff(sp.expand_log(sp.log(G4_kas.ginv[2, 2]), force=True), r_in))
    kasner_window_roots = sorted(sp.solve(sp.Eq(kasner_pot_exp - kasner_curv_exp, 0), c_psi))  # [4 - 2*sqrt(3), 4 + 2*sqrt(3)]

    # Exponents of the centrifugal term, read from the Jordan-frame static metric G5_stat of THM 22 (r is the Jordan areal radius):
    #   fibre mass term P_2^2 (-g^{22}) = P_2^2 / Phi^2, and rescaled inverse angular metric g~^{theta theta} = e^{-c_Weyl psi} g^{theta theta}.
    P2_cent = sp.Symbol('P2_cent', positive=True)
    routh_mass_term = P2_cent**2 * (-G5_stat.ginv[1, 1])
    weyl_metric_shift = 1 / (sp.exp(c_Weyl_ded * psi_r) * G5_stat.g[3, 3])
    routh_mass_exp = sp.simplify(sp.diff(sp.expand_log(sp.log(routh_mass_term), force=True), psi_r))
    weyl_inv_metric_exp = sp.simplify(sp.diff(sp.expand_log(sp.log(weyl_metric_shift), force=True), psi_r))
    centrifugal_psi_exp = sp.simplify(routh_mass_exp + weyl_inv_metric_exp)
    centrifugal_routhian_diff = sp.simplify(centrifugal_psi_exp - (-2 * beta_chrono_ded))

    # O(4)-symmetric throat in the harmonic regime (potential term of Eq. 151 neglected, rho << 1/m_psi):
    # Laplacian of psi(rho) on the Euclidean metric d rho^2 + rho^2 d Omega_3^2, solved with dsolve.
    rho_euc, C_throat = sp.symbols('rho_euc C_throat', positive=True)
    chi_s3, th_s3, ph_s3 = sp.symbols('chi_s3 th_s3 ph_s3', positive=True)
    E4_diag = [sp.Integer(1), rho_euc**2, rho_euc**2 * sp.sin(chi_s3)**2, rho_euc**2 * sp.sin(chi_s3)**2 * sp.sin(th_s3)**2]
    sqrtg_E4 = rho_euc**3 * sp.sin(chi_s3)**2 * sp.sin(th_s3)
    E4_volume_check = sp.simplify(sqrtg_E4**2 - E4_diag[0] * E4_diag[1] * E4_diag[2] * E4_diag[3])
    psi_thr = sp.Function('psi_thr')(rho_euc)
    lap_E4 = sp.simplify(sp.diff(sqrtg_E4 / E4_diag[0] * sp.diff(psi_thr, rho_euc), rho_euc) / sqrtg_E4)
    psi_thr_sol = sp.dsolve(sp.Eq(lap_E4, 0), psi_thr).rhs
    throat_first_integral = sp.simplify(rho_euc**3 * sp.diff(psi_thr_sol, rho_euc))          # constant of the motion rho^3 psi'
    harmonic_check = sp.simplify(sp.diff(throat_first_integral, rho_euc))
    throat_consts = [s_c for s_c in throat_first_integral.free_symbols if s_c != rho_euc]
    psi_prime_throat = sp.simplify(sp.diff(psi_thr_sol, rho_euc).subs(
        throat_consts[0], sp.solve(sp.Eq(throat_first_integral, C_throat), throat_consts[0])[0]))
    throat_profile_diff = sp.simplify(psi_prime_throat - C_throat / rho_euc**3)
    s3_ranges = ((chi_s3, 0, sp.pi), (th_s3, 0, sp.pi), (ph_s3, 0, 2 * sp.pi))
    vol_S3 = sp.integrate(sqrtg_E4 / rho_euc**3, *s3_ranges)
    ghy_flux_S3 = sp.simplify(sp.integrate((-c_GC_ded) * sqrtg_E4 * psi_prime_throat, *s3_ranges))
    print(f"    [SUB-LOG THM 23.1] Deductive 5D curved-S^2 Kasner K_5 diff={kasner_K5_diff}, K_5(c_psi=2)={K5_at_cpsi_2}, "
          f"NEC Eq.149 diff={nec_null_diff}, centrifugal exp={centrifugal_psi_exp} & O(4) S^3 flux={ghy_flux_S3} checked in {time.time() - t_sub:.2f}s")

    thm23_ok = _require_all("THM 23", [
        ("A_interior_sign == -1 & A_exterior_sign == +1 from Eq. (85)",
         (A_interior_sign, A_exterior_sign, A_interior_formula_diff, A_kasner_consistency_sign) == (-1, 1, 0, -1),
         (A_interior_sign, A_exterior_sign, A_interior_formula_diff, A_kasner_consistency_sign)),
        ("kasner_K5_diff == 0, quartic_sos_diff == 0, exp_sos_diff == 0 & K5_at_cpsi_2 == 4",
         (kasner_K5_diff, quartic_sos_diff, exp_sos_diff, K5_at_cpsi_2) == (0, 0, 0, 4),
         (kasner_K5_diff, quartic_sos_diff, exp_sos_diff, K5_at_cpsi_2)),
        ("kasner_window_roots == [4 - 2*sqrt(3), 4 + 2*sqrt(3)]",
         kasner_window_roots == [4 - 2 * sp.sqrt(3), 4 + 2 * sp.sqrt(3)],
         kasner_window_roots),
        ("Hilbert stress Eq. (88) & null contraction Eq. (149) exact",
         (nec_null_diff, nec_general_diff, eq88_tt_diff, eq88_rr_diff) == (0, 0, 0, 0),
         (nec_null_diff, nec_general_diff, eq88_tt_diff, eq88_rr_diff)),
        ("centrifugal_psi_exp == -3 & centrifugal_routhian_diff == 0",
         (centrifugal_psi_exp, centrifugal_routhian_diff) == (-3, 0),
         (centrifugal_psi_exp, centrifugal_routhian_diff)),
        ("O(4) harmonic profile psi' == C_throat / rho^3 from dsolve", (E4_volume_check, harmonic_check, throat_profile_diff) == (0, 0, 0),
         (E4_volume_check, harmonic_check, throat_profile_diff)),
        ("ghy_flux_S3 == 4*pi^2*C_throat", ghy_flux_S3 == 4 * sp.pi**2 * C_throat, ghy_flux_S3),
    ])

    state.deduce_theorem(
        "interior_nec_centrifugal_and_instanton_flux",
        {"A_interior_sign": A_interior_sign, "A_exterior_sign": A_exterior_sign,
         "K5_kasner_coeff": K5_kasner_coeff, "K5_at_cpsi_2": K5_at_cpsi_2,
         "T_kk_null": nec_d['T_kk_in_A'], "centrifugal_psi_exp": centrifugal_psi_exp, "ghy_flux_S3": ghy_flux_S3},
        "Interior A_Killing < 0 (Eq. 85), deductive 5D curved-S^2 Kasner K_5, variational NEC (Eqs. 88, 149), e^{-3 psi} & O(4) S^3 flux",
        thm23_ok
    )
    state.deduce_corollary("interior_nec_centrifugal_and_instanton_flux", "interior_signature_flip_and_nec_violation",
                           (A_interior_sign, A_exterior_sign, A_interior_formula_diff, kasner_K5_diff,
                            quartic_sos_diff, exp_sos_diff, K5_at_cpsi_2, nec_null_diff, eq88_tt_diff, eq88_rr_diff),
                           (-1, 1, 0, 0, 0, 0, 4, 0, 0, 0),
                           "Eq. (85) A == -e^{-psi+2nu} < 0 inside, deductive 5D Kasner K_5, K_5(2)==4 & Hilbert variation Eqs. (88, 149)")
    state.deduce_corollary("interior_nec_centrifugal_and_instanton_flux", "routhian_centrifugal_barrier_scaling",
                           (centrifugal_psi_exp, centrifugal_routhian_diff), (-3, 0),
                           "Exponent count from G5_stat: fibre mass term e^{-2 psi} times rescaled inverse angular metric e^{-psi}/r^2 (Jordan areal radius) gives e^{-3 psi}/r^2; consistent with -2 beta_chrono of THM 13")
    state.deduce_corollary("interior_nec_centrifugal_and_instanton_flux", "euclidean_o4_throat_flux_quantization",
                           (harmonic_check, throat_profile_diff, vol_S3, ghy_flux_S3), (0, 0, 2 * sp.pi**2, 4 * sp.pi**2 * C_throat),
                           "4D Euclidean harmonic throat psi'(rho)==C_throat/rho^3 over Vol(S^3)==2 pi^2 yields topological flux 4 pi^2 C_throat")

    # =========================================================================
    # THM 24: ISO(3,1) Warped Fiber Curvature G^(5)_{22} = +6(sigma')^2 >= 0 &
    #         Finite-Volume Orbifold Yukawa RKHS Closure to THM #07 Cauchy-Gram Minor
    # =========================================================================
    t_sub = time.time()
    print("  [LOG] Module 5 -> Computing THM 24 (ISO(3,1) Warped Fiber G^(5)_{22} = 6(sigma')^2 & Cauchy-Gram Closure to THM #07)...")
    x0_w, y_w, x1_w, x2_w, x3_w = sp.symbols('x0_w y_w x1_w x2_w x3_w', real=True)
    W_y = sp.Function('W_y')(y_w)
    G5_warp = _FastDiagGeometry(
        [-W_y**2, sp.Integer(-1), W_y**2, W_y**2, W_y**2],
        [x0_w, y_w, x1_w, x2_w, x3_w],
        W_y**4
    )
    sigma_prime_sq = (sp.diff(W_y, y_w) / W_y)**2
    c_warp_22 = sp.simplify(G5_warp.Ein[1, 1] / sigma_prime_sq)

    m_vec = sorted(list(cauchy_minors_thm7[2].free_symbols), key=str)  # [mu1, mu2, mu3] from THM #07
    y_int, L0_rk, m_sum = sp.symbols('y_int L0_rk m_sum', positive=True)
    antideriv_yuk = -sp.exp(-m_sum * y_int) / m_sum
    ftc_check = sp.expand(sp.diff(antideriv_yuk, y_int) - sp.exp(-m_sum * y_int))
    K_finite_L = antideriv_yuk.subs(y_int, sp.pi * L0_rk) - antideriv_yuk.subs(y_int, 0)
    K_uv_lim = sp.limit(K_finite_L, L0_rk, sp.oo)  # 1 / m_sum

    # Positivity of the leading minors for arbitrary mu_a > 0 and L_0 > 0 (no sample point):
    # Andreief identity  det K_L[1..k] == (1/k!) int_{[0, pi L_0]^k} D_k(y)^2 d^k y  with  D_k(y) = det[e^{-mu_a y_b}],
    # verified term by term from the Leibniz expansion of D_k and the kernel integral K_L; D_k is not identically zero for
    # distinct mu_a because its mixed Taylor coefficient at y = 0 is the Vandermonde product.
    E_g = sp.symbols('E_g1:4', positive=True)                           # E_a stands for exp(-pi L_0 mu_a)
    y_g = sp.symbols('y_g1:4', positive=True)

    def _kernel_E(a_i, b_i):
        return (1 - E_g[a_i] * E_g[b_i]) / (m_vec[a_i] + m_vec[b_i])

    gram_kernel_link = [
        sp.simplify(K_finite_L.subs(m_sum, m_vec[a_i] + m_vec[b_i])
                    - _kernel_E(a_i, b_i).subs({E_g[c_i]: sp.exp(-sp.pi * L0_rk * m_vec[c_i]) for c_i in range(3)}))
        for a_i in range(3) for b_i in range(a_i, 3)
    ]
    K_E_mat = sp.Matrix(3, 3, _kernel_E)
    gram_leibniz_diffs, gram_andreief_diffs, gram_taylor_sq_diffs = [], [], []
    for k_m in (1, 2, 3):
        perms_k = [(p_k, sp.combinatorics.Permutation(list(p_k)).signature()) for p_k in itertools.permutations(range(k_m))]
        D_leibniz = sum(sgn_k * sp.prod([sp.exp(-m_vec[p_k[b_i]] * y_g[b_i]) for b_i in range(k_m)]) for p_k, sgn_k in perms_k)
        D_matrix = sp.Matrix(k_m, k_m, lambda a_i, b_i: sp.exp(-m_vec[a_i] * y_g[b_i])).det()
        gram_leibniz_diffs.append(sp.simplify(sp.expand(D_leibniz - D_matrix)))
        # int of each product term of D_k^2 factorises into kernel integrals K_L(mu_{sigma(b)}, mu_{tau(b)}):
        andreief_k = sum(
            sg_1 * sg_2 * sp.prod([_kernel_E(p_1[b_i], p_2[b_i]) for b_i in range(k_m)])
            for p_1, sg_1 in perms_k for p_2, sg_2 in perms_k
        ) / sp.factorial(k_m)
        gram_andreief_diffs.append(sp.cancel(sp.together(andreief_k - K_E_mat[:k_m, :k_m].det())))
        taylor_k = D_matrix
        for b_i in range(k_m):
            taylor_k = sp.diff(taylor_k, y_g[b_i], b_i)
        taylor_k = taylor_k.subs({y_g[b_i]: 0 for b_i in range(k_m)})
        vandermonde_k = sp.prod([m_vec[a_i] - m_vec[b_i] for a_i in range(k_m) for b_i in range(a_i + 1, k_m)])
        gram_taylor_sq_diffs.append(sp.expand(taylor_k**2 - vandermonde_k**2))
    fin_minors_pos = (
        all(d_k == 0 for d_k in gram_kernel_link) and gram_leibniz_diffs == [0, 0, 0]
        and gram_andreief_diffs == [0, 0, 0] and gram_taylor_sq_diffs == [0, 0, 0]
    )

    G_uv_mat = sp.Matrix(3, 3, lambda a_i, b_i: K_uv_lim.subs(m_sum, m_vec[a_i] + m_vec[b_i]))
    prod_diag_norms = G_uv_mat[0, 0] * G_uv_mat[1, 1] * G_uv_mat[2, 2]
    det_corr_uv = sp.factor(G_uv_mat.det() / prod_diag_norms)
    num_diff, _ = sp.fraction(sp.together(det_corr_uv - cauchy_minors_thm7[2]))
    closure_diff = sp.expand(num_diff)
    print(f"    [SUB-LOG THM 24.1] ISO(3,1) G^(5)_22/(sigma')^2={c_warp_22} & Cauchy-Gram closure diff={closure_diff} "
          f"checked in {time.time() - t_sub:.2f}s")

    thm24_ok = _require_all("THM 24", [
        ("c_warp_22 == 6", c_warp_22 == 6, c_warp_22),
        ("ftc_check == 0", ftc_check == 0, ftc_check),
        ("K_uv_lim == 1/m_sum", K_uv_lim == 1 / m_sum, K_uv_lim),
        ("Andreief identity for the k = 1, 2, 3 leading minors & non-vanishing Vandermonde Taylor coefficient", fin_minors_pos,
         (gram_kernel_link, gram_leibniz_diffs, gram_andreief_diffs, gram_taylor_sq_diffs)),
        ("closure_diff == 0", closure_diff == 0, closure_diff),
    ])

    state.deduce_theorem(
        "iso31_warped_fiber_and_cauchy_gram_closure",
        {"c_warp_22": c_warp_22, "K_finite_L": K_finite_L, "det_corr_uv": det_corr_uv, "closure_diff": closure_diff},
        "ISO(3,1) warped fiber G^(5)_{22} == +6(sigma')^2 >= 0 & exact orbifold Yukawa RKHS closure to THM #07 Cauchy-Gram minor",
        thm24_ok
    )
    state.deduce_corollary("iso31_warped_fiber_and_cauchy_gram_closure", "iso31_warped_fiber_positivity",
                           c_warp_22, 6,
                           "Poincare-invariant ISO(3,1) warped 5D metric has strictly non-negative fiber Einstein curvature G^(5)_{22} == 6(sigma')^2 >= 0")
    state.deduce_corollary("iso31_warped_fiber_and_cauchy_gram_closure", "finite_volume_rkhs_gram_positivity",
                           (ftc_check, K_uv_lim, fin_minors_pos), (0, 1 / m_sum, True),
                           "Finite-volume kernel K_L(mu_a, mu_b) == (1 - e^{-pi L_0(mu_a+mu_b)})/(mu_a+mu_b); leading minors == (1/k!) int D_k^2 > 0 for all L_0 > 0 and distinct mu_a > 0 (Andreief identity)")
    state.deduce_corollary("iso31_warped_fiber_and_cauchy_gram_closure", "asymptotic_closure_to_thm7_cauchy_vandermonde",
                           closure_diff, 0,
                           "Normalized UV correlation Gram determinant det(C_uv) matches THM #07 Cauchy-Vandermonde minor identically (Q.E.D. Loop Closed!)")


def print_principia_summary(state: DeductiveState) -> None:
    banner("CBG PRINCIPIA GEOMETRICA: DEDUCTIVE DAG SUMMARY")
    print(f"  * Total Explicit Axioms Injected : {len(state.axioms)} ({', '.join(state.axioms.keys())})")
    print(f"  * Total Pure Deductive Theorems  : {state.thm_count}")
    print(f"  * Total Pure Deductive Corollaries: {state.cor_count}")
    print("-" * 88)
    for idx, (thm_key, thm_val) in enumerate(state.theorems.items(), start=1):
        sub_cors = [sub_k for (p_k, sub_k) in state.corollaries.keys() if p_k == thm_key]
        print(f"  [THM #{idx:02d}] {thm_key:<48} | {len(sub_cors)} Corollaries Verified [Q.E.D.]")
    print("=" * 88)
    print("  [STATUS] All registered theorems and corollaries verified symbolically.")
    print("           Declared regime conditions and inputs inside nodes: quasi-static limit (THM 14), sign convention of Eq. 107 and")
    print("           uniqueness of analytic continuation for zeta (THM 17), semigroup hypothesis (THM 18), radiation test fluid (THM 21),")
    print("           harmonic regime of the throat (THM 23).")
    print("=" * 88)


# ========================================================================================
# COLAB CELL 5 EXECUTION & FINAL PRINCIPIA GEOMETRICA VERIFICATION
# ========================================================================================
if __name__ == "__main__":
    if "_COLAB_CHECKPOINT_M4" not in globals():
        raise RuntimeError("Please run Colab Cell 4 (Module 4) first so `_COLAB_CHECKPOINT_M4` is in memory!")
    t0_m5 = time.time()
    master_state = copy.deepcopy(globals()["_COLAB_CHECKPOINT_M4"])
    print("  [INSTANT RESUME] Loaded Modules 1-4 state (19 THMs, 79 CORs) from `_COLAB_CHECKPOINT_M4`!")
    run_module_5(master_state)
    globals()["_COLAB_CHECKPOINT_M5"] = copy.deepcopy(master_state)
    print(f"\n  --> [COLAB CELL 5 READY] Module 5 completed in {time.time() - t0_m5:.2f}s "
          f"(Cumulative: {master_state.thm_count} THMs, {master_state.cor_count} CORs saved in `_COLAB_CHECKPOINT_M5`)\n")
    print_principia_summary(master_state)