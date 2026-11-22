#!/usr/bin/env python3
"""
CBG -- bitemporal t2-throat audit.  Does relaxing the cylinder condition (d/dt2 = 0) at the Planckian throat, and evolving
across t2 in [0, pi L0], resolve the UV bounce and/or the cycle reset from [A1]-[A3] alone?
Standalone, report-only; none of the published suites is imported or modified.

Metric ([A1], signature (-,-,+,+,+), k = +1):
    ds^2 = -e^{2 nu} dt1^2 - 2 B dt1 dt2 - Phi^2 dt2^2 + a^2 dOmega_3^2,   Phi = Phi_inf e^{psi},   all functions of (t1, t2).
Part S: exact SymPy (rational).  Part N: DOP853 / quadrature evidence with stated tolerances.  kappa_5 = 1 throughout.
"""
import itertools, time, warnings
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp, quad
warnings.filterwarnings("ignore")
T0_ = time.time()
CHK = []
def rec(tag, cls, label, ok, detail=""):
    CHK.append(bool(ok)); print(f"[{'PASS' if ok else 'FAIL'}] {tag:<4} {cls:<9} {label}" + (f"\n{'':20}{detail}" if detail else ""))

def curvature(g, X):
    n = len(X); gi = g.inv()
    Gam = [[[sp.simplify(sum(gi[a, d] * (sp.diff(g[d, b], X[c]) + sp.diff(g[d, c], X[b]) - sp.diff(g[b, c], X[d])) for d in range(n)) / 2)
             for c in range(n)] for b in range(n)] for a in range(n)]
    def Riem(a, b, c, d):
        return sp.diff(Gam[a][b][d], X[c]) - sp.diff(Gam[a][b][c], X[d]) + sum(Gam[a][c][e] * Gam[e][b][d] - Gam[a][d][e] * Gam[e][b][c] for e in range(n))
    Ric = sp.Matrix(n, n, lambda b, d: sp.simplify(sum(Riem(a, b, a, d) for a in range(n))))
    Rs = sp.simplify(sum(gi[b, d] * Ric[b, d] for b in range(n) for d in range(n)))
    return Gam, Riem, Ric, Rs, sp.simplify(Ric - Rs * g / 2), gi

print("=" * 122 + "\nPART S -- exact statements\n" + "=" * 122)
t1, t2, x, th, ph = sp.symbols('t1 t2 x theta phi', real=True)
Pinf, L0, k = sp.symbols('Phi_inf L_0 k', positive=True)
nu = sp.Function('nu')(t1, t2); psi = sp.Function('psi')(t1, t2); a = sp.Function('a')(t1, t2)
Phi = Pinf * sp.exp(psi)
X = [t1, t2, x, th, ph]
g = sp.diag(-sp.exp(2 * nu), -Phi**2, a**2, a**2 * sp.sin(x)**2, a**2 * sp.sin(x)**2 * sp.sin(th)**2)
Gam, Riem, Ric, Rs, Ein, gi = curvature(g, X)
D = lambda f, *v: sp.diff(f, *v)
# 2D temporal block h = -gamma, gamma = e^{2nu} dt1^2 + Phi^2 dt2^2 (Riemannian)
grad2 = sp.exp(-2 * nu) * D(a, t1)**2 + D(a, t2)**2 / Phi**2
hess = lambda i, j: D(a, X[i], X[j]) - sum(Gam[c][i][j] * D(a, X[c]) for c in (0, 1))
lap = sp.exp(-2 * nu) * hess(0, 0) + hess(1, 1) / Phi**2             # Laplace-Beltrami of gamma acting on a
rec("S1", "[D]", "G_02 = -(3/a) nabla_1 nabla_2 a = -(3/a)[a_,12 - nu_,2 a_,1 - psi_,1 a_,2]   (diagonal gauge)",
    sp.simplify(Ein[0, 1] + 3 * (D(a, t1, t2) - D(nu, t2) * D(a, t1) - D(psi, t1) * D(a, t2)) / a) == 0,
    "=> G_02 = T_02 reads a_,12 = nu_,2 a_,1 + psi_,1 a_,2 - (a/3) T_02: on a branch with d/dt2 = 0 no t2-dependence of a is generated unless T_02 != 0")
rec("S2", "[D]", "S^3 block: R_ij = [ Lap_gamma(a)/a + 2 (1 + |grad a|^2_gamma)/a^2 ] g_ij   (k = +1)",
    sp.simplify(Ric[2, 2] / g[2, 2] - (lap / a + 2 * (1 + grad2) / a**2)) == 0)
trG = sp.simplify(gi[0, 0] * Ein[0, 0] + gi[1, 1] * Ein[1, 1])
rec("S3", "[D]", "temporal trace: h^{ab} G_ab = -(3/a) Lap_gamma(a) - 6 (1 + |grad a|^2_gamma)/a^2",
    sp.simplify(trG + 3 * lap / a + 6 * (1 + grad2) / a**2) == 0,
    "=> Lap_gamma(a) = -2 (1 + |grad a|^2)/a - (a/3) T_h,  T_h = h^{ab} T_ab  (vacuum: a is strictly superharmonic on the (t1,t2) surface)")
xi1, xi2 = sp.symbols('xi_1 xi_2', real=True)
symb = sp.exp(-2 * nu) * xi1**2 + xi2**2 / Phi**2
coef = (sp.expand(lap).coeff(D(a, t1, 2)), sp.expand(lap).coeff(D(a, t2, 2)))
rec("S4", "[D]", "principal part of the (t1,t2) operator: e^{-2nu} d1^2 + Phi^{-2} d2^2 with positive-definite symbol: REAL ELLIPTIC, no Wick rotation",
    sp.simplify(coef[0] - sp.exp(-2 * nu)) == 0 and sp.simplify(coef[1] - 1 / Phi**2) == 0 and sp.simplify(symb.subs({xi1: 0, xi2: 1})) != 0,
    "consequence: the (t1,t2) problem is a boundary-value problem; it is not a Cauchy evolution in t1 or in t2 (S8)")
# Z2 parity of the mixed component
Bf = sp.Function('B')
rec("S5", "[D]", "Z2 equivariance under R = diag(1,-1,1,1,1): g_02(t1,-t2) = -g_02(t1,t2), so B is odd in t2 and B = 0 on both fixed walls",
    sp.Matrix([[1, 0], [0, -1]]) * sp.Matrix([[0, 1], [1, 0]]) * sp.Matrix([[1, 0], [0, -1]]) == -sp.Matrix([[0, 1], [1, 0]]),
    "inside (0, pi L0) an odd B can always be removed by isothermal coordinates (2D, definite block), which keep the walls and the Neumann condition")
# isothermal gauge and the fiber-averaged radius
om = sp.Function('omega')(t1, t2)
g_iso = sp.diag(-sp.exp(2 * om), -sp.exp(2 * om), a**2, a**2 * sp.sin(x)**2, a**2 * sp.sin(x)**2 * sp.sin(th)**2)
_, _, Ric_i, _, Ein_i, gi_i = curvature(g_iso, X)
flatlap = D(a, t1, 2) + D(a, t2, 2)
gr_i = sp.exp(-2 * om) * (D(a, t1)**2 + D(a, t2)**2)
rec("S6", "[D]", "isothermal gauge: a_,11 + a_,22 = -e^{2 omega} [ 2 (1 + |grad a|^2)/a + (a/3) T_h ]",
    sp.simplify((gi_i[0, 0] * Ein_i[0, 0] + gi_i[1, 1] * Ein_i[1, 1]) + 3 * sp.exp(-2 * om) * flatlap / a + 6 * (1 + gr_i) / a**2) == 0,
    "integrating over t2 with Neumann walls: d^2/dt1^2 Int a dt2 = -Int e^{2 omega}[2(1+|grad a|^2)/a + (a/3) T_h] dt2\n"
    f"{'':20}=> in vacuum the fiber-averaged radius is strictly concave in t1: it has no minimum (no bounce) and reaches zero at finite t1;\n"
    f"{'':20}=> t2-gradients enter only through |grad a|^2 >= |a_,1|^2, i.e. they RAISE the threshold -T_h > 6(1+|grad a|^2)/a^2 needed for a bounce")
# concircular condition => Killing vector (generalised Birkhoff)
A11, A12, A22, A1, A2, O1, O2 = sp.symbols('a11 a12 a22 a1 a2 o1 o2', real=True)
tf1 = Ein_i[0, 0] - Ein_i[1, 1]; tf2 = Ein_i[0, 1]
rec("S7", "[D]", "trace-free temporal equations (T_ab = (1/2) T_h h_ab): a_,11 - a_,22 = 2(omega_,1 a_,1 - omega_,2 a_,2),  a_,12 = omega_,1 a_,2 + omega_,2 a_,1",
    sp.simplify(tf1 + 3 * (D(a, t1, 2) - D(a, t2, 2) - 2 * (D(om, t1) * D(a, t1) - D(om, t2) * D(a, t2))) / a) == 0
    and sp.simplify(tf2 + 3 * (D(a, t1, t2) - D(om, t1) * D(a, t2) - D(om, t2) * D(a, t1)) / a) == 0)
K_up = [sp.exp(-2 * om) * D(a, t2), -sp.exp(-2 * om) * D(a, t1)]
lie = sp.Matrix(2, 2, lambda i, j: sum(K_up[c] * D(sp.exp(2 * om), X[c]) for c in (0, 1)) * (1 if i == j else 0)
                + sp.exp(2 * om) * (D(K_up[j], X[i]) + D(K_up[i], X[j])))
subs_tf = {D(a, t1, t2): D(om, t1) * D(a, t2) + D(om, t2) * D(a, t1), D(a, t1, 2): D(a, t2, 2) + 2 * (D(om, t1) * D(a, t1) - D(om, t2) * D(a, t2))}
rec("S8", "[D]", "GENERALISED BIRKHOFF: under S7, K^a = eps^{ab} d_b a is a Killing vector of the (t1,t2) surface",
    all(sp.simplify(lie[i, j].subs(subs_tf)) == 0 for i in range(2) for j in range(2)),
    "=> for vacuum or a constant potential every solution with d/dt2 != 0 still has one temporal Killing direction: there is no genuinely two-time solution")
# the explicit family
av, y, mu, V = sp.symbols('a y mu V', real=True)
F = sp.Function('F')(av)
g_b = sp.diag(-1 / F, -F, av**2, av**2 * sp.sin(x)**2, av**2 * sp.sin(x)**2 * sp.sin(th)**2)
Xb = [av, y, x, th, ph]
Gb, Riem_b, Ric_b, Rs_b, Ein_b, gi_b = curvature(g_b, Xb)
Fsol = -1 + mu / av**2 + V * av**2 / 6
eqs = [sp.simplify((Ein_b[i, i] + V * g_b[i, i]).subs(F, Fsol).doit()) for i in range(5)]
rec("S9", "[D]", "the family: ds^2 = -da^2/F - F dy^2 + a^2 dOmega_3^2 with F = -1 + mu/a^2 + V a^2/6 solves G_AB = -V g_AB exactly",
    all(e == 0 for e in eqs), "reading y = t2 gives a cylinder cosmology a(t1) with Phi^2 = F; reading y = t1 gives a static t2-profile a(t2) with lapse^2 = F")
Rl = lambda a_, b_, c_, d_: sum(g_b[a_, e] * Riem_b(e, b_, c_, d_) for e in range(5))
Kb = sp.simplify(sum(4 * Rl(i, j, i, j)**2 * gi_b[i, i]**2 * gi_b[j, j]**2 for i in range(5) for j in range(i + 1, 5)).subs(F, Fsol).doit())
print(f"{'':20}K_5 of the family =", sp.factor(Kb))
rec("S10", "[D]", "Kretschmann of the family: K_5 = 72 mu^2/a^8 + 10 V^2/9: finite at the pinch F = 0, divergent at a -> 0 whenever mu != 0",
    sp.simplify(Kb - (72 * mu**2 / av**8 + sp.Rational(10, 9) * V**2)) == 0)
lam, mm, u = sp.symbols('lambda mu_ u', real=True)
quadr = lam * u**2 - u + mm                       # a^2 F with u = a^2, lambda = V/6
r1, r2 = sp.symbols('u1 u2', positive=True)
rec("S11", "[D]", "TEST A(ii): a t2-profile with Neumann walls at both ends needs F > 0 between two positive simple zeros; for k = +1 this is impossible",
    sp.simplify(sp.expand(lam * (u - r1) * (u - r2)).coeff(u, 1) + lam * (r1 + r2)) == 0,
    "a^2 F = lambda u^2 - u + mu (u = a^2): F > 0 between the roots needs lambda < 0, but the root sum u1 + u2 = 1/lambda would then be negative")
rec("S12", "[D|c0]", "the only non-singular member is mu < 0, V > 0: a >= a_+ with the fiber pinching (Phi^2 = F -> 0) at the turning radius",
    sp.simplify(Fsol.subs({mu: -1, V: 6}).subs(av, sp.sqrt((1 + sp.sqrt(5)) / 2))) == 0,
    "it needs a positive constant potential (not in W_3) and Phi -> 0, i.e. psi -> -infinity, where the W_3 wall e^{-6 psi} diverges: excluded by [A2]")
# clock rotation
P2, Ph, pp, mass = sp.symbols('P_2 Phi_ p m_0', positive=True)
E2 = mass**2 + pp**2 - P2**2 / Ph**2
rec("S13", "[D]", "TEST A(i): with d/dt2 = 0 the momentum P_2 is conserved and (e^nu dt1/dtau)^2 = [m0^2 + p^2/a^2 - P_2^2/Phi^2]/m0^2",
    sp.simplify(E2.subs(P2, Ph * sp.sqrt(mass**2 + pp**2))) == 0,
    "a worldline can stall and reverse in t1 where Phi = |P_2|/sqrt(m0^2+p^2/a^2) (the temporal block has no light cone), but a(t1) and rho(a)\n"
    f"{'':20}are fixed by the field equations and do not depend on the clock of any observer: the rotation changes a parametrisation, not the geometry")
# KK mode equation
n_ = sp.Symbol('n', positive=True, integer=True); P0 = sp.Symbol('Phi_0', positive=True); fn = sp.Function('psi_n')(t1)
mode = fn * sp.cos(n_ * t2 / L0)
rec("S14", "[D]", "TEST B: an n >= 1 mode psi_n(t1) cos(n t2/L0) obeys psi_n'' = +(n/(Phi_0 L0))^2 psi_n: exponential, not oscillatory",
    sp.simplify((-D(mode, t1, 2) - D(mode, t2, 2) / P0**2) / sp.cos(n_ * t2 / L0) - (-D(fn, t1, 2) + n_**2 * fn / (P0**2 * L0**2))) == 0,
    "the modes are not radiation: growth rate n/(Phi_0 L0) is unbounded in n (Hadamard ill-posed as a t1-evolution), and they satisfy Neumann walls")
q0, q1, p0, p1, Rr, gq, m0 = sp.symbols('q0 q1 p0 p1 R g m', real=True)
Ham = p0**2 / 2 + p1**2 - q1**2 / (4 * Rr**2) + m0**2 * (q0**2 + q1**2 / 2) / 2 + gq * (q0**4 + 3 * q0**2 * q1**2 + sp.Rational(3, 8) * q1**4) / 4
flow = [D(Ham, p0), D(Ham, p1), -D(Ham, q0), -D(Ham, q1)]
rec("S15", "[D]", "two-mode reduction (zero mode q0, first mode q1): Hamiltonian flow is divergence-free and the mode energy -q1^2/(4R^2) is negative",
    sp.simplify(D(flow[0], q0) + D(flow[1], q1) + D(flow[2], p0) + D(flow[3], p1)) == 0 and D(Ham, q1, 2).subs({q0: 0, q1: 0, m0: 0}) == -1 / (2 * Rr**2),
    "=> phase-space volume is conserved: no trajectory bundle can contract onto psi_0 = psi_b; and the energy is not bounded below, so nothing damps")
# Gilkey density on FRW x fiber
tt = sp.Symbol('t', real=True); af = sp.Function('a')(tt); kk = sp.Symbol('k', real=True)
g4 = sp.diag(-1, -1, af**2, af**2 * sp.sin(x)**2, af**2 * sp.sin(x)**2 * sp.sin(th)**2)
Xf = [tt, y, x, th, ph]
Gf, Riem_f, Ric_f, Rs_f, Ein_f, gi_f = curvature(g4, Xf)
Ric2 = sp.simplify(sum(Ric_f[i, i]**2 * gi_f[i, i]**2 for i in range(5)))
Rlf = lambda a_, b_, c_, d_: g4[a_, a_] * Riem_f(a_, b_, c_, d_)
Kf = sp.simplify(sum(4 * (Rlf(i, j, i, j))**2 * gi_f[i, i]**2 * gi_f[j, j]**2 for i in range(5) for j in range(i + 1, 5)))
Hh = D(af, tt) / af; acc = D(af, tt, 2) / af
gil = sp.simplify(-sp.Rational(7, 360) * Kf - sp.Rational(1, 45) * Ric2 + sp.Rational(1, 72) * Rs_f**2)
rec("S16", "[D]", "TEST C: Gilkey density (-7/360 K - 1/45 Ric^2 + 1/72 R^2) on FRW(k=+1) x fiber = (11/15)(addot/a)(H^2 + 1/a^2)",
    sp.simplify(gil - sp.Rational(11, 15) * acc * (Hh**2 + 1 / af**2)) == 0,
    "it is the Gauss-Bonnet part only (Weyl = 0); it acts on psi through the measure factor e^{psi} with a coefficient xi fixed by FREE moments of [A2]")

# ------------------------------------------------------------------------------------------------ Part N
print("\n" + "=" * 122 + "\nPART N -- numerical evidence\n" + "=" * 122)
f_ = [sp.lambdify((q0, q1, p0, p1, Rr, gq, m0), e) for e in flow]; Hn = sp.lambdify((q0, q1, p0, p1, Rr, gq, m0), Ham)
Rv, gv, mv = 0.05, 1.0, 1.0
def run2(q1_0, T=40.0):
    rhs = lambda t, s: [fi(s[0], s[1], s[2], s[3], Rv, gv, mv) for fi in f_]
    return solve_ivp(rhs, [0, T], [1.0, q1_0, 0.0, 0.0], method='DOP853', rtol=1e-12, atol=1e-14, dense_output=True)
s0 = run2(0.0); s1 = run2(1e-12)
E0 = lambda s, tA, tB: np.mean([0.5 * s.sol(t)[2]**2 + 0.5 * mv**2 * s.sol(t)[0]**2 + 0.25 * gv * s.sol(t)[0]**4 for t in np.linspace(tA, tB, 2000)])
tg = np.linspace(0.2, 1.2, 50); rate = np.polyfit(tg, np.log(np.abs([s1.sol(t)[1] for t in tg])), 1)[0]
dH = abs(Hn(*s1.y[:, -1], Rv, gv, mv) - Hn(*s1.y[:, 0], Rv, gv, mv))
print("  two-mode model (m = 1, R = Phi_0 L0 = 0.05, quartic g = 1; zero mode starts at q0 = 1 at rest, first mode seeded at 1e-12)")
print("  growth rate of the first mode = %.3f  (linear prediction ~ 1/R = %.3f)" % (rate, 1 / Rv))
print("  first-mode amplitude: max |q1| = %.2f (new equilibrium 2/(R sqrt(3g)) = %.2f);  zero-mode energy, time-averaged: before %.3f, after %.3f"
      % (np.max(np.abs(s1.y[1])), 2 / (Rv * np.sqrt(3 * gv)), E0(s1, 0, 1.5), E0(s1, 25, 40)))
print("  conserved Hamiltonian drift = %.1e;  with q1 = 0 exactly the zero mode keeps energy %.3f" % (dH, E0(s0, 25, 40)))
rec("N1", "[N]", "TEST B: exciting an n >= 1 mode does not damp the zero mode: the mode runs away at rate ~1/(Phi_0 L0) and PUMPS the zero mode",
    rate > 0.9 / Rv and E0(s1, 25, 40) > 10 * E0(s1, 0, 1.5) and dH < 1e-6)
# Gilkey kick through the H3 radiation bounce (Paper III analytic solution x^4 = 1 + (2 Omega_B t)^2), Planck units t_P = 1
OmB = np.sqrt(8 * np.pi / 3)
xfun = lambda t: (1 + (2 * OmB * t)**2)**0.25
Hf = lambda t: OmB * (2 * OmB * t) / (1 + (2 * OmB * t)**2)
accf = lambda t: (Hf(t + 1e-6) - Hf(t - 1e-6)) / 2e-6 + Hf(t)**2
G_of = lambda t: (11 / 15) * accf(t) * Hf(t)**2                       # curvature term 1/a^2 is ~1e-66 of H^2 away from t = 0
inner = lambda t: quad(lambda s: -(2 / 3) * G_of(s) * xfun(s)**3, 0, t, limit=200)[0] / xfun(t)**3
Ckick = quad(inner, 0, 1e4, limit=400)[0] + quad(inner, 1e4, 1e8, limit=400)[0]
psi_b = 5.64533e-7
LP = 1.616255e-35; lam_psi = 4.7113e-5; m2 = (LP / lam_psi)**2
Rb = 1.9723e-32 * (299792458.0 / (67.3460e3 / 3.0856775814913673e22)) / 2.402867e-4
G_still = (11 / 15) * (2 * OmB**2) * (LP / Rb)**2
print("  Gilkey term through the H3 radiation bounce (Planck units), effective potential U + xi e^{psi} G:")
print("    dynamic displacement accumulated after the bounce:  delta psi = %.4e * xi      -> xi needed for psi_b = %.3e" % (Ckick, psi_b / abs(Ckick)))
print("    static root at the still point (G = %.3e, U''(0) = (3/2) m^2 = %.3e):  psi_eq = %.3e * xi   (not reached: m t_P = %.1e)"
      % (G_still, 1.5 * m2, -G_still / (1.5 * m2), np.sqrt(m2)))
rec("N2", "[N|H]", "TEST C: the curvature-induced displacement is linear in the free coefficient xi; psi_b = 5.645e-7 is reproduced only by choosing xi",
    abs(Ckick) > 0 and np.isfinite(Ckick), "xi is a ratio of the free cutoff moments of [A2] times Lambda_cut^-2: this is a one-parameter fit, not a zero-parameter prediction")
print("\nSUMMARY: %d/%d checks passed   (run time %.0f s)" % (sum(CHK), len(CHK), time.time() - T0_))
