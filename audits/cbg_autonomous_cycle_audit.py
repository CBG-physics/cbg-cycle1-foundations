#!/usr/bin/env python3
"""
CBG -- audit of an autonomous bounce -> apogee -> bounce cycle in the Paper 0 homogeneous system
(THM #21), k = +1, WITH khronon charge J_Theta, Branch-A dust and radiation.  Report-only:
this file does not replace or modify any Paper III suite.

Lagrangian (Paper 0, Thm 5.1(ii), Jordan frame, plus matter with Phi^0 normalisation, H1a):
    L = a e^psi [ -6 adot^2/N - 6 a adot psidot/N + 6 k N - 2 a^2 N V(psi, A) ] - 2 N (R/a + M),
    A = N^2 / Thetadot^2.
Potential (H2, taken in this Jordan Lagrangian):
    V = (3/4) m^2 [psi - psi_t(A)]^2,  psi_t(A) = psi_max [exp(-A^2/Ac^2) - exp(-1/Ac^2)].
Part S: exact SymPy statements.  Part N: DOP853 evidence, tolerances stated.
"""
import warnings
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
warnings.filterwarnings("ignore")
OK = []
def rec(tag, label, ok, detail=""):
    OK.append(bool(ok)); print(f"[{'PASS' if ok else 'FAIL'}] {tag:<4} {label}" + (f"\n{'':12}{detail}" if detail else ""))

# ------------------------------------------------------------------------------- Part S
print("=== Part S: exact statements ===")
t = sp.Symbol('t')
a = sp.Function('a')(t); ps = sp.Function('psi')(t); Th = sp.Function('Theta')(t)
N, k = sp.symbols('N k')
m, M, R, pmax, Ac = sp.symbols('m M R psi_max A_c', positive=True)
Asym = sp.Symbol('A', positive=True)
psit = lambda A_: pmax * (sp.exp(-A_**2 / Ac**2) - sp.exp(-1 / Ac**2))
Vfun = lambda p_, A_: sp.Rational(3, 4) * m**2 * (p_ - psit(A_))**2
def Lag(Nv, ad, pd, Td):
    return (a * sp.exp(ps) * (-6 * ad**2 / Nv - 6 * a * ad * pd / Nv + 6 * k * Nv - 2 * a**2 * Nv * Vfun(ps, Nv**2 / Td**2))
            - 2 * Nv * (R / a + M))
ad_, pd_, Td_ = a.diff(t), ps.diff(t), Th.diff(t)
L = Lag(N, ad_, pd_, Td_)
rec("S1", "time reversal: L(N, -adot, -psidot, -Thetadot) = L, so the equations are invariant under t -> -t",
    sp.simplify(Lag(N, -ad_, -pd_, -Td_) - L) == 0)
J = sp.diff(L, Td_).subs(N, 1)
A_of = 1 / Td_**2
VA = sp.diff(Vfun(ps, Asym), Asym)
Tpos = sp.Symbol('Thetadot', positive=True)
rec("S2", "Theta is cyclic: J_Theta = dL/dThetadot = 4 a^3 e^psi A^(3/2) dV/dA is conserved",
    sp.diff(L, Th) == 0 and sp.simplify((J - (4 * a**3 * sp.exp(ps) * Asym**sp.Rational(3, 2) * VA).subs(Asym, A_of)).subs(Td_, Tpos)) == 0)

# equations at N = 1 in first-order symbols
av, pv, a1, p1, T1, a2, p2, T2 = sp.symbols('a_v psi_v a1 p1 T1 a2 p2 T2')
L1 = L.subs(N, 1)
def EL(q):
    return sp.diff(L1, q) - sp.diff(sp.diff(L1, q.diff(t)), t)
def flat(e):
    return e.subs({a.diff(t, 2): a2, ps.diff(t, 2): p2, Th.diff(t, 2): T2}).subs({a.diff(t): a1, ps.diff(t): p1, Th.diff(t): T1}).subs({a: av, ps: pv})
Ea, Ep, ET = flat(EL(a)), flat(EL(ps)), flat(EL(Th))
EN = flat(sp.diff(L, N).subs(N, 1))
Aval = 1 / T1**2
rho_r = sp.exp(-pv) * R / av**4
rho_m = sp.exp(-pv) * M / av**3
V_ = Vfun(pv, Asym); Vp_ = sp.diff(V_, pv); VA_ = sp.diff(V_, Asym)
rho_T = 2 * Asym * VA_
Ccon = 3 * (a1**2 / av**2 + a1 * p1 / av + k / av**2) - (V_ + rho_T + rho_r + rho_m)
rec("S3", "lapse constraint: 3(H^2 + H psidot + k/a^2) = V + 2A dV/dA + rho_r + rho_m,  rho_r = e^-psi R/a^4, rho_m = e^-psi M/a^3",
    sp.simplify(EN / (2 * av**3 * sp.exp(pv)) - Ccon.subs(Asym, Aval)) == 0,
    "the khronon enters as a pressureless component rho_Theta = 2A dV/dA = J/(2 a^3 e^psi sqrt(A))")
sol = sp.solve([Ea, Ep, ET], [a2, p2, T2], dict=True)[0]
fix = {a1: 0, p1: 0}
kfix = sp.solve(Ccon.subs(fix), k)[0]                         # constraint solved for k/a^2 at the still point
acc = sp.simplify((sol[a2] / av).subs(fix).subs(T1, 1 / sp.sqrt(Asym)).subs(k, kfix))
psd = sp.simplify(sol[p2].subs(fix).subs(T1, 1 / sp.sqrt(Asym)).subs(k, kfix))
Tdd = sp.simplify(sol[T2].subs(fix))
rec("S4", "still point (adot = psidot = 0): addot/a = (1/3)[V_psi - rho_r - rho_m - rho_Theta]",
    sp.simplify(acc - (Vp_ - rho_r - rho_m - rho_T) / 3) == 0,
    "dust enters with coefficient 1 (not 1/2), exactly like the khronon term")
rec("S5", "still point: psiddot = (2/3)[V - V_psi] + (1/3)(rho_m + rho_Theta), and Thetaddot = 0 (A is stationary there)",
    sp.simplify(psd - (sp.Rational(2, 3) * (V_ - Vp_) + (rho_m + rho_T) / 3)) == 0 and Tdd == 0,
    "=> the state (a, psi, A) with all velocities zero is a fixed set of time reversal: the solution is even about it")
x = sp.Symbol('x', positive=True)
rT = sp.simplify(rho_T / (6 * m**2 * pmax * (pv - psit(Asym))))
peak = sp.solve(sp.diff(x * sp.exp(-x), x), x)
rec("S6", "khronon energy under H2: rho_Theta = 6 m^2 psi_max (psi - psi_t) (A^2/Ac^2) exp(-A^2/Ac^2), so |rho_Theta| <= (6/e) m^2 psi_max |psi - psi_t|",
    sp.simplify(rT - (Asym**2 / Ac**2) * sp.exp(-Asym**2 / Ac**2)) == 0 and peak == [1])
d_ = sp.Symbol('delta', real=True)
low = sp.Rational(3, 4) * m**2 * d_**2 - (6 / sp.E) * m**2 * pmax * sp.Abs(d_)
dmin = sp.solve(sp.diff(sp.Rational(3, 4) * m**2 * x**2 - (6 / sp.E) * m**2 * pmax * x, x), x)[0]
rec("S7", "NO-GO for J_Theta < 0: V + rho_Theta >= -(12/e^2) m^2 psi_max^2 (bounded); it cannot offset kinetic energy growing as a^-6",
    sp.simplify((sp.Rational(3, 4) * m**2 * dmin**2 - (6 / sp.E) * m**2 * pmax * dmin) + 12 * m**2 * pmax**2 / sp.E**2) == 0)
# radiation-curvature bound
ab, a0, pb, p0, OK_, Or_, H0 = sp.symbols('a_b a_0 psi_b psi_0 Omega_K Omega_r H_0', positive=True)
Rv = sp.solve(sp.Eq(sp.exp(-p0) * R / a0**4, 3 * Or_ * H0**2), R)[0]
lhs = (sp.exp(-pb) * Rv / ab**4) * ab**2 / 3                 # rho_r(a_b) a_b^2 / 3  <= 1 at a still point with V + rho_m + rho_Theta >= 0
lhs = sp.simplify(lhs.subs(H0, sp.sqrt(1 / (OK_ * a0**2))))  # k/a_0^2 = Omega_K H_0^2
rec("S8", "RADIATION-CURVATURE BOUND: a still point with V + rho_m + rho_Theta >= 0 and conserved R needs (a_0/a_b)^2 <= (Omega_K/Omega_r) e^(psi_b - psi_0)",
    sp.simplify(lhs - (a0 / ab)**2 * sp.exp(p0 - pb) * Or_ / OK_) == 0)
PhiInf = 2.4029e-4; OmK = PhiInf**2; OmR = 9.2e-5
zmax = lambda dpsi, ok=OmK: np.sqrt(ok / OmR) * np.exp(dpsi / 2)
need = lambda z, ok=OmK: 2 * np.log(z / np.sqrt(ok / OmR))
print(f"{'':12}[N] Paper III inputs Omega_K0 = Phi_inf^2 = {OmK:.3e}, Omega_r0 ~ {OmR:.1e}:  (1+z_b)_max = {zmax(0):.4f} at psi_b = psi_0,  {zmax(2):.4f} at psi_b - psi_0 = 2")
print(f"{'':12}[N] psi_b - psi_0 needed for a bounce before z* = 1089.8: {need(1090.8):.1f};  before T ~ 4.1 TeV (z ~ 1.7e16): {need(1.7e16):.1f}")
rec("S9", "[N] with Paper III's curvature and psi_b - psi_0 < 2 the bound gives (1+z_b) < 1: no still-point bounce lies in the past", zmax(2) < 1)

# ------------------------------------------------------------------------------- Part N
print("\n=== Part N: DOP853, A = 1 (J_Theta = 0), m = 1, k = +1 ===")
subsN = {k: 1, m: 1, pmax: 0, Ac: 1, T1: 1}
fa = sp.lambdify((av, pv, a1, p1, M, R), sol[a2].subs(subsN), 'numpy')
fp = sp.lambdify((av, pv, a1, p1, M, R), sol[p2].subs(subsN), 'numpy')
fC = sp.lambdify((av, pv, a1, p1, M, R), Ccon.subs(Asym, 1).subs(subsN), 'numpy')
def rhs(Mv, Rv_):
    return lambda t_, s: [s[2], s[3], fa(s[0], s[1], s[2], s[3], Mv, Rv_), fp(s[0], s[1], s[2], s[3], Mv, Rv_)]
def a_still(p_, Mv, Rv_):
    g = lambda x_: 3 / x_**2 - 0.75 * p_**2 - np.exp(-p_) * (Mv / x_**3 + Rv_ / x_**4)
    xs = np.logspace(-3.0007, 6.0003, 4001); gs = g(xs)
    return [brentq(g, xs[i], xs[i + 1], xtol=1e-15) for i in range(len(xs) - 1) if gs[i] * gs[i + 1] < 0][-1]
def first_max(p_, Mv, Rv_, tmax=200):
    f = rhs(Mv, Rv_); s0 = [a_still(p_, Mv, Rv_), p_, 0.0, 0.0]
    ev = lambda t_, s: s[2]; ev.direction = -1; ev.terminal = True
    s1 = solve_ivp(f, [0, 1e-6], s0, method='DOP853', rtol=1e-13, atol=1e-15).y[:, -1]
    so = solve_ivp(f, [1e-6, tmax], s1, method='DOP853', rtol=1e-12, atol=1e-14, events=[ev])
    return so.t_events[0][0], so.y_events[0][0]
allok = True; mults = []
for Mv, Rv_, lo, hi in ((0.0, 0.0, 1.1073, 1.1147), (0.5, 0.0, 1.1147, 1.1221), (1.0, 1.0, 1.1590, 1.1664)):
    pbv = brentq(lambda p_: first_max(p_, Mv, Rv_)[1][3], lo, hi, xtol=1e-15, rtol=1e-15)
    ab_ = a_still(pbv, Mv, Rv_); tap, sa = first_max(pbv, Mv, Rv_); f = rhs(Mv, Rv_)
    accb = f(0, [ab_, pbv, 0, 0])[2] / ab_; acca = f(0, [sa[0], sa[1], 0, 0])[2] / sa[0]
    mn = lambda t_, s: s[2]; mn.direction = +1
    so = solve_ivp(f, [0, 6 * tap], [ab_, pbv, 0.0, 0.0], method='DOP853', rtol=1e-13, atol=1e-15, events=[mn], dense_output=True)
    Cmax = np.max(np.abs(fC(so.y[0], so.y[1], so.y[2], so.y[3], Mv, Rv_)))
    tt = np.linspace(0, tap, 200)
    mirror = np.max(np.abs(so.sol(tap + tt)[:2] - so.sol(tap - tt)[:2]))
    bounce2 = so.sol(2 * tap); ret = max(abs(bounce2[0] - ab_), abs(bounce2[1] - pbv), abs(bounce2[2]), abs(bounce2[3]))
    # instability: growth of a small offset in psi_b, read at successive bounces
    so2 = solve_ivp(f, [0, 8.4 * tap], [a_still(pbv + 1e-9, Mv, Rv_), pbv + 1e-9, 0.0, 0.0], method='DOP853', rtol=1e-13, atol=1e-15, events=[mn])
    devs = [y[0] - ab_ for y in so2.y_events[0][:4]]
    mult = devs[2] / devs[1]; mults.append(mult)
    print(f"  M={Mv:.1f} R={Rv_:.1f}: psi_b={pbv:.12f} a_b={ab_:.8f} (addot/a={accb:+.4f}) -> t_ap={tap:.8f} a_ap={sa[0]:.8f} psi_ap={sa[1]:+.8f} "
          f"psidot_ap={sa[3]:.1e} (addot/a={acca:+.4f})")
    print(f"      a_ap/a_b={sa[0] / ab_:.4f} ({np.log(sa[0] / ab_):.3f} e-folds); 3 cycles, no reflection: max|C|={Cmax:.1e}; mirror error about apogee={mirror:.1e}; "
          f"return error at bounce 2={ret:.1e}; offset growth per cycle ~ {mult:+.0f}")
    allok &= (abs(sa[3]) < 1e-12 and accb > 0 and acca < 0 and Cmax < 1e-10 and mirror < 1e-8 and ret < 1e-8)
rec("N1", "EXISTENCE: still-point bounce -> still-point apogee -> bounce 2 -> cycles 2,3 with no reflection and |C| <= 1e-10 (vacuum, dust, dust+radiation)", allok)
rec("N2", "INSTABILITY: an offset of the bounce datum grows by a factor of order 10^2 per cycle (isolated, hyperbolic orbit)", all(abs(mu) > 10 for mu in mults))
ex = []
for p_ in (1.2, 1.5, 1.8, 1.95, 1.99):
    tap, sa = first_max(p_, 0, 0, tmax=2000); ex.append(np.log(sa[0] / (2 / p_)))
print("      vacuum expansion to the first maximum of a, psi_b = 1.2, 1.5, 1.8, 1.95, 1.99: e-folds =", np.round(ex, 2))
rec("N3", "expansion grows without bound as psi_b -> 2 (static point psiddot = 0 at psi = 2): e-folds are available by tuning the datum", ex[-1] > ex[0] and sp.simplify(psd.subs({M: 0, pmax: 0, pv: 2})) == 0)
print("\nSUMMARY: %d/%d" % (sum(OK), len(OK)))
