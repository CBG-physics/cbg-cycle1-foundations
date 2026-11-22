#!/usr/bin/env python3
"""
CBG -- complete 18 + 18 stage dual-branch audit:  [A1]-[A4] + H1 (dual branch) + H2 + H3, k = +1 (S^3).
Standalone, report-only.  It does not import, replace or modify verify_cbg_paper3_v13.py.

EQUATIONS (Einstein metric g~ = e^psi g, units H_0 = 1, rho = 3 Omega / a^n; Jordan observables a = a~ e^{-psi/2}):
  I   (H3, Paper III Eq. "Postulated Series-Compliance UV Bounce"):
          H^2 = (rho/3)(1 - rho/rho_P) - k/a~^2 ,   Hdot = k/a~^2 - (1/2)(rho + p)(1 - 2 rho/rho_P)
  II  (Paper 0 Eq. 120, kappa_psi = -3/2):      psiddot + 3 H psidot + (2/3) U_psi = rho_b / 3
  III/IV: A = 1 (J_Theta = 0); the constraint is Eq. I; khronon and P_2 terms are discussed in the report, not evolved.
  rho = (3/4) psidot^2 + U + rho_r + rho_b ,   p = (3/4) psidot^2 - U + rho_r / 3
  Branch A      : baryons, rho_b = 3 Omega_b e^{-psi/2} / a~^3   (H1a: constant mass in g; sources psi)
  Branch A n B  : radiation, rho_r = 3 Omega_r / a~^4            (H1a Phi^0 Maxwell; does not source psi)
  Branch B      : the fiber field itself, (3/4) psidot^2 + U     (condensate when it oscillates)
  U(psi) = (3/2) m_psi^2 Pi_2(psi) + c0 ,  Pi_2 = (1/8) e^{-6 psi} (e^{2 psi} - 1)^2  in W_3 ([A2]),  U''(0) = (3/2) m_psi^2 (H2).
H3 is used in the form printed in Paper III (dressing factor w_UV = 1 - rho/rho_P on the source density, ordinary continuity).
No exhaustion term, no x^-6 recoil, no x^-4 dilution, no velocity reflection.
"""
import time
import warnings
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq
warnings.filterwarnings("ignore")
T_START = time.time()

# ------------------------------------------------------------------------------------------------ constants (Paper III closure)
c = 299792458.0; hbar = 1.054571817e-34; G = 6.67430e-11; kB = 1.380649e-23; eV = 1.602176634e-19
alpha = 7.2973525693e-3; Mpc = 3.0856775814913673e22; Gyr = 3.15576e16; yr = 3.15576e7
Phi = alpha / 30 * (1 - 5 * alpha / 3)
Ob = 427 * np.sqrt(19) / 37620; Om = (2 * np.pi / 20) * (1 + Ob / 9); OK = Phi**2
LP = np.sqrt(hbar * G / c**3); tP = np.sqrt(hbar * G / c**5); EP = np.sqrt(hbar * c**5 / G) / eV
H0 = 67.3460e3 / Mpc; T0 = 2.7246; kT0 = kB * T0 / eV
rho_crit = 3 * H0**2 / (8 * np.pi * G)
Or = (4 * (np.pi**2 * kB**4 / (60 * hbar**3 * c**2)) * T0**4 / c**3 / rho_crit) * (1 + 3 * (7 / 8) * (4 / 11)**(4 / 3))
m_eV = hbar * c / (np.sqrt(LP * c / H0) * eV); m = (m_eV * eV / hbar) / H0
rhoP = 3 * (c**5 / (hbar * G**2)) / rho_crit
OL = 1 - Om - Or + OK
psi_crest = 0.5 * np.log(3); psi_P3 = psi_crest - Phi / (4 * np.pi)
E3 = Phi**6 * EP
STAGES = [(1, "Sub-Planck / UHECR", Phi * EP), (2, "EeV instanton rung", Phi**2 * EP), (3, "PeV soliton rung", Phi**3 * EP),
          (4, "TeV instanton rung", Phi**4 * EP), (5, "Electroweak", 4 * Phi**5 * EP * (1 + 2 * Ob)), (6, "Multi-GeV QGP", Phi**5 * EP),
          (7, "Yang-Mills gap", (23 / 6) * 188 * E3), (8, "QCD hadronization", 92 * E3), (9, "n/p freeze", E3), (10, "BBN", E3 / 30),
          (11, "Soft X-ray shell", Phi**7 * EP), (12, "Recombination", 1094.99 * kT0), (13, "Neutrino rung", Phi**8 * EP),
          (14, "Dilaton bath", m_eV), (15, "Present (2.7246 K)", kT0), (16, "Sub-K rung", Phi**9 * EP), (17, "Coherence rung", Phi**10 * EP)]
P3_T = {0: 5.3912e-44, 1: 1.4198e-37, 2: 2.4591e-30, 3: 4.2621e-23, 4: 7.9588e-16, 5: 7.1416e-10, 6: 1.3800e-08, 7: 4.6020e-07, 8: 2.8238e-05,
        9: 2.3901e-01, 10: 2.1511e+02, 11: 4.1376e+06, 12: 1.1652e+13, 13: 3.3475e+13, 14: 7.1630e+15, 15: 4.3521e+17, 16: 1.4923e+18,
        17: 6.1090e+18, 18: 1.0016e+19}
def ft(sec):
    return "%.3e s" % sec if abs(sec) < 1e3 * yr else "%.3e yr" % (sec / yr)
CHK = []
def rec(tag, label, ok, detail=""):
    CHK.append(bool(ok)); print(f"[{'PASS' if ok else 'FAIL'}] {tag:<4} {label}" + (f"\n{'':12}{detail}" if detail else ""))

# ------------------------------------------------------------------------------------------------ Part S: exact statements
print("=" * 124 + "\nPART S -- exact SymPy statements\n" + "=" * 124)
ps_, y_ = sp.symbols('psi y', real=True)
P2 = sp.Rational(1, 8) * sp.exp(-6 * ps_) * (sp.exp(2 * ps_) - 1)**2
rec("S1", "Pi_2 = (1/8)(e^-2psi - 2 e^-4psi + e^-6psi) lies in W_3 with no constant term",
    sp.simplify(P2 - sp.Rational(1, 8) * (sp.exp(-2 * ps_) - 2 * sp.exp(-4 * ps_) + sp.exp(-6 * ps_))) == 0)
rec("S2", "Pi_2(0) = 0, Pi_2'(0) = 0, Pi_2''(0) = 1 (so U''(0) = (3/2) m^2, the H2 Hooke curvature); crest Pi_2'((1/2)ln3) = 0, Pi_2 = 1/54",
    P2.subs(ps_, 0) == 0 and sp.diff(P2, ps_).subs(ps_, 0) == 0 and sp.diff(P2, ps_, 2).subs(ps_, 0) == 1
    and sp.simplify(sp.diff(P2, ps_).subs(ps_, sp.log(3) / 2)) == 0 and sp.simplify(P2.subs(ps_, sp.log(3) / 2) - sp.Rational(1, 54)) == 0)
cub = sp.Rational(1, 8) * (y_ - 2 * y_**2 + y_**3)
rec("S3", "as a cubic in y = e^-2psi the potential has exactly two critical points (y = 1 minimum, y = 1/3 crest) and Pi_2 -> 0+ as psi -> +inf",
    sorted(sp.solve(sp.diff(cub, y_), y_)) == [sp.Rational(1, 3), 1] and sp.limit(P2, ps_, sp.oo) == 0,
    "=> beyond the crest the fiber runs away to psi -> +infinity; there is no second minimum and no plateau")
t_ = sp.Symbol('t'); a_ = sp.Function('a')(t_); pf = sp.Function('psi')(t_); Uf = sp.Function('U')
rb_ = sp.Function('rho_b')(t_); rr_ = sp.Function('rho_r')(t_); rP_, k_ = sp.symbols('rho_P k', positive=True)
Hs = a_.diff(t_) / a_
rho_s = sp.Rational(3, 4) * pf.diff(t_)**2 + Uf(pf) + rr_ + rb_
p_s = sp.Rational(3, 4) * pf.diff(t_)**2 - Uf(pf) + rr_ / 3
Hdot_s = k_ / a_**2 - sp.Rational(1, 2) * (rho_s + p_s) * (1 - 2 * rho_s / rP_)
eoms = {pf.diff(t_, 2): -3 * Hs * pf.diff(t_) - sp.Rational(2, 3) * Uf(pf).diff(pf) + rb_ / 3,
        rr_.diff(t_): -4 * Hs * rr_, rb_.diff(t_): -3 * Hs * rb_ - pf.diff(t_) * rb_ / 2}
Cs = Hs**2 - (rho_s / 3) * (1 - rho_s / rP_) + k_ / a_**2
dCs = Cs.diff(t_).subs(a_.diff(t_, 2), a_ * (Hdot_s + Hs**2)).subs(eoms).doit()
rec("S4", "H3 system is closed: with Eq. II and the Branch A exchange term, d/dt[H^2 - (rho/3)(1 - rho/rho_P) + k/a^2] = 0 identically", sp.simplify(dCs) == 0)
rho0 = sp.Symbol('rho', positive=True); p0 = sp.Symbol('p', real=True)
rec("S5", "H3 bounce: at rho = rho_P the constraint gives H^2 = -k/a^2 ~ 0 and Hdot = k/a^2 + (rho + p)/2 > 0 for any content with rho + p > 0",
    sp.simplify((k_ / a_**2 - sp.Rational(1, 2) * (rho0 + p0) * (1 - 2 * rho0 / rP_)).subs(rho0, rP_) - (k_ / a_**2 + (rP_ + p0) / 2)) == 0,
    "=> every contraction that reaches rho_P turns round without a singularity, whatever dominates  [D | H3]")
OmS, OKS, OLS, aS = sp.symbols('Omega_m Omega_K Omega_L a', positive=True)
disc = sp.discriminant(sp.expand((OmS / aS**3 - OKS / aS**2 + OLS) * aS**3), aS)
rec("S6", "apogee with a constant sector: Omega_m/a^3 - Omega_K/a^2 + Omega_L = 0 has a positive root iff Omega_L <= 4 Omega_K^3 / (27 Omega_m^2)",
    sp.simplify(disc - (4 * OKS**3 * OLS - 27 * OmS**2 * OLS**2)) == 0,
    "[N] bound for Paper III values: Omega_L <= %.3e, against the closure value %.4f" % (4 * OK**3 / (27 * Om**2), OL))
flip = {a_.diff(t_): -a_.diff(t_), pf.diff(t_): -pf.diff(t_)}
rec("S7", "time reversal: the right-hand sides of Eq. II and of the Hdot equation are even under (adot, psidot) -> (-adot, -psidot)",
    sp.simplify(eoms[pf.diff(t_, 2)].subs(flip, simultaneous=True) - eoms[pf.diff(t_, 2)]) == 0
    and sp.simplify(Hdot_s.subs(flip, simultaneous=True) - Hdot_s) == 0,
    "=> a state with H = psidot = 0 is its own mirror: the solution is even in time about it (exact periodicity needs two such states)")

# ------------------------------------------------------------------------------------------------ numerical core
def Pi2(p): return 0.125 * np.exp(-6 * p) * np.expm1(2 * p)**2
def dPi2(p):
    E = np.expm1(2 * p); return 0.25 * np.exp(-6 * p) * E * (2 - E)
def comps(N, ps, pd):
    a = np.exp(N)
    return 1.5 * m * m * Pi2(ps), 1.5 * m * m * dPi2(ps), 0.75 * pd * pd, 3 * Or / a**4, 3 * Ob * np.exp(-ps / 2) / a**3, OK / a**2
def tot(N, ps, pd):
    U, Up, kin, rr, rb, ck = comps(N, ps, pd); return kin + U + rr + rb
def Hsq(N, ps, pd):
    r = tot(N, ps, pd); return (r / 3) * (1 - r / rhoP) - OK / np.exp(2 * N)
def rhs_t(t, s):                     # state (N, H, psi, psidot): used only across a bounce
    N, H, ps, pd = s; U, Up, kin, rr, rb, ck = comps(N, ps, pd); rho = kin + U + rr + rb; p = kin - U + rr / 3
    return [H, ck - 0.5 * (rho + p) * (1 - 2 * rho / rhoP), pd, -3 * H * pd - (2 / 3) * Up + rb / 3]
def rhs_N(N, s, sgn=1.0):            # state (psi, v = psidot/m, t); H from the constraint, sgn = sign of H
    ps, v, t = s; pd = v * m; U, Up, kin, rr, rb, ck = comps(N, ps, pd); rho = kin + U + rr + rb
    H = sgn * np.sqrt((rho / 3) * (1 - rho / rhoP) - ck)
    return [pd / H, (-3 * H * pd - (2 / 3) * Up + rb / 3) / (m * H), 1.0 / H]
def ray_res(sol, N, sgn=1.0, d=1e-4):   # independent check: Raychaudhuri (Hdot equation) against the constraint-derived H
    def lnH(n_):
        y = sol(n_); return 0.5 * np.log(Hsq(n_, y[0], y[1] * m))
    y = sol(N); pd = y[1] * m; U, Up, kin, rr, rb, ck = comps(N, y[0], pd); rho = kin + U + rr + rb; p = kin - U + rr / 3
    h2 = Hsq(N, y[0], pd)
    return abs((lnH(N + d) - lnH(N - d)) / (2 * d) - (ck - 0.5 * (rho + p) * (1 - 2 * rho / rhoP)) / h2)
def bounce_N(psi_i, pd=0.0):
    N0 = 0.25 * np.log(3 * Or / rhoP); return brentq(lambda N: tot(N, psi_i, pd) / rhoP - 1.0, N0 - 1.0, N0 + 1.0, xtol=1e-16)
def through_bounce(s0, thr=1e-6):
    ev = lambda t, s: tot(s[0], s[2], s[3]) / rhoP - thr if s[1] > 0 else 1.0; ev.terminal = True; ev.direction = -1
    return solve_ivp(rhs_t, [0.0, 1e-50], s0, method='DOP853', rtol=1e-12, atol=[1e-14, 1e-30, 1e-14, 1e-30], events=[ev], first_step=1e-66,
                     dense_output=True)
def expansion(psi_i, ratio=1e4):
    Nb = bounce_N(psi_i); tb = through_bounce([Nb, 0.0, psi_i, 0.0]); N1, H1, p1, pd1 = tb.y[:, -1]
    ev2 = lambda N, s: m / np.sqrt(Hsq(N, s[0], s[1] * m)) - ratio; ev2.terminal = True
    sc = max(abs(psi_i), 1e-12)
    so = solve_ivp(rhs_N, [N1, N1 + 130], [p1, pd1 / m, tb.t[-1]], method='DOP853', rtol=1e-11, atol=[1e-13 * sc, 1e-13 * sc, 1e-80], events=[ev2],
                   dense_output=True)
    return Nb, tb, so
def Oc_of(N, ps, v): return (0.75 * (v * m)**2 + 1.5 * m * m * Pi2(ps)) * np.exp(3 * N) / 3
def t_log(la, lb, Hf):
    edges = np.append(np.arange(la, lb, 2.0), lb)
    return sum(quad(lambda L: 1.0 / Hf(np.exp(L)), edges[i], edges[i + 1], epsabs=0, epsrel=1e-11)[0] for i in range(len(edges) - 1))
def t_from(a_start, a_x, a_max, Hf):
    if a_max is None or a_x <= a_max / 2: return t_log(np.log(a_start), np.log(a_x), Hf)
    s_lo = np.sqrt(max(1 - a_x / a_max, 0.0))
    near = quad(lambda s_: 2 * s_ / ((1 - s_ * s_) * Hf(a_max * (1 - s_ * s_))), s_lo, np.sqrt(0.5), limit=400, epsabs=0, epsrel=1e-11)[0]
    return t_log(np.log(a_start), np.log(a_max / 2), Hf) + near

# ------------------------------------------------------------------------------------------------ Part N1: Bounce #1 and the release of the fiber
print("\n" + "=" * 124 + "\nPART N1 -- Bounce #1 (H3) and the fiber release; test (a) 'autobalancing'\n" + "=" * 124)
print("inputs: Phi_inf = %.6e, Omega_b = %.6f, Omega_m = %.6f, Omega_r = %.4e, Omega_K = %.4e, m_psi = %.4f meV, rho_P/rho_crit = %.4e" % (Phi, Ob, Om, Or, OK, m_eV * 1e3, rhoP / 3))
res = {}
for lab, p_i in (("Paper III crest datum psi_b = (1/2)ln3 - Phi_inf/4pi", psi_P3), ("psi_b = 0.3", 0.3), ("psi_b = 1e-6", 1e-6)):
    Nb_, tb_, so_ = expansion(p_i); res[p_i] = Oc_of(so_.t[-1], so_.y[0, -1], so_.y[1, -1])
    print("  %-52s -> condensate Omega_c = %.4e   (target Omega_m - Omega_b = %.4f)" % (lab, res[p_i], Om - Ob))
psi_i = 1e-6 * np.sqrt((Om - Ob) / res[1e-6])
Nb, tb, so = expansion(psi_i); Nh = so.t[-1]; psh, vh, th = so.y[:, -1]; Oc = Oc_of(Nh, psh, vh)
a_b = np.exp(Nb)
print("  required datum for Omega_c = Omega_m - Omega_b:  psi_b = %.5e  -> Omega_c = %.6f  (quadratic law checked: %.1e)" % (psi_i, Oc, abs(Oc / (Om - Ob) - 1)))
rec("N1", "Bounce #1 under H3 is a non-singular still point: rho = rho_P, H = 0, Hdot > 0, a_b = %.4e (Paper III: 1.9728e-32)" % a_b,
    rhs_t(0, [Nb, 0.0, psi_i, 0.0])[1] > 0 and abs(a_b / 1.9728e-32 - 1) < 2e-3)
drift = abs(so.sol(Nb + 30)[0] / psi_i - 1)
rec("N2", "(a) NO autobalancing: from the bounce to H = m_psi the fiber does not move (|dpsi/psi| = %.1e over 30 e-folds); Omega_c is fixed by psi_b^2" % drift,
    drift < 1e-3 and res[psi_P3] > 1e10,
    "the crest datum of Paper III overshoots the target by %.1e; the target needs psi_b = %.3e as an initial datum" % (res[psi_P3] / (Om - Ob), psi_i))

# ------------------------------------------------------------------------------------------------ Part N2: the 18 stages of expansion
a_h = np.exp(Nh)
def Hw(a, c0=0.0): return np.sqrt(Or / a**4 + (Ob + Oc) / a**3 - OK / a**2 + c0)
a_ap = brentq(lambda a: Or / a**4 + (Ob + Oc) / a**3 - OK / a**2, 1.0, 1e12)
t_ap = th + t_from(a_h, a_ap, a_ap, Hw)
rows = [(0, "Bounce #1 (rho = rho_P)", kT0 / a_b, a_b, 0.0, 0.0, 3 * Ob / a_b**3 / rhoP, 0.0, 1.0, psi_i, abs(tb.y[1, 0]**2 - Hsq(Nb, psi_i, 0.0)) / (rhoP / 3))]
for idx, name, E in STAGES:
    a_t = kT0 / E; N = np.log(a_t)
    if a_t < a_h:
        if N < so.t[0]:
            tloc = brentq(lambda t__: tb.sol(t__)[0] - N, 0.0, tb.t[-1], xtol=1e-90, rtol=1e-14); y = tb.sol(tloc); psx, pdx, tx, Hx = y[2], y[3], tloc, y[1]
            resid = abs(Hx**2 - Hsq(N, psx, pdx)) / Hx**2
        else:
            y = so.sol(N); psx, pdx, tx = y[0], y[1] * m, y[2]; Hx = np.sqrt(Hsq(N, psx, pdx)); resid = ray_res(so.sol, N)
        U, Up, kin, rr, rb, ck = comps(N, psx, pdx); r = kin + U + rr + rb
        rows.append((idx, name, E, a_t, tx, Hx, rb / r, (kin + U) / r, rr / r, psx, resid))
    else:
        r = Or / a_t**4 + (Ob + Oc) / a_t**3
        rows.append((idx, name, E, a_t, th + t_from(a_h, a_t, a_ap, Hw), Hw(a_t), (Ob / a_t**3) / r, (Oc / a_t**3) / r, (Or / a_t**4) / r,
                     2 * np.sqrt(Oc / a_t**3) / m, 0.0))
r = Or / a_ap**4 + (Ob + Oc) / a_ap**3
rows.append((18, "Apogee (H = 0)", kT0 / a_ap, a_ap, t_ap, 0.0, (Ob / a_ap**3) / r, (Oc / a_ap**3) / r, (Or / a_ap**4) / r, 2 * np.sqrt(Oc / a_ap**3) / m, 0.0))
print("\n" + "=" * 124 + "\nPART N2 -- CONFIGURATION A (c0 = 0): 18 stages of expansion\n" + "=" * 124)
print("  psi: field value (stages 0-5), oscillation amplitude (stages 6-18).  res: constraint/Raychaudhuri residual (exact legs); 0 where the dust law is used.")
hdr = "  %-3s %-26s %10s %10s %14s %10s %9s %9s %9s %10s %8s %14s"
print(hdr % ("#", "stage", "k_B T[eV]", "a", "t", "H/H_0", "A:baryon", "B:fiber", "AnB:rad", "psi", "res", "Paper III t"))
for (idx, name, E, a_t, tx, Hx, fb, fc, fr, psx, resid) in rows:
    print("  %-3d %-26s %10.3e %10.3e %14s %10.3e %9.2e %9.2e %9.2e %+10.2e %8.1e %14s" % (idx, name, E, a_t, ft(tx / H0), Hx, fb, fc, fr, psx, resid, ft(P3_T[idx])))
t15 = [r_ for r_ in rows if r_[0] == 15][0]
print("  at 2.7246 K: age = %.3f Gyr (Paper III 13.791), H = %.2f km/s/Mpc (Paper III 67.35);  apogee a = %.4e (Paper III 1.732e7), t_ap = %.4e Gyr (Paper III 317.39)"
      % (t15[4] / H0 / Gyr, 67.3460 * t15[5], a_ap, t_ap / H0 / Gyr))
rec("N3", "(b) apogee by curvature alone exists in configuration A (H = 0, addot < 0) with no exhaustion term", a_ap > 1 and (Ob + Oc) / a_ap**3 > 0)
rec("N4", "residuals on the exactly integrated legs <= 1e-6 (constraint imposed; Raychaudhuri equation checked independently)", max(r_[10] for r_ in rows) < 1e-6,
    "max residual = %.1e" % max(r_[10] for r_ in rows))

# ------------------------------------------------------------------------------------------------ Part N3: configuration B (constant sector)
print("\n" + "=" * 124 + "\nPART N3 -- CONFIGURATION B (baseline c0 = 3 H_0^2 Omega_L added by hand [H]): expansion only\n" + "=" * 124)
HwB = lambda a: Hw(a, OL)
tB = {idx: th + t_from(a_h, kT0 / E, None, HwB) for idx, name, E in STAGES if kT0 / E >= a_h}
print("  " + "   ".join("#%d: %s" % (i_, ft(tB[i_] / H0)) for i_ in (12, 14, 15, 16, 17)))
print("  at 2.7246 K: age = %.4f Gyr, H = %.3f km/s/Mpc;  a = Phi_inf^-2 reached at t = %.2f Gyr with H = %.4f H_0 > 0 (still expanding)"
      % (tB[15] / H0 / Gyr, 67.3460 * HwB(1.0), (th + t_from(a_h, Phi**-2, None, HwB)) / H0 / Gyr, HwB(Phi**-2)))
rec("N5", "(b) with the constant sector the present epoch is reproduced (age %.3f Gyr) but there is NO apogee: Omega_L exceeds the bound of S6 by %.1e"
    % (tB[15] / H0 / Gyr, OL / (4 * OK**3 / (27 * Om**2))), abs(tB[15] / H0 / Gyr - 13.79) < 0.05 and OL > 4 * OK**3 / (27 * Om**2))
print("  m_psi / H_0 = %.2e: the fiber is pinned at its minimum, so its vacuum value cannot roll; W_3 has no plateau (S3)." % m)

# ------------------------------------------------------------------------------------------------ Part N4: contraction and Bounce #2
print("\n" + "=" * 124 + "\nPART N4 -- CONFIGURATION A: 18 stages of contraction and Bounce #2 under H3; test (c)\n" + "=" * 124)
print("  Stages 18 -> 6 (dust era): exact mirror of the expansion; stage n is passed again at t = 2 t_ap - t_n  (2 t_ap = %.4e Gyr)." % (2 * t_ap / H0 / Gyr))
A_c = np.sqrt(psh**2 + vh**2); ph_c = np.arctan2(-vh, psh)
print("  Re-entry at k_B T = %.2f GeV (m/|H| = 1e4).  The condensate phase there, after ~%.0e oscillations, is not computable; it is scanned." % (kT0 / a_h / 1e9, m * 2 * t_ap / (2 * np.pi)))
def inward(dphi):
    ps0 = A_c * np.cos(ph_c + dphi); v0 = -A_c * np.sin(ph_c + dphi)
    ev = lambda N, s: tot(N, s[0], s[1] * m) / rhoP - 1e-6; ev.terminal = True
    c1 = solve_ivp(lambda N, s: rhs_N(N, s, -1.0), [Nh, Nh - 80], [ps0, v0, 0.0], method='DOP853', rtol=1e-11, atol=[1e-13 * A_c, 1e-13 * A_c, 1e-80],
                   events=[ev], dense_output=True)
    N1, p1, v1, t1 = c1.t[-1], *c1.y[:, -1]
    b2 = through_bounce([N1, -np.sqrt(Hsq(N1, p1, v1 * m)), p1, v1 * m])
    tmin = brentq(lambda t__: b2.sol(t__)[1], 0.0, b2.t[-1], xtol=1e-90, rtol=1e-14); ymin = b2.sol(tmin)
    N2, H2_, p2, pd2 = b2.y[:, -1]
    stop = lambda N, s: m * np.sqrt(abs(1.5 * (dPi2(s[0] + 1e-6) - dPi2(s[0] - 1e-6)) / 2e-6)) / np.sqrt(Hsq(N, s[0], s[1] * m)) - 30.0
    stop.terminal = True; stop.direction = 1
    c2 = solve_ivp(lambda N, s: rhs_N(N, s, 1.0), [N2, 0.0], [p2, pd2 / m, 0.0], method='DOP853', rtol=1e-10, atol=[1e-12, 1e-12, 1e-80], events=[stop],
                   dense_output=True)
    return c1, b2, tmin, ymin, c2
cases = {}
print("  %-18s %12s %12s %10s %10s %10s %12s %10s  %s" % ("phase offset", "a_min/a_b", "kT_max [eV]", "rho/rho_P", "Hdot>0", "psi at B2", "psi frozen", "res", "fiber after Bounce #2"))
for dphi, lab in ((np.pi / 2, "+pi/2"), (-np.pi / 2, "-pi/2"), (np.pi / 4, "+pi/4"), (1e-3, "+1e-3")):
    c1, b2, tmin, ymin, c2 = inward(dphi); cases[lab] = (c1, b2, tmin, ymin, c2)
    Nmin, Hmin, pmin, pdmin = ymin
    Nfr = min(c2.t[-1], np.log(kT0 / 1e9)); pfz = c2.sol(Nfr)[0]
    kin_share = 0.75 * pdmin**2 / tot(Nmin, pmin, pdmin)
    resid = abs(b2.y[1, -1]**2 - Hsq(*b2.y[[0, 2, 3], -1])) / b2.y[1, -1]**2
    if kin_share < 1e-3: fate = "radiation-dominated bounce; fiber essentially undisturbed"
    elif pfz > psi_crest: fate = "beyond the crest (%.4f): runs away to psi -> +inf" % psi_crest
    else: fate = "left at psi = %+.2f inside the well" % pfz
    cases[lab] += (pfz, kin_share)
    print("  %-18s %12.4e %12.3e %10.6f %10s %+10.3f %+12.3f %10.1e  %s" % (lab, np.exp(Nmin - Nb), kT0 / np.exp(Nmin), tot(Nmin, pmin, pdmin) / rhoP,
          str(rhs_t(0, ymin)[1] > 0), pmin, pfz, resid, fate))
rec("N6", "(c) Bounce #2 occurs for every phase without any velocity reflection: H = 0 at rho = rho_P with Hdot > 0  [D | H3]",
    all(abs(tot(*v_[3][[0, 2, 3]]) / rhoP - 1) < 1e-4 and rhs_t(0, v_[3])[1] > 0 for v_ in cases.values()))
rec("N7", "(c) but for a generic phase Bounce #2 is kinetic-dominated: it happens at k_B T_max ~ 1e25 eV (not E_P) and displaces the fiber by tens of units",
    cases["+pi/2"][6] > 0.99 and abs(cases["+pi/2"][5]) > 5 and kT0 / np.exp(cases["+pi/2"][3][0]) < 1e-2 * EP)
a_f = brentq(lambda N: m / np.sqrt(Hsq(N, *(so.sol(N)[:2] * np.array([1, m])))) - 1.0, so.t[0], Nh)
amp = (Oc * np.exp(a_f) / Or) * (np.exp(a_f) / a_b)**2
print("  phase tolerance (linear estimate): fiber kinetic share at Bounce #2 ~ %.2e sin^2(dphi); a radiation-dominated Bounce #2 (share < 1e-3)" % amp)
print("  needs |dphi| < %.1e rad.  Offsets below ~1e-3 rad cannot be resolved by this float64 run (integrator tolerance 1e-11 times the same" % np.sqrt(1e-3 / amp))
print("  amplification); the exact mirror phase dphi = 0 is the time reverse of the expansion table and reproduces Bounce #1 exactly (S7).")

# inward stage table for the generic phase +pi/2
c1, b2, tmin, ymin, c2, pfz, ks = cases["+pi/2"]
print("\n  Inward stages for the generic phase +pi/2 (time shown as time remaining before Bounce #2):")
print(hdr % ("#", "stage", "k_B T[eV]", "a~", "t before B2", "|H|/H_0", "A:baryon", "B:fiber", "AnB:rad", "psi", "res", ""))
t_rem = lambda N: tmin + quad(lambda n_: 1.0 / np.sqrt(Hsq(n_, c1.sol(n_)[0], c1.sol(n_)[1] * m)), c1.t[-1], N, limit=400, epsabs=0, epsrel=1e-9)[0]
for idx, name, E in reversed(STAGES[:5]):
    N = np.log(kT0 / E)
    if N < ymin[0]:
        print("  %-3d %-26s %10.3e   not reached: Bounce #2 turns the universe round at k_B T = %.3e eV" % (idx, name, E, kT0 / np.exp(ymin[0]))); continue
    if N >= c1.t[-1]:
        y = c1.sol(N); psx, pdx, tx = y[0], y[1] * m, t_rem(N); resid = ray_res(c1.sol, N)
    else:
        tloc = brentq(lambda t__: b2.sol(t__)[0] - N, 0.0, tmin, xtol=1e-90, rtol=1e-14); y = b2.sol(tloc); psx, pdx, tx = y[2], y[3], tmin - tloc
        resid = abs(y[1]**2 - Hsq(N, psx, pdx)) / y[1]**2
    U, Up, kin, rr, rb, ck = comps(N, psx, pdx); r = kin + U + rr + rb
    print("  %-3d %-26s %10.3e %10.3e %14s %10.3e %9.2e %9.2e %9.2e %+10.2e %8.1e" % (idx, name, E, np.exp(N), ft(tx / H0), np.sqrt(Hsq(N, psx, pdx)), rb / r, (kin + U) / r, rr / r, psx, resid))
print("  B2  %-26s %10.3e %10.3e %14s %10.3e %9.2e %9.2e %9.2e %+10.2e" % ("Bounce #2 (rho = rho_P)", kT0 / np.exp(ymin[0]), np.exp(ymin[0]), "0", 0.0,
      comps(*ymin[[0, 2, 3]])[4] / rhoP, (comps(*ymin[[0, 2, 3]])[2] + comps(*ymin[[0, 2, 3]])[0]) / rhoP, comps(*ymin[[0, 2, 3]])[3] / rhoP, ymin[2]))
Nend = c2.t[-1]; ye = c2.sol(Nend); U, Up, kin, rr, rb, ck = comps(Nend, ye[0], ye[1] * m); r = kin + U + rr + rb
print("  cycle 2, followed to k_B T = %.3e eV: psi = %+.3f, shares baryon/fiber/radiation = %.2e / %.2e / %.2e, Phi/Phi_inf = e^psi = %.3e"
      % (kT0 / np.exp(Nend), ye[0], rb / r, (kin + U) / r, rr / r, np.exp(ye[0])))

# float64 mirror retrace over the unstable stretch (for comparison with the multi-precision run in mp_result.txt)
N1 = so.t[0]; Nm = brentq(lambda N: m / np.sqrt(Hsq(N, *(so.sol(N)[:2] * np.array([1, m])))) - 30.0, N1, Nh)
bk = solve_ivp(lambda N, s: rhs_N(N, s, 1.0), [Nm, N1], so.sol(Nm), method='DOP853', rtol=1e-12, atol=[1e-20, 1e-20, 1e-80])
print("\n  mirror retrace in float64 from m/H = 30 back to the bounce exit (%.1f e-folds): psi returns to %.6e (start %.6e), relative error %.1e"
      % (Nm - N1, bk.y[0, -1], so.sol(N1)[0], abs(bk.y[0, -1] / so.sol(N1)[0] - 1)))
print("\nSUMMARY: %d/%d checks passed   (run time %.0f s)" % (sum(CHK), len(CHK), time.time() - T_START))
