#!/usr/bin/env python3
"""
CBG -- 18-stage expansion + 18-stage compression probe of the AUTONOMOUS Paper 0 system
(THM #21, Eqs. 118-121 with k = +1, A = 1), started from the Paper III boundary data.

Report-only companion.  It does not replace, import or modify verify_cbg_paper3_v13.py.

System (Einstein-frame form of the Paper 0 Lagrangian; Jordan scale factor a = a_E e^{-psi/2}; units H_0 = 1):
    3 (H_E^2 + k/a_E^2) = (3/4) psi'^2 + U(psi) + rho_r + rho_m
    psi'' + 3 H_E psi' + (2/3) U_psi = rho_m / 3
    H_E' = k/a_E^2 - (3/4) psi'^2 - rho_m/2 - (2/3) rho_r
    U = (3/4) m^2 psi^2 e^{-psi}   (H2 Hooke potential of the Jordan Lagrangian),
    rho_r = 3 Omega_r / a_E^4,  rho_m = 3 Omega_m e^{-psi/2} / a_E^3  (Branch A dust, H1a),  k/a_E^2 = Omega_K / a_E^2.
There is NO exhaustion term, NO x^-6 recoil term, NO x^-4 dilution of the potential, NO velocity reflection.

Boundary data taken from Paper III (Section 4.1-4.2): a_b = k_B T_0 / E_P, psi_b = (1/2) ln 3 - Phi_inf/(4 pi),
psidot_b = 0, and the closure values Omega_m, Omega_r, Omega_K = Phi_inf^2, H_0^(CBG), T_CMB,0, m_psi = hbar c / sqrt(L_P c/H_0).
Stage thresholds: the same 18 temperatures k_B T = k_B T_0 / a as the Paper III chain.
"""
import warnings
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq
warnings.filterwarnings("ignore")

c = 299792458.0; hbar = 1.054571817e-34; G = 6.67430e-11; kB = 1.380649e-23; eV = 1.602176634e-19
alpha = 7.2973525693e-3; Mpc = 3.0856775814913673e22; Gyr = 3.15576e16
Phi = alpha / 30 * (1 - 5 * alpha / 3)
Ob = 427 * np.sqrt(19) / 37620; Om = (2 * np.pi / 20) * (1 + Ob / 9); OK = Phi**2
LP = np.sqrt(hbar * G / c**3); tP = np.sqrt(hbar * G / c**5); EP = np.sqrt(hbar * c**5 / G) / eV
H0 = 67.3460e3 / Mpc; T0 = 2.7246; kT0 = kB * T0 / eV
rho_c = 3 * H0**2 / (8 * np.pi * G); sig = np.pi**2 * kB**4 / (60 * hbar**3 * c**2)
Or = (4 * sig * T0**4 / c**3 / rho_c) * (1 + 3 * (7 / 8) * (4 / 11)**(4 / 3))
m_eV = hbar * c / (np.sqrt(LP * c / H0) * eV); m = (m_eV * eV / hbar) / H0
psi_b = 0.5 * np.log(3) - Phi / (4 * np.pi); a_b = kT0 / EP
E3 = Phi**6 * EP; zstar = 1093.99
stages = [(1, "Sub-Planck / UHECR", Phi * EP), (2, "EeV instanton rung", Phi**2 * EP), (3, "PeV soliton rung", Phi**3 * EP),
          (4, "TeV instanton rung", Phi**4 * EP), (5, "Electroweak", 4 * Phi**5 * EP * (1 + 2 * Ob)), (6, "Multi-GeV QGP", Phi**5 * EP),
          (7, "Yang-Mills gap", (23 / 6) * 188 * E3), (8, "QCD hadronization", 92 * E3), (9, "n/p freeze", E3), (10, "BBN", E3 / 30),
          (11, "Soft X-ray shell", Phi**7 * EP), (12, "Recombination", (1 + zstar) * kT0), (13, "Neutrino rung", Phi**8 * EP),
          (14, "Dilaton bath", m_eV), (15, "Present (T = 2.7246 K)", kT0), (16, "Sub-K rung", Phi**9 * EP), (17, "Coherence rung", Phi**10 * EP)]
P3_t4D = {0: 5.3912e-44, 1: 1.4198e-37, 2: 2.4591e-30, 3: 4.2621e-23, 4: 7.9588e-16, 5: 7.1416e-10, 6: 1.3800e-08, 7: 4.6020e-07, 8: 2.8238e-05,
          9: 2.3901e-01, 10: 2.1511e+02, 11: 4.1376e+06, 12: 1.1652e+13, 13: 3.3475e+13, 14: 7.1630e+15, 15: 4.3521e+17, 16: 1.4923e+18,
          17: 6.1090e+18, 18: 1.0016e+19}      # Paper III v13 summary table, 4D times [s]

def dens(Nt, ps, Om_d=Om):
    aE = np.exp(Nt)
    U = 0.75 * m * m * ps * ps * np.exp(-ps); Up = 0.75 * m * m * np.exp(-ps) * (2 * ps - ps * ps)
    return U, Up, 3 * Or / aE**4, 3 * Om_d * np.exp(-ps / 2) / aE**3, OK / aE**2
def H2(Nt, ps, w, Om_d=Om):
    U, Up, rr, rm, ck = dens(Nt, ps, Om_d); return (U + rr + rm - 3 * ck) / (3 - 0.75 * w * w)
def rhs(Nt, s, Om_d=Om):
    ps, w, tJ = s; U, Up, rr, rm, ck = dens(Nt, ps, Om_d); h2 = (U + rr + rm - 3 * ck) / (3 - 0.75 * w * w)
    Hd = ck - 0.75 * w * w * h2 - 0.5 * rm - (2 / 3) * rr
    return [w, (-3 * h2 * w - (2 / 3) * Up + rm / 3) / h2 - w * Hd / h2, np.exp(-ps / 2) / np.sqrt(h2)]
def Ec(Nt, ps, w, Om_d=Om):      # condensate density coefficient: rho_c = 3 Omega_c / a^3
    return (0.75 * w * w * H2(Nt, ps, w, Om_d) + 0.75 * m * m * ps * ps * np.exp(-ps)) * np.exp(3 * Nt) / 3
def exact_leg(psi_start, Om_d=Om, ratio=1e4):
    Nb = np.log(a_b) + psi_start / 2
    ev = lambda Nt, s: m / np.sqrt(H2(Nt, s[0], s[1], Om_d)) - ratio; ev.terminal = True
    so = solve_ivp(lambda Nt, s: rhs(Nt, s, Om_d), [Nb, Nb + 120], [psi_start, 0.0, 0.0], method='DOP853', rtol=1e-11, atol=1e-14,
                   events=[ev], dense_output=True)
    return Nb, so

print("=" * 118)
print("INPUTS (Paper III closure values): Phi_inf = %.6e, Omega_m = %.6f, Omega_r = %.4e, Omega_K = Phi_inf^2 = %.4e" % (Phi, Om, Or, OK))
print("   H_0 = 67.3460 km/s/Mpc, m_psi = %.4f meV = %.4e H_0, a_b = %.4e, psi_b = %.9f, psidot_b = 0" % (m_eV * 1e3, m, a_b, psi_b))
Nb, so = exact_leg(psi_b)
U, Up, rr, rm, ck = dens(Nb, psi_b)
Hb = np.sqrt(H2(Nb, psi_b, 0.0))
print("\n[1] ADMISSIBILITY OF THE PAPER III STILL POINT IN THE PAPER 0 CONSTRAINT")
print("    (U + rho_r + rho_m) / (3k/a^2) at the Paper III bounce data = %.3e   (a still point needs exactly 1)" % ((U + rr + rm) / (3 * ck)))
print("    => adot = 0 is not admissible; the constraint forces |H_b| = %.4e 1/s  (1/(2 H_b) = %.3e s = %.3f t_P)" % (Hb * H0, 1 / (2 * Hb * H0), 1 / (2 * Hb * H0) / tP))
back = solve_ivp(rhs, [Nb, Nb - 30], [psi_b, 0.0, 0.0], method='DOP853', rtol=1e-11, atol=1e-14)
h2b = np.array([H2(n_, p_, w_) for n_, p_, w_ in zip(back.t, back.y[0], back.y[1])])
print("    backward from x = 1 (30 e-folds): H^2 > 0 throughout = %s, |H| grows by %.2e, psi stays %.6f, proper time used = %.3e s"
      % (bool(np.all(h2b > 0)), np.sqrt(h2b[-1] / h2b[0]), back.y[0, -1], abs(back.y[2, -1]) / H0))
print("    => before x = 1 the solution runs into a -> 0 within %.2f t_P: no bounce precedes the Paper III data in this system" % (1 / (2 * Hb * H0) / tP))

Nh = so.t[-1]; psh, wh, tJh = so.y[:, -1]; a_h = np.exp(Nh - psh / 2)
Oc = Ec(Nh, psh, wh)
g1 = lambda Nt: m / np.sqrt(H2(Nt, *so.sol(Nt)[:2])) - 1
N1 = brentq(g1, Nb, Nh); y1 = so.sol(N1)
g3 = lambda Nt: m / np.sqrt(H2(Nt, *so.sol(Nt)[:2])) - 1e3
N3 = brentq(g3, Nb, Nh); y3 = so.sol(N3)
print("\n[2] FIBER RELEASE AND CONDENSATE")
print("    H = m_psi at k_B T = %.3f TeV (a = %.3e); the fiber then oscillates as pressureless condensate" % (kT0 / np.exp(N1 - y1[0] / 2) / 1e12, np.exp(N1 - y1[0] / 2)))
print("    condensate coefficient Omega_c = rho_c a^3 / rho_crit,0 = %.4e  (at m/H = 1e3: %.4e; drift %.1e => WKB dust law verified)" % (Oc, Ec(N3, y3[0], y3[1]), abs(Oc / Ec(N3, y3[0], y3[1]) - 1)))
print("    Omega_c / Omega_m(Paper III) = %.3e ;  condensate/radiation at handover (k_B T = %.2f GeV) = %.1f" % (Oc / Om, kT0 / a_h / 1e9, Oc * a_h / Or))

Hw = lambda a_: np.sqrt(Or / a_**4 + (Om + Oc) / a_**3 - OK / a_**2)
a_ap = brentq(lambda a_: Or / a_**4 + (Om + Oc) / a_**3 - OK / a_**2, 1.0, 1e30)
def t_log(la, lb, Hf):                           # int d ln a / H between ln a = la and lb, in 2-e-fold panels
    edges = np.append(np.arange(la, lb, 2.0), lb)
    return sum(quad(lambda L: 1.0 / Hf(np.exp(L)), edges[i], edges[i + 1], epsabs=0, epsrel=1e-11)[0] for i in range(len(edges) - 1))
def t_from(a_start, a_, a_max, Hf):              # proper time from a_start to a_ (units 1/H0), regular at the apogee a_max
    if a_ <= a_max / 2:
        return t_log(np.log(a_start), np.log(a_), Hf)
    s_lo = np.sqrt(max(1 - a_ / a_max, 0.0))
    near = quad(lambda s_: 2 * s_ / ((1 - s_ * s_) * Hf(a_max * (1 - s_ * s_))), s_lo, np.sqrt(0.5), limit=400, epsabs=0, epsrel=1e-11)[0]
    return t_log(np.log(a_start), np.log(a_max / 2), Hf) + near
t_wkb = lambda a_: t_from(a_h, a_, a_ap, Hw)
t_ap = tJh + t_wkb(a_ap)
rows = [(0, "Paper III bounce data (x = 1)", EP, a_b, 0.0, psi_b, -Hb, 0.0)]
for idx, name, E in stages:
    a_t = kT0 / E
    if a_t < a_h:
        Ns = brentq(lambda Nt: Nt - so.sol(Nt)[0] / 2 - np.log(a_t), Nb, Nh); y = so.sol(Ns)
        rows.append((idx, name, E, a_t, y[2], y[0], np.sqrt(H2(Ns, y[0], y[1])), Ec(Ns, y[0], y[1]) / a_t**3 / (Or / a_t**4 + Om / a_t**3 + Ec(Ns, y[0], y[1]) / a_t**3)))
    else:
        amp = 2 * np.sqrt(Oc / a_t**3) / m
        rows.append((idx, name, E, a_t, tJh + t_wkb(a_t), amp, Hw(a_t), (Oc / a_t**3) / (Or / a_t**4 + (Om + Oc) / a_t**3)))
rows.append((18, "Apogee (H = 0)", kT0 / a_ap, a_ap, t_ap, 2 * np.sqrt(Oc / a_ap**3) / m, 0.0, Oc / (Om + Oc)))
def fmt_t(sec):
    return "%.4e s" % sec if sec < 3.15576e7 * 1e3 else ("%.4e yr" % (sec / 3.15576e7))
print("\n[3] 18 STAGES OF EXPANSION AND 18 STAGES OF COMPRESSION (time-mirrored about the apogee)")
print("    psi column: field value while frozen/rolling (stages 0-5); oscillation amplitude afterwards.  f_c = condensate share of the density.")
print("    compression: the same stage is passed again at t = 2 t_ap - t_exp, i.e. t_exp before the end (mirror of the dust era; see [4] for the last leg).")
print("    %-3s %-30s %11s %11s %16s %12s %11s %9s %16s" % ("#", "stage", "k_B T [eV]", "a", "t_exp", "H / H_0", "psi / amp", "f_c", "Paper III t4D"))
for idx, name, E, a_t, tt, pv_, Hh, fc in rows:
    te = tt / H0
    print("    %-3d %-30s %11.4e %11.4e %16s %12.3e %+11.3e %9.2e %16s" % (idx, name, E, a_t, fmt_t(te), abs(Hh), pv_, fc, fmt_t(P3_t4D[idx])))
print("    apogee: a_ap = %.4e  (Paper III: Phi_inf^-2 = %.4e);  t_ap = %.4e yr  (Paper III: 3.1739e+11 yr);  full span 2 t_ap = %.4e yr" % (a_ap, Phi**-2, t_ap / H0 / 3.15576e7, 2 * t_ap / H0 / 3.15576e7))
t15 = [r for r in rows if r[0] == 15][0]
print("    at T = 2.7246 K: age = %.1f yr (Paper III: 1.379e+10 yr), H = %.3e H_0^(CBG)" % (t15[4] / H0 / 3.15576e7, t15[6]))
print("    a = Phi_inf^-2 is passed at t = %.4e yr while still expanding" % ((tJh + t_wkb(Phi**-2)) / H0 / 3.15576e7))

print("\n[4] END OF THE COMPRESSION (re-entry at m/|H| = 1e4, k_B T = %.2f GeV; integration continued inward)" % (kT0 / a_h / 1e9))
A_h = np.sqrt(psh**2 + (wh * np.sqrt(H2(Nh, psh, wh)) / m)**2); ph_h = np.arctan2(-wh * np.sqrt(H2(Nh, psh, wh)) / m, psh)
print("    columns: state at the point where the run stops, i.e. where |dpsi/dN| is within 1e-7 of the kinetic limit 2, or at 12 e-folds inside x = 1.")
print("    %-22s %14s %14s %12s %12s  %s" % ("condensate phase", "k_B T at stop", "a at stop", "psi", "dpsi/dN", "outcome"))
for dphi, lab in ((0.0, "mirror (exact)"), (1e-9, "mirror + 1e-9 rad"), (1e-3, "mirror + 1e-3 rad"), (np.pi / 2, "mirror + pi/2"), (np.pi, "mirror + pi")):
    ph = ph_h + dphi; ps0 = A_h * np.cos(ph)
    w0 = brentq(lambda w_: w_ * np.sqrt(H2(Nh, ps0, w_)) / m + A_h * np.sin(ph), -1.9, 1.9) if dphi != 0.0 else wh
    if dphi == 0.0: ps0 = psh
    kin = lambda Nt, s: 2.0 - 1e-7 - abs(s[1]); kin.terminal = True
    cs = solve_ivp(rhs, [Nh, Nb - 12], [ps0, w0, 0.0], method='DOP853', rtol=1e-11, atol=1e-14, events=[kin], dense_output=True)
    Ne = cs.t[-1]; pe, we, te = cs.y[:, -1]; aJ = np.exp(Ne - pe / 2)
    h2s = np.array([H2(n_, p_, w_) for n_, p_, w_ in zip(cs.t, cs.y[0], cs.y[1])])
    aJmin = np.min(np.exp(cs.t - cs.y[0] / 2))
    if len(cs.t_events[0]):
        out = "kinetic run-away (|dpsi/dN| -> 2): " + ("fiber collapses, psi -> -inf, a stalls" if we > 0 else "fiber decompactifies, psi -> +inf")
    else:
        out = "passes x = 1 with H = -|H_b| and continues to a -> 0"
    print("    %-22s %11.3e eV %14.4e %+12.4f %+12.6f  %s%s" % (lab, kT0 / aJmin, aJmin, pe, we, out, "" if np.all(h2s > 0) else " [H^2 <= 0 met]"))
print("    in every case H_E^2 > 0 throughout: no second bounce (no H = 0) occurs before the singular end.")
print("    the 'mirror (exact)' row is the time-reversed expansion state; analytically it retraces to x = 1 and then follows [1] into a -> 0.")
print("    numerically it leaves the mirror curve because round-off (1e-16) in the scalar velocity is amplified: linear estimate a^-1 over the")
print("    %.1f e-folds from re-entry to x = 1, i.e. a factor %.1e." % (Nh - Nb, np.exp(Nh - Nb)))

print("\n[5] WHAT THE AUTONOMOUS SYSTEM WOULD NEED TO KEEP THE PAPER III MATTER BUDGET (condensate = Omega_m - Omega_b, dust = Omega_b)")
target = Om - Ob
def oc_of(p_):
    Nb_, so_ = exact_leg(p_, Om_d=Ob, ratio=1e3); return Ec(so_.t[-1], so_.y[0, -1], so_.y[1, -1], Ob) - target
p_need = brentq(oc_of, 1e-9, 1e-3, xtol=1e-18, rtol=1e-10)
Hn = lambda a_: np.sqrt(Or / a_**4 + Om / a_**3 - OK / a_**2)
a_apn = brentq(lambda a_: Or / a_**4 + Om / a_**3 - OK / a_**2, 1.0, 1e12)
t0n = t_from(1e-12, 1.0, a_apn, Hn)
t_apn = t_from(1e-12, a_apn, a_apn, Hn)
print("    required initial displacement psi_b = %.4e  (Paper III: %.6f, a factor %.2e larger)" % (p_need, psi_b, psi_b / p_need))
print("    then, with no constant sector in the Paper 0 equations: H(today) = %.2f km/s/Mpc, age = %.3f Gyr, a_ap = Omega_m/Omega_K = %.4e, t_ap = %.4e Gyr"
      % (67.3460 * Hn(1.0), t0n / H0 / Gyr, a_apn, t_apn / H0 / Gyr))
print("=" * 118)
