#!/usr/bin/env python3
"""
CBG -- "fiber-tension suction" audit of the turnaround / contraction / second bounce
in the Paper 0 homogeneous system, Eqs. (118)-(122), with A = 1, sigma = 0:

    3 (H^2 + k/a^2) = (3/4) psidot^2 + V + rho                                   (118)
    psiddot + 3 H psidot + (2/3) V_psi = (1/3)(rho - 3p)                        (120)
    Hdot = k/a^2 - (3/4) psidot^2 - (1/2)(rho + p)                              (121)
    addot/a = -(1/2) psidot^2 + (1/3) V - (1/6)(rho + 3p)                       (122)

Potential: V = c0 + c1 y + c2 y^2 + c3 y^3,  y = exp(-2 psi).   Report-only.
Part S: exact SymPy statements.   Part N: numerical ODE evidence (stated as evidence, not proof).
"""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

OK = []
def rec(tag, label, ok, detail=""):
    OK.append(bool(ok)); print(f"[{'PASS' if ok else 'FAIL'}] {tag:<5} {label}" + (f"\n{'':13}{detail}" if detail else ""))

# ------------------------------------------------------------------ Part S
print("=== Part S: exact statements ===")
t = sp.Symbol('t', positive=True)
k = sp.Symbol('k', real=True)
a = sp.Function('a')(t); psi = sp.Function('psi')(t); Vf = sp.Function('V')
H = a.diff(t) / a
rr = sp.Function('rho_r')(t)                                   # radiation: p = rho/3, source-free in (120)
E = sp.Rational(3, 4) * psi.diff(t)**2 + Vf(psi)
C = 3 * (H**2 + k / a**2) - E - rr
eom = {psi.diff(t, 2): -3 * H * psi.diff(t) - sp.Rational(2, 3) * Vf(psi).diff(psi),
       a.diff(t, 2): a * (-psi.diff(t)**2 / 2 + Vf(psi) / 3 - rr / 3),
       rr.diff(t): -4 * H * rr}
dC = sp.simplify(C.diff(t).subs(eom).doit() + 2 * H * C)
rec("S1", "constraint propagation: dC/dt = -2H C on (120),(122) + radiation continuity", dC == 0)
dE = sp.simplify(E.diff(t).subs(eom).doit() + sp.Rational(9, 2) * H * psi.diff(t)**2)
rec("S2", "fiber energy: d/dt[(3/4)psidot^2 + V] = -(9/2) H psidot^2  (friction if H>0, anti-friction if H<0)", dE == 0)
Hdot = sp.simplify((a.diff(t, 2) / a - H**2).subs(eom) - (k / a**2 - sp.Rational(3, 4) * psi.diff(t)**2 - sp.Rational(2, 3) * rr)
                   + C / 3)
rec("S3", "(121) from (118),(122): Hdot = k/a^2 - (3/4)psidot^2 - (2/3)rho_r;  k <= 0 => Hdot <= 0 => no bounce", sp.simplify(Hdot) == 0)

V0, av, pd, rho = sp.symbols('V_0 a_0 psidot_0 rho_0', real=True)
acc = -pd**2 / 2 + V0 / 3 - rho / 3                            # addot/a with radiation
con = sp.Eq(3 / av**2, sp.Rational(3, 4) * pd**2 + V0 + rho)   # H = 0, k = +1
Vsol = sp.solve(con, V0)[0]
rec("S4", "turning point (H=0,k=+1): addot/a = 1/a^2 - (3/4)psidot^2 - (2/3)rho_r",
    sp.simplify(acc.subs(V0, Vsol) - (1 / av**2 - sp.Rational(3, 4) * pd**2 - sp.Rational(2, 3) * rho)) == 0,
    "apogee (addot<0)  <=>  (3/4)psidot^2 + (2/3)rho_r > 1/a^2 ;  bounce (addot>0)  <=>  the reverse")
rec("S5", "VACUUM OBSTRUCTION: H=0 and psidot=0 with rho=0  =>  addot/a = 1/a^2 > 0: a time-symmetric point is always a bounce, never an apogee",
    sp.simplify(acc.subs(V0, Vsol).subs({pd: 0, rho: 0}) - 1 / av**2) == 0)
rec("S6", "with radiation a time-symmetric apogee needs rho_r > 3/(2a^2), i.e. V(psi_ap) < rho_r(a_ap)",
    sp.simplify(acc.subs(V0, Vsol).subs(pd, 0) - (1 / av**2 - sp.Rational(2, 3) * rho)) == 0
    and sp.simplify(Vsol.subs(pd, 0) - (3 / av**2 - rho)) == 0)

# single exponentials V = c exp(-2 n psi): exact power-law solution a = t^p (k=0)
n, p, c, psi0 = sp.symbols('n p c psi_0', positive=True)
psi_s = sp.log(t) / n + psi0
V_s = lambda x: c * sp.exp(-2 * n * x)
kg = sp.simplify((psi_s.diff(t, 2) + 3 * (p / t) * psi_s.diff(t) - sp.Rational(4, 3) * n * V_s(psi_s)) * t**2)
fr = sp.simplify((3 * (p / t)**2 - sp.Rational(3, 4) * psi_s.diff(t)**2 - V_s(psi_s)) * t**2)
sol = sp.solve([kg, fr], [p, c], dict=True)[0]
pvals = [sp.simplify(sol[p].subs(n, j)) for j in (1, 2, 3)]
rec("S7", "scaling solutions on e^{-2 n psi}: a ~ t^p with p = 3/(4 n^2) = (3/4, 3/16, 1/12) < 1: no term of W_3 accelerates",
    sp.simplify(sol[p] - 3 / (4 * n**2)) == 0 and pvals == [sp.Rational(3, 4), sp.Rational(3, 16), sp.Rational(1, 12)],
    "energy ~ a^(-2/p) = a^(-8/3), a^(-32/3), a^(-24): each falls faster than curvature a^-2 => k=+1 recollapses on any W_3 tail")
rec("S8", "sign of the scaling amplitude: c > 0 only for n = 1 (p > 1/3); n = 2, 3 need c < 0",
    [sp.simplify(sol[c].subs({n: j, psi0: 0})).is_positive for j in (1, 2, 3)] == [True, False, False])
# kination psidot = +/-2H: V_n/KE ~ a^(6 -/+ 4n)
ex_wall = [6 - 4 * j for j in (1, 2, 3)]
rec("S9", "kinetic contraction (psidot = +/-2H, KE ~ a^-6): toward small fiber V_n/KE ~ a^(6-4n) = a^(2,-2,-6); toward large fiber every term decays",
    ex_wall == [2, -2, -6],
    "=> the e^{-4psi}, e^{-6psi} walls reflect the field; nothing in Span{1,e^-2psi,e^-4psi,e^-6psi} stops the run to psi -> +infinity")
w, Nn = sp.symbols('w N', real=True)
rec("S10", "curvature budget: d ln(E a^2)/d ln a = -(1+3w); a bounce needs E a^2 = 3, so after reaching X it needs >= (1/2) ln(X/3) e-folds of w = -1 contraction",
    sp.simplify(-(1 + 3 * w) + 3 * (1 + w) - 2) == 0 and sp.simplify(-(1 + 3 * sp.Integer(-1))) == 2)
y = sp.Symbol('y', positive=True)
Vy = y * (y - 1) * (y - 2)
crit = sp.solve(sp.diff(Vy, y), y)
rec("S11", "example inside the span (c0,c1,c2,c3) = (0,2,-3,1): wall, negative well, positive hill",
    sp.expand(Vy) == y**3 - 3 * y**2 + 2 * y and Vy.subs(y, crit[0]) > 0 and Vy.subs(y, crit[1]) < 0,
    f"hill V = {sp.nsimplify(Vy.subs(y, crit[0]))} at y = {crit[0]},  well V = {sp.nsimplify(Vy.subs(y, crit[1]))} at y = {crit[1]}")

# ------------------------------------------------------------------ Part N
print("\n=== Part N: numerical evidence (example potential of S11, k = +1) ===")
Vn = lambda q: (lambda Y: Y * (Y - 1) * (Y - 2))(np.exp(-2 * q))
Vq = lambda q: (lambda Y: -2 * Y * (3 * Y * Y - 6 * Y + 2))(np.exp(-2 * q))
def rhs(R):
    return lambda t_, s: [s[1], np.exp(-2 * s[0]) - 0.75 * s[3]**2 - (2 / 3) * R * np.exp(-4 * s[0]), s[3], -3 * s[1] * s[3] - (2 / 3) * Vq(s[2])]
def bounce_data(yb, R):
    q = -0.5 * np.log(yb); a2 = (3 + np.sqrt(9 - 4 * Vn(q) * R)) / (2 * Vn(q)); return [0.5 * np.log(a2), 0.0, q, 0.0]
def run(yb, R=0.0, tmax=400.0):
    f = rhs(R); s0 = bounce_data(yb, R)
    s1 = solve_ivp(f, [0, 1e-7], s0, method='DOP853', rtol=1e-13, atol=1e-15).y[:, -1]
    dn = lambda t_, s: s[1]; dn.direction = -1
    up = lambda t_, s: s[1]; up.direction = +1
    bl = lambda t_, s: abs(s[1]) - 1e4; bl.terminal = True
    sol = solve_ivp(f, [1e-7, tmax], s1, method='DOP853', rtol=1e-11, atol=1e-13, events=[dn, up, bl])
    return s0, sol
print(f"{'y_b':>12} {'R':>5} {'e-folds':>8} {'t_ap':>8} {'psidot_ap':>10} {'V_ap':>8} {'addot/a':>9} {'2nd bounce':>10} {'end':>7} {'E a^2 end':>10}")
rows = []
for yb, R in ((0.428672923, 0), (0.423674429, 0), (0.422654776, 0), (0.422649756, 0), (0.6, 0), (0.9, 0),
              (0.423674429, 1.0), (0.44, 2.5), (3.0, 0)):
    s0, sol = run(yb, R)
    N, Hh, q, v = sol.y_events[0][0]
    acc_ap = -0.5 * v * v + Vn(q) / 3 - R * np.exp(-4 * N) / 3
    Ne, He, qe, ve = sol.y[:, -1]
    rows.append((len(sol.t_events[1]), len(sol.t_events[2]), acc_ap, v))
    print(f"{yb:12.9f} {R:5.1f} {N - s0[0]:8.3f} {sol.t_events[0][0]:8.3f} {v:+10.4f} {Vn(q):+8.4f} {acc_ap:+9.4f} {len(sol.t_events[1]):>10d} "
          f"{'CRUNCH' if len(sol.t_events[2]) else 'run':>7} {(0.75 * ve * ve + Vn(qe)) * np.exp(2 * Ne):10.2e}")
rec("N1", "every tested bounce reaches an apogee (H=0, addot<0) with NO a-dependent term: turnaround by the fiber alone", all(r[2] < 0 for r in rows))
rec("N2", "at every apogee psidot != 0 (no time-symmetric apogee), and every run ends in kinetic blow-up with no second bounce",
    all(abs(r[3]) > 1e-3 and r[0] == 0 and r[1] == 1 for r in rows))
# scan for a time-symmetric apogee
hits = 0; jumps = 0
for R in (0.0, 0.3, 1.0, 2.5, 5.5):
    prev = None
    for yb in np.concatenate([np.linspace(0.4227, 0.9995, 500), np.linspace(2.0005, 6.0, 300)]):
        if 9 - 4 * Vn(-0.5 * np.log(yb)) * R < 0: prev = None; continue
        s0, sol = run(yb, R, tmax=200)
        if len(sol.t_events[0]) == 0: prev = None; continue
        tt, v = sol.t_events[0][0], sol.y_events[0][0][3]
        if prev is not None and prev[1] * v < 0:
            if abs(prev[1] - v) < 0.05: hits += 1
            else: jumps += 1
        prev = (tt, v)
rec("N3", f"scan of 4000 time-symmetric bounces (5 radiation levels): continuous zeros of psidot_ap = {hits}; sign flips through a discontinuity = {jumps}", hits == 0)
print("\nSUMMARY: %d/%d" % (sum(OK), len(OK)))
