#!/usr/bin/env python3
"""
CBG -- multi-precision check of the mirror (time-reversed) leg of cbg_complete_18stage_dualbranch_audit.py.
Same equations (H3 constraint, W_3 potential, Branch A baryons, radiation), 40 significant digits (mpmath Taylor integrator).
Forward: from 3.5 e-folds after Bounce #1 through the release of the fiber (33.5 e-folds, to m/H ~ 30).
Backward: the same stretch in reverse.  In float64 this retrace fails (relative error ~ 6e2, printed by the main audit);
with 40 digits it returns to the start, which separates round-off from dynamics.  Run time: tens of minutes.
"""
import time
import numpy as np
import mpmath as mp

c = 299792458.0; hbar = 1.054571817e-34; G = 6.67430e-11; kB = 1.380649e-23; eV = 1.602176634e-19
alpha = 7.2973525693e-3; Mpc = 3.0856775814913673e22
Phi = alpha / 30 * (1 - 5 * alpha / 3); Ob_f = 427 * np.sqrt(19) / 37620; OK_f = Phi**2
LP = np.sqrt(hbar * G / c**3); H0 = 67.3460e3 / Mpc; T0 = 2.7246
rho_crit = 3 * H0**2 / (8 * np.pi * G)
Or_f = (4 * (np.pi**2 * kB**4 / (60 * hbar**3 * c**2)) * T0**4 / c**3 / rho_crit) * (1 + 3 * (7 / 8) * (4 / 11)**(4 / 3))
m_f = ((hbar * c / (np.sqrt(LP * c / H0) * eV)) * eV / hbar) / H0
rhoP_f = 3 * (c**5 / (hbar * G**2)) / rho_crit

def run(dps=40, span=33.5, psi_start='5.64533e-7', N_start=-69.5):
    mp.mp.dps = dps
    m = mp.mpf(m_f); rP = mp.mpf(rhoP_f); OR = mp.mpf(Or_f); OB = mp.mpf(Ob_f); Ok = mp.mpf(OK_f)
    def F(sg):
        def f(x, y):
            N = sg * x; ps, v, t = y; pd = v * m; a = mp.e**N; E = mp.expm1(2 * ps)
            U = mp.mpf(3) / 16 * m * m * mp.e**(-6 * ps) * E * E; Up = mp.mpf(3) / 8 * m * m * mp.e**(-6 * ps) * E * (2 - E)
            rr = 3 * OR / a**4; rb = 3 * OB * mp.e**(-ps / 2) / a**3; ck = Ok / a**2
            rho = mp.mpf(3) / 4 * pd * pd + U + rr + rb; H = mp.sqrt(rho / 3 * (1 - rho / rP) - ck)
            return [sg * pd / H, sg * (-3 * H * pd - mp.mpf(2) / 3 * Up + rb / 3) / (m * H), sg / H]
        return f
    N1 = mp.mpf(N_start); y0 = [mp.mpf(psi_start), mp.mpf(0), mp.mpf(0)]
    t0 = time.time(); y1 = mp.odefun(F(1), N1, y0)(N1 + span); tf = time.time() - t0
    t0 = time.time(); y2 = mp.odefun(F(-1), -(N1 + span), y1)(-N1); tb = time.time() - t0
    return tf, tb, y1, max(abs(y2[0] - y0[0]) / y0[0], abs(y2[1]))

if __name__ == "__main__":
    tf, tb, y1, err = run()
    line = "dps=40 span=33.5 e-folds: forward %.0f s, backward %.0f s; state at the far end psi=%s v=%s; return error=%s" % (
        tf, tb, mp.nstr(y1[0], 12), mp.nstr(y1[1], 12), mp.nstr(err, 4))
    print(line); open("mp_result.txt", "w").write(line + "\n")
