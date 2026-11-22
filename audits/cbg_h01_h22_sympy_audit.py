#!/usr/bin/env python3
# CBG Cycle I hypothesis audit (H01-H22): SymPy checks behind the Category 2 statements and the sign questions.
# Test script only. It does not import or modify any verify_cbg_*.py file.
import sympy as sp, itertools
ok = lambda name, cond: print(("[PASS] " if cond else "[FAIL] ") + name) or (cond or (_ for _ in ()).throw(SystemExit(1)))
t = sp.Symbol('t', real=True)

# ---------------- H12: khronon first integral from the Paper 0 minisuperspace Lagrangian (Thm 5.1(ii)) -------------
a, psi, Th, N = [sp.Function(n)(t) for n in ('a', 'psi', 'Theta', 'N')]
k = sp.Symbol('k'); V = sp.Function('V'); A_ = sp.Symbol('A', positive=True)
Aexpr = N**2 / Th.diff(t)**2
L = a*sp.exp(psi)*(-6*a.diff(t)**2/N - 6*a*a.diff(t)*psi.diff(t)/N + 6*k*N - 2*a**2*N*V(psi, Aexpr))
J = sp.diff(L, Th.diff(t))                                   # Theta is cyclic  =>  dJ/dt = 0 is the Theta equation
VA = sp.Derivative(V(psi, A_), A_)
J_claim = 4*a**3*sp.exp(psi)*N**3/Th.diff(t)**3 * sp.Subs(VA, A_, Aexpr)
ok("H12a  J = dL/dTheta_dot = 4 a^3 e^psi A^(3/2) dV/dA  (Theta cyclic, so J is conserved)", sp.simplify((J - J_claim).doit()) == 0)
ok("H12b  L has no explicit Theta dependence (Noether first integral)", sp.diff(L, Th) == 0)
dLdN_V = sp.diff(-2*a**3*sp.exp(psi)*N*V(psi, Aexpr), N)
claim = -2*a**3*sp.exp(psi)*(V(psi, Aexpr) + 2*Aexpr*sp.Subs(VA, A_, Aexpr))
ok("H12c  lapse equation carries V + 2 A dV/dA, i.e. the 'dropped' term is 2A V_A = J / (2 a^3 e^psi sqrt(A))", sp.simplify((dLdN_V - claim).doit()) == 0)
m, pm, Ac, ps = sp.symbols('m psi_max A_c psi_s', positive=True)
pt = pm*(sp.exp(-A_**2/Ac**2) - sp.exp(-1/Ac**2)); VH = sp.Rational(3, 4)*m**2*(ps - pt)**2
ok("H12d  Hooke + Gaussian kernel: V_A = 0  <=>  psi = psi_target(A)  (then V = 0 and V_psi = 0)",
   sp.simplify(sp.diff(VH, A_) - 3*m**2*(ps - pt)*pm*A_*sp.exp(-A_**2/Ac**2)/Ac**2) == 0)
# exact homogeneous anisotropic S^3 (Bianchi IX): 3-curvature depends on the anisotropy (not k=+1 plus an a^-6 fluid)
th, ph, ch = sp.symbols('theta phi chi', real=True); a1, a2, a3 = sp.symbols('a1 a2 a3', positive=True)
X = (th, ph, ch)
s1 = [-sp.sin(ch), sp.sin(th)*sp.cos(ch), 0]; s2 = [sp.cos(ch), sp.sin(th)*sp.sin(ch), 0]; s3 = [0, sp.cos(th), 1]
g = sp.Matrix(3, 3, lambda i, j: a1**2*s1[i]*s1[j] + a2**2*s2[i]*s2[j] + a3**2*s3[i]*s3[j]); gi = g.inv()
Ga = [[[sum(gi[i, l]*(g[l, j].diff(X[kk]) + g[l, kk].diff(X[j]) - g[j, kk].diff(X[l])) for l in range(3))/2 for kk in range(3)] for j in range(3)] for i in range(3)]
Ric = sp.Matrix(3, 3, lambda j, kk: sum(Ga[i][j][kk].diff(X[i]) - Ga[i][j][i].diff(X[kk]) + sum(Ga[i][i][l]*Ga[l][j][kk] - Ga[i][kk][l]*Ga[l][j][i] for l in range(3)) for i in range(3)))
R3_raw = sum(gi[i, j]*Ric[i, j] for i in range(3) for j in range(3))
R3_claim = (2*(a1**2*a2**2 + a2**2*a3**2 + a3**2*a1**2) - a1**4 - a2**4 - a3**4)/(2*a1**2*a2**2*a3**2)
# the space is homogeneous, so the scalar curvature is evaluated exactly at three sample points (symbolic in a1, a2, a3)
R3_pt = sp.simplify(R3_raw.subs({th: sp.pi/3, ph: 0, ch: sp.pi/5}))          # exact, symbolic in (a1, a2, a3)
num_pts = [({th: 0.7, ph: 1.0, ch: 0.4}, {a1: 1.3, a2: 0.8, a3: 2.1}), ({th: 2.0, ph: 2.0, ch: 1.1}, {a1: 0.6, a2: 1.7, a3: 0.9})]
ok("H12e  Bianchi IX: 3R = [2(a1^2a2^2+a2^2a3^2+a3^2a1^2) - a1^4-a2^4-a3^4]/(2 a1^2a2^2a3^2): depends on the anisotropy (3/(2a^2) only if isotropic)",
   sp.simplify(R3_pt - R3_claim) == 0 and all(abs(sp.N((R3_raw - R3_claim).subs(pa).subs(pb))) < 1e-10 for pa, pb in num_pts))

# ---------------- H14: homogeneous gas carrying fiber momentum P_2 on signature (-,-,+,+,+) ----------------------
E, p, P2, Phi, m5, aa, n0 = sp.symbols('E p P_2 Phi m_5 a n_0', positive=True)
shell = -E**2 - P2**2/Phi**2 + p**2/aa**2 + m5**2            # g^{AB} p_A p_B = -m_5^2
ok("H14a  mass shell: E^2 = m_5^2 + p^2/a^2 - P_2^2/Phi^2  (fiber momentum lowers E^2: timelike fiber)", sp.simplify(sp.solve(shell, E**2)[0] - (m5**2 + p**2/aa**2 - P2**2/Phi**2)) == 0)
rho_dust = (n0/aa**3)*sp.sqrt(m5**2 - P2**2/Phi**2)
ok("H14b  cold P_2-gas: rho = n_0 a^-3 sqrt(m_5^2 - P_2^2/Phi^2): scales as a^-3 at fixed Phi, never as -C/a^6", sp.simplify(sp.diff(sp.log(rho_dust), aa)*aa + 3) == 0)
x, vps, C, am, aK, aL, ek, xa = sp.symbols('x v_psi C alpha_m alpha_K alpha_L ek x_ap', positive=True); Vt = sp.Symbol('Vt')
br = (1 + Vt/6)/x**4 + am*sp.exp(ek*0)/x**3 + aK/x**2 + aL*(1 - x**2/xa**2) - C/x**6
H2 = vps**2 + br; acc = -2*vps**2 - (1 + Vt/6)/x**4 - am/(2*x**3) + aL*(1 - 2*x**2/xa**2) + 2*C/x**6
# ---------------- H19: tensor pump a''/a in conformal time on the five-sector background --------------------------
pump = sp.simplify(x**2*(H2 + acc))
ok("H19a  five-sector: a''/a = C/x^4 + alpha_m/(2x) + alpha_K + alpha_L x^2 (2 - 3x^2/x_ap^2) - x^2 v_psi^2",
   sp.simplify(pump - (C/x**4 + am/(2*x) + aK + aL*x**2*(2 - 3*x**2/xa**2) - x**2*vps**2)) == 0)
eta, Ne = sp.symbols('eta N_e', positive=True); z = eta**(-(1 + 1/Ne))
ok("H19b  ansatz z ~ |eta|^-(1+1/N): z''/z = (1+1/N)(2+1/N)/eta^2", sp.simplify(z.diff(eta, 2)/z - (1 + 1/Ne)*(2 + 1/Ne)/eta**2) == 0)

# ---------------- H06: 5D Dirac operator on M_{3,2}: reduction and Z_2 parity of the bulk mass -------------------
s0 = sp.eye(2); sx = sp.Matrix([[0, 1], [1, 0]]); sy = sp.Matrix([[0, -sp.I], [sp.I, 0]]); sz = sp.Matrix([[1, 0], [0, -1]])
kr = sp.kronecker_product
Gm = [kr(sp.I*sy, s0), kr(sp.I*sz, s0), kr(sx, sx), kr(sx, sy), kr(sx, sz)]      # index 1 = secondary time
eta5 = sp.diag(-1, -1, 1, 1, 1)
ok("H06a  Cl(3,2): {Gamma_a, Gamma_b} = 2 eta_ab with eta = diag(-,-,+,+,+)", all(sp.simplify(Gm[i]*Gm[j] + Gm[j]*Gm[i] - 2*eta5[i, j]*sp.eye(4)) == sp.zeros(4) for i in range(5) for j in range(5)))
Acj = Gm[0]*Gm[1]; S = Gm[1]                                   # Dirac conjugation matrix; spinor image of t_2 -> -t_2
def sign_of(M1, M2):
    r = sp.simplify(M1*M2.inv()); assert r == r[0, 0]*sp.eye(4); return r[0, 0]
s_m = sign_of(S.H*Acj*S, Acj)
s_k = [sign_of(S.H*Acj*Gm[b]*S*(-1 if b == 1 else 1), Acj*Gm[b]) for b in range(5)]
ok("H06b  kinetic bilinear transforms with one common sign under t_2 -> -t_2", len(set(s_k)) == 1)
ok("H06c  bulk mass bilinear is Z_2-ODD relative to the kinetic term (constant bulk Dirac mass is projected out)", sp.simplify(s_m/s_k[0]) == -1)
# reduction of the massless operator on ds^2 = -dt1^2 - Phi(t1,x)^2 dt2^2 + dx^2 for t2-independent spinors
t1, t2, x1, x2, x3 = sp.symbols('t1 t2 x1 x2 x3', real=True); Xs = (t1, t2, x1, x2, x3); Ph = sp.Function('Phi', positive=True)(t1, x1)
g5 = sp.diag(-1, -Ph**2, 1, 1, 1); g5i = g5.inv(); e = sp.diag(1, Ph, 1, 1, 1); ei = e.inv()      # e[a,M], ei[M,a]
G5 = [[[sum(g5i[i, l]*(g5[l, j].diff(Xs[kk]) + g5[l, kk].diff(Xs[j]) - g5[j, kk].diff(Xs[l])) for l in range(5))/2 for kk in range(5)] for j in range(5)] for i in range(5)]
def omega(M, a_, b_):   # omega_M^{a}{}_{b} = e^a_N (d_M E_b^N + Gamma^N_{MK} E_b^K)
    return sp.simplify(sum(e[a_, Nn]*(ei[Nn, b_].diff(Xs[M]) + sum(G5[Nn][M][K]*ei[K, b_] for K in range(5))) for Nn in range(5)))
chi = sp.Matrix([sp.Function('chi%d' % i)(t1, x1, x2, x3) for i in range(4)]); psi5 = chi/sp.sqrt(Ph)
def D5(sp_):
    out = sp.zeros(4, 1)
    for a_ in range(5):
        for M in range(5):
            if ei[M, a_] == 0: continue
            conn = sp.zeros(4)
            for b_ in range(5):
                for c_ in range(5):
                    conn += sp.Rational(1, 4)*omega(M, b_, c_)*eta5[c_, c_]*Gm[b_]*Gm[c_]*eta5[b_, b_]*eta5[b_, b_]
            out += eta5[a_, a_]*Gm[a_]*ei[M, a_]*(sp_.diff(Xs[M]) + conn*sp_)
    return out
flat4 = sum((eta5[a_, a_]*Gm[a_]*chi.diff(Xs[a_]) for a_ in (0, 2, 3, 4)), sp.zeros(4, 1))
ok("H06d  massless 5D Dirac operator on psi = Phi^(-1/2) chi equals Phi^(-1/2) x (4D operator in the induced metric g_mu_nu)", sp.simplify(D5(psi5) - flat4/sp.sqrt(Ph)) == sp.zeros(4, 1))

# ---------------- H21: 5D Chern-Simons density on the Z_2-even, t2-independent photon sector ---------------------
Af = [sp.Function('A%d' % i)(t1, x1, x2, x3) if i != 1 else sp.Integer(0) for i in range(5)]   # A_2 odd => no zero mode
F = sp.Matrix(5, 5, lambda i, j: Af[j].diff(Xs[i]) - Af[i].diff(Xs[j]))
cs = sum(sp.LeviCivita(*pm_)*Af[pm_[0]]*F[pm_[1], pm_[2]]*F[pm_[3], pm_[4]] for pm_ in itertools.permutations(range(5)))
ok("H21a  eps^{ABCDE} A_A F_BC F_DE = 0 identically for A_2 = 0, d/dt2 = 0 (no parity-odd photon term survives)", sp.simplify(cs) == 0)

# ---------------- H03: the integer 17 ------------------------------------------------------------------------------
Rz = sp.diag(1, -1, 1, 1, 1); even = [(i, j) for i in range(5) for j in range(5) if Rz[i, i]*Rz[j, j] == 1]
ok("H03a  dim (V5* x V5*)^{Z_2-even} = 17 = 11 (Sym^2) + 6 (Lambda^2)", len(even) == 17 and len([p_ for p_ in even if p_[0] <= p_[1]]) == 11 and len([p_ for p_ in even if p_[0] < p_[1]]) == 6)
ok("H03b  Delta_IR = 1 - 4*(V0/12) = -17 only for the constitutive normalization V0 = 54", 1 - 4*sp.Rational(54, 12) == -17)

# ---------------- H16: closed-S^3 curvature alone cannot turn the expansion around while Omega_Lambda > 0 ---------
Om, OK, OL = 0.31588635, -5.77376871e-8, 0.68402215; av = sp.Symbol('a', positive=True)
crit = sp.nsolve(sp.diff(Om/av**3 + OK/av**2, av), av, 1e7); Hmin = Om/crit**3 + OK/crit**2 + OL
ok("H16a  with Omega_Lambda = 0.684: min over a of (Omega_m/a^3 + Omega_K/a^2) + Omega_Lambda > 0 (H never reaches 0)", Hmin > 0.68)
ok("H16b  with Omega_Lambda = 0 the curvature turnaround is at a = Omega_m/|Omega_K| = 5.47e6 (not Phi_inf^-2 = 1.73e7)", abs(Om/abs(OK) - 5.471e6) < 1e4)
print("ALL SYMPY AUDIT CHECKS PASSED")
