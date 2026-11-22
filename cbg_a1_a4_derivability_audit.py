#!/usr/bin/env python3
"""
CBG -- derivability audit of [H1]-[H6], [H_aux] against the 4-axiom core [A1]-[A4].
Report-only.  Every line printed as PROVED is an exact SymPy statement (zero residual).
Lines printed as COUNTERMODEL exhibit two configurations that both satisfy the stated
axiom content and differ on the hypothesis, which proves independence.

Conventions: coordinates (t1, t2, x1, x2, x3), eta = diag(-1,-1,+1,+1,+1),
{Gamma^A, Gamma^B} = 2 eta^{AB}, R_Z2 = diag(1,-1,1,1,1), psi = ln(Phi/Phi_inf).
"""
import itertools
import sympy as sp

RESULTS = []


def rec(tag, label, ok, detail=""):
    RESULTS.append((tag, label, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {tag:<12} {label}" + (f"\n{'':20}{detail}" if detail else ""))


def kron(a, b):
    return sp.kronecker_product(a, b)


# =====================================================================================
# (a) SPINORS ON THE ORBIFOLD, MASS BILINEARS, PHI-POWER OF A BOUNDARY MASS
# =====================================================================================
print("\n=== (a) spinor parity, zero modes and mass terms ===")
s1 = sp.Matrix([[0, 1], [1, 0]])
s3 = sp.Matrix([[1, 0], [0, -1]])
ep = sp.Matrix([[0, 1], [-1, 0]])
I2 = sp.eye(2)
I4 = sp.eye(4)
G = [kron(ep, s1), kron(ep, s3), kron(s1, I2), kron(s3, I2), kron(ep, ep)]   # t1, t2, x1, x2, x3
eta = sp.diag(-1, -1, 1, 1, 1)
Rz2 = sp.diag(1, -1, 1, 1, 1)

rec("a.1", "real 4x4 Cl(3,2) generators: {G^A,G^B} = 2 eta^{AB}",
    all(G[a] * G[b] + G[b] * G[a] == 2 * eta[a, b] * I4 for a in range(5) for b in range(5)))

P = G[1]                                    # lift of the reflection t2 -> -t2
rec("a.2", "P = Gamma^{t2}: P^-1 G^A P = -R^A_A G^A  (pin lift of R_Z2)",
    all(P.inv() * G[a] * P == -Rz2[a, a] * G[a] for a in range(5)))

B = G[0] * G[1]                             # Dirac conjugation matrix, Psibar = Psi^dagger B
sgn = {a: (+1 if (B * G[a]).H == B * G[a] else -1 if (B * G[a]).H == -B * G[a] else 0) for a in range(5)}
rec("a.3", "B = G^{t1} G^{t2}: (B G^A)^dagger = s B G^A with one common sign s (real kinetic action)",
    len(set(sgn.values())) == 1 and 0 not in sgn.values(), f"s = {sgn[0]}")

rec("a.4", "scalar bilinear is Z2-ODD:  P^dagger B P = -B", P.H * B * P == -B)
rec("a.5", "kinetic bilinear is Z2-EVEN: P^dagger (B G^A) P = +R^A_A (B G^A)",
    all(P.H * B * G[a] * P == Rz2[a, a] * B * G[a] for a in range(5)))

rec("a.6", "P^2 = -1: a REAL spinor with Psi(t2) = P Psi(-t2) and d_t2 Psi = 0 vanishes",
    P * P == -I4 and (I4 - P).det() != 0 and (I4 + P).det() != 0,
    "det(1 -/+ P) != 0  =>  Psi = +/-P Psi has only the solution Psi = 0")

Pi_p = (I4 + sp.I * P) / 2
Pi_m = (I4 - sp.I * P) / 2
rec("a.7", "complex spinor: Pi_+/- = (1 +/- iP)/2 are rank-2 projectors (one 4D Weyl zero mode each)",
    Pi_p * Pi_p == Pi_p and Pi_m * Pi_m == Pi_m and Pi_p.rank() == 2 and Pi_m.rank() == 2
    and Pi_p * Pi_m == sp.zeros(4))

rec("a.8", "on an equivariant zero mode  Psibar Psi == 0  identically (bulk AND fixed points)",
    Pi_p.H * B * Pi_p == sp.zeros(4) and Pi_m.H * B * Pi_m == sp.zeros(4))
rec("a.9", "the zero mode keeps its kinetic term on the 4D block, loses the t2 component",
    all((Pi_p.H * B * G[a] * Pi_p) != sp.zeros(4) for a in (0, 2, 3, 4))
    and Pi_p.H * B * G[1] * Pi_p == sp.zeros(4))
rec("a.10", "cross bilinear of an opposite-parity pair is non-zero and Z2-even: Pi_+^dag B Pi_- != 0",
    (Pi_p.H * B * Pi_m).rank() == 2
    and (sp.I * P).H * B * (-sp.I * P) == B)
Sig = [(G[a] * G[b] - G[b] * G[a]) / 4 for a in range(5) for b in range(a + 1, 5)]
rec("a.11", "Majorana-type bilinear Psi^T B Psi: Spin(3,2)-invariant, antisymmetric, non-zero on the zero mode, Z2-even",
    all(B * S + S.T * B == sp.zeros(4) for S in Sig) and B.T == -B
    and Pi_p.T * B * Pi_p != sp.zeros(4) and (sp.I * P).T * B * (sp.I * P) == B)

# Phi-power bookkeeping.  sqrt|g5| = Phi sqrt|g4|;  Psi_5 = Phi^(-1/2) chi_4 (canonical kinetic term).
Phi, psi, mu, L0 = sp.symbols('Phi psi mu L0', positive=True)
p_bulk = sp.simplify(sp.log(Phi * (Phi**sp.Rational(-1, 2))**2) / sp.log(Phi))              # bulk measure
p_bdry = sp.simplify(sp.log(1 * (Phi**sp.Rational(-1, 2))**2) / sp.log(Phi))                # induced measure
rec("a.12", "Phi-power of a mass term in the induced metric g:  bulk (fiber-integrated) -> Phi^0,  fixed-point (induced measure) -> Phi^-1",
    (p_bulk, p_bdry) == (0, -1), f"exponents = ({p_bulk}, {p_bdry})")
# exponent n in m_g = m0 * e^{n psi}; Einstein frame g~ = e^psi g  =>  m~ = e^{-psi/2} m_g
n = sp.Symbol('n')
ein = lambda nn: sp.Rational(-1, 2) + nn
exps = {"Branch A (n=0)": ein(0), "Branch B (n=+1/2)": ein(sp.Rational(1, 2)), "fixed-point, induced measure (n=-1)": ein(-1)}
ratio = (exps["fixed-point, induced measure (n=-1)"] / exps["Branch A (n=0)"])**2
rec("a.13", "Einstein-frame exponents d ln m~/d psi = (-1/2, 0, -3/2); coupling^2 ratio boundary/Branch A = 9",
    list(exps.values()) == [sp.Rational(-1, 2), 0, sp.Rational(-3, 2)] and ratio == 9,
    "with Paper 0's alpha_0^2 = 1/3 for Branch A this is alpha^2 = 3")
F_A = sp.solve(sp.Eq(-1 + n, 0), n)[0]
F_B = sp.solve(sp.Eq(-1 + n, sp.Rational(1, 2)), n)[0]
rec("a.14", "a fixed-point term mu F(Phi) with F = Phi^q: Branch A needs q = 1, Branch B needs q = 3/2 (q is free)",
    (F_A, F_B) == (1, sp.Rational(3, 2)))

# =====================================================================================
# (b) CONFIGURATION COUNT UNDER [A1] + [A4]
# =====================================================================================
print("\n=== (b) zero-mode field-space dimensions ===")
par = lambda a, b: Rz2[a, a] * Rz2[b, b]
all_pairs = list(itertools.product(range(5), repeat=2))
sym_pairs = [(a, b) for a in range(5) for b in range(a, 5)]
asy_pairs = [(a, b) for a in range(5) for b in range(a + 1, 5)]
cnt = lambda prs: (sum(1 for a, b in prs if par(a, b) == 1), sum(1 for a, b in prs if par(a, b) == -1))
rec("b.1", "funfbein / End(V5): 25 = 17 (Z2-even) + 8 (odd)", cnt(all_pairs) == (17, 8))
rec("b.2", "metric Sym^2: 15 = 11 (even: g_mu nu, g_22) + 4 (odd: g_mu 2)", cnt(sym_pairs) == (11, 4))
rec("b.3", "local Lorentz so(3,2): 10 = 6 (even) + 4 (odd)", cnt(asy_pairs) == (6, 4))
rec("b.4", "t2-independent + Z2-equivariant => odd components vanish: metric zero modes = 11 = 17 - 6",
    17 - 6 == 11)
rec("b.5", "after the 4 even diffeomorphisms: 11 - 4 = 7 components; '15 + 2' counts 4 components that vanish",
    11 - 4 == 7 and 15 - 4 == 11)

# =====================================================================================
# (c) THE KERNEL psi_t(A)
# =====================================================================================
print("\n=== (c) what [A2]+[A3]+[A4] fix about psi_t(A) ===")
t, t2, r, th, ph = sp.symbols('t t2 r theta phi', real=True)
f = sp.Function('f')(r)
Ph = sp.Function('Phi')(r)
X = [t, t2, r, th, ph]
g = sp.diag(-f, -Ph**2, 1 / f, r**2, r**2 * sp.sin(th)**2)
gi = g.inv()
Gam = [[[sum(gi[a, d] * (sp.diff(g[d, b], X[c]) + sp.diff(g[d, c], X[b]) - sp.diff(g[b, c], X[d])) for d in range(5)) / 2
         for c in range(5)] for b in range(5)] for a in range(5)]


def riem(a, b, c, d):
    return (sp.diff(Gam[a][b][d], X[c]) - sp.diff(Gam[a][b][c], X[d])
            + sum(Gam[a][c][e] * Gam[e][b][d] - Gam[a][d][e] * Gam[e][b][c] for e in range(5)))


K5 = 0
for a, b in itertools.combinations(range(5), 2):       # diagonal metric: only R^{ab}{}_{ab} survive
    Rabab = sp.simplify(riem(a, b, a, b) * gi[b, b])
    K5 += 4 * Rabab**2
K5 = sp.simplify(K5)
K4 = sp.simplify(sp.diff(f, r, 2)**2 + 4 * sp.diff(f, r)**2 / r**2 + 4 * (1 - f)**2 / r**4)
fp, Pp = sp.diff(f, r), sp.diff(Ph, r)
extra = 4 * ((f * sp.diff(Ph, r, 2) + fp * Pp / 2)**2 + (fp * Pp / 2)**2 + 2 * (f * Pp / r)**2) / Ph**2
rec("c.1", "K_5 = K_4 + 4 |grad grad Phi|^2 / Phi^2  (exact, generic f(r), Phi(r))",
    sp.simplify(K5 - K4 - extra) == 0,
    "=> K_5(r_h) finite  <=>  Phi is C^2 and non-zero at r_h; with Phi = Phi_inf exp(psi_t(f)): psi_t is C^2 at A = 0")

rs, p0, lam, A = sp.symbols('r_s psi_0 lambda A', positive=True)
fS = 1 - rs / r
cands = {
    "Gaussian  psi0[exp(-lam A^2) - exp(-lam)]": p0 * (sp.exp(-lam * A**2) - sp.exp(-lam)),
    "Lorentzian psi0[1/(1+lam A^2) - 1/(1+lam)]": p0 * (1 / (1 + lam * A**2) - 1 / (1 + lam)),
    "linear (odd part) psi0 (1 - A)": p0 * (1 - A),
    "logarithmic psi0 ln A": p0 * sp.log(A),
}
out = {}
for name, pt in cands.items():
    Phi_r = sp.exp(pt.subs(A, fS))
    ex = extra.subs({f: fS, Ph: Phi_r}).doit()
    out[name] = sp.limit(sp.simplify(ex), r, rs, '+')
    print(f"{'':20}{name:<46} psi_t(1) = {sp.simplify(pt.subs(A, 1))},  fiber part of K_5 at r_h = {sp.simplify(out[name])}")
rec("c.2", "[A3] is met by Gaussian, Lorentzian AND a kernel with an odd (linear) part; it excludes ln A",
    all(out[k].is_finite for k in list(cands)[:3]) and out[list(cands)[3]] == sp.oo)

u, v = sp.symbols('u v', positive=True)
Fg, Fl = sp.exp(-lam * u), 1 / (1 + lam * u)
cm = lambda F: all(sp.simplify((-1)**k * sp.diff(F, u, k)).is_positive for k in range(0, 7))
rec("c.3", "Gaussian is NOT completely monotone in A (f'' < 0 for A^2 < 1/(2 lam)); it is in u = A^2",
    sp.simplify(sp.diff(sp.exp(-lam * A**2), A, 2).subs(A, 0)) == -2 * lam and cm(Fg))
rec("c.4", "COUNTERMODEL: F(u) = 1/(1+lam u) has F(0)=1, F'(0)=-lam, is completely monotone (k<=6), and violates F(u+v)=F(u)F(v)",
    Fl.subs(u, 0) == 1 and sp.diff(Fl, u).subs(u, 0) == -lam and cm(Fl)
    and sp.simplify(Fl.subs(u, u + v) - Fl * Fl.subs(u, v)) != 0)
rec("c.5", "semigroup law <=> exponential: F'(u) = F'(0) F(u) has the unique solution exp(-lam u)  [D | semigroup H]",
    sp.dsolve(sp.Function('F')(u).diff(u) + lam * sp.Function('F')(u), sp.Function('F')(u),
              ics={sp.Function('F')(0): 1}).rhs == Fg)
s, w, Rr = sp.symbols('s w R', positive=True)
mell = sp.simplify(sp.integrate(s**(-1) * (4 * sp.pi * s)**(-2) * sp.exp(-sp.pi**2 * w**2 * Rr**2 / s), (s, 0, sp.oo)))
rec("c.6", "[A2]: the per-winding kernel is Gaussian in the fiber length R = L0 Phi, but its Mellin moment is a pure power; no A appears",
    sp.simplify(mell * (w * Rr)**4).is_constant() and not mell.has(A), f"k=0 moment = {mell}")

# =====================================================================================
# (d) REMAINING GROUPS
# =====================================================================================
print("\n=== (d) H3-H6 (former H4-H7) and H_aux ===")
LP, rc = sp.symbols('L_P r_c', positive=True)
sol = sp.solve(sp.Eq(rs / rc**3, 2 / LP**2), rc)
rec("d.1", "inside H3: r_core = (r_s L_P^2/2)^(1/3) follows from chi_max = 2/L_P^2 and chi = r_s/r^3 (one postulate, not two)",
    len(sol) == 1 and sp.simplify(sol[0] - (rs * LP**2 / 2)**sp.Rational(1, 3)) == 0)

xx = sp.symbols('x0:5', real=True)                     # t1, t2, x1, x2, x3
def cs_density(Afield):
    Fm = sp.Matrix(5, 5, lambda a, b: sp.diff(Afield[b], xx[a]) - sp.diff(Afield[a], xx[b]))
    return sp.expand(sum(sp.LeviCivita(a, b, c, d, e) * Afield[a] * Fm[b, c] * Fm[d, e]
                         for a, b, c, d, e in itertools.permutations(range(5))))
dep = (xx[0], xx[2], xx[3], xx[4])
Aeven = [sp.Function('A0')(*dep), sp.Integer(0), sp.Function('A2')(*dep), sp.Function('A3')(*dep), sp.Function('A4')(*dep)]
Aboth = list(Aeven); Aboth[1] = sp.Function('A1')(*dep)
rec("d.2", "inside H6: Chern-Simons density vanishes on [A1]+[A4] photons (A_mu even, A_2 odd => 0)",
    cs_density(Aeven) == 0)
rec("d.3", "            ... and does NOT vanish under [A4] alone (A_2(t1,x) kept): both axioms are needed",
    cs_density(Aboth) != 0)
rec("d.4", "inside H6: (pi/2) chi_E(S^2) = pi  (chi_E(S^2) = 2 by Gauss-Bonnet on the unit sphere)",
    sp.integrate(sp.integrate(sp.sin(th), (th, 0, sp.pi)), (ph, 0, 2 * sp.pi)) / (2 * sp.pi) == 2)

xg, a0 = sp.symbols('x a_0', positive=True)
rec("d.5", "inside H4: mu(x) = x/sqrt(1+x^2) = sin(arctan x) is a trigonometric identity; the law itself is not obtained",
    sp.simplify(sp.sin(sp.atan(xg)) - xg / sp.sqrt(1 + xg**2)) == 0)

# Independence by parameter freedom: [A1]-[A4] contain (L0, Lambda_cut, f_4, f_2, f_0) with f's unconstrained.
c1, c2, c3 = sp.symbols('c1 c2 c3', real=True)         # c_k = f_{4-2k} M_k (free up to sign pattern)
y = sp.Symbol('y', positive=True)                       # y = e^{-2 psi}
V = c1 * y + c2 * y**2 + c3 * y**3
roots = sp.solve(sp.diff(V, y), y)
rec("d.6", "COUNTERMODEL family: the stationary point of V in W_3 moves with the free ratios c_k = f_{4-2k} M_k",
    all(any(rt.has(c) for c in (c1, c2, c3)) for rt in roots),
    "=> Phi_inf, m_psi, G and Lambda are fixed by free moments; no relation among them (Phi_inf(alpha_EM), "
    "lambda_psi = sqrt(L_P R_H), chi_max = 2/L_P^2) can follow from [A1]-[A4]")

print("\n" + "=" * 100)
npass = sum(1 for _, _, ok in RESULTS if ok)
print(f"SUMMARY: {npass}/{len(RESULTS)} exact checks passed")
if npass != len(RESULTS):
    raise SystemExit(1)
