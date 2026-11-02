"""Log-density ODE along trajectories of a 2D linear field (chapter 2).

The field is u_t(x) = g(t) A0 x with g(t) = 1 + 0.8 sin(3 pi t) and an
upper-triangular A0, so the flow matrix and the Gaussian law are exact.
"""
import math
import numpy as np

A0 = np.array([[0.6, 1.5], [0.0, -1.0]])
a, c, b = A0[0, 0], A0[0, 1], A0[1, 1]
TRACE = a + b


def g(t):
    return 1.0 + 0.8 * math.sin(3.0 * math.pi * t)


def big_g(t):
    return t + 0.8 * (1.0 - math.cos(3.0 * math.pi * t)) / (3.0 * math.pi)


def flow_matrix(t):
    """Exact Phi(t) = expm(G(t) A0) for the triangular A0."""
    s = big_g(t)
    off = c * (math.exp(a * s) - math.exp(b * s)) / (a - b)
    return np.array([[math.exp(a * s), off], [0.0, math.exp(b * s)]])


def gauss_logpdf(x, cov):
    """Log density of N(0, cov) in 2D; x has shape (n, 2)."""
    prec = np.linalg.inv(cov)
    quad = np.einsum("ni,ij,nj->n", x, prec, x)
    return -math.log(2.0 * math.pi) - 0.5 * math.log(np.linalg.det(cov)) \
        - 0.5 * quad


def rhs(t, x):
    """Joint right side: dx/dt = u_t(x), d(log p)/dt = -div u_t."""
    return x @ (g(t) * A0).T, -g(t) * TRACE


def integrate(x0, ell0, steps, method):
    h = 1.0 / steps
    x, ell = x0.copy(), ell0.copy()
    for k in range(steps):
        t = k * h
        k1x, k1l = rhs(t, x)
        if method == "euler":
            x, ell = x + h * k1x, ell + h * k1l
            continue
        k2x, k2l = rhs(t + h / 2, x + h / 2 * k1x)
        k3x, k3l = rhs(t + h / 2, x + h / 2 * k2x)
        k4x, k4l = rhs(t + h, x + h * k3x)
        x = x + h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        ell = ell + h / 6 * (k1l + 2 * k2l + 2 * k3l + k4l)
    return x, ell


rng = np.random.default_rng(2026)
phi1 = flow_matrix(1.0)
sigma1 = phi1 @ phi1.T
print(f"J_1 = det Phi(1) = {np.linalg.det(phi1):.12f}, "
      f"exp(G(1) tr A0) = {math.exp(big_g(1.0) * TRACE):.12f}")
x0 = rng.standard_normal((1000, 2))
ell0 = gauss_logpdf(x0, np.eye(2))
identity = ell0 - big_g(1.0) * TRACE
exact_end = gauss_logpdf(x0 @ phi1.T, sigma1)
print(f"identity log p0 - int div vs Gaussian at exact endpoint: "
      f"max error {np.max(np.abs(identity - exact_end)):.2e}")
x1, ell1 = integrate(x0, ell0, 64, "rk4")
for i in range(3):
    ref = gauss_logpdf(x1[i:i + 1], sigma1)[0]
    print(f"x0=({x0[i, 0]:+.4f},{x0[i, 1]:+.4f}) "
          f"x1=({x1[i, 0]:+.4f},{x1[i, 1]:+.4f}) "
          f"ODE log p1={ell1[i]:+.10f} exact={ref:+.10f}")
print("steps  Euler total   RK4 total     RK4 log p only  RK4 position")
for steps in (8, 16, 32, 64):
    errs = []
    for method in ("euler", "rk4"):
        xn, ln = integrate(x0, ell0, steps, method)
        errs.append(np.max(np.abs(ln - gauss_logpdf(xn, sigma1))))
    errs.append(np.max(np.abs(ln - identity)))
    errs.append(np.max(np.abs(xn - x0 @ phi1.T)))
    print(f"{steps:5d}  " + "  ".join(f"{e:.6e}" for e in errs))
sym = 0.5 * (A0 + A0.T)
frob2 = float(np.sum(sym * sym))
diag2 = float(np.sum(np.diag(sym) ** 2))
probes = {"Gaussian": rng.standard_normal((20000, 2)),
          "Rademacher": rng.choice([-1.0, 1.0], size=(20000, 2))}
theory = {"Gaussian": 2 * frob2, "Rademacher": 2 * (frob2 - diag2)}
for name, eta in probes.items():
    est = np.einsum("ni,ij,nj->n", eta, A0, eta)
    print(f"Hutchinson {name:10s}: mean {est.mean():+.4f} (tr A0 = "
          f"{TRACE:+.4f}), variance {est.var():.4f}, theory "
          f"{theory[name]:.4f}")
d, n_steps = 2, 64
print("conditional path, d=2: t, Lipschitz 1/(1-t), exact rise "
      "-d log(1-t), RK4 rise (64 steps)")
for t_end in (0.9, 0.99, 0.999):
    h = t_end / n_steps
    rise = 0.0
    for k in range(n_steps):
        s = k * h
        f = [d / (1.0 - s - w * h) for w in (0.0, 0.5, 1.0)]
        rise += h / 6 * (f[0] + 4 * f[1] + f[2])
    print(f"{t_end:.3f}  {1 / (1 - t_end):8.1f}  "
          f"{-d * math.log(1 - t_end):.8f}  {rise:.8f}")
atom = math.erf(0.25 / math.sqrt(2.0))
print(f"collapse u=-2 sign(x) sqrt|x|: atom mass at 0 when t=0.5 is "
      f"{atom:.6f}")
