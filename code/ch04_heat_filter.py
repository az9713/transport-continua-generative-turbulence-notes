"""Gaussian filters as heat flow, semigroup tests and backward growth.

Chapter 4 of Transport, Continua and Generative Models of Turbulent Flow.
Standard library and NumPy only; periodic box [0, 2*pi) in every direction.
"""
import numpy as np

NU = 1.0
TWO_PI = 2 * np.pi


def heat(f, s, nu=NU):
    """Exact spectral heat flow d_s f = nu * lap f in any dimension."""
    axes = [np.fft.fftfreq(m, d=1.0 / m) for m in f.shape]
    ks = np.meshgrid(*axes, indexing="ij")
    k2 = sum(k * k for k in ks)
    return np.real(np.fft.ifftn(np.exp(-nu * k2 * s) * np.fft.fftn(f)))


def gauss_matrix(n, width):
    """Circulant matrix of the periodised Gaussian kernel (quadrature)."""
    h = TWO_PI / n
    x = np.arange(n) * h
    r = x[:, None] - x[None, :]
    images = r[..., None] + TWO_PI * np.arange(-3, 4)
    g = np.sqrt(6 / np.pi) / width * np.exp(-6 * images**2 / width**2)
    return h * g.sum(axis=-1)


def random_field_1d(n, rng, slope=5 / 6):
    """Unit-rms periodic signal with rfft amplitudes |k|**(-slope)."""
    k = np.arange(n // 2 + 1)
    amp = np.zeros(n // 2 + 1)
    amp[1:-1] = k[1:-1] ** (-slope)
    c = amp * (rng.normal(size=k.size) + 1j * rng.normal(size=k.size))
    c /= np.sqrt(2)
    f = np.fft.irfft(c, n)
    scale = 1 / f.std()
    return f * scale, amp * scale


def rel(g, f):
    """Relative L2 error of an estimate g of f."""
    return np.linalg.norm(g - f) / np.linalg.norm(f)


def snr(kap, lam, scale, nus, floor):
    """Per-mode SNR: scale * exp(-2 nu s kap^2) * kap^-lam / floor."""
    return scale * np.exp(-2 * nus * kap**2) * kap ** (-lam) / floor


def level_crossing(lam, scale, nus, floor, level):
    """Wavenumber where the monotone per-mode SNR equals level."""
    lo, hi = 1e-3, 1e4
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        big = snr(mid, lam, scale, nus, floor) > level
        lo, hi = (mid, hi) if big else (lo, mid)
    return mid


rng = np.random.default_rng(4)
print("1. Gaussian filter of width Delta versus heat flow to s=Delta^2/24nu")
n, width = 256, TWO_PI / 16
s = width**2 / (24 * NU)
f, _ = random_field_1d(n, rng)
m = gauss_matrix(n, width)
diff = np.abs(m @ f - heat(f, s)).max()
x = np.arange(n) * TWO_PI / n
r = np.where(x < np.pi, x, x - TWO_PI)
print(f"  1D N={n}: max|conv - heat| = {diff:.2e}")
print(f"  kernel second moment {np.sum(r**2 * m[:, 0]):.10f}, "
      f"Delta^2/12 {width**2 / 12:.10f}, 2 nu s {2 * NU * s:.10f}")
n3 = 32
f3 = rng.normal(size=(n3, n3, n3))
for ratio in (4, 8):
    width3 = ratio * TWO_PI / n3
    m3 = gauss_matrix(n3, width3)
    g3 = np.einsum("ia,abc->ibc", m3, f3)
    g3 = np.einsum("jb,abc->ajc", m3, g3)
    g3 = np.einsum("kc,abc->abk", m3, g3)
    diff3 = np.abs(g3 - heat(f3, width3**2 / (24 * NU))).max()
    print(f"  3D N={n3}^3, Delta={ratio}h, separable kernel: "
          f"max|conv - heat| = {diff3:.2e}, "
          f"G_hat(pi/h) = {np.exp(-np.pi**2 * ratio**2 / 24):.2e}")

print("2. Independent check: explicit finite-difference heat equation")
phase = rng.uniform(0, TWO_PI, size=8)
modes = np.arange(1, 9)
previous = None
steps = int(np.ceil(s / (0.2 * (TWO_PI / 128) ** 2 / NU)))
for n in (128, 256, 512):
    h = TWO_PI / n
    x = np.arange(n) * h
    u = np.sum(modes[:, None] ** (-5 / 6)
               * np.cos(modes[:, None] * x + phase[:, None]), axis=0)
    dt = s / steps
    for _ in range(steps):
        u = u + NU * dt * (np.roll(u, 1) - 2 * u + np.roll(u, -1)) / h**2
    exact = np.sum(modes[:, None] ** (-5 / 6)
                   * np.exp(-NU * modes[:, None] ** 2 * s)
                   * np.cos(modes[:, None] * x + phase[:, None]), axis=0)
    err = np.abs(u - exact).max()
    ratio = "" if previous is None else f", ratio {previous / err:.2f}"
    print(f"  N={n}: {steps} steps, max error {err:.3e}{ratio}")
    previous = err
    steps *= 4

print("3. Composition of filters")
n = 256
w1, w2 = TWO_PI / 16, TWO_PI / 8
both = gauss_matrix(n, w1) @ gauss_matrix(n, w2)
one = gauss_matrix(n, np.hypot(w1, w2))
print(f"  Gaussian D1, D2 vs sqrt(D1^2+D2^2): max diff "
      f"{np.abs(both - one).max() / one.max():.2e} (relative)")
q = np.linspace(1e-9, 6 * np.pi, 20001)
twice = np.sinc(q / TWO_PI) ** 2
best = min(np.abs(twice - np.sinc(a * q / TWO_PI)).max()
           for a in np.linspace(0.5, 3.0, 2501))
print(f"  box twice vs every box of width 0.5D..3D: min sup gap {best:.4f}")
print(f"  box transfer minimum {np.sinc(q / TWO_PI).min():.6f}")

print("4. Backward heat problem: amplification at the grid Nyquist mode")
for ratio in (1, 2, 4, 8):
    print(f"  Delta/h = {ratio}: exp(pi^2 (Delta/h)^2 / 24) = "
          f"{np.exp(np.pi**2 * ratio**2 / 24):.4e}")
n, sigma = 64, 1e-3
f, amp = random_field_1d(n, rng)
noise = sigma * rng.normal(size=n)
k = np.arange(n // 2 + 1)
for ratio in (4, 8):
    s = (ratio * TWO_PI / n) ** 2 / (24 * NU)
    gain = np.exp(-NU * k**2 * s)
    d_hat = gain * np.fft.rfft(f) + np.fft.rfft(noise)
    data = np.fft.irfft(d_hat, n)
    naive = np.fft.irfft(d_hat / gain, n)
    print(f"  N={n}, Delta={ratio}h, sigma={sigma}: data error "
          f"{rel(data, f):.4f}, naive inverse {rel(naive, f):.3e}")
    for order in (5, 20, 100):
        q_n = sum((1 - gain) ** j for j in range(order + 1))
        est = np.fft.irfft(q_n * d_hat, n)
        print(f"    van Cittert N={order:3d}: max gain {q_n.max():8.3f}, "
              f"error {rel(est, f):.4f}")
    prior = amp**2
    wiener = gain * prior / (gain**2 * prior + n * sigma**2)
    bound = amp[1:-1] / (2 * np.sqrt(n) * sigma)
    print(f"    Wiener (Gaussian prior): max gain {wiener.max():.3f}, "
          f"max gain/bound {np.max(wiener[1:-1] / bound):.4f}, error "
          f"{rel(np.fft.irfft(wiener * d_hat, n), f):.4f}")

print("5. Two spectral orderings, per-mode density kappa^-lambda")
kc = 16.0
for lam in (5 / 3, 11 / 3):
    t = 1 / (1 + kc ** (-lam / 2))
    floor = 1e-8
    nus = (np.log(1 / floor) - lam * np.log(kc)) / (2 * kc**2)
    ratio2 = (t / (1 - t)) ** 2
    print(f"  lambda={lam:.4f}: noise t={t:.6f}; blur nu*s={nus:.6f}, "
          f"Delta={np.sqrt(24 * nus):.4f}, floor {floor:.0e}")
    for kap in (4, 8, 16, 24, 32):
        print(f"    kappa={kap:2d}: SNR noise "
              f"{snr(kap, lam, ratio2, 0.0, 1.0):9.3e}, blur "
              f"{snr(kap, lam, 1.0, nus, floor):9.3e}")
    band = [level_crossing(lam, *p, 0.1) / level_crossing(lam, *p, 10.0)
            for p in ((ratio2, 0.0, 1.0), (1.0, nus, floor))]
    print(f"    band kappa(SNR=0.1)/kappa(SNR=10): noise {band[0]:.3f}, "
          f"blur {band[1]:.3f}")

print("6. A physical reading of s: Gaussian 80 % width for the jet")
delta, nu_air, k_tke, eps = 12.95e-3, 1.4607e-5, 6.0, 80.0
s_jet = delta**2 / (24 * nu_air)
print(f"  s = Delta^2/(24 nu) = {s_jet:.4f} s; k/eps = {k_tke / eps:.4f} s; "
      f"ratio {s_jet / (k_tke / eps):.2f}")
