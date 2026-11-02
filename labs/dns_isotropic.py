"""Forced isotropic turbulence by pseudo-spectral DNS, for Lab 2 and ch. 3, 6.

Method (the scheme of the JHTDB isotropic1024coarse README, with two
changes): periodic box [0, 2 pi)^3, incompressible Navier-Stokes
in vorticity (rotational) form, viscous term integrated exactly with an
integrating factor, nonlinear term by third-order SSP Runge-Kutta
instead of Adams-Bashforth, de-aliased by the 2/3 rule instead of phase
shift with 2 sqrt(2)/3 truncation. Energy is injected as in JHTDB: after
each step the modes with 0 < |k| <= 2 are rescaled so that their total
energy keeps its initial value. Fourier coefficients use the course
convention fftn / N^3 (torch norm='forward'); arrays are (3, N, N, N) in
the order (component, x, y, z), as in labs/jhtdb_fetch.py.

Outputs in --out (default data/dns):
  params.json         run parameters
  stats.csv           time series: k, eps, u', L, T_L, lambda, Re_lambda,
                      eta, kmax*eta, gradient skewness, CFL, ms/step
  spectrum.csv        E(kappa) of every snapshot (one row per snapshot)
  corr.csv            rho(lag) = <u(t_s) . u(t_s + lag)> / (2 k) between
                      each snapshot t_s and the logged times after it
  filtered/ssss.npy   64^3 field: sharp cutoff |k_i| < 32, then the 64^3
                      grid (periodic, alias-free; training data)
  crop/ssss_jj.npy    64^3 contiguous blocks of the N^3 field (full
                      resolution, not periodic)
  full/ssss.npy       N^3 velocity of the first --full snapshots
  state.pt            checkpoint for --resume
File names start with the snapshot number, so a sorted split is a split
in time (chapter 6).

Run:  python labs/dns_isotropic.py --check          (32^3 tests, seconds)
      python labs/dns_isotropic.py --n 256 --steps 20 --snaps 0  (timing)
      python labs/dns_isotropic.py --n 256          (full run, hours)
"""
import argparse
import json
import math
import pathlib
import time
import numpy as np
import torch

FFT = dict(norm='forward')


class Grid:
    def __init__(self, n, nu, dt, dev):
        self.n, self.nu, self.dt = n, nu, dt
        k1 = torch.fft.fftfreq(n, 1.0 / n, device=dev)
        kz = torch.fft.rfftfreq(n, 1.0 / n, device=dev)
        self.k = (k1.view(-1, 1, 1), k1.view(1, -1, 1), kz.view(1, 1, -1))
        self.k2 = self.k[0]**2 + self.k[1]**2 + self.k[2]**2
        self.k2inv = 1.0 / torch.where(self.k2 == 0, 1.0, self.k2)
        cut = n // 3   # 2/3 rule: keep |k_i| <= N/3
        self.mask = ((self.k[0].abs() <= cut) & (self.k[1].abs() <= cut)
                     & (self.k[2].abs() <= cut))
        self.kmax = cut
        self.decay = torch.exp(-nu * self.k2 * dt)
        self.half = torch.exp(-nu * self.k2 * dt / 2)
        # Weight of each stored mode in sums over the full spectrum.
        w = torch.full((n // 2 + 1,), 2.0, device=dev)
        w[0] = w[-1] = 1.0
        self.w = w.view(1, 1, -1)
        self.shell = torch.round(self.k2.sqrt()).long()
        self.band = (self.k2 > 0) & (self.k2 <= 4)

    def project(self, vh):
        kdot = (self.k[0] * vh[0] + self.k[1] * vh[1]
                + self.k[2] * vh[2]) * self.k2inv
        return torch.stack([vh[i] - self.k[i] * kdot for i in range(3)])

    def curl(self, vh):
        kx, ky, kz = self.k
        return 1j * torch.stack([ky * vh[2] - kz * vh[1],
                                 kz * vh[0] - kx * vh[2],
                                 kx * vh[1] - ky * vh[0]])

    def rhs(self, uh):
        """Projected, de-aliased u x omega (pressure is in the projection)."""
        s = (self.n,) * 3
        u = torch.fft.irfftn(uh, s=s, dim=(1, 2, 3), **FFT)
        w = torch.fft.irfftn(self.curl(uh), s=s, dim=(1, 2, 3), **FFT)
        uxw = torch.stack([u[1] * w[2] - u[2] * w[1],
                           u[2] * w[0] - u[0] * w[2],
                           u[0] * w[1] - u[1] * w[0]])
        del u, w
        nh = torch.fft.rfftn(uxw, dim=(1, 2, 3), **FFT) * self.mask
        return self.project(nh)

    def step(self, uh):
        """SSP Runge-Kutta 3 with integrating factor (exact viscous term).

        Heun (RK2) was tried first: it amplifies pure advection, and at
        256^3 the high wavenumbers grew faster than viscosity damped them.
        """
        dt, a, h = self.dt, self.decay, self.half
        u1 = a * (uh + dt * self.rhs(uh))
        u2 = 0.75 * h * uh + 0.25 / h * (u1 + dt * self.rhs(u1))
        del u1
        return a * uh / 3 + 2 / 3 * h * (u2 + dt * self.rhs(u2))

    def clean(self, uh):
        """Remove the three parts that the nonlinear term cannot control.

        1. The kz = 0 plane stores k and -k separately. Its non-Hermitian
        part is roundoff that is not part of any real field. The forcing
        amplifies it and cuFFT maps it into the Nyquist plane, where it
        aliases into the mean flow. Measured without this: mean-flow energy
        2e-6 at t = 52, 0.19 at t = 111.
        2. The mean flow (k = 0) has no viscous damping.
        3. The divergent part of u: rhs() projects only the nonlinear term,
        so roundoff in k . u gets no transfer, and the forcing amplifies it
        at rate eps / (2 E_f) - nu k^2. Measured without this at 256^3:
        energy 4e-12 at t = 12, 0.17 at t = 111, at amplitude rate 0.146
        (predicted 0.150), with gradient flatness 4.5 -> 28.
        """
        n, p = self.n, uh[..., 0]
        i = (-torch.arange(n, device=uh.device)) % n
        uh[..., 0] = 0.5 * (p + p[:, i][:, :, i].conj())
        uh[:, 0, 0, 0] = 0
        return self.project(uh)

    def corr(self, ah, bh):
        """<a . b> / sqrt(<a . a> <b . b>) from Fourier coefficients."""
        ab = float(((ah * bh.conj()).real.sum(0) * self.w).sum())
        return 0.5 * ab / math.sqrt(self.energy(ah) * self.energy(bh))

    def energy(self, vh, weight=None):
        e = (vh.real**2 + vh.imag**2).sum(0) * self.w
        if weight is not None:
            e = e * weight
        return 0.5 * float(e.sum())

    def spectrum(self, vh):
        e = 0.5 * (vh.real**2 + vh.imag**2).sum(0) * self.w
        return torch.bincount(self.shell.ravel(), e.ravel(),
                              self.n).cpu().numpy()

    def stats(self, uh):
        # Sums over modes with the exact |k|, not over rounded shells.
        e_k = self.spectrum(uh)
        k_e = self.energy(uh)
        eps = 2 * self.nu * self.energy(uh, self.k2)
        up = math.sqrt(2 * k_e / 3)
        length = math.pi / (2 * up**2) * self.energy(uh, self.k2inv.sqrt())
        lam = math.sqrt(15 * self.nu * up**2 / eps)
        eta = (self.nu**3 / eps)**0.25
        s = (self.n,) * 3
        dudx = torch.fft.irfftn(1j * self.k[0] * uh[0], s=s, **FFT)
        skew = float((dudx**3).mean() / (dudx**2).mean()**1.5)
        umax = float(torch.fft.irfftn(uh, s=s, dim=(1, 2, 3), **FFT)
                     .abs().max())
        return dict(k=k_e, eps=eps, urms=up, L=length, TL=length / up,
                    lam=lam, Re_lam=up * lam / self.nu, eta=eta,
                    kmax_eta=self.kmax * eta, skew=skew,
                    cfl=umax * self.dt * self.n / (2 * math.pi)), e_k


def initial_field(g, energy, k0, gen):
    """Random solenoidal field with E(k) ~ k^4 exp(-2 (k/k0)^2)."""
    shape = (3,) + tuple(g.k2.shape)
    vh = torch.complex(torch.randn(shape, generator=gen, device=g.k2.device),
                       torch.randn(shape, generator=gen, device=g.k2.device))
    kk = g.k2.sqrt()
    spec = kk**4 * torch.exp(-2 * (kk / k0)**2)
    vh = g.project(vh * torch.sqrt(spec * g.k2inv) * g.mask)
    s = (g.n,) * 3
    # Round trip through real space makes the half-spectrum Hermitian.
    # cuFFT can put the non-Hermitian part into the Nyquist plane, so
    # the mask is applied again.
    vh = torch.fft.rfftn(torch.fft.irfftn(vh, s=s, dim=(1, 2, 3), **FFT),
                         dim=(1, 2, 3), **FFT) * g.mask
    return vh * math.sqrt(energy / g.energy(vh))


def filtered64(g, uh, m=64):
    """Sharp cutoff |k_i| < m/2, then the m^3 grid (alias-free)."""
    h = m // 2
    idx = torch.cat([torch.arange(h, device=uh.device),
                     torch.arange(g.n - h, g.n, device=uh.device)])
    c = uh[:, idx][:, :, idx][..., :h + 1].clone()
    c[:, h] = 0
    c[:, :, h] = 0
    c[..., h] = 0
    return torch.fft.irfftn(c, s=(m,) * 3, dim=(1, 2, 3), **FFT)


def check(dev):
    """Self-checks at 32^3; each assert names what it tests."""
    n, nu = 32, 0.05
    g = Grid(n, nu, 0.01, dev)
    # 1. A shear wave u = (sin 3y, 0, 0) has zero nonlinear term, so it
    #    decays exactly as exp(-nu k^2 t).
    y = torch.arange(n, device=dev) * 2 * math.pi / n
    u = torch.zeros(3, n, n, n, device=dev)
    u[0] = torch.sin(3 * y).view(1, -1, 1)
    uh = torch.fft.rfftn(u, dim=(1, 2, 3), **FFT)
    e0 = g.energy(uh)
    for _ in range(100):
        uh = g.step(uh)
    err = abs(g.energy(uh) / e0 / math.exp(-2 * nu * 9 * 1.0) - 1)
    print(f'shear-wave decay, relative error {err:.1e}')
    assert err < 1e-5, 'viscous term or time stepper'
    # 2. Decaying random field: divergence stays zero, and the energy
    #    budget dk/dt = -eps closes to the O(dt^2) error of the scheme.
    gen = torch.Generator(device=dev).manual_seed(1)
    g = Grid(n, 0.02, 0.005, dev)
    uh = initial_field(g, 0.5, 3.0, gen)
    ks, eps = [], []
    for _ in range(201):
        st, _ = g.stats(uh)
        ks.append(st['k'])
        eps.append(st['eps'])
        uh = g.step(uh)
    ks, eps = np.array(ks), np.array(eps)
    dkdt = (ks[2:] - ks[:-2]) / (2 * g.dt)
    res = np.abs(dkdt + eps[1:-1]).max() / eps.max()
    div = (g.k[0] * uh[0] + g.k[1] * uh[1] + g.k[2] * uh[2]).abs().max()
    div = float(div / (g.k2.sqrt() * uh.abs().sum(0)).max())
    print(f'energy budget residual / eps {res:.1e}, divergence {div:.1e}, '
          f'k fell {ks[0]:.4f} -> {ks[-1]:.4f}')
    assert res < 1e-2, 'energy budget (nonlinear term must conserve k)'
    assert div < 1e-5, 'projection'
    # 3. The 64^3 filtered field of a field that already has |k_i| < 32
    #    equals the field sampled at every other point.
    g = Grid(128, 0.02, 0.005, dev)
    uh = initial_field(g, 0.5, 8.0, gen)
    uh = uh * ((g.k[0].abs() < 32) & (g.k[1].abs() < 32) & (g.k[2] < 32))
    full = torch.fft.irfftn(uh, s=(128,) * 3, dim=(1, 2, 3), **FFT)
    err = float((filtered64(g, uh) - full[:, ::2, ::2, ::2]).abs().max()
                / full.abs().max())
    print(f'filtered64 against subsampling of a band-limited field {err:.1e}')
    assert err < 1e-5, 'filtered64 index mapping'
    # 4. clean() removes a mean flow, a non-Hermitian part and a divergent
    #    part, nothing else.
    g = Grid(32, 0.02, 0.005, dev)
    good = initial_field(g, 0.5, 3.0, gen)
    bad = good.clone()
    bad[:, 0, 0, 0] = 0.1
    bad[:, 1, 2, 0] += 0.01j   # k and -k with equal, not conjugate, values
    bad[:, -1, -2, 0] += 0.01j
    bad[:, 1, 1, 1] += 0.01    # parallel to k = (1, 1, 1)
    err = float((g.clean(bad) - g.clean(good)).abs().max())
    nyq = float((g.clean(good) - good).abs().max() / good.abs().max())
    print(f'clean(): left over {err:.1e}, change to a real field {nyq:.1e}')
    assert nyq < 1e-5, 'clean() must not change a real field'
    assert err < 1e-6, 'clean() must remove mean, non-Hermitian, divergence'
    print('all checks pass')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--nu', type=float, default=2.0e-3)
    ap.add_argument('--dt', type=float, default=4.0e-3)
    ap.add_argument('--energy', type=float, default=0.7)
    ap.add_argument('--spinup', type=float, default=20.0,
                    help='time before the first snapshot')
    ap.add_argument('--every', type=float, default=1.0,
                    help='time between snapshots')
    ap.add_argument('--snaps', type=int, default=100)
    ap.add_argument('--crops', type=int, default=8)
    ap.add_argument('--full', type=int, default=4)
    ap.add_argument('--steps', type=int, default=0,
                    help='if > 0: run this many steps, report, stop')
    ap.add_argument('--log', type=int, default=25)
    ap.add_argument('--out', default='data/dns')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.check:
        return check(dev)
    out = pathlib.Path(args.out)
    for d in ('filtered', 'crop', 'full'):
        (out / d).mkdir(parents=True, exist_ok=True)
    g = Grid(args.n, args.nu, args.dt, dev)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.resume:
        st = torch.load(out / 'state.pt', map_location=dev, weights_only=True)
        uh, t, step, snap, e_f = (st['uh'], st['t'], st['step'],
                                  st['snap'], st['e_f'])
        rng.bit_generator.state = st['rng']
    else:
        uh = initial_field(g, args.energy, 2.0, gen)
        t, step, snap = 0.0, 0, 0
        e_f = g.energy(uh, g.band)
        (out / 'params.json').write_text(json.dumps(
            dict(vars(args), forced_band_energy=e_f, kmax=g.kmax,
                 device=torch.cuda.get_device_name() if dev == 'cuda'
                 else 'cpu', torch=torch.__version__), indent=1))
        (out / 'stats.csv').write_text(
            't,step,k,eps,urms,L,TL,lam,Re_lam,eta,kmax_eta,skew,cfl,'
            'ms_per_step\n')
        (out / 'spectrum.csv').unlink(missing_ok=True)
    band = g.band
    ref = None
    next_snap = args.spinup + snap * args.every
    stop = args.spinup + max(args.snaps - 1, 0) * args.every - 0.5 * args.dt
    if dev == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t0, s0 = time.time(), step
    while True:
        uh = g.clean(g.step(uh))
        uh = torch.where(band, uh * math.sqrt(e_f / g.energy(uh, band)), uh)
        step += 1
        t += args.dt
        if step % args.log == 0 or step - s0 == args.steps:
            if dev == 'cuda':
                torch.cuda.synchronize()
            ms = 1e3 * (time.time() - t0) / (step - s0)
            st, _ = g.stats(uh)
            if not math.isfinite(st['k']):
                raise SystemExit(f'blow-up at t = {t:.3f}; lower --dt')
            with open(out / 'stats.csv', 'a') as f:
                f.write(f'{t:.4f},{step},' + ','.join(
                    f'{v:.6g}' for v in st.values()) + f',{ms:.1f}\n')
            print(f't {t:7.3f}  k {st["k"]:.4f}  eps {st["eps"]:.4f}  '
                  f'Re_lam {st["Re_lam"]:5.1f}  kmax*eta '
                  f'{st["kmax_eta"]:.2f}  skew {st["skew"]:+.3f}  '
                  f'CFL {st["cfl"]:.2f}  {ms:.0f} ms/step', flush=True)
            if ref is not None:
                with open(out / 'corr.csv', 'a') as f:
                    f.write(f'{snap - 1},{t - ref[0]:.4f},'
                            f'{g.corr(ref[1], uh):.6f}' + '\n')
        if args.steps and step - s0 == args.steps:
            if dev == 'cuda':
                print(f'peak memory: allocated '
                      f'{torch.cuda.max_memory_allocated() / 2**20:.0f} MiB'
                      f', reserved '
                      f'{torch.cuda.max_memory_reserved() / 2**20:.0f} MiB')
            return
        if snap < args.snaps and t >= next_snap - 0.5 * args.dt:
            st, e_k = g.stats(uh)
            with open(out / 'spectrum.csv', 'a') as f:
                f.write(f'{snap},{t:.4f},' +
                        ','.join(f'{e:.6e}' for e in e_k) + '\n')
            np.save(out / 'filtered' / f'{snap:04d}.npy',
                    filtered64(g, uh).cpu().numpy())
            u = torch.fft.irfftn(uh, s=(args.n,) * 3, dim=(1, 2, 3), **FFT)
            for j in range(args.crops):
                x, y, z = rng.integers(0, args.n - 64 + 1, 3)
                np.save(out / 'crop' / f'{snap:04d}_{j:02d}.npy',
                        u[:, x:x + 64, y:y + 64, z:z + 64].cpu().numpy())
            if snap < args.full:
                np.save(out / 'full' / f'{snap:04d}.npy', u.cpu().numpy())
            del u
            ref = (t, uh.clone())
            snap += 1
            next_snap += args.every
            torch.save(dict(uh=uh, t=t, step=step, snap=snap, e_f=e_f,
                            rng=rng.bit_generator.state), out / 'state.pt')
            print(f'snapshot {snap - 1} saved at t = {t:.3f}', flush=True)
        if t >= stop and snap == args.snaps:
            break
    torch.save(dict(uh=uh, t=t, step=step, snap=snap, e_f=e_f,
                    rng=rng.bit_generator.state), out / 'state.pt')
    print(f'done: {snap} snapshots, t = {t:.3f}, {step} steps')


if __name__ == '__main__':
    main()
