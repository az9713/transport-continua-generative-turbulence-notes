"""Lab 2 (week 12): flow matching on 3D velocity fields, judged by physics.

Course convention: X_t = t z + (1 - t) xi, t = 0 noise, t = 1 data, with
target velocity z - xi. The model is a small 3D U-Net that fits a 4 GB GPU
(measured: 64^3, batch 4, width 32 uses about 2.5 GB with fp16 autocast).

Data sources
  --data synthetic  Gaussian random solenoidal fields with a model spectrum.
                    They have the right second-order statistics but none of
                    the non-Gaussian structure of real turbulence; use them
                    to test the pipeline and to study spectral collapse.
  --data filtered   64^3 periodic fields made by labs/dns_isotropic.py
                    (sharp spectral cutoff of the 256^3 DNS, then the 64^3
                    grid); circular padding.
  --data crop       64^3 contiguous blocks of the same DNS; full resolution
                    but not periodic; zero padding.
  --data npy        Any (3, N, N, N) cubes in --dir, e.g. from the optional
                    labs/jhtdb_fetch.py; zero padding.
File names start with the snapshot number, so the last 20 % of the sorted
files, the test set, are the latest in time (chapter 6). --guard G drops the
G files before the test set, so that the split has a guard band.

Evaluation (chapter 5): shell-summed energy spectrum E(kappa) of generated
versus held-out fields, the L2 distance of log spectra, per-component
standard deviation, skewness and flatness, and velocity-gradient skewness.
Shells stop at N/2 - 1, the last shell without self-conjugate modes.
Spectral-collapse detector (section 5.11): for one field against one field,
the departure shell is the smallest n_d with |log(E_a / E_b)| > 6 / sqrt(M_n)
in every shell from n_d to the top, where M_n is the number of wavevectors
in shell n. It is applied to generated/held-out pairs and, as the control,
to pairs of held-out fields as far apart in time as the test set allows.
Run:  python labs/lab2_generative_turbulence_3d.py [--n 32] [--iters 1500]
"""
import argparse
import glob
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def synthetic_fields(m, n, rng):
    """m solenoidal Gaussian fields on [0, 2 pi)^3 with a model spectrum."""
    k1 = np.fft.fftfreq(n, 1.0 / n)
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing='ij')
    kk = np.sqrt(kx**2 + ky**2 + kz**2)
    kk[0, 0, 0] = 1.0
    ke, kd = 2.0, n / 4
    spec = kk**4 / (1 + (kk / ke)**2)**(17 / 6) * np.exp(-(kk / kd)**2)
    amp = np.sqrt(spec / (4 * np.pi * kk**2))
    amp[0, 0, 0] = 0.0
    out = np.empty((m, 3, n, n, n), np.float32)
    for i in range(m):
        w = rng.standard_normal((3, n, n, n))
        wh = np.fft.fftn(w, axes=(1, 2, 3)) * amp
        kdot = (kx * wh[0] + ky * wh[1] + kz * wh[2]) / kk**2
        wh = wh - np.stack([kx, ky, kz]) * kdot
        out[i] = np.fft.ifftn(wh, axes=(1, 2, 3)).real
    return out


def shells(n):
    k1 = np.fft.fftfreq(n, 1.0 / n)
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing='ij')
    return np.rint(np.sqrt(kx**2 + ky**2 + kz**2)).astype(int).ravel()


def spectra(v):
    """E(kappa) of each field (S, 3, N, N, N), convention fftn / N^3."""
    n = v.shape[-1]
    shell = shells(n)
    return np.stack([np.bincount(
        shell, (0.5 * (np.abs(np.fft.fftn(f, axes=(1, 2, 3)) / n**3)**2)
                .sum(0)).ravel(), n) for f in v])


def departure(ea, eb, mn):
    """Departure shell of section 5.11, or 0 if the top shell is inside."""
    out = np.abs(np.log(ea[1:] / eb[1:])) > 6 / np.sqrt(mn[1:])
    n_d = 0
    for n in range(len(out), 0, -1):
        if not out[n - 1]:
            break
        n_d = n
    return n_d


def report_departures(name, ea, eb, top, mn):
    nd = [departure(a[:top + 1], b[:top + 1], mn[:top + 1])
          for a, b in zip(ea, eb)]
    hit = [d for d in nd if d]
    print(f'departure shell, {name}: {len(hit)} of {len(nd)} pairs depart'
          + (f', median n_d {int(np.median(hit))}, range {min(hit)}'
             f'..{max(hit)}' if hit else ''))


def gradient_skewness(v):
    """Skewness of du_i/dx_i by one-sided differences, no wrap-around."""
    out = []
    for i in range(3):
        g = np.diff(v[:, i], axis=i + 1)
        out.append((g**3).mean() / (g**2).mean()**1.5)
    return np.array(out)


def blk(i, o, pad):
    return nn.Sequential(
        nn.Conv3d(i, o, 3, padding=1, padding_mode=pad), nn.GroupNorm(8, o),
        nn.SiLU(),
        nn.Conv3d(o, o, 3, padding=1, padding_mode=pad), nn.GroupNorm(8, o),
        nn.SiLU())


class UNet3D(nn.Module):
    """Two-level 3D U-Net; time enters as an extra constant channel."""

    def __init__(self, c=32, pad='circular'):
        super().__init__()
        self.e1, self.e2 = blk(4, c, pad), blk(c, 2 * c, pad)
        self.e3 = blk(2 * c, 4 * c, pad)
        self.d2, self.d1 = blk(6 * c, 2 * c, pad), blk(3 * c, c, pad)
        self.out = nn.Conv3d(c, 3, 1)

    def forward(self, x, t):
        tt = t.view(-1, 1, 1, 1, 1).expand(-1, 1, *x.shape[2:])
        a = self.e1(torch.cat([x, tt], 1))
        b = self.e2(F.avg_pool3d(a, 2))
        c = self.e3(F.avg_pool3d(b, 2))
        b = self.d2(torch.cat([F.interpolate(c, scale_factor=2), b], 1))
        a = self.d1(torch.cat([F.interpolate(b, scale_factor=2), a], 1))
        return self.out(a)


def moments(v):
    """Per-component std, skewness and flatness over all samples."""
    u = v.transpose(1, 0, 2, 3, 4).reshape(3, -1)
    u = u - u.mean(1, keepdims=True)
    s = u.std(1)
    return s, (u**3).mean(1) / s**3, (u**4).mean(1) / s**4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', choices=['synthetic', 'filtered', 'crop',
                                       'npy'], default='synthetic')
    ap.add_argument('--dir', help='default data/dns/<data> or data/jhtdb')
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--fields', type=int, default=320)
    ap.add_argument('--iters', type=int, default=1500)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--samples', type=int, default=32)
    ap.add_argument('--euler', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--guard', type=int, default=0,
                    help='files dropped between the train and test sets')
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if args.data == 'synthetic':
        data = synthetic_fields(args.fields, args.n, rng)
        pad = 'circular'
    else:
        d = args.dir or ('data/jhtdb' if args.data == 'npy'
                         else f'data/dns/{args.data}')
        files = sorted(glob.glob(f'{d}/*.npy'))
        if not files:
            raise SystemExit(f'no .npy cubes in {d}; run dns_isotropic.py')
        data = np.stack([np.load(f) for f in files]).astype(np.float32)
        pad = 'circular' if args.data == 'filtered' else 'zeros'
    n_test = max(len(data) // 5, 1)
    train, test = data[:-n_test - args.guard], data[-n_test:]
    scale = float(train.std())
    print(f'{len(train)} train / {len(test)} test fields of '
          f'{data.shape[-1]}^3, std {scale:.4f}')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet3D(args.width, pad).to(dev)
    opt = torch.optim.AdamW(model.parameters(), 3e-4)
    xs = torch.from_numpy(train / scale)
    t0 = time.time()
    for it in range(1, args.iters + 1):
        z = xs[torch.randint(len(xs), (args.batch,))].to(dev)
        xi = torch.randn_like(z)
        t = torch.rand(args.batch, device=dev)
        xt = t.view(-1, 1, 1, 1, 1) * z + (1 - t.view(-1, 1, 1, 1, 1)) * xi
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == 'cuda'):
            loss = F.mse_loss(model(xt, t).float(), z - xi)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % max(args.iters // 5, 1) == 0:
            print(f'iter {it:5d}  CFM loss {loss.item():.4f}  '
                  f'{time.time() - t0:6.1f} s')
    model.eval()
    gen = []
    with torch.no_grad():
        for _ in range(math.ceil(args.samples / args.batch)):
            x = torch.randn(args.batch, *xs.shape[1:], device=dev)
            for i in range(args.euler):
                t = torch.full((args.batch,), i / args.euler, device=dev)
                with torch.autocast(dev, dtype=torch.float16,
                                    enabled=dev == 'cuda'):
                    x = x + model(x, t).float() / args.euler
            gen.append(x.cpu().numpy() * scale)
    gen = np.concatenate(gen)[:args.samples]
    n = data.shape[-1]
    top = n // 2 - 1
    ks = np.arange(1, top + 1)
    es, rs = spectra(gen), spectra(test)
    eg, er = es.mean(0), rs.mean(0)
    mn = np.bincount(shells(n), minlength=n)
    ratio = eg[ks] / er[ks]
    dlog = float(np.sqrt(np.sum(np.log(ratio)**2)))
    print(' kappa     M_n   E_ref        E_gen        gen/ref')
    for k in ks:
        print(f'{k:6d}  {mn[k]:6d}  {er[k]:.4e}  {eg[k]:.4e}  '
              f'{eg[k] / er[k]:7.3f}')
    print(f'TKE ref {er.sum():.4e}  gen {eg.sum():.4e}')
    print(f'L2 distance of log spectra over 1..{top}: {dlog:.3f}')
    m = len(rs) // 2
    report_departures('generated vs held-out', es,
                      rs[np.arange(len(es)) % len(rs)], top, mn)
    report_departures('held-out vs held-out', rs[:m], rs[m:2 * m], top, mn)
    for name, a in (('ref', test), ('gen', gen)):
        s, sk, fl = moments(a)
        print(f'{name}: std {np.round(s, 4)}  skew {np.round(sk, 3)}  '
              f'flatness {np.round(fl, 3)}  gradient skewness '
              f'{np.round(gradient_skewness(a), 3)}')


if __name__ == '__main__':
    main()
