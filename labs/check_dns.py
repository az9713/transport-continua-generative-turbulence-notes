"""Acceptance check of a DNS folder, from its 64^3 filtered fields.

Run: python labs/check_dns.py data/dns
Fails if the mean flow or the divergent energy is not roundoff, or if the
gradient flatness of any direction stays above mean + 3 sd of the first half
of the run for 5 snapshots in a row. Prints window means of skewness too.
"""
import glob
import sys
import numpy as np
n = 64
k1, kz = np.fft.fftfreq(n, 1 / n), np.fft.rfftfreq(n, 1 / n)
K = np.meshgrid(k1, k1, kz, indexing='ij')
k2 = K[0]**2 + K[1]**2 + K[2]**2
inv = 1 / np.where(k2 == 0, 1, k2)
w = np.where((kz == 0) | (kz == n // 2), 1.0, 2.0).reshape(1, 1, -1)
files = sorted(glob.glob(sys.argv[1] + '/filtered/*.npy'))
rows = []
for f in files:
    uh = np.fft.rfftn(np.load(f).astype(np.float64), axes=(1, 2, 3)) / n**3
    d = K[0] * uh[0] + K[1] * uh[1] + K[2] * uh[2]
    S, F = [], []
    for i in range(3):
        g = np.fft.irfftn(1j * K[i] * uh[i], s=(n,) * 3,
                          axes=(0, 1, 2)) * n**3
        v = (g**2).mean()
        S.append((g**3).mean() / v**1.5)
        F.append((g**4).mean() / v**2)
    rows.append((0.5 * (abs(uh[:, 0, 0, 0])**2).sum(),
                 (0.5 * abs(d)**2 * inv * w).sum(), *S, *F))
r = np.array(rows)
print(f'{len(files)} snapshots; max mean-flow energy {r[:, 0].max():.1e}; '
      f'max divergent energy {r[:, 1].max():.1e}')
for a in range(0, len(r), 20):
    b = r[a:a + 20]
    print(f'snaps {a:3d}-{a + len(b) - 1:3d}  skew',
          np.round(b[:, 2:5].mean(0), 2),
          ' flat', np.round(b[:, 5:8].mean(0), 2))
half = r[:len(r) // 2, 5:8]
limit = half.mean(0) + 3 * half.std(0)
print('flatness limit per direction', np.round(limit, 2))
ok = r[:, 0].max() < 1e-10 and r[:, 1].max() < 1e-9
for i in range(3):
    high = r[:, 5 + i] > limit[i]
    run = next((s for s in range(len(r) - 4) if high[s:s + 5].all()), None)
    print(f'direction {i}: first 5 snapshots in a row above limit: {run}')
    ok = ok and run is None
print('PASS' if ok else 'FAIL')
