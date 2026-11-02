"""Read the course DNS output: the range procedure of section 3.5 (ch. 3.8).

Reads only the published files data/dns/params.json, stats.csv, spectrum.csv
and sampling_spectra.csv (from labs/dns_sampling_spectra.py), not the
snapshots. Requires Python 3.13 and NumPy. Writes no files.
"""
import csv
import json
from pathlib import Path
import numpy as np

DNS = Path(__file__).resolve().parents[1] / "data" / "dns"
par = json.loads((DNS / "params.json").read_text())
NU, TOP = par["nu"], par["kmax"]

print("Part 1. Window means of stats.csv, t >= first snapshot")
rows = list(csv.DictReader(open(DNS / "stats.csv")))
st = {c: np.array([float(r[c]) for r in rows]) for c in rows[0]}
win = st["t"] >= par["spinup"] - 1e-9
for c in ("k", "eps", "urms", "L", "Re_lam", "eta", "kmax_eta", "skew"):
    x = st[c][win]
    print(f"  {c:8s} mean {x.mean():.4g}  range {x.min():.4g} .. "
          f"{x.max():.4g}")
print(f"  {win.sum()} logged times, N = {par['n']}, nu = {NU}, "
      f"retained |k_i| <= {TOP}")

print("Part 2. Shell sums, mean over the snapshots of spectrum.csv")
raw = np.loadtxt(DNS / "spectrum.csv", delimiter=",")
snaps, e_all = raw[:, 1], raw[:, 2:]
e = e_all.mean(0)
m = np.arange(len(e))
dis = 2 * NU * m**2 * e
k_n, eps_n = e.sum(), dis.sum()
eta = (NU**3 / eps_n) ** 0.25
print(f"  {len(snaps)} snapshots, t = {snaps[0]:.0f} .. {snaps[-1]:.0f}")
print(f"  k_N = {k_n:.4f}, eps_N = {eps_n:.5f}, eta = {eta:.5f}, "
      f"kmax*eta = {TOP * eta:.3f}; top occupied shell {m[e > 0].max()}")
print(f"  beyond shell {TOP}: share of k {e[TOP + 1:].sum() / k_n:.1e}, "
      f"of eps {dis[TOP + 1:].sum() / eps_n:.4f}")
q = np.arange(-TOP, TOP + 1)
kk = q[:, None, None]**2 + q[None, :, None]**2 + q[None, None, :]**2
nm = np.bincount(np.rint(np.sqrt(kk)).astype(int).ravel())[:TOP + 1]
vol = 4 * np.pi * (m[:TOP + 1]**2 + 1 / 12)
print("  m  N_m  N_m^(-1/2)  std/mean over snapshots")
for s in (1, 2, 3, 5, 9, 20, 40, 70):
    print(f"  {s:2d} {nm[s]:6d} {nm[s]**-0.5:8.3f}"
          f" {e_all[:, s].std(ddof=1) / e[s]:10.3f}")

print("Part 3. The range procedure on shells 1 ..", TOP)
mm = m[1:TOP + 1]
psi = mm**(5 / 3) * e[1:TOP + 1] / eps_n**(2 / 3)
fix = vol[1:] / nm[1:]
pcc = psi * fix
for name, y in (("raw", psi), ("count-corrected", pcc)):
    b = mm[y >= 0.95 * y.max()]
    print(f"  Psi {name}: max {y.max():.3f} at m = {mm[y.argmax()]};"
          f" within 5 %: m = {b[0]}..{b[-1]}, {len(b)} shell(s)")
print("  m:   " + " ".join(f"{s:5d}" for s in range(1, 13)))
print("  Psi: " + " ".join(f"{y:5.2f}" for y in psi[:12]))
print("  cc:  " + " ".join(f"{y:5.2f}" for y in pcc[:12]))
d = dis[1:TOP + 1]
cum = np.cumsum(dis) / eps_n
print(f"  peak of E at m = {mm[e[1:TOP + 1].argmax()]}; peak of D at m = "
      f"{mm[d.argmax()]}, kappa*eta = {mm[d.argmax()] * eta:.3f}; 90 % of "
      f"eps at m = {np.argmax(cum >= 0.9)}, {np.argmax(cum >= 0.9) * eta:.3f}")
b = mm[pcc >= 0.95 * pcc.max()]
for name, lo, hi in (("below band", 1, b[0] - 1), ("band", b[0], b[-1]),
                     ("above band", b[-1] + 1, TOP)):
    print(f"  {name:10s} m={lo:2d}..{hi:2d}: share of k "
          f"{e[lo:hi + 1].sum() / k_n:.3f}, of eps "
          f"{dis[lo:hi + 1].sum() / eps_n:.3f}")
ell = st["L"][win].mean()
print(f"  L_11 = {ell:.3f}, L_11/eta = {ell / eta:.0f}; kappa_EI = "
      f"{12 * np.pi / ell:.1f}, kappa_DI = {2 * np.pi / (60 * eta):.2f}")
slope = np.gradient(np.log(e[1:TOP + 1]), np.log(mm))
scc = np.gradient(np.log(e[1:TOP + 1] * fix), np.log(mm))
print(f"  local slope on m = 4..12: raw {slope[3:12].max():.2f} .. "
      f"{slope[3:12].min():.2f}; count-corrected {scc[3:12].max():.2f} .. "
      f"{scc[3:12].min():.2f}")
for a, b in ((3, 10), (4, 12), (4, 16), (5, 15), (8, 20)):
    fit = np.polyfit(np.log(m[a:b + 1]), np.log(e[a:b + 1]), 1)[0]
    print(f"  hand-picked fit m = {a}..{b}: slope {fit:.3f}")

print("Part 4. The two 64^3 samples, sampling_spectra.csv")
tab = {}
for r in csv.DictReader(open(DNS / "sampling_spectra.csv")):
    tab.setdefault(r["kind"], {})[int(r["kappa"])] = float(r["E"])
wt = np.array([0.5, 1, 1, 1, 0.5]) / 4
print("  kappa  parent E  rectangle/parent  Hann/parent")
hann_high = []
for kap in range(4, 125, 4):
    ref = (wt * e[kap - 2:kap + 3]).sum()
    if kap >= 24:
        hann_high.append(tab["crop_hann"][kap] / ref)
    if kap in (4, 8, 12, 16, 20, 24, 28, 36, 40, 64, 84, 124):
        print(f"  {kap:4d} {ref:9.2e} {tab['crop_rect'][kap] / ref:12.3f}"
              f" {tab['crop_hann'][kap] / ref:12.3f}")
hann_high.sort()
print(f"  Hann/parent for kappa >= 24: {hann_high[0]:.3f} .. "
      f"{hann_high[-2]:.3f}, and {hann_high[-1]:.3f} once")
k_rect = sum(tab["crop_rect"].values()) * 4
k_hann = sum(tab["crop_hann"].values()) * 4
h0 = tab["crop_hann"][0] * 4
print(f"  k: parent {k_n:.4f}, rectangle {k_rect:.4f} ({k_rect / k_n:.3f}),"
      f" Hann {k_hann:.4f} ({k_hann / k_n:.3f}), of which shell 0: {h0:.4f}")
e4 = e_all[:4].mean(0)
sh, no = tab["stride4_sharp"], tab["stride4_none"]
miss = max(abs(sh[s] / e4[s] - 1) for s in range(1, 32))
print(f"  first 4 snapshots: filtered/parent, max rel. diff {miss:.1e}")
print("  m  stride-4 no prefilter / filtered")
for s in (8, 16, 24, 28, 31):
    print(f"  {s:2d} {no[s] / sh[s]:8.4f}")
print(f"  k: parent {e4.sum():.4f}, stride-4 {sum(no.values()):.4f},"
      f" filtered {sum(sh.values()):.4f}")
d4 = (2 * NU * m**2 * e4).sum()
d_sh = sum(2 * NU * s**2 * x for s, x in sh.items())
print(f"  whole filtered field: {sum(sh.values()) / e4.sum():.4f} of k, "
      f"{d_sh / d4:.3f} of eps; cube corners reach kappa*eta = "
      f"{np.sqrt(3) * 31 * eta:.2f}")
print(f"  shells 1..31 of the parent hold {e[:32].sum() / k_n:.4f} of k, "
      f"{dis[:32].sum() / eps_n:.3f} of eps; kappa*eta at 31 = {31 * eta:.2f}")
print(f"  shells 1..84 hold {dis[:85].sum() / eps_n:.3f} of eps; "
      f"kappa*eta at 84 = {84 * eta:.2f}")
