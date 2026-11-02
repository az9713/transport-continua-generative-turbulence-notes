"""Lab 1 (week 7): conditional flow matching on a 2D toy distribution.

Course convention: X_t = t z + (1 - t) xi, t = 0 noise, t = 1 data.
The conditional velocity is u_t(x | z) = (z - x) / (1 - t), and along the
path it equals z - xi, which is the regression target.

The lab trains a small MLP, samples with explicit Euler at several step
counts, and measures (a) the endpoint error against a fine-step reference
of the same learned ODE and (b) a sliced Wasserstein distance to the data.
Run:  python labs/lab1_flow_matching_2d.py [--steps 3000] [--seed 0]
"""
import argparse
import math
import torch
import torch.nn as nn


def ring_of_gaussians(n, k=8, radius=4.0, std=0.35, gen=None):
    """Mixture of k isotropic Gaussians on a circle (the data law)."""
    j = torch.randint(0, k, (n,), generator=gen)
    ang = 2 * math.pi * j / k
    centre = radius * torch.stack([torch.cos(ang), torch.sin(ang)], 1)
    return centre + std * torch.randn(n, 2, generator=gen)


class Velocity(nn.Module):
    """u_theta(x, t): an MLP on (x, t) with sinusoidal time features."""

    def __init__(self, width=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 8, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 2))

    def forward(self, x, t):
        f = torch.arange(1, 5, device=x.device) * math.pi * t[:, None]
        return self.net(torch.cat([x, torch.sin(f), torch.cos(f)], 1))


def euler(model, x, steps):
    """Integrate dx/dt = u_theta(x, t) from t = 0 to t = 1."""
    h = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), i * h, device=x.device)
        x = x + h * model(x, t)
    return x


def sliced_w2(a, b, n_proj=256, gen=None):
    """Sliced 2-Wasserstein distance between equal-size samples."""
    th = torch.rand(n_proj, generator=gen) * math.pi
    d = torch.stack([torch.cos(th), torch.sin(th)], 1).to(a.device)
    pa = torch.sort(a @ d.T, 0).values
    pb = torch.sort(b @ d.T, 0).values
    return ((pa - pb) ** 2).mean().sqrt().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Velocity().to(dev)
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    for it in range(1, args.steps + 1):
        z = ring_of_gaussians(512, gen=gen).to(dev)
        xi = torch.randn(512, 2, generator=gen).to(dev)
        t = torch.rand(512, generator=gen).to(dev)
        xt = t[:, None] * z + (1 - t[:, None]) * xi
        loss = ((model(xt, t) - (z - xi)) ** 2).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % (args.steps // 5) == 0:
            print(f'iter {it:5d}  CFM loss {loss.item():.4f}')
    model.eval()
    with torch.no_grad():
        x0 = torch.randn(4000, 2, generator=gen).to(dev)
        ref = euler(model, x0, 1024)
        data = ring_of_gaussians(4000, gen=gen).to(dev)
        floor = sum(sliced_w2(ring_of_gaussians(4000, gen=gen).to(dev),
                              data, gen=gen) for _ in range(5)) / 5
        print(f'sliced W2, data vs fresh data, mean of 5: {floor:.4f}')
        print(' steps   h        endpoint err   ratio   sliced W2')
        prev = None
        for n in (4, 8, 16, 32, 64, 128):
            x1 = euler(model, x0, n)
            err = (x1 - ref).norm(dim=1).mean().item()
            ratio = '' if prev is None else f'{prev / err:5.2f}'
            print(f'{n:6d}  {1 / n:.5f}  {err:12.5f}   {ratio:>5}'
                  f'   {sliced_w2(x1, data, gen=gen):.4f}')
            prev = err


if __name__ == '__main__':
    main()
