"""Pool the per-step debug dumps of one run into distribution statistics.

The metric is a single number per step and its step-to-step spread is ~0.0015, so a 2-step A/B is
weakly powered. Pooling the |dlogp| distribution over every dumped step (~2k tokens each) gives a
much tighter comparison, and the tail counts are what actually carry the variance.
"""
import glob
import os
import sys

import torch


def pears(a, b):
    a = a.double(); b = b.double()
    a = a - a.mean(); b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm())


def run(dirname):
    rows = []
    for f in sorted(glob.glob(os.path.join(dirname, "batch_step*.pt")),
                    key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        rl = d["rollout_log_probs"].float()
        ol = d["old_log_probs"].float()
        m = d["response_mask"].bool()
        r, o = rl[m], ol[m]
        dd = o - r
        pa, pr = o.exp(), r.exp()
        rows.append(dict(f=os.path.basename(f), n=r.numel(), d=dd, pa=pa, pr=pr))
    if not rows:
        return None
    d = torch.cat([x["d"] for x in rows])
    pa = torch.cat([x["pa"] for x in rows])
    pr = torch.cat([x["pr"] for x in rows])
    per_step = [pears(x["pa"], x["pr"]).item() for x in rows]
    return dict(name=os.path.basename(dirname), n=len(rows), tokens=d.numel(),
                pooled_pearson=pears(pa, pr).item(),
                per_step=per_step,
                mean_step=sum(per_step) / len(per_step),
                med= d.abs().median().item(), std=d.std().item(),
                gt05=(d.abs() > 0.5).float().mean().item() * 100,
                gt10=(d.abs() > 1.0).float().mean().item() * 100,
                var_gt05=d[d.abs() > 0.5].pow(2).sum().item() / d.pow(2).sum().item() * 100,
                hi_p=pa[pa > 0.5],
                hi_rel=((pa[pa > 0.5] - pr[pa > 0.5]).abs() / pa[pa > 0.5]).median().item())


dirs = sys.argv[1:]
print(f"{'run':>14} {'steps':>5} {'tokens':>7} {'pooled':>8} {'mean_step':>9} "
      f"{'med|d|':>7} {'std':>6} {'>0.5%':>6} {'>1.0%':>6} {'var>0.5%':>8} {'hi-p rel':>8}")
for dd in dirs:
    s = run(dd)
    if s is None:
        continue
    print(f"{s['name']:>14} {s['n']:>5} {s['tokens']:>7} {s['pooled_pearson']:>8.5f} "
          f"{s['mean_step']:>9.5f} {s['med']:>7.4f} {s['std']:>6.4f} {s['gt05']:>6.2f} "
          f"{s['gt10']:>6.2f} {s['var_gt05']:>8.1f} {s['hi_rel']:>8.4f}")