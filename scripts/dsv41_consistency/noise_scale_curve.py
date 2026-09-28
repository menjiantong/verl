"""How does probs-pearson respond to a *uniform shrink* of the train/rollout logprob gap?

p_actor(a) = exp(rollout_logp + a*(actor_logp - rollout_logp));  a=1 is the real run.
This is a proxy for "the two stacks agree k times better": if some fix divides the
total mismatch by k, the metric should read about the a=1/k row.
"""
import torch

def load(p):
    d = torch.load(p, map_location='cpu', weights_only=False)
    return d['rollout_log_probs'].float(), d['old_log_probs'].float(), d['response_mask'].bool()

def pears(a, b):
    a = a.double(); b = b.double()
    a = a - a.mean(); b = b - b.mean()
    return (a*b).sum() / (a.norm()*b.norm())

print(f"{'alpha':>6} {'pearson(actor moved)':>21} {'pearson(rollout moved)':>23}")
for f in ['/tmp/dsv41_batch_real4_tis/batch_step0.pt',
          '/tmp/dsv41_batch_real4_tis/batch_step1.pt']:
    rl, ol, m = load(f)
    d = ol - rl
    print(f"--- {f.split('/')[-1]} ---")
    for a in [1.0, 0.75, 0.6, 0.5, 0.4, 0.33, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0]:
        pa = (rl + a*d)[m].exp()
        pr = rl[m].exp()
        p2 = (ol - a*d)[m].exp()
        print(f"{a:6.2f} {pears(pa,pr).item():21.5f} {pears(p2,ol[m].exp()).item():23.5f}")

print("\n=== variance decomposition (step0) ===")
rl, ol, m = load('/tmp/dsv41_batch_real4_tis/batch_step0.pt')
pa, pr = ol[m].exp(), rl[m].exp()
dp = pa - pr
for name, x in [('p_actor', pa), ('p_rollout', pr), ('delta_p', dp)]:
    print(f"{name:10s} mean={x.mean():.5f} std={x.std():.5f} var={x.var():.3e} "
          f"rms={x.pow(2).mean().sqrt():.5f} skew={( (x-x.mean())**3 ).mean()/x.std()**3:.3f}")
print(f"E[p^2] / E[(diff)^2]: var(p)={pa.var():.4e} var(dp)={dp.var():.4e} "
      f"ratio={dp.var()/pa.var():.5f} -> 1/pearson_approx={1/(1-0.5*dp.var()/pa.var()):.5f}")
# top-k concentration of p
s, _ = pa.sort(descending=True)
print(f"top 1% tokens hold {s[:len(s)//100].sum()/s.sum()*100:.1f}% of sum(p); "
      f"top 3% -> {s[:3*len(s)//100].sum()/s.sum()*100:.1f}%; "
      f"top 8% -> {s[:8*len(s)//100].sum()/s.sum()*100:.1f}%")
d2 = dp.pow(2)
s2, _ = d2.sort(descending=True)
print(f"top 1% tokens hold {s2[:len(s2)//100].sum()/s2.sum()*100:.1f}% of sum(dp^2); "
      f"top 3% -> {s2[:3*len(s2)//100].sum()/s2.sum()*100:.1f}%; "
      f"top 8% -> {s2[:8*len(s2)//100].sum()/s2.sum()*100:.1f}%")
# correlation restricted to high-p tokens
for thr in [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
    sel = pa > thr
    if sel.sum() < 10: continue
    print(f"pearson on p_actor>{thr:.2f}: n={sel.sum():5d} pearson={pears(pa[sel],pr[sel]).item():.5f} "
          f"logspace={pears(ol[m][sel],rl[m][sel]).item():.5f}")