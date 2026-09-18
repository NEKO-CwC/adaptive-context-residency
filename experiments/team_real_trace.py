"""Phase-0 headline experiment: five concurrent copies of one real Claude Code
session on one engine, replayed through every residency policy at two host-tier sizes.

Reproduce:  PYTHONPATH=src python3 experiments/team_real_trace.py
Logged output: experiments/logs/2026-09-18-team-real.log (the numbers quoted in docs/05).
"""
import sys, time

from dataclasses import replace
from pathlib import Path
from acr.config import default_config
from acr.replay import run_policy
from acr.trace import Turn, from_jsonl
real = from_jsonl(Path(__file__).resolve().parent.parent / 'trace' / 'ncu-cc-sample.jsonl')
base = default_config()
def team(k, hint_scale=1.0):
    out=[]
    for i in range(k):
        for t in real:
            out.append(replace(t, session_id=f"{t.session_id}-{i}", t=t.t*0.25 + i*240.0,
                               tool_eta_s=None if t.tool_eta_s is None else t.tool_eta_s*hint_scale))
    out.sort(key=lambda x: x.t)
    return out
NAMES=("lru","lfu","fixed_ttl","continuum_ttl","adaptive_value","oracle")
for k in (1, 3, 5):
  for gib in (40, 100):
    cfg=replace(base,tiers=tuple(replace(t,bytes_capacity=int(gib*1024**3)) if t.name=="ram" else t for t in base.tiers))
    turns=team(k)
    print(f"=== team x{k} on real CC trace, ram {gib} GiB @0.55 eff, {len(turns)} turns", flush=True)
    for n in NAMES:
        t0=time.time(); r=run_policy(cfg,turns,n)
        sup=r.per_role["supervisor"]
        print(f"  {n:15s} rec={r.recompute_tokens/1000:8.1f}k restore={r.restore_bytes/1024**3:7.1f}GiB wasted={r.objective():8.1f}s thrash={r.thrash:6d} ttft_p50={sup['ttft_p50']:5.2f} p95={sup['ttft_p95']:6.2f} [{time.time()-t0:.0f}s]", flush=True)
