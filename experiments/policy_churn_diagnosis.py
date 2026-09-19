#!/usr/bin/env python3
"""Why does AcrValuePolicy lose to the built-in LRU under pressure? Three arms, one capacity.

The replay (experiments/offload_replay_real.py, 2026-09-20) measured, at 300 blocks:
  lru  hit 98.4%  stored 1402  wasted 1102  recompute 1.14M tok
  acr  hit 83.6%  stored 14044 wasted 13744 recompute 11.46M tok
so our policy evicts blocks that the next turn of the same session re-reads in full, then re-stores
them every turn, and every re-store displaces something else. Candidates, tested here by removing
one signal at a time rather than by reading the code and hoping:
  * the announced tool ETA dominates value → an append-only coder session with a 90 s ETA is ranked
    below a 3 s patient, even though its next turn is certain and huge;
  * the role/pin asymmetry alone is enough to starve coders.
"""
import copy
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from offload_replay_real import ACR, LRU, Runner, stream_mixed  # noqa: E402

EVENTS = stream_mixed(coders=12, patients=20, shared_repo=120, turns=40)
BLOCKS = 300


def variant(kind: str):
    out = []
    for sess, keys, p in EVENTS:
        q = copy.deepcopy(p)
        if kind == "uniform_eta":
            q["acr"]["tool_eta_s"] = 30.0
        elif kind == "role_blind":
            q["acr"]["role"] = "coding"
        elif kind == "no_leases":
            q["acr"].pop("leases", None)
        out.append((sess, keys, q))
    return out


def run(label, stream, pol):
    r = Runner(pol, BLOCKS)
    for sess, keys, p in stream:
        r.touch_stream(sess, keys, p)
    s = r.stats
    print(f"{label:18s} hit={s['hits']/max(s['lookups'],1):6.2%} stored={s['stored']:>6} "
          f"wasted={s['wasted_stores']:>6} recompute={s['recomputed_tokens']/1e3:>8.1f}K tok errs={s['errors']}")
    return s


print(f"capacity {BLOCKS} blocks × 816 tokens = {BLOCKS*816/1e3:.0f}K tokens of RAM")
base_lru = run("lru baseline", EVENTS, LRU)
acr = run("acr as-implemented", EVENTS, ACR)
v1 = run("acr uniform eta", variant("uniform_eta"), ACR)
v2 = run("acr role-blind", variant("role_blind"), ACR)
v3 = run("acr no leases", variant("no_leases"), ACR)
for name, s in (("uniform-eta", v1), ("role-blind", v2), ("no-leases", v3)):
    print(f"  removing {name:11s} recovers {(acr['recomputed_tokens']-s['recomputed_tokens'])/1e3:>8.1f}K "
          f"of the {(acr['recomputed_tokens']-base_lru['recomputed_tokens'])/1e3:.1f}K tok gap "
          f"({100*(acr['recomputed_tokens']-s['recomputed_tokens'])/max(acr['recomputed_tokens']-base_lru['recomputed_tokens'],1):5.1f}%)")
