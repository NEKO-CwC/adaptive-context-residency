#!/usr/bin/env python3
"""Is our policy's ranking path even reachable under the manager's real call order?

The replay showed AcrValuePolicy responding to *nothing* — uniform ETA, role-blind and lease-free
streams produced byte-identical results. Either the signal is irrelevant at this capacity, or the
ranking code never runs. This counts the calls to find out which, because the difference decides
whether the fix is a tuning change or a wiring change.
"""
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from offload_replay_real import ACR, Runner, stream_mixed  # noqa: E402

import acr.vllm.policy as P  # noqa: E402

C = {"touch": 0, "known": 0, "insert": 0, "remove": 0, "evict_calls": 0, "evict_returned": 0,
     "get": 0}
orig = {n: getattr(P.AcrValuePolicy, n) for n in ("touch", "insert", "remove", "evict", "get")}


def touch(self, keys, rc):
    ks = list(keys)
    C["touch"] += len(ks)
    C["known"] += sum(1 for kk in ks if kk in self.meta)
    return orig["touch"](self, ks, rc)


def insert(self, key, block):
    C["insert"] += 1
    return orig["insert"](self, key, block)


def remove(self, key):
    C["remove"] += 1
    return orig["remove"](self, key)


def evict(self, n, prot):
    out = orig["evict"](self, n, prot)
    C["evict_calls"] += 1
    C["evict_returned"] += 0 if out is None else len(out)
    return out


def get(self, key):
    C["get"] += 1
    return orig["get"](self, key)


P.AcrValuePolicy.touch, P.AcrValuePolicy.insert, P.AcrValuePolicy.remove = touch, insert, remove
P.AcrValuePolicy.evict, P.AcrValuePolicy.get = evict, get

r = Runner(ACR, 300)
for sess, keys, params in stream_mixed(coders=6, patients=10, shared_repo=120, turns=25):
    r.touch_stream(sess, keys, params)
s = r.stats
print(f"stream: lookups={s['lookups']} hits={s['hits']} stored={s['stored']} "
      f"wasted={s['wasted_stores']} recompute={s['recomputed_tokens']} errors={s['errors']}")
pct = 100 * C["known"] / max(C["touch"], 1)
print(f"policy calls: touch_keys={C['touch']} of_which_in_meta={C['known']} ({pct:.1f}%) "
      f"insert={C['insert']} remove={C['remove']} get={C['get']} "
      f"evict_calls={C['evict_calls']} evict_returned={C['evict_returned']}")
print(f"meta size at end: {len(r.m.__dict__.get('_policy', getattr(r.m, '_policy', None)).meta) if hasattr(r.m, '_policy') else 'no _policy attr'}")
print("VERDICT:", "ranking path is DEAD (touch never finds its own metadata)" if pct < 5 else
      "ranking path runs; the signal genuinely does not separate these streams" if pct > 80 else
      f"partially wired: {pct:.0f}% of touches are on keys we know")
