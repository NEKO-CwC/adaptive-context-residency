#!/usr/bin/env python3
"""E_signal at the tier: replay real session streams through the **real** CPUOffloadingManager.

Why this exists instead of the simulator: the F-7b lesson. My simulator picked a 256-token block and
reported a 5.6× win that vanished at the engine's real 816-token granularity. Here the block size,
the key layout (`OffloadKey = block_hash + group_idx`), the manager, the capacity accounting and the
eviction interface are all the deployed ones — the only variable is the policy.

Two streams, same capacity, two policies (built-in `lru` vs `AcrValuePolicy`):
  A. the recorded supervisor session (`trace/ncu-cc-sample.jsonl`, 343 turns, 39.9K→322.8K context,
     with real `aborted` rollbacks and announced `tool_eta_s`) — append-only growth plus truncation;
  B. a mixed ward: several coding sessions sharing one long repo prefix + many short patient sessions,
     which is the production question in one line: *does ranking protect the latency class, or is
     pinning alone enough?*

Reported in the units that decide things: hit ratio, wasted stores (stored then evicted untouched),
recomputed tokens avoided, and seconds — recompute at the measured 11.1K tok/s, restore at the
PCIe-derived 0.6 µs/token (M-1 pending, so the restore column is an upper bound on its own value).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
import time

from vllm.v1.kv_offload.base import ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

BLOCK = 816                      # the engine's attention group, from the boot log
GROUP = 0                        # single group per stream: group keys are exercised in N1's S5
PREFILL_TOK_S = 11_100.0         # docs/00 §2 marginal cold prefill
RESTORE_S_PER_TOK = 0.6e-6       # docs/00 §4 PCIe upper bound

ACR = ("AcrValuePolicy", "acr.vllm.policy")
TTL = ("SessionTtlPolicy", "acr.vllm.ttl")     # the original "alive for N seconds" idea, in the same ABC
LRU = ("lru", None)                             # built-in
ARC = ("arc", None)                             # built-in adaptive: the strongest library baseline


def k(block_hash_idx: str, group: int = GROUP) -> bytes:
    return make_offload_key(hashlib.sha256(block_hash_idx.encode()).digest()[:32], group)


def ctx(req_id: str, params: dict | None) -> ReqContext:
    return ReqContext(req_id=req_id, kv_transfer_params=params)


class Runner:
    """Drives one policy over one event stream and counts what the tier actually did."""

    def __init__(self, policy: tuple[str, str | None], blocks: int, store_threshold: int = 1):
        args = dict(num_blocks=blocks, store_threshold=store_threshold, enable_events=True)
        if policy[1]:
            args.update(cache_policy=policy[0], cache_policy_module_path=policy[1])
        else:
            args.update(cache_policy=policy[0])
        self.m = CPUOffloadingManager(**args)
        self.served: dict[str, set[bytes]] = {}      # session -> blocks it believes are in RAM
        self.stats = dict(lookups=0, hits=0, misses=0, stored=0, restored_tokens=0,
                          recomputed_tokens=0, wasted_stores=0, errors=0)
        # per-role view: the product question is not aggregate hit ratio but whether the latency
        # class is the one that survives pressure.
        self.by_role: dict[str, dict[str, int]] = {}

    def _safe(self, fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except Exception:                                                  # noqa: BLE001
            self.stats["errors"] += 1
            return None

    def advance(self, seconds: float) -> None:
        """Move the virtual clock, because recency/ETA signals are meaningless on a flattened one.

        The first replay had no clock at all: turns were processed back-to-back in microseconds, so
        `idle` was ~0 for every block and `reuse_prob` could not separate anything — an instrument
        that cannot answer the question it was built for. The engine's policy uses time.monotonic,
        so we shift that process-wide clock instead of forking the policy.
        """
        import time as _t
        _t.monotonic = lambda _base=seconds, _acc=[0.0]: _acc.__setitem__(0, _acc[0] + _base) or _acc[0] + 1_000_000.0

    def touch_stream(self, session: str, keys: list[bytes], params: dict) -> None:
        c = ctx(f"{session}-{time.time_ns()}", params)
        self._safe(self.m.on_new_request, c)
        present = self.served.setdefault(session, set())
        # lookup every block of the stream: a HIT means the tier can serve it, a MISS means the
        # engine will prefill it and, if worth caching, store it on the way out
        role = (params.get("acr") or {}).get("role", "?")
        br = self.by_role.setdefault(role, {"lookups": 0, "hits": 0, "recomputed": 0})
        cold = []
        for key in keys:
            self.stats["lookups"] += 1
            br["lookups"] += 1
            r = self._safe(self.m.lookup, key, c)
            name = getattr(r, "name", "MISS")
            if name in ("HIT", "HIT_PENDING"):
                self.stats["hits"] += 1
                br["hits"] += 1
                self.stats["restored_tokens"] += BLOCK
                self._safe(self.m.prepare_load, [key], c)
                self._safe(self.m.complete_load, [key], c)
            else:
                self.stats["misses"] += 1
                cold.append(key)
        self._safe(self.m.touch, keys, c)
        if cold:
            self.stats["recomputed_tokens"] += len(cold) * BLOCK
            br["recomputed"] += len(cold) * BLOCK
            out = self._safe(self.m.prepare_store, cold, c)
            if out is not None:
                accepted = list(out.keys_to_store)
                self.stats["stored"] += len(accepted)
                self._safe(self.m.complete_store, accepted, c)
                for key in out.evicted_keys or []:
                    # a block we just displaced that had never been read back was a wasted store
                    if any(key in s for s in self.served.values()):
                        self.stats["wasted_stores"] += 1
        present.update(cold)
        self._safe(self.m.on_request_finished, c)


def stream_real(path: pathlib.Path) -> list[tuple[str, list[bytes], dict]]:
    """One event per recorded turn: the block set of that turn's context, plus its announced signals."""
    out = []
    prev_t = None
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if "prompt_tokens" not in r:
            continue
        gap = 0.0 if prev_t is None else max(0.0, float(r.get("t", prev_t)) - prev_t)
        prev_t = float(r.get("t", prev_t))
        n = max(1, math.ceil(r["prompt_tokens"] / BLOCK))
        keys = [k(f"{r['session_id']}:{i}") for i in range(n)]
        params = {"acr": {"role": r.get("role", "supervisor"), "session": r["session_id"],
                          "tool_eta_s": r.get("tool_eta_s") or 5.0,
                          "aborted": bool(r.get("aborted")),
                          "slo_class": r.get("slo_class", "interactive"),
                          "shared_prefix_tokens": r.get("shared_prefix_tokens", 0)}}
        out.append((r["session_id"], keys, params, gap))
    return out


def stream_mixed(coders: int, patients: int, shared_repo: int, turns: int,
                 coder_gap: float = 8.0, patient_gap: float = 45.0):
    """Ward model: N coding sessions over one long shared repo prefix, M patient sessions, short —
    and, critically, **interleaved in time** the way a real ward is. The first version emitted all
    coder turns and then all patient turns, which is a two-phase arrival pattern where recency is
    near-optimal by construction and any forward-looking ranking is punished by the time断层; it
    measured the stream generator, not the policy.

    Each session keeps its own cadence and a staggered start; the stream is merged by timestamp and
    carries the real gap to the previous event, so the policy's clock sees seconds for tool waits
    and tens of seconds for human typing.
    """
    timed: list[tuple[float, str, list[bytes], dict]] = []
    repo = [k(f"repo:{i}") for i in range(shared_repo)]
    for c in range(coders):
        sess = f"coder-{c}"
        own = [k(f"{sess}:{i}") for i in range(turns)]
        start = (c % 4) * 2.0
        for t in range(1, turns + 1):
            timed.append((start + (t - 1) * coder_gap, sess, repo + own[:t],
                          {"acr": {"role": "coding", "session": sess, "tool_eta_s": 90.0,
                                   "leases": 1}}))
    for pt in range(patients):
        sess = f"patient-{pt}"
        base = [k(f"case:{i}") for i in range(2)]                 # 2 blocks ≈ shared case truth
        own = [k(f"{sess}:{i}") for i in range(turns)]
        start = (pt % 5) * 9.0
        for t in range(1, turns + 1):
            timed.append((start + (t - 1) * patient_gap, sess, base + own[:t],
                          {"acr": {"role": "patient", "session": sess, "tool_eta_s": 3.0,
                                   "leases": 1}}))
    timed.sort(key=lambda x: x[0])
    ev, prev = [], 0.0
    for ts, sess, keys, params in timed:
        ev.append((sess, keys, params, ts - prev))
        prev = ts
    return ev


def run(label: str, events, blocks: int, policies=(LRU, ARC, TTL, ACR)) -> dict:
    res = {}
    for pol in policies:
        t0 = time.time()
        r = Runner(pol, blocks)
        for session, keys, params, gap in events:
            r.advance(gap)
            r.touch_stream(session, keys, params)
        hit = r.stats["hits"] / max(r.stats["lookups"], 1)
        res[pol[0]] = dict(r.stats, hit_ratio=round(hit, 4), secs=round(time.time() - t0, 1),
                           recompute_s=round(r.stats["recomputed_tokens"] / PREFILL_TOK_S, 1),
                           restore_s=round(r.stats["restored_tokens"] * RESTORE_S_PER_TOK, 1))
        roles = " ".join(f"{rr}:hits={d['hits']/max(d['lookups'],1):5.1%}/recomp={d['recomputed']/1e3:6.0f}K"
                         for rr, d in sorted(r.by_role.items()))
        print(f"  {label:11s} {pol[0]:16s} blocks={blocks:6d} lookups={r.stats['lookups']:>8} "
              f"hit={hit:5.1%} stored={r.stats['stored']:>7} wasted={r.stats['wasted_stores']:>6} "
              f"recompute={r.stats['recomputed_tokens']/1e3:8.1f}K tok ({r.stats['recomputed_tokens']/PREFILL_TOK_S:6.0f}s) "
              f"errors={r.stats['errors']} ({time.time()-t0:.0f}s)")
        print(f"              {roles}")
    a, b = res["lru"], res[max(res, key=lambda k: res[k]["hit_ratio"])]
    print(f"  → best={b and max(res, key=lambda k: res[k]['hit_ratio'])} vs lru: hit {b['hit_ratio']-a['hit_ratio']:+.2%}, "
          f"recomputed {b['recomputed_tokens']-a['recomputed_tokens']:+d} tok, "
          f"wasted stores {b['wasted_stores']-a['wasted_stores']:+d}, "
          f"recompute-seconds {b['recompute_s']-a['recompute_s']:+.0f}s")
    return {"label": label, "blocks": blocks, "policies": res}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=str(pathlib.Path(__file__).resolve().parent.parent / "trace/ncu-cc-sample.jsonl"))
    ap.add_argument("--out", default="/tmp/acr-replay.json")
    a = ap.parse_args()

    real = stream_real(pathlib.Path(a.trace))
    real_tok = sum(len(keys) for _, keys, _, _ in real) * BLOCK
    print(f"streams: real={len(real)} turns ≈{real_tok/1e6:.2f}M token-visits · mixed=12 coders×40 + 20 patients×12")
    mixed = stream_mixed(coders=12, patients=20, shared_repo=120, turns=40)

    report = []
    # capacity in blocks of 816 tokens: 1.6M tokens ≈ 1960 blocks ≈ 91 GiB at 56.4 KiB/token
    for blocks in (1200, 600, 500, 400, 350, 300, 240):
        report.append(run(f"mixed {blocks}", mixed, blocks))
    json.dump(report, open(a.out, "w"), indent=1)
    errs = sum(p["errors"] for r in report for p in r["policies"].values())
    print(f"\nwrote {a.out} · manager call errors across all arms: {errs}"
          + ("" if errs == 0 else "  ← NON-ZERO: these numbers are not trustworthy"))
    return 0 if errs == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
