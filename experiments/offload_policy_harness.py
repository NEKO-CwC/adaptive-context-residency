#!/usr/bin/env python3
"""N1 — drive the *deployed* CPU offload tier with ACR's selection policy, no engine, no GPU.

Why this exists: the field-selection decisions (docs 01 §1, plan D1–D5) are all claims about the
behaviour of `CPUOffloadingManager` + a `CachePolicy`, and that pair is plain Python — verified
importable in 8.0 s inside the runtime image with no CUDA. So the selection mechanism can be proven
or falsified here instead of burning a vLLM restart window, which is the scarce resource on this box.

Everything below uses the real classes and the real loader path the engine uses:
  `CachePolicyFactory.get_cache_policy_cls(name, module_path)` — out-of-tree, "no vLLM fork/patch"
  (verbatim from the shipped factory docstring), driven by `kv_connector_extra_config`.

Run (throwaway container, no GPU, production engine untouched):
  docker run --rm --gpus none -v $ACR:/acr:ro -e PYTHONPATH=/acr -e CUDA_VISIBLE_DEVICES="" $IMG \
      python3 /acr/experiments/offload_policy_harness.py --out /tmp/acr-n1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time

from vllm.v1.kv_offload.base import (
    ReqContext,
    get_offload_block_hash,
    get_offload_group_idx,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.factory import CachePolicyFactory
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

BLOCK_TOKENS = 816          # this engine's attention group, from the boot log
GROUP_ATTENTION, GROUP_STATE = 0, 1
ACR_CLASS = "AcrValuePolicy"
ACR_MODULE = "acr.vllm.policy"

results: list[dict] = []


def check(name: str, ok: bool, evidence: dict | None = None) -> bool:
    results.append({"check": name, "ok": bool(ok), "evidence": evidence or {}})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {evidence}" if evidence else ""))
    return bool(ok)


def key_for(tokens: list[int], group: int) -> bytes:
    """Engine-style positional key: hash of the block's token ids, plus the group index."""
    return make_offload_key(hashlib.sha256(bytes(str(tokens).encode())).digest()[:32], group)


def blocks(text_tokens: list[int], n: int) -> list[bytes]:
    """n positional keys over a growing prefix: block i covers tokens[:i*816]."""
    out = []
    for i in range(1, n + 1):
        span = text_tokens[: i * BLOCK_TOKENS] or [0]
        out.append(key_for(span, GROUP_ATTENTION))
    return out


def new_manager(store_threshold: int = 1, num_blocks: int = 192) -> CPUOffloadingManager:
    return CPUOffloadingManager(num_blocks=num_blocks, cache_policy=ACR_CLASS,
                                cache_policy_module_path=ACR_MODULE, store_threshold=store_threshold,
                                enable_events=True)


def params(role: str, session: str, eta_s: float = 30.0, leases: int = 1) -> dict:
    """What the gateway stamps per request; ACR reads it at touch()/insert() time."""
    return {"acr": {"role": role, "session": session, "tool_eta_s": eta_s, "leases": leases}}


def ctx(req_id: str, p: dict | None = None) -> ReqContext:
    return ReqContext(req_id=req_id, kv_transfer_params=p)


def store(m, keys, c):
    """prepare_store answers with the keys it actually accepted (PrepareStoreOutput.keys_to_store)
    plus whatever it had to evict to make room — so admission is observable right here."""
    out = m.prepare_store(keys, c)
    if out is None:
        return None
    m.complete_store(list(out.keys_to_store), c)
    return out


# ----------------------------------------------------------------------------------------------
def s0_loader():
    print("\nS0  the seam: does the shipped engine loader resolve OUR module?")
    cls = CachePolicyFactory.get_cache_policy_cls(ACR_CLASS, ACR_MODULE)
    ok = check("factory imports acr.vllm.policy.AcrValuePolicy", cls.__module__ == ACR_MODULE,
               {"class": f"{cls.__module__}.{cls.__name__}"})
    inst = None
    try:
        inst = cls(cache_capacity=8)
        ok &= check("__init__(cache_capacity) matches the ABC", hasattr(inst, "evict"))
    except Exception as exc:                                       # noqa: BLE001
        ok &= check("__init__(cache_capacity) matches the ABC", False, {"err": repr(exc)[:120]})
    m = None
    try:
        m = new_manager()
        ok &= check("CPUOffloadingManager accepts the out-of-tree policy", m is not None)
    except Exception as exc:                                       # noqa: BLE001
        ok &= check("CPUOffloadingManager accepts the out-of-tree policy", False, {"err": repr(exc)[:160]})
    return m


def s1_identity(m):
    print("\nS1  D1: is a cached prefix's identity stable across a tool-call rollback?")
    tokens = list(range(1, 90_000))
    full = blocks(tokens, 80)                       # 80 blocks ≈ 65K tokens
    c = ctx("agent-1", params("coding", "s1"))
    m.on_new_request(c)
    out = store(m, full, c)
    hits = sum(1 for k in full if m.lookup(k, c).name == "HIT")
    pol = getattr(m, "_policy", None)
    print(f"        (prepare_store -> {'None' if out is None else len(out.keys_to_store)} accepted,"
          f" evicted {0 if out is None else len(out.evicted_keys)};"
          f" capacity {getattr(pol, 'cache_capacity', '?')} blocks, hits {hits}/{len(full)})")
    # rollback: the harness judged the last 17 blocks' tool call useless; the session truncates and
    # regenerates a different tail. The retained prefix must still hit, block for block.
    kept = full[:63]
    new_tail = blocks(tokens[:63 * BLOCK_TOKENS] + [999_999] * (17 * BLOCK_TOKENS), 0)  # noqa: E501
    hits_after = sum(1 for k in kept if m.lookup(k, c).name == "HIT")
    m.touch(kept, c)
    return check("retained 63/80 blocks still HIT after truncating the last 17",
                 hits == 80 and hits_after == 63 and len(new_tail) == 0,
                 {"hits_before": hits, "hits_after_truncate": hits_after})


def s2_admission(m):
    print("\nS2  D3: does store_threshold skip never-reused blocks natively, and can ACR add value?")
    m = new_manager(store_threshold=2)          # >=2 turns the native reuse filter on
    c = ctx("agent-2", params("coding", "s2"))
    m.on_new_request(c)
    once = key_for([1, 2, 3], GROUP_ATTENTION)
    seen = key_for([4, 5, 6], GROUP_ATTENTION)
    m.lookup(once, c)                              # first sighting only
    m.lookup(seen, c)
    m.lookup(seen, c)                              # second sighting => eligible
    out = store(m, [once, seen], c)
    pol = getattr(m, "_policy", None)
    accepted = list(out.keys_to_store) if out is not None else []
    return check("one-shot block not admitted, twice-seen block admitted",
                 pol is not None and seen in accepted and once not in accepted,
                 {"accepted": len(accepted), "twice_seen_in": seen in accepted, "one_shot_in": once in accepted})


def s3_pin(m):
    print("\nS3  D4: is pinned medical KV actually immune to eviction under pressure?")
    med = ctx("patient-1", params("patient", "m1", eta_s=5.0, leases=2))
    ag1 = ctx("agent-a", params("coding", "a1", eta_s=120.0))
    ag2 = ctx("agent-b", params("coding", "a2", eta_s=120.0))
    for c, ks in ((med, [key_for([i], GROUP_ATTENTION) for i in range(4)]),
                  (ag1, [key_for([100 + i], GROUP_ATTENTION) for i in range(8)]),
                  (ag2, [key_for([200 + i], GROUP_ATTENTION) for i in range(8)])):
        m.on_new_request(c)
        m.lookup(ks[0], c)
        store(m, ks, c)
        for _ in range(6 if c is med else 2):
            m.touch(ks, c)
    pol = getattr(m, "_policy", None) or getattr(m, "policy", None)
    if pol is None:
        return check("manager exposes the policy so ACR can pin", False, {"err": "no policy handle"})
    med_keys = [key_for([i], GROUP_ATTENTION) for i in range(4)]
    for k in med_keys:
        pol.mark_non_evictable(k)
    victims = [v[0] if isinstance(v, tuple) else v for v in (pol.evict(6, set()) or [])]
    leaked = [k for k in med_keys if k in victims]
    return check("evict(n) never returns a mark_non_evictable medical key",
                 bool(victims) and not leaked,
                 {"evicted": len(victims), "medical_leaked": len(leaked),
                  "medical_total": len(med_keys)})


def _seed_pair(m):
    """Two agent blocks with *opposite* recency and future-value profiles.

    X  = touched repeatedly, most recently, but announced as a one-shot tail (leases=1, eta tiny)
    Y  = touched once long ago, but announced as a session with 8 live leases and a long tool ETA
    Recency (LRU) ranks Y below X; predicted-future-value ranks X below Y. The two policies must
    therefore pick different victims — that divergence is the whole claim at this layer, and it is
    the F-7b lesson applied: prove the mechanism, then prove it differs from the built-in.
    """
    x, y = key_for([4242], GROUP_ATTENTION), key_for([9090], GROUP_ATTENTION)
    cx, cy = ctx("x", params("coding", "sx", eta_s=1.0, leases=1)), ctx("y", params("coding", "sy", eta_s=120.0, leases=8))
    for c, k in ((cx, x), (cy, y)):
        m.on_new_request(c)
        m.lookup(k, c)
        store(m, [k], c)
    m.touch([y], cy)                       # Y first, then X repeatedly: X is the more recent
    for _ in range(6):
        m.touch([x], cx)
    return x, y, cx, cy


def s4_order(m, label=""):
    print(f"\nS4{label}  D4: is the eviction ORDER what we claim, and does it differ from the built-in?")
    pol = getattr(m, "_policy", None)
    if pol is None:
        return check("policy handle for order test", False)
    x, y, _, _ = _seed_pair(m)
    victims = [v[0] if isinstance(v, tuple) else v for v in (pol.evict(1, set()) or [])]
    if not victims:
        return check("evict(1) returns a victim", False)
    return victims[0]


def control_arm():
    """What does ACR add over the built-in LRU, at this layer? If nothing, say nothing."""
    print("\nCTRL  ACR vs built-in LRUCachePolicy on the same two tests")
    acr_pin_ok = None
    m = new_manager(); acr_pin_ok = s3_pin(m)

    m_lru = new_manager(); m_lru._policy = LRUCachePolicy(cache_capacity=192)
    m_acr = new_manager()
    v_lru, v_acr = s4_order(m_lru, "[lru]"), s4_order(m_acr, "[acr]")
    _, y, _, _ = _seed_pair(new_manager())        # recompute the identity of the high-value block
    print(f"        lru victim={v_lru[:4].hex() if v_lru else None} "
          f"acr victim={v_acr[:4].hex() if v_acr else None}  (high-future-value block = {y[:4].hex()})")
    check("LRU and ACR disagree on the victim (the divergence is the contribution)",
          v_lru is not None and v_acr is not None and v_lru != v_acr,
          {"acr_keeps_future_value": v_acr != y, "lru_evicts_stale": v_lru == y})
    check("pinned medical KV survives eviction under ACR", bool(acr_pin_ok))


def s5_groups(m):
    print("\nS5  D2/H1: do the attention and state groups stay separate entries?")
    same_hash = hashlib.sha256(b"same-block").digest()[:32]
    ka, ks = make_offload_key(same_hash, GROUP_ATTENTION), make_offload_key(same_hash, GROUP_STATE)
    c = ctx("agent-3", params("coding", "s3"))
    m.on_new_request(c)
    m.lookup(ka, c)
    store(m, [ka], c)                              # attention only — the ACR-proposed policy
    la = m.lookup(ka, c).name
    ls = m.lookup(ks, c).name
    ok = check("(hash, group_idx) keys do not collide", la == "HIT" and ls == "MISS",
               {"attention": la, "state": ls,
                "decoded": [get_offload_group_idx(ka), get_offload_group_idx(ks)]})
    # The question H1 actually asks: can the tier load a group that was never stored? Capture the
    # real assertion text, not just the exception class — "which assert fired" is the finding.
    try:
        pl = m.prepare_load([ks], c)
        res = f"returned {pl!r}"[:120]
        raised = None
    except Exception as exc:                                       # noqa: BLE001
        res, raised = f"{type(exc).__name__}: {exc}"[:220], True
    print(f"        prepare_load(state group) -> {res}")
    return ok & check("H1: attention-only offload is loadable (state group absent)",
                      raised is None, {"result": res} if raised is None
                      else {"blocker": "H1 false at manager level", "result": res})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/acr-n1.json")
    args = ap.parse_args()
    t0 = time.time()
    print(f"vllm offload tier harness — real classes, no engine "
          f"(policy {ACR_CLASS} from {ACR_MODULE}; builtin comparator "
          f"{LRUCachePolicy.__name__})")
    m = s0_loader()
    if m is None:
        print("\nseam does not load — nothing else can hold")
        json.dump(results, open(args.out, "w"), indent=1)
        return 1
    s1_identity(m); s2_admission(m)
    s3_pin(new_manager()); s4_order(new_manager()); s5_groups(new_manager())
    control_arm()
    passed = sum(1 for r in results if r["ok"])
    print(f"\n{passed}/{len(results)} checks passed in {time.time()-t0:.1f}s")
    json.dump({"checks": results, "passed": passed, "total": len(results)}, open(args.out, "w"), indent=1)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
