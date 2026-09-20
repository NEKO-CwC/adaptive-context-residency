#!/usr/bin/env python3
"""N9 — quantify the ring-correct classification of patch 05 (revised 2026-09-20).

connector_path_probe.py (N8) answered the config-side question on the PRISTINE
tree. This probe answers the behavioral question on the PATCHED tree (acr set,
revised 05), with the real classes from
`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`:

  1. what `get_sliding_window_size_in_chunks` now returns for the ring at the
     real geometry (block_size=8, --prefix-match-unit 8 => tokens_per_chunk=8);
  2. the resulting per-group `alignment_chunk_count` / `hashes_per_chunk` from
     the real `SchedulerOffloadConfig.from_spec`;
  3. how many store keys each classification actually offers the manager for a
     4080-token prompt, using the real `storable_chunks` + the real
     `is_store_reachable_swa_chunk` and the ring's real block-table geometry
     (one pinned block per request -- CircularBufferManager);
  4. end-to-end: the real `_lookup_complete_chunks` over a real
     `CPUOffloadingManager`, for three variants:
       old05          -- shipped patch 05 (ring treated as full attention)
       classify_only  -- ring windowed (cdiv) but still gating the lookup
       fixed          -- revised patch 05 (window + excluded from store/load/lookup)
     under the production MTP shape (is_eagle_group=True on all groups) and a
     non-eagle control.

The point of variant `classify_only` is falsification: a positive window value
alone does NOT unblock the hybrid load path, because the ring's single pinned
block can never hold a boundary-keyed copy, so every boundary-windowed lookup
misses and `_lookup_complete_chunks` still returns a hard 0. That is why the
revised patch also excludes ring groups from store/load/lookup.

Run inside the patched container:
  python3 /acr/experiments/connector_ring_classify_probe.py
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field

from vllm.config import CacheConfig, ParallelConfig, VllmConfig
from vllm.config.device import DeviceConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import scheduler as SCH
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.base import ReqContext
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

FA16 = __import__("torch").float16

N_TOKENS = 4080            # 5 full-attention chunks
HASH_STRIDE = 8            # --prefix-match-unit 8
GROUPS = {
    "fa": FullAttentionSpec(block_size=816, num_kv_heads=1, head_size=256, dtype=FA16),
    "qsa": CircularBufferSpec(block_size=8, num_kv_heads=1, head_size=256, dtype=FA16),
    "mamba": MambaSpec(block_size=16, shapes=[(1, 2560)], dtypes=[FA16],
                       mamba_cache_mode="align"),
}
ORDER = ["fa", "qsa", "mamba"]

results: list[tuple[str, bool, str]] = []


def say(name: str, ok: bool, note: str = "") -> None:
    results.append((name, ok, note))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {note}" if note else ""))


def group_specs(eagle: bool):
    return [
        KVCacheGroupSpec(layer_names=[f"l.{i}"], kv_cache_spec=GROUPS[name],
                         is_eagle_group=eagle)
        for i, name in enumerate(ORDER)
    ]


class _NoEvents:
    enable_kv_cache_events = False


class FakeSpec:
    tokens_per_block = tuple(GROUPS[n].block_size for n in ORDER)
    blocks_per_chunk = 1
    tokens_per_hash = HASH_STRIDE
    offload_prompt_only = False
    kv_events_config = _NoEvents()

    def __init__(self, manager):
        self._manager = manager

    def get_manager(self):
        return self._manager


def vllm_config() -> VllmConfig:
    return VllmConfig(device_config=DeviceConfig(device="cpu"),
                      cache_config=CacheConfig(enable_prefix_caching=True),
                      parallel_config=ParallelConfig(tensor_parallel_size=4,
                                                     enable_expert_parallel=True))


def block_hashes():
    return [hashlib.sha256(f"chunk-{i}".encode()).digest()
            for i in range(N_TOKENS // HASH_STRIDE)]


@dataclass
class FakeReq:
    request_id: str
    num_tokens: int = N_TOKENS
    num_prompt_tokens: int = N_TOKENS
    kv_transfer_params: None = None
    block_hashes: list = field(default_factory=block_hashes)
    skip_reading_prefix_cache: bool = False


# the shipped patch-05 classification, reproduced for the old05 variant
_REAL_CLASSIFY = SCH.get_sliding_window_size_in_chunks


def old05_classify(kv_cache_spec, tokens_per_chunk):
    if isinstance(kv_cache_spec, CircularBufferSpec):
        if kv_cache_spec.block_size >= tokens_per_chunk:
            return None
        return (kv_cache_spec.block_size + tokens_per_chunk - 1) // tokens_per_chunk
    return _REAL_CLASSIFY(kv_cache_spec, tokens_per_chunk)


def build_scheduler(mode: str, eagle: bool):
    """mode: old05 | classify_only | fixed (the patched tree as shipped)."""
    mgr = CPUOffloadingManager(num_blocks=50_000, store_threshold=1,
                               cache_policy="lru")
    kvc = KVCacheConfig(num_blocks=1442, kv_cache_tensors=[],
                        kv_cache_groups=group_specs(eagle))
    if mode == "old05":
        orig = SCH.get_sliding_window_size_in_chunks
        SCH.get_sliding_window_size_in_chunks = old05_classify
        try:
            sched = SCH.OffloadingConnectorScheduler(FakeSpec(mgr), vllm_config(), kvc)
        finally:
            SCH.get_sliding_window_size_in_chunks = orig
        # old05 classified the ring as full attention -> stock __init__ put it in
        # full_attention_groups; the revised __init__ excluded it. Restore.
        full = [gc.group_idx for gc in sched.config.kv_group_configs
                if gc.sliding_window_size_in_chunks is None]
        sliding = sorted(
            (gc.group_idx for gc in sched.config.kv_group_configs
             if gc.sliding_window_size_in_chunks is not None),
            key=lambda i: sched.config.kv_group_configs[i].sliding_window_size_in_chunks,
            reverse=True)
        sched._sliding_window_groups = tuple(sliding)
        sched._lookup_groups = tuple(full) + tuple(sliding)
    elif mode == "classify_only":
        sched = SCH.OffloadingConnectorScheduler(FakeSpec(mgr), vllm_config(), kvc)
        # keep the ring out of nothing: put it back in the sliding-window lists
        sliding = sorted(
            (gc.group_idx for gc in sched.config.kv_group_configs
             if gc.sliding_window_size_in_chunks is not None),
            key=lambda i: sched.config.kv_group_configs[i].sliding_window_size_in_chunks,
            reverse=True)
        sched._sliding_window_groups = tuple(sliding)
        sched._lookup_groups = tuple(
            gc.group_idx for gc in sched.config.kv_group_configs
            if gc.sliding_window_size_in_chunks is None) + tuple(sliding)
    else:
        sched = SCH.OffloadingConnectorScheduler(FakeSpec(mgr), vllm_config(), kvc)
    return sched


def simulate_store(sched, mode: str):
    """Reproduce _build_store_jobs' per-chunk gating with the REAL helpers:
    RequestOffloadState.storable_chunks + is_store_reachable_swa_chunk + the
    ring's real one-pinned-block table. Returns per-group stored-key counts."""
    req = FakeReq(request_id="store-req")
    ctx = ReqContext(req_id=req.request_id)
    octx = sched.manager.on_new_request(ctx)
    state = SCH.RequestOffloadState(config=sched.config, req=req, req_context=ctx,
                                    offloading_context=octx)
    state.update_offload_keys()
    # block tables the core scheduler hands the connector: FA/mamba grow one
    # block per chunk; CircularBufferManager claims exactly one pinned block.
    tables = {
        "fa": list(range(1, 1 + N_TOKENS // 816)),
        "qsa": [7],
        "mamba": list(range(100, 100 + N_TOKENS // 16)),
    }
    counts = {}
    for gc in sched.config.kv_group_configs:
        name = ORDER[gc.group_idx]
        gs = state.group_states[gc.group_idx]
        gs.block_ids = list(tables[name])
        if mode == "fixed" and gc.is_circular_buffer_group:
            counts[name] = 0            # the revised patch skips the group
            continue
        num_chunks = state.storable_chunks(gc, gs, N_TOKENS)
        keys = []
        for idx in range(gs.next_stored_chunk_idx, num_chunks):
            bid = gs.block_ids[idx * sched.config.blocks_per_chunk
                               + sched.config.blocks_per_chunk - 1] \
                if idx < len(gs.block_ids) else 0
            if bid == 0:
                continue
            if not SCH.is_store_reachable_swa_chunk(
                    idx, num_chunks, gc.alignment_chunk_count,
                    gc.sliding_window_size_in_chunks, gc.is_eagle_group):
                continue
            keys.append(gs.offload_keys[idx])
        out = sched.manager.prepare_store(keys, ctx)
        stored = list(out.keys_to_store) if out else []
        sched.manager.complete_store(stored, ctx)
        counts[name] = len(stored)
    return counts


def lookup_after_store(sched, mode: str) -> int | None:
    counts = simulate_store(sched, mode)
    req2 = FakeReq(request_id="restore-req")
    ctx2 = ReqContext(req_id=req2.request_id)
    octx2 = sched.manager.on_new_request(ctx2)
    state2 = SCH.RequestOffloadState(config=sched.config, req=req2, req_context=ctx2,
                                     offloading_context=octx2)
    state2.update_offload_keys()
    state2.num_locally_computed_tokens = 0
    hit = sched._lookup_complete_chunks(state2)
    return hit, counts


def main() -> int:
    print("N9 — ring classification probe (patched tree, acr set, revised 05)")

    # 1. classification at the real geometry
    c = SCH.get_sliding_window_size_in_chunks
    ring = GROUPS["qsa"]
    say("ring(block 8) vs tokens_per_chunk 8 -> 1 (window over capacity, NOT None)",
        c(ring, 8) == 1, f"got {c(ring, 8)!r}")
    say("ring(block 8) vs tokens_per_chunk 816 -> 1", c(ring, 816) == 1,
        f"got {c(ring, 816)!r}")
    say("accepted specs unchanged (FA None / mamba 1 / swa 4096@816 -> 6)",
        c(GROUPS["fa"], 816) is None
        and c(GROUPS["mamba"], 816) == 1
        and c(SlidingWindowSpec(block_size=816, num_kv_heads=1, head_size=256,
                                dtype=FA16, sliding_window=4096), 816) == 6)

    # 2. from_spec config table (production MTP shape)
    sched = build_scheduler("fixed", eagle=True)
    table = {
        ORDER[gc.group_idx]: (gc.tokens_per_chunk, gc.hashes_per_chunk,
                              gc.sliding_window_size_in_chunks,
                              gc.alignment_chunk_count, gc.is_circular_buffer_group)
        for gc in sched.config.kv_group_configs
    }
    for k, v in table.items():
        print(f"    {k:6s} tokens_per_chunk/hashes/window/alignment/ring = {v}")
    say("config table fa(816,102,None,None) qsa(8,1,1,102,True) mamba(16,2,1,51)",
        table["fa"] == (816, 102, None, None, False)
        and table["qsa"] == (8, 1, 1, 102, True)
        and table["mamba"] == (16, 2, 1, 51, False))
    say("ring excluded from lookup groups",
        table["qsa"][4] and 1 not in sched._lookup_groups
        and 1 not in sched._sliding_window_groups)

    # 3+4. store volumes and end-to-end lookups, MTP shape
    for mode in ("old05", "classify_only", "fixed"):
        s = build_scheduler(mode, eagle=True)
        hit, counts = lookup_after_store(s, mode)
        total = sum(counts.values())
        print(f"    {mode:14s} stores={counts} total={total} "
              f"eagle lookup={hit!r} tokens")
        if mode == "old05":
            say("old05 reproduces the hard-0 hybrid match (measured defect)",
                hit == 0 and counts["mamba"] == N_TOKENS // 16)
        if mode == "classify_only":
            say("classify_only still 0: positive window without exclusion is "
                "insufficient (ring block can never hold a boundary copy)",
                hit == 0)
        if mode == "fixed":
            say("fixed restores a large hybrid match", isinstance(hit, int) and hit > 0,
                f"hit={hit}")
            old = build_scheduler("old05", eagle=True)
            _, old_counts = lookup_after_store(old, "old05")
            say("store volume: mamba 25.5x down (alignment filter restored), "
                "ring 1->0, total 261->15",
                counts["mamba"] == 10 and old_counts["mamba"] == 255
                and counts["qsa"] == 0 and sum(counts.values()) == 15)

    # non-eagle control. The non-eagle store filter keeps only the true segment
    # tail (reachable_tail = window + 0), so the 4064-capped FA hit cannot be
    # served by a resident mamba state and convergence lands on the previous
    # full-attention chunk — the connector's own alignment semantics.
    s = build_scheduler("fixed", eagle=False)
    hit, counts = lookup_after_store(s, "fixed")
    print(f"    fixed/no-eagle stores={counts} lookup={hit!r} tokens")
    say("fixed non-eagle match > 0 (full-attention aligned, 3264) with mamba "
        "stores cut to segment tails", hit == 3264 and counts["mamba"] == 5)

    bad = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed"
          + (f"; failed: {bad}" if bad else ""))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
