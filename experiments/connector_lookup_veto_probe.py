#!/usr/bin/env python3
"""N10 — is the stock `_lookup_complete_chunks` veto real, and what separates 0 from a hit?

No engine, no GPU. Runs inside the PINNED PRISTINE image
(vllm/vllm-openai@sha256:0aea3024...) with `CUDA_VISIBLE_DEVICES=""`.

The claim under test (from the team lead): in
`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`,
`_lookup_complete_chunks` converges ONE hit boundary across all lookup groups,
and inside the per-group loop

    max_hit_size_tokens = min(max_hit_size_tokens, len(offload_keys) * tokens_per_chunk)
    if max_hit_size_tokens - num_computed_tokens < tokens_per_chunk:   # scheduler.py:748-753
        return 0

so a single group whose `tokens_per_chunk` is coarser than the remaining hit
zeroes the WHOLE request, which is why a populated CPU tier never serves a load
on this attention(816) / GDN(16) / QSA-ring(8) tree.

WHAT IS REAL HERE: `_lookup_complete_chunks`, `_maximal_prefix_lookup`,
`_sliding_window_lookup`, `SchedulerOffloadConfig.from_spec`,
`RequestOffloadState.update_offload_keys`, `make_offload_key`, and the group
classification are the pristine library's own code — this script never replaces
them. Only the *source of HIT/MISS* (the manager) and the *Request* record are
controlled fakes, because the veto is pure scheduler arithmetic over which keys
are resident. Every claim the script prints is EXECUTED unless tagged inferred.

Two group sets:
  * SET2 [FA 816, Mamba 16] — stock-legal, no classification patch. This is the
    cleanest isolation of the heterogeneous *chunk-width* veto.
  * SET3 [FA 816, ring 8, Mamba 16] — the ring crashes stock `from_spec`
    (scheduler.py:125 assert), so a classification shim is applied ONLY to let
    `from_spec` build; the lookup method itself stays unpatched.

Cases (per the team lead's brief):
  a  all lookup groups complete, boundary-consistent          -> does it EVER hit?
  b  ring excluded from offload_keys (patch-05 exclusion)     -> = SET2 complete
  c  attention complete, Mamba short by exactly one chunk     -> width veto?
  d  prompt misaligned to coarsest chunk (86000 vs 86016)     -> alignment veto?
  e  uniform chunk width (both groups 816) complete           -> reconciliation fix?
  a-real SET3 with the ring's ONE storable pinned block       -> production ring

For every case the script also recomputes the scalar arithmetic independently
and flags any disagreement with the executed return.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field

import torch

from vllm.config import CacheConfig, ParallelConfig, VllmConfig
from vllm.config.device import DeviceConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import scheduler as SCH
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.kv_offload.base import LookupResult, ReqContext, make_offload_key

FA16 = torch.float16
HASH = 8                     # --prefix-match-unit 8 -> tokens_per_hash
N_TOKENS = 86_000            # production prompt size in the measured windows

_REAL_CLASSIFY = SCH.get_sliding_window_size_in_chunks


# --------------------------------------------------------------------------- #
# controlled HIT/MISS source and Request record (the fakes)                    #
# --------------------------------------------------------------------------- #
class FakeManager:
    """Only `lookup` matters to `_lookup_complete_chunks`. HIT iff key resident."""

    def __init__(self) -> None:
        self.stored: set[bytes] = set()

    def lookup(self, key, req_context):  # noqa: ANN001
        return LookupResult.HIT if key in self.stored else LookupResult.MISS

    def touch(self, keys, req_context):  # noqa: ANN001  (not on the lookup path)
        pass

    def on_new_request(self, req_context):  # noqa: ANN001
        return None


class _NoEvents:
    enable_kv_cache_events = False


@dataclass
class FakeReq:
    request_id: str
    num_tokens: int
    num_prompt_tokens: int
    block_hashes: list = field(default_factory=list)
    kv_transfer_params: None = None
    skip_reading_prefix_cache: bool = False


def fake_spec(manager, specs, bpc=1):  # noqa: ANN001
    tpbs = tuple(g.kv_cache_spec.block_size for g in specs)

    class _Spec:
        def __init__(self):
            self._m = manager

        def get_manager(self):
            return self._m

        tokens_per_block = tpbs
        blocks_per_chunk = bpc
        tokens_per_hash = HASH
        kv_events_config = _NoEvents()
        offload_prompt_only = False

    return _Spec()


def vllm_config() -> VllmConfig:
    return VllmConfig(device_config=DeviceConfig(device="cpu"),
                      cache_config=CacheConfig(enable_prefix_caching=True),
                      parallel_config=ParallelConfig(tensor_parallel_size=4,
                                                     enable_expert_parallel=True))


# --------------------------------------------------------------------------- #
# group sets                                                                   #
# --------------------------------------------------------------------------- #
def set2_specs(eagle: bool):
    g = [KVCacheGroupSpec(layer_names=["l0.attn"],
                          kv_cache_spec=FullAttentionSpec(block_size=816, num_kv_heads=1,
                                                          head_size=256, dtype=FA16),
                          is_eagle_group=eagle),
         KVCacheGroupSpec(layer_names=["l1.gdn"],
                          kv_cache_spec=MambaSpec(block_size=16, shapes=[(1, 2560)],
                                                  dtypes=[FA16], mamba_cache_mode="align"),
                          is_eagle_group=eagle)]
    return g


def set2_uniform_specs(eagle: bool):
    # case (e): both groups share one chunk width -> no cross-group width mismatch
    g = [KVCacheGroupSpec(layer_names=["l0.attn"],
                          kv_cache_spec=FullAttentionSpec(block_size=816, num_kv_heads=1,
                                                          head_size=256, dtype=FA16),
                          is_eagle_group=eagle),
         KVCacheGroupSpec(layer_names=["l1.gdn"],
                          kv_cache_spec=MambaSpec(block_size=816, shapes=[(1, 2560)],
                                                  dtypes=[FA16], mamba_cache_mode="align"),
                          is_eagle_group=eagle)]
    return g


def set3_specs(eagle: bool):
    g = [KVCacheGroupSpec(layer_names=["l0.attn"],
                          kv_cache_spec=FullAttentionSpec(block_size=816, num_kv_heads=1,
                                                          head_size=256, dtype=FA16),
                          is_eagle_group=eagle),
         KVCacheGroupSpec(layer_names=["l1.qsa"],
                          kv_cache_spec=CircularBufferSpec(block_size=8, num_kv_heads=1,
                                                           head_size=256, dtype=FA16),
                          is_eagle_group=eagle),
         KVCacheGroupSpec(layer_names=["l2.gdn"],
                          kv_cache_spec=MambaSpec(block_size=16, shapes=[(1, 2560)],
                                                  dtypes=[FA16], mamba_cache_mode="align"),
                          is_eagle_group=eagle)]
    return g


def ring_as_fullattn_classify(kv_spec, tokens_per_chunk):
    """Reproduces shipped patch-05: the ring is NOT a sliding window here
    (block 8 >= chunk 8 -> None -> full-attention lookup group). Stock `from_spec`
    cannot classify a CircularBufferSpec at all (assert at :125), so this shim
    exists ONLY to let the config build; the lookup method is untouched."""
    if isinstance(kv_spec, CircularBufferSpec):
        return None if kv_spec.block_size >= tokens_per_chunk else 1
    return _REAL_CLASSIFY(kv_spec, tokens_per_chunk)


def build(specs, *, ring_shim: bool, eagle: bool):
    mgr = FakeManager()
    if ring_shim:
        orig = SCH.get_sliding_window_size_in_chunks
        SCH.get_sliding_window_size_in_chunks = ring_as_fullattn_classify
        try:
            s = SCH.OffloadingConnectorScheduler(fake_spec(mgr, specs), vllm_config(),
                                                 KVCacheConfig(num_blocks=1442,
                                                               kv_cache_tensors=[],
                                                               kv_cache_groups=specs))
        finally:
            SCH.get_sliding_window_size_in_chunks = orig
    else:
        s = SCH.OffloadingConnectorScheduler(fake_spec(mgr, specs), vllm_config(),
                                             KVCacheConfig(num_blocks=1442,
                                                           kv_cache_tensors=[],
                                                           kv_cache_groups=specs))
    return s, mgr


# --------------------------------------------------------------------------- #
# populate offload_keys + residency                                          #
# --------------------------------------------------------------------------- #
def block_hashes(n_tokens: int):
    return [hashlib.sha256(f"h{i}".encode()).digest() for i in range(n_tokens // HASH)]


def make_state(sched, n_tokens: int):
    req = FakeReq(request_id="r", num_tokens=n_tokens, num_prompt_tokens=n_tokens,
                  block_hashes=block_hashes(n_tokens))
    ctx = ReqContext(req_id=req.request_id)
    st = SCH.RequestOffloadState(config=sched.config, req=req, req_context=ctx,
                                 offloading_context=None)
    st.update_offload_keys()
    st.num_locally_computed_tokens = 0
    return st


def store_group(state, group_idx, keep) -> set[bytes]:
    """Resident set for one group. keep: 'all' | int (# leading chunks) | 'tail-1'."""
    keys = list(state.group_states[group_idx].offload_keys)
    if keep == "all":
        return set(keys)
    if keep == "none":
        return set()
    if keep == "tail-1":
        return set(keys[-1:]) if keys else set()
    return set(keys[:keep])


# --------------------------------------------------------------------------- #
# independent arithmetic cross-check of _lookup_complete_chunks               #
# --------------------------------------------------------------------------- #
def cdiv(a, b):  # noqa: ANN001
    return -(-a // b)


def round_down(a, b):  # noqa: ANN001
    return (a // b) * b


def predict(sched, state) -> tuple[int | None, list[str], str]:
    """Mirror the scalar loop of scheduler.py:697-868 over the fake manager's
    HIT/MISS. Returns (predicted return, per-group trace lines, halt-reason)."""
    trace: list[str] = []
    num_computed = state.num_locally_computed_tokens
    max_hit = state.req.num_tokens
    if sched._sliding_window_groups:
        max_hit -= 1
        if sched._mamba_align_size is not None:
            max_hit = round_down(max_hit, sched._mamba_align_size)
    trace.append(f"init max_hit={max_hit} (computed={num_computed}, "
                 f"mamba_align={sched._mamba_align_size})")

    def leading_hits(keys):  # noqa: ANN001
        n = 0
        for k in keys:
            if k in sched.manager.stored:
                n += 1
            else:
                break
        return n

    def trailing_run(keys, window):  # noqa: ANN001
        # faithful mirror of scheduler.py:633-664: scan from the end and return
        # the END INDEX (chunk count) of the last full window of consecutive
        # hits, not the window length itself.
        run = 0
        for idx in range(len(keys) - 1, -1, -1):
            if keys[idx] in sched.manager.stored:
                run += 1
            else:
                run = 0
            if run == window:
                return idx + window
        return run

    groups = sched._lookup_groups
    eagle_verified: set[int] = set()
    defer = False
    num_hit_tokens = 0
    guard = 0
    while groups:
        guard += 1
        if guard > 20:
            trace.append("LOOP (convergence did not settle in 20 passes)")
            return None, trace, "loop-guard"
        looked_up_sw = False
        giter = list(groups)
        groups = []
        for gidx in giter:
            gc = sched.config.kv_group_configs[gidx]
            gstate = state.group_states[gidx]
            tpc = gc.tokens_per_chunk
            keys = gstate.offload_keys
            assert len(keys) >= state.req.num_tokens // tpc, "offload_keys assert (739)"
            is_eagle_unver = gc.is_eagle_group and gidx not in eagle_verified
            max_hit = min(max_hit, len(keys) * tpc)
            if max_hit - num_computed < tpc:
                trace.append(f"g{gidx} tpc={tpc} keys={len(keys)} "
                             f"max_hit->{max_hit}: {max_hit - num_computed} < {tpc}")
                return 0, trace, f"g{gidx} @748-753 (cap {tpc*len(keys)} < computed+chunk)"
            qmax = max_hit
            if is_eagle_unver and gc.sliding_window_size_in_chunks is not None:
                qmax = min(max_hit + tpc, len(keys) * tpc)
            nchunks = min(cdiv(qmax, tpc), len(keys))
            start = num_computed // tpc
            sliced = keys[start:nchunks]
            if gc.sliding_window_size_in_chunks is None:
                nhit = leading_hits(sliced)
            else:
                win = gc.sliding_window_size_in_chunks + (1 if is_eagle_unver else 0)
                nhit = trailing_run(sliced, win)
            if nhit == 0:
                trace.append(f"g{gidx} lookup nhit=0")
                return 0, trace, f"g{gidx} @792-793 (zero hit chunks)"
            if is_eagle_unver:
                nhit -= 1
                eagle_verified.add(gidx)
            max_hit = min(max_hit, tpc * (start + nhit))
            new = max_hit - num_computed
            trace.append(f"g{gidx} tpc={tpc} keys={len(keys)} nhit={nhit} "
                         f"max_hit->{max_hit} new_hit={new}")
            if new < tpc:
                return 0, trace, f"g{gidx} @808-810 (hit {new} < chunk {tpc})"
            if new < num_hit_tokens:
                if not gc.is_eagle_group:
                    eagle_verified.clear()
            looked_up_sw |= gc.sliding_window_size_in_chunks is not None
            num_hit_tokens = new
    if defer:
        return None, trace, "defer"
    return num_hit_tokens, trace, "return num_hit_tokens"


# --------------------------------------------------------------------------- #
# runner                                                                       #
# --------------------------------------------------------------------------- #
rows: list[dict] = []


def run_case(label: str, sched, mgr, pop, *, n_tokens=N_TOKENS, computed=0):
    """pop: callable(state) -> None that sets sched.manager.stored."""
    state = make_state(sched, n_tokens)
    state.num_locally_computed_tokens = computed
    pop(state)
    exec_hit = sched._lookup_complete_chunks(state)
    pred, trace, reason = predict(sched, state)
    agree = (pred == exec_hit) or (isinstance(exec_hit, int) and isinstance(pred, int)
                                   and pred == exec_hit)
    per_group = {f"g{gc.group_idx}":
                 (gc.tokens_per_chunk, len(state.group_states[gc.group_idx].offload_keys),
                  len([k for k in state.group_states[gc.group_idx].offload_keys
                       if k in mgr.stored]))
                 for gc in sched.config.kv_group_configs}
    rows.append(dict(label=label, exec=exec_hit, pred=pred, agree=agree,
                     reason=reason, per_group=per_group, trace=trace))
    tag = "OK " if agree else "!! "
    print(f"\n=== {tag}{label} ===")
    print(f"    per-group (tokens_per_chunk, #offload_keys, #resident): {per_group}")
    print(f"    EXECUTED _lookup_complete_chunks -> {exec_hit!r}")
    print(f"    arithmetic predict                -> {pred!r}   [{reason}]")
    if not agree:
        print("    *** DISAGREEMENT between executed method and cross-check ***")
    for t in trace:
        print(f"      · {t}")
    return exec_hit


def main() -> int:
    print("N10 — stock `_lookup_complete_chunks` veto probe (pristine image, no GPU)")
    print(f"    prompt = {N_TOKENS} tokens, tokens_per_hash = {HASH}")

    # ---- SET2 (FA816 + Mamba16), production MTP shape (eagle on all groups) ----
    s2, m2 = build(set2_specs(eagle=True), ring_shim=False, eagle=True)
    print(f"\n  SET2 config: lookup_groups={s2._lookup_groups} "
          f"sliding={s2._sliding_window_groups} mamba_align={s2._mamba_align_size} "
          f"partial_tail={s2.config.supports_partial_tail}")

    def a_all(st):  # noqa: ANN001
        m2.stored = store_group(st, 0, "all") | store_group(st, 1, "all")
    def c_mamba_short(st):  # noqa: ANN001
        keys1 = st.group_states[1].offload_keys
        m2.stored = store_group(st, 0, "all") | store_group(st, 1, len(keys1) - 1)
    def d_mamba_all_attn_aligned(st):  # noqa: ANN001
        m2.stored = store_group(st, 0, "all") | store_group(st, 1, "all")

    run_case("(a) SET2 all complete [eagle]", s2, m2, a_all)
    run_case("(b) = SET2 all complete (ring excluded) [eagle]", s2, m2, a_all)
    run_case("(c) SET2 attn complete, mamba -1 chunk [eagle]", s2, m2, c_mamba_short)
    run_case("(d) SET2 86000 misaligned, all complete [eagle]", s2, m2, d_mamba_all_attn_aligned,
             n_tokens=86_000)
    run_case("(d') SET2 86016 aligned to 816, all complete [eagle]", s2, m2,
             d_mamba_all_attn_aligned, n_tokens=86_016)

    # control: non-eagle (no MTP) — does eagle change the veto boundary?
    s2n, m2n = build(set2_specs(eagle=False), ring_shim=False, eagle=False)
    run_case("(a) SET2 all complete [no eagle]", s2n, m2n,
             lambda st: m2n.__setattr__("stored", store_group(st, 0, "all")
                                        | store_group(st, 1, "all")))

    # ---- (e) uniform width: both groups tokens_per_chunk 816 ----
    sU, mU = build(set2_uniform_specs(eagle=True), ring_shim=False, eagle=True)
    run_case("(e) uniform width (both 816), all complete [eagle]", sU, mU,
             lambda st: mU.__setattr__("stored", store_group(st, 0, "all")
                                       | store_group(st, 1, "all")))

    # ---- SET3 with ring as a full-attention lookup group (patch-05 shipped) ----
    s3, m3 = build(set3_specs(eagle=True), ring_shim=True, eagle=True)
    print(f"\n  SET3 config: lookup_groups={s3._lookup_groups} "
          f"sliding={s3._sliding_window_groups} mamba_align={s3._mamba_align_size}")

    def a3_all_incl_ring(st):  # noqa: ANN001
        m3.stored = (store_group(st, 0, "all") | store_group(st, 1, "all")
                     | store_group(st, 2, "all"))
    def a3_ring_one(st):  # noqa: ANN001
        # production ring: CircularBufferManager pins ONE block -> one stored chunk
        m3.stored = (store_group(st, 0, "all") | store_group(st, 1, 1)
                     | store_group(st, 2, "all"))

    run_case("(a) SET3 all complete incl ring [eagle]", s3, m3, a3_all_incl_ring)
    run_case("a-real SET3 attn+mamba complete, ring has 1 stored chunk [eagle]",
             s3, m3, a3_ring_one)
    run_case("(b) SET3 ring excluded, attn+mamba complete [eagle]", s3, m3, a_all_excl_ring(s3, m3))

    # the literal quoted veto (scheduler.py:748-753): a NON-eagle ring lookup
    # group that can hold only its single pinned chunk caps max_hit to 8, and
    # the NEXT group's chunk check (mamba, 16) trips `return 0` at line 751.
    s3n, m3n = build(set3_specs(eagle=False), ring_shim=True, eagle=False)
    run_case("ring-sparse NON-eagle SET3 (hits literal 748-753)", s3n, m3n,
             lambda st: m3n.__setattr__(
                 "stored", store_group(st, 0, "all") | store_group(st, 1, 1)
                 | store_group(st, 2, "all")))

    # ---- summary table ----
    print("\n\n================ SUMMARY ================")
    print(f"{'case':<58} {'exec':>8} {'pred':>8} agree  halt")
    for r in rows:
        print(f"{r['label']:<58} {str(r['exec']):>8} {str(r['pred']):>8} "
              f"{'Y' if r['agree'] else 'N':>4}   {r['reason']}")
    n_bad = sum(1 for r in rows if not r["agree"])
    print(f"\n{len(rows)} cases, {n_bad} arithmetic disagreements "
          f"(a disagreement = harness bug, not a finding)")
    return 0 if n_bad == 0 else 1


def a_all_excl_ring(sched, mgr):  # noqa: ANN001
    def pop(st):  # noqa: ANN001
        # drop ring from the lookup groups exactly as revised patch-05 does
        sched._lookup_groups = tuple(g for g in sched._lookup_groups if g != 1)
        sched._sliding_window_groups = tuple(g for g in sched._sliding_window_groups if g != 1)
        mgr.stored = (store_group(st, 0, "all") | store_group(st, 2, "all"))
    return pop


if __name__ == "__main__":
    sys.exit(main())
