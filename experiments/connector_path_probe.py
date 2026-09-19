#!/usr/bin/env python3
"""N8 — is the tier really blocked by 5 lines, or by the whole design? No engine, no GPU.

W1d proved the stock connector dies at `offloading/scheduler.py:125`
(`assert isinstance(kv_cache_spec, FullAttentionSpec)`) because this model's QSA group is a
`CircularBufferSpec`. But the byte-moving side (`canonical_mapping.py:271/392`) dispatches on the
parent `AttentionSpec`, which `CircularBufferSpec` *is*. So the question for tomorrow's window is
whether a minimal group-aware branch is enough to get the library tier booting, or whether the
connector's chunk model fundamentally cannot represent a ring buffer.

This script answers the config-side half of that, offline, in three moves:
  1. reproduce the assert on a real CircularBufferSpec (so the diagnosis is not inferred);
  2. install the proposed 5-line branch as a monkeypatch and drive `SchedulerOffloadConfig.from_spec`
     over a group set faithful to the two boots we observed (attention 816 / circular 8 / mamba 16 —
     the only combination that both raises `tokens_per_block=8 … 816` in W1b and is cleared by
     `--prefix-match-unit 8` in W1c/W1d);
  3. say plainly what this does NOT prove: store/load byte correctness for a ring whose live slots
     move, which is exactly the failure mode that is not visible in any config and only G-1 catches.

Run:
  docker run --rm -v $ACR:/acr:ro -e CUDA_VISIBLE_DEVICES="" -e NVIDIA_VISIBLE_DEVICES="" \
      --entrypoint /bin/bash $IMG -c 'python3 /acr/experiments/connector_path_probe.py'
"""
from __future__ import annotations

import sys

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
    SlidingWindowSpec,
)

FA16 = torch.float16
results: list[tuple[str, bool, str]] = []


def say(name: str, ok: bool, note: str = "") -> None:
    results.append((name, ok, note))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {note}" if note else ""))


def groups():
    """(attention, QSA ring, GDN state) with the block sizes the two boots implied."""
    return [
        KVCacheGroupSpec(layer_names=["model.layers.0.self_attn"],
                         kv_cache_spec=FullAttentionSpec(block_size=816, num_kv_heads=1,
                                                         head_size=256, dtype=FA16)),
        KVCacheGroupSpec(layer_names=["model.layers.1.qsa"],
                         kv_cache_spec=CircularBufferSpec(block_size=8, num_kv_heads=1,
                                                          head_size=256, dtype=FA16)),
        KVCacheGroupSpec(layer_names=["model.layers.1.linear_attn"],
                         kv_cache_spec=MambaSpec(block_size=16, shapes=[(1, 2560)],
                                                 dtypes=[FA16], mamba_cache_mode="align")),
    ]


class FakeSpec:
    """Everything `SchedulerOffloadConfig.from_spec` reads off the spec (enumerated from source:
    tokens_per_block, blocks_per_chunk, tokens_per_hash, offload_prompt_only, sliding_window,
    attention_chunk_size, mamba_cache_mode, kv_events_config). tokens_per_hash=8 is what
    `--prefix-match-unit 8` resolves to on this tree."""
    tokens_per_block = tuple(g.kv_cache_spec.block_size for g in groups())
    blocks_per_chunk = 1
    tokens_per_hash = 8
    offload_prompt_only = False
    sliding_window = None
    attention_chunk_size = None
    mamba_cache_mode = "align"
    kv_events_config = None


def build_config():
    # speculative_config=None keeps the EAGLE/MTP branch out of this probe; the engine sets
    # is_eagle_group separately, which is listed below as NOT PROVEN rather than quietly tested.
    # device_config must be explicit: with no GPU visible this process cannot infer a device type,
    # and the failure surfaces far from here as "Failed to infer device type".
    return VllmConfig(device_config=DeviceConfig(device="cpu"),
                      cache_config=CacheConfig(enable_prefix_caching=True),
                      parallel_config=ParallelConfig(tensor_parallel_size=4,
                                                     enable_expert_parallel=True))


def main() -> int:
    print("N8 — connector path probe (config-side only)")

    # 1. reproduce the wall with the real class.
    try:
        SCH.get_sliding_window_size_in_chunks(CircularBufferSpec(
            block_size=8, num_kv_heads=1, head_size=256, dtype=FA16), 816)
        say("reproduces scheduler.py:125 assert on CircularBufferSpec", False, "no AssertionError")
    except AssertionError as exc:
        say("reproduces scheduler.py:125 assert on CircularBufferSpec", True, f"AssertionError {exc!r}"[:80])

    # sanity: the accepted kinds still behave as documented, so the probe is honest about what the
    # branch must not break.
    ok = (SCH.get_sliding_window_size_in_chunks(
              MambaSpec(block_size=16, shapes=[(1, 2560)], dtypes=[FA16],
                        mamba_cache_mode="align"), 816) == 1
          and SCH.get_sliding_window_size_in_chunks(
              FullAttentionSpec(block_size=816, num_kv_heads=1, head_size=256, dtype=FA16), 816) is None
          and SCH.get_sliding_window_size_in_chunks(
              SlidingWindowSpec(block_size=816, num_kv_heads=1, head_size=256, dtype=FA16,
                                sliding_window=4096), 816) == 6)
    say("existing accepted specs unchanged by our reading of the function", ok)

    # 2. install the candidate branch and drive from_spec over our group set.
    orig = SCH.get_sliding_window_size_in_chunks

    def patched(kv_cache_spec, tokens_per_chunk):
        # the proposed 5 lines: a ring is window-like over its own capacity
        if isinstance(kv_cache_spec, CircularBufferSpec):
            return None if kv_cache_spec.block_size >= tokens_per_chunk else 1
        return orig(kv_cache_spec, tokens_per_chunk)

    SCH.get_sliding_window_size_in_chunks = patched
    try:
        cfg = None
        try:
            gspec = groups()
            cfg = SCH.SchedulerOffloadConfig.from_spec(
                FakeSpec(), build_config(),
                KVCacheConfig(num_blocks=1442, kv_cache_tensors=[], kv_cache_groups=gspec))
            say("from_spec completes with the 5-line branch", True,
                f"alignment_chunk_count={getattr(cfg, 'alignment_tokens', '?')} "
                f"groups={len(groups())} num_workers={getattr(cfg, 'num_workers', '?')}")
        except Exception as exc:                                    # noqa: BLE001
            say("from_spec completes with the 5-line branch", False,
                f"{type(exc).__name__}: {str(exc)[:150]}")
    finally:
        SCH.get_sliding_window_size_in_chunks = orig

    # 3. what this cannot tell us.
    print("\n  NOT PROVEN by this script (must be gated, not assumed):")
    for line in (
        "store/load byte correctness for a ring whose live slots move (only G-1 restored-vs-cold "
        "equivalence catches this)",
        "the worker side: canonical_mapping dispatches on AttentionSpec so it *should* handle "
        "CircularBufferSpec, but its page geometry is only exercised with real KV tensors",
        "whether EAGLE/MTP draft-group exclusion interacts with the QSA group "
        "(is_eagle_group defaults False in this probe; the engine may set it)",
    ):
        print(f"    - {line}")

    bad = [n for n, ok_, _ in results if not ok_]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed" + (f"; failed: {bad}" if bad else ""))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
