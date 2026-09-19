"""Cost-model and index invariants — the claims the design rests on."""
from __future__ import annotations

import math

from acr.config import CostModel, TierConfig, default_config
from acr.index import PrefixIndex
from acr.trace import synthetic_mix


def test_breakeven_bandwidth_tracks_the_resolved_prefill_rate():
    """restore beats recompute iff BW > bytes_per_token x prefill_rate (docs/00 §4).

    C-1 resolved 2026-09-19 by direct measurement (cold 4K-150K -> 6.8-11.6K tok/s), so the
    break-even is ~0.63 GB/s, not the ~0.445 GB/s the first draft used. The conclusion is
    unchanged and now stronger: the measured /data tier (0.413-0.432 GB/s) sits well below it.
    """
    cost = CostModel()
    assert math.isclose(cost.breakeven_bw_gbs(), 0.634, rel_tol=0.03)
    assert cost.breakeven_bw_gbs() > 0.432      # measured /data read+write speed


def test_ram_tier_clears_break_even_and_disk_does_not():
    cfg = default_config()
    ram = cfg.tier("ram")
    disk = TierConfig("disk", bytes_capacity=1024 * 1024**3, bandwidth_gbs=0.413,
                      efficiency=1.0)
    assert ram is not None and cfg.cost.is_worth_tier(ram)
    assert not cfg.cost.is_worth_tier(disk)


def test_shared_root_is_stored_once_not_once_per_session():
    idx = PrefixIndex(block_tokens=256, cost=CostModel())
    for i in range(4):
        idx.touch_path(f"s{i}", "supervisor", 320_000, 320_000, now=1.0 + i)
    shared = [b for b in idx.blocks.values() if b.shared]
    tokens = sum(b.tokens for b in shared)
    assert tokens == 320_000 // 256 * 256          # one copy of the common prefix
    assert all(len(b.refs) == 4 for b in shared)   # leased by all four sessions


def test_role_namespaces_do_not_false_share():
    idx = PrefixIndex(block_tokens=256, cost=CostModel())
    idx.touch_path("code", "supervisor", 100_000, 100_000, now=0.0)
    idx.touch_path("patient", "patient", 100_000, 100_000, now=0.0)
    code_keys = set(idx.session_path["code"])
    patient_keys = set(idx.session_path["patient"])
    assert not (code_keys & patient_keys)


def test_lease_aggregation_monotonic_in_leases():
    idx = PrefixIndex(block_tokens=256, cost=CostModel())
    idx.touch_path("a", "supervisor", 4096, 4096, now=0.0)
    one = idx.live_reuse_prob("root:supervisor:b0", now=10.0, horizon=60.0)
    idx.touch_path("b", "supervisor", 4096, 4096, now=1.0)
    two = idx.live_reuse_prob("root:supervisor:b0", now=10.0, horizon=60.0)
    assert two > one > 0.0


def test_abort_truncates_path_and_records_invalidation():
    idx = PrefixIndex(block_tokens=256, cost=CostModel())
    idx.touch_path("s", "supervisor", 20_000, 4_096, now=0.0)
    long_path = list(idx.session_path["s"])
    idx.touch_path("s", "supervisor", 10_000, 4_096, now=1.0, aborted=True)
    short_path = idx.session_path["s"]
    assert len(short_path) < len(long_path)
    assert set(idx.invalidations) == set(long_path[len(short_path):])


def test_dead_session_leases_expire():
    idx = PrefixIndex(block_tokens=256, cost=CostModel())
    idx.touch_path("s", "supervisor", 4_096, 4_096, now=0.0)
    key = "root:supervisor:b0"
    assert idx.live_leases(key, now=100.0, stale_after=900.0) == ["s"]
    assert idx.live_leases(key, now=10_000.0, stale_after=900.0) == []


def test_synthetic_mix_oversubscribes_hbm():
    """The experiment needs pressure: distinct working set must exceed the HBM pool."""
    turns = synthetic_mix(seed=3, coding_sessions=12, patient_sessions=20)
    idx = PrefixIndex(block_tokens=256, cost=CostModel())
    for turn in turns:
        idx.touch_path(turn.session_id, turn.role, turn.prompt_tokens,
                       turn.shared_prefix_tokens, now=turn.t)
    distinct = sum(b.tokens for b in idx.blocks.values())
    assert distinct > 1_003_197, distinct
