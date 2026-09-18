"""Replay-simulator behaviour: the claims we would otherwise have to assert in prose.

Note on scope: the oracle is a Belady bound on the **host tier only**. HBM is modelled as the
engine's own LRU set, which ACR fills but cannot order, so "oracle" is not an upper bound on the
system — it is the regret baseline for the part we control.
"""
from __future__ import annotations

from dataclasses import replace

from acr.config import TierConfig, default_config
from acr.replay import run_policy
from acr.trace import Turn

BPT = 57_591.0


def _turn(t, sid, prompt, completion=8, shared=0, eta=None):
    return Turn(t=t, session_id=sid, role="supervisor", prompt_tokens=prompt,
                completion_tokens=completion, shared_prefix_tokens=shared, tool_eta_s=eta)


def _cfg(ram_gib: float, hbm_tokens: int = 1_003_197, efficiency: float = 1.0):
    base = default_config()
    tiers = tuple(replace(t, bytes_capacity=int(ram_gib * 1024**3), efficiency=efficiency)
                  if t.name == "ram" else t for t in base.tiers)
    return replace(base, tiers=tiers, hbm_pool_tokens=hbm_tokens)


def test_warm_repeat_is_pure_hbm_hit():
    turns = [_turn(0.0, "s", 8_000), _turn(5.0, "s", 8_100)]
    report = run_policy(_cfg(160), turns, "lru")
    assert report.recompute_tokens > 0        # first turn is cold
    assert report.restore_bytes == 0          # nothing had to come back from host
    assert report.turns == 2


def test_host_tier_rescues_context_that_hbm_evicted():
    """Two 600K sessions cannot both live in a 1M pool; when the first returns, the host tier
    must supply it instead of the engine re-prefilling 600K tokens."""
    cfg = _cfg(160)
    turns = [_turn(0.0, "a", 600_000), _turn(1.0, "b", 600_000), _turn(2.0, "a", 600_000)]
    with_host = run_policy(cfg, turns, "lru")
    without_host = run_policy(replace(cfg, tiers=(TierConfig(
        "ram", bytes_capacity=1, bandwidth_gbs=96.0),)), turns, "lru")
    assert with_host.restore_bytes > 0
    # the rescued context is exactly the work the host tier removed from the recompute bill
    assert without_host.recompute_tokens - with_host.recompute_tokens >= 500_000


def test_wasted_time_shrinks_as_host_tier_grows():
    turns = [_turn(i % 3 * 1.0 + (i // 3) * 10.0, f"s{i % 3}", 500_000) for i in range(24)]
    small = run_policy(_cfg(4), turns, "lru")
    big = run_policy(_cfg(120), turns, "lru")
    assert big.objective() < small.objective()
    assert big.recompute_tokens < small.recompute_tokens


def test_run_is_deterministic():
    turns = [_turn(float(i), f"s{i % 4}", 40_000 + 500 * i, shared=40_000) for i in range(40)]
    a = run_policy(_cfg(160), turns, "adaptive_value").as_row()
    b = run_policy(_cfg(160), turns, "adaptive_value").as_row()
    assert a == b


def test_policies_diverge_under_pressure():
    """If every policy produced the same numbers the harness is not measuring anything."""
    turns = [_turn(i * 2.0, f"s{i % 8}", 260_000, shared=120_000) for i in range(64)]
    rows = {name: run_policy(_cfg(30), turns, name).as_row()
            for name in ("lru", "lfu", "fixed_ttl", "continuum_ttl", "adaptive_value", "oracle")}
    wasted = {name: r["wasted_s"] for name, r in rows.items()}
    assert len(set(wasted.values())) > 1, wasted


def test_admission_filter_drops_blocks_that_are_not_worth_caching():
    """One-shot requests should not buy residency; LRU lets them push out real sessions."""
    one_shots = [Turn(t=i * 10.0, session_id=f"throw{i}", role="rag",
                      prompt_tokens=20_000, completion_tokens=8) for i in range(60)]
    keepers = [_turn(1.0, "live", 200_000)] + [
        _turn(600.0 + i * 10.0, "live", 200_000) for i in range(20)]
    turns = sorted(one_shots + keepers, key=lambda t: t.t)
    lru = run_policy(_cfg(6), turns, "lru")
    adaptive = run_policy(_cfg(6), turns, "adaptive_value")
    assert adaptive.policy == "adaptive_value"
    # the session we care about must not be the one paying for the junk we cached
    assert adaptive.per_role["supervisor"]["recompute_ktok"] <= lru.per_role["supervisor"]["recompute_ktok"]
    assert "rag" in lru.per_role and "rag" in adaptive.per_role


def test_policy_choice_is_immaterial_when_the_tier_is_not_contended():
    """A negative result we want to keep honest: eviction policy only matters under pressure.

    When the host tier is larger than the distinct working set, every policy degenerates to
    "keep everything" and the numbers are identical. Any claim of policy improvement must
    therefore state the capacity band it was measured in (docs/00, docs/02).
    """
    turns = [_turn(i * 2.0, f"s{i % 4}", 260_000, shared=120_000) for i in range(32)]
    rows = {name: run_policy(_cfg(160), turns, name).as_row()
            for name in ("lru", "lfu", "fixed_ttl", "adaptive_value", "oracle")}
    assert len({r["wasted_s"] for r in rows.values()}) == 1, rows


def test_engine_owned_hbm_lru_churn_is_visible_as_thrash():
    """Most host-tier restores are blocks the engine dropped from HBM moments ago.

    That is the quantitative case for phase 3 (an HBM residency hint): the host tier is doing
    work that a GPU-side retention decision would not need at all.
    """
    turns = [_turn(i * 2.0, f"s{i % 8}", 260_000, shared=120_000) for i in range(64)]
    report = run_policy(_cfg(60), turns, "lru")
    assert report.restore_bytes > 0
    assert report.thrash > 0
