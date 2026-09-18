"""Trace-driven multi-tier replay.

One pass = one policy. The model is first-order on purpose:

* **HBM** is engine-owned (vLLM APC). We approximate it as a byte-capped LRU set that ACR can
  fill but not pin — matching what phase 2 will actually be able to do.
* **Two contended servers**: prefill compute and the host→device PCIe path. This is what makes
  interference observable rather than assumed; the measured 9.2× short-request TTFT blowup from
  one 30K cold prefill reproduces from queueing alone, with no fudge factor.
* **Prefix contiguity**: the longest resident prefix is serviceable, the rest must be recomputed.
  A hole in the middle forces recompute/restore from there on, like a real connector lookup.

Anything this cannot see is stated in docs/00 rather than hidden in a constant.
"""
from __future__ import annotations

import statistics
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .config import SimConfig, TierConfig
from .index import PrefixIndex
from .policies import OraclePolicy, Probe, TierPolicy, build
from .trace import Turn


@dataclass
class HBMTier:
    """Engine-side pool: fillable, evictable by the engine, not steerable by us (yet)."""

    capacity_bytes: int
    _blocks: "OrderedDict[str, int]" = field(default_factory=OrderedDict)
    _used: int = 0

    def contains(self, key: str) -> bool:
        return key in self._blocks

    def touch(self, keys: Iterable[str]) -> None:
        for key in keys:
            if key in self._blocks:
                self._blocks.move_to_end(key)

    def store(self, items: list[tuple[str, int]], protected: set[str]) -> list[str]:
        """Fill, then let the engine LRU whatever it wants. Returns keys it dropped."""
        for key, nbytes in items:
            if key in self._blocks:
                self._blocks.move_to_end(key)
                continue
            self._blocks[key] = nbytes
            self._used += nbytes
        evicted: list[str] = []
        if self._used <= self.capacity_bytes:
            return evicted
        # One forward pass over the LRU order (oldest first); a per-victim rescan would make
        # every turn quadratic in pool size, which is how this harness used to take minutes.
        for victim in [k for k in self._blocks if k not in protected]:
            self._used -= self._blocks.pop(victim)
            evicted.append(victim)
            if self._used <= self.capacity_bytes:
                break
        return evicted

    @property
    def occupancy(self) -> int:
        return self._used


@dataclass
class TurnMetric:
    t: float
    role: str
    prompt_tokens: int
    ttft_s: float
    e2e_s: float
    recompute_tokens: int
    restore_bytes: int
    source_tiers: tuple[str, ...]


@dataclass
class Report:
    policy: str
    turns: int = 0
    window_s: float = 0.0
    prompt_tokens: int = 0
    recompute_tokens: int = 0
    restore_bytes: int = 0
    prefill_seconds: float = 0.0
    transfer_seconds: float = 0.0
    thrash: int = 0
    reacquires: int = 0
    demotions: int = 0
    invalidations: int = 0
    peak_ram_bytes: int = 0
    hbm_bytes: int = 0
    per_role: dict[str, dict[str, float]] = field(default_factory=dict)

    def objective(self) -> float:
        """Wasted time: recompute we could have avoided plus transfer we paid to avoid it."""
        return self.prefill_seconds + self.transfer_seconds

    def as_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "policy": self.policy,
            "turns": self.turns,
            "recompute_ktok": round(self.recompute_tokens / 1000, 1),
            "wasted_s": round(self.objective(), 1),
            "prefill_s": round(self.prefill_seconds, 1),
            "xfer_s": round(self.transfer_seconds, 2),
            "ram_hit_%": round(100 * self.restore_bytes / max(1, self.restore_bytes
                                                              + self.recompute_tokens
                                                              * 57_591), 1),
            "demotions": self.demotions,
            "thrash": self.thrash,
            "reacquire": self.reacquires,
        }
        for role, m in sorted(self.per_role.items()):
            row[f"{role}_ttft_p50"] = round(m["ttft_p50"], 2)
            row[f"{role}_ttft_p95"] = round(m["ttft_p95"], 2)
            if "slo_viol" in m:
                row[f"{role}_slo_viol"] = int(m["slo_viol"])
        return row


class Simulator:
    def __init__(self, cfg: SimConfig, policy: TierPolicy, ram_tier: TierConfig,
                 index: PrefixIndex | None = None):
        self.cfg = cfg
        self.policy = policy
        self.ram_tier = ram_tier
        self.index = index or PrefixIndex(cfg.block_tokens, cfg.cost)
        self.hbm = HBMTier(capacity_bytes=cfg.hbm_bytes)
        self.probe = Probe(index=self.index, cost=cfg.cost, tier=ram_tier)
        self.pcie_busy = 0.0
        self.prefill_busy = 0.0
        self.metrics: list[TurnMetric] = []
        # Two exact, cheaply-computable policy mistakes:
        #   thrash     - restored a block from host that left HBM moments ago
        #   reacquires - re-stored a block into host after having evicted it (we threw away
        #                KV we later needed, i.e. the recompute we paid for is on us)
        self.left_hbm_at: dict[str, float] = {}
        self.left_host_at: dict[str, float] = {}
        self.thrash = 0
        self.reacquires = 0
        self.demotions = 0
        self.peak_ram = 0
        self.report = Report(policy=policy.name)

    # ---- oracle support ------------------------------------------------------

    @staticmethod
    def future_accesses(cfg: SimConfig, turns: Sequence[Turn]) -> dict[str, list[float]]:
        sched: dict[str, list[float]] = {}
        index = PrefixIndex(cfg.block_tokens, cfg.cost)
        for turn in sorted(turns, key=lambda t: t.t):
            keys = index.chain(turn.session_id, turn.prompt_tokens, turn.shared_prefix_tokens,
                               root_id=f"root:{turn.role}")
            for key, _, _ in keys:
                sched.setdefault(key, []).append(turn.t)
        return {k: sorted(v) for k, v in sched.items()}

    def _refresh_oracle(self, keys: Sequence[str], now: float,
                        sched: dict[str, list[float]]) -> None:
        for key in keys:
            entry = self.policy.get(key)
            if entry is None:
                continue
            nxt = next((t for t in sched.get(key, ()) if t > now), float("inf"))
            entry.next_use = nxt

    # ---- one turn ------------------------------------------------------------

    def run(self, turns: Sequence[Turn]) -> Report:
        sched = (self.future_accesses(self.cfg, turns)
                 if isinstance(self.policy, OraclePolicy) else {})
        last_turn: dict[str, Turn] = {}
        for turn in sorted(turns, key=lambda t: t.t):
            self._one(turn, sched)
            last_turn[turn.session_id] = turn
        # Sessions that ended release their leases (no future knowledge used: the engine's own
        # staleness rule is time-based, and stale_after mirrors the gateway's idle timeout).
        for sid in list(self.index.session_path):
            self.index.release_session(sid)
        self._finalize()
        return self.report

    def _one(self, turn: Turn, sched: dict[str, list[float]]) -> None:
        now = turn.t
        cost = self.cfg.cost
        keys = self.index.touch_path(turn.session_id, turn.role, turn.prompt_tokens,
                                     turn.shared_prefix_tokens, now, aborted=turn.aborted)
        if turn.tool_eta_s is not None:
            self.probe.tool_eta[turn.session_id] = turn.tool_eta_s
        self.probe.now = now

        # longest serviceable prefix, and where each block must come from
        restore_bytes = 0
        cut = len(keys)
        for i, key in enumerate(keys):
            if self.hbm.contains(key):
                continue
            entry = self.policy.get(key)
            if entry is None:
                cut = i
                break
            restore_bytes += entry.nbytes
            idle = now - self.left_hbm_at.get(key, -1e18)
            if 0.0 <= idle <= self.cfg.thrash_window_s:
                self.thrash += 1
        recompute_tokens = sum(self.index.blocks[k].tokens for k in keys[cut:])

        transfer_s = self.ram_tier.restore_seconds(restore_bytes)
        prefill_s = cost.recompute_seconds(recompute_tokens)

        pcie_start = max(now, self.pcie_busy)
        pcie_end = pcie_start + transfer_s
        self.pcie_busy = pcie_end
        prefill_start = max(now, self.prefill_busy, pcie_end if transfer_s else 0.0)
        prefill_end = prefill_start + prefill_s
        self.prefill_busy = prefill_end
        ttft = cost.request_overhead_s + max(0.0, prefill_end - now)
        e2e = ttft + turn.completion_tokens / cost.decode_tokens_per_s

        # HBM fill (engine-owned LRU, we only fill) then host-tier admission (ours)
        path_bytes = [(k, self.index.blocks[k].nbytes) for k in keys]
        for victim in self.hbm.store(path_bytes, set(keys)):
            self._left_hbm(victim)
        self._store_batch_to_host(keys, now, sched)

        for key in keys:
            self.policy.touch([key], now, self.probe)
        self._refresh_oracle(keys, now, sched)

        self.metrics.append(TurnMetric(t=now, role=turn.role, ttft_s=ttft, e2e_s=e2e,
                                       prompt_tokens=turn.prompt_tokens,
                                       recompute_tokens=recompute_tokens,
                                       restore_bytes=restore_bytes,
                                       source_tiers=(self.ram_tier.name,) if restore_bytes else ()))
        self.peak_ram = max(self.peak_ram, self.policy.occupancy)

    def _store_batch_to_host(self, keys: Sequence[str], now: float,
                             sched: dict[str, list[float]]) -> None:
        """Make room for this turn's context in the host tier, in one eviction round.

        Per-block eviction would re-sort the whole candidate set for every block; batching is
        both faster and closer to what the manager does (it asks for N blocks at once).
        """
        candidates: list[tuple[str, int]] = []
        for key in keys:
            if self.policy.get(key) is not None:
                continue
            candidates.append((key, self.index.blocks[key].nbytes))
        fresh = self.policy.admit_chain(candidates, now, self.probe)
        if not fresh:
            return
        needed = sum(n for _, n in fresh) - max(
            0, self.ram_tier.bytes_capacity - self.policy.occupancy)
        room = max(0, self.ram_tier.bytes_capacity - self.policy.occupancy)
        if needed > 0:
            freed = self.policy.evict(needed, {k for k, _ in fresh}, now, self.probe)
            if freed is None:
                return            # nothing evictable at all (all protected or empty tier)
            for victim in freed:
                self._left_host(victim)
                room += self.index.blocks[victim].nbytes
        for key, nbytes in fresh:
            if nbytes > room:
                break             # store the prefix that fits, in path order
            room -= nbytes
            if key in self.left_host_at:
                self.reacquires += 1
            self.policy.insert(key, nbytes, now, self.probe)
            if isinstance(self.policy, OraclePolicy):
                entry = self.policy.get(key)
                if entry is not None:
                    entry.next_use = next((t for t in sched.get(key, ()) if t > now),
                                          float("inf"))

    def _left_hbm(self, key: str) -> None:
        self.demotions += 1
        self.left_hbm_at[key] = self.probe.now

    def _left_host(self, key: str) -> None:
        self.left_host_at[key] = self.probe.now

    # ---- reporting -----------------------------------------------------------

    def _finalize(self) -> None:
        rep = self.report
        rep.turns = len(self.metrics)
        rep.window_s = self.metrics[-1].t if self.metrics else 0.0
        rep.recompute_tokens = sum(m.recompute_tokens for m in self.metrics)
        rep.restore_bytes = sum(m.restore_bytes for m in self.metrics)
        rep.prompt_tokens = sum(m.prompt_tokens for m in self.metrics)
        rep.prefill_seconds = self.cfg.cost.recompute_seconds(rep.recompute_tokens)
        rep.transfer_seconds = self.ram_tier.restore_seconds(rep.restore_bytes)
        rep.thrash = self.thrash
        rep.reacquires = self.reacquires
        rep.demotions = self.demotions
        rep.peak_ram_bytes = self.peak_ram
        rep.hbm_bytes = self.hbm.occupancy
        rep.invalidations = len(self.index.invalidations)
        by_role: dict[str, list[TurnMetric]] = {}
        for m in self.metrics:
            by_role.setdefault(m.role, []).append(m)
        for role, items in by_role.items():
            ttfts = [i.ttft_s for i in items]
            stats = {
                "n": float(len(items)),
                "ttft_p50": statistics.median(ttfts),
                "ttft_p95": _pct(ttfts, 0.95),
                "ttft_max": max(ttfts),
                "recompute_ktok": sum(i.recompute_tokens for i in items) / 1000,
            }
            if role == "patient":
                stats["slo_viol"] = float(sum(
                    1 for t in ttfts if t > self.cfg.patient_slo_ttft_s))
            rep.per_role[role] = stats

def _pct(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def run_policy(cfg: SimConfig, turns: Sequence[Turn], name: str) -> Report:
    ram = cfg.tier("ram")
    if ram is None:
        raise ValueError("config has no enabled `ram` tier")
    policy = build(name, ram.bytes_capacity)
    return Simulator(cfg, policy, ram).run(turns)


def compare(cfg: SimConfig, turns: Sequence[Turn],
            names: Iterable[str]) -> list[Report]:
    return [run_policy(cfg, turns, name) for name in names]


def markdown_table(reports: Sequence[Report]) -> str:
    rows = [r.as_row() for r in reports]
    if not rows:
        return ""
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}".rstrip("0").rstrip(".") if value else "0"
    return str(value)
