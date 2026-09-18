"""Cost model + tier configuration.

Every default here is a number from docs/00-evidence.md with its confidence class noted.
Nothing in this module is a magic constant invented for a demo.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import yaml


@dataclass(frozen=True)
class TierConfig:
    """One storage tier below HBM (or HBM itself, with capacity in tokens)."""

    name: str
    bytes_capacity: int
    bandwidth_gbs: float          # sustained bytes/s toward the tier above, /1e9
    enabled: bool = True
    # Fraction of nominal bandwidth actually reached by the connector path.
    efficiency: float = 1.0

    @property
    def effective_gbs(self) -> float:
        return self.bandwidth_gbs * self.efficiency

    def restore_seconds(self, nbytes: int) -> float:
        if nbytes <= 0:
            return 0.0
        return nbytes / (self.effective_gbs * 1e9)


@dataclass(frozen=True)
class CostModel:
    """Physical constants that turn a residency decision into seconds."""

    kv_bytes_per_token: float = 57_591.0     # 13.5 GiB x 4 ranks / 1,003,197 tokens (measured)
    prefill_tokens_per_s: float = 7_900.0    # 30K cold prefix -> 3.8 s (measured; C-1 contested)
    prefill_tokens_per_s_alt: float = 41_700.0  # the conflicting 400K/9.6s figure
    decode_tokens_per_s: float = 118.0       # single-stream decode (measured)
    request_overhead_s: float = 0.36         # warm short-request TTFT floor (measured)

    def recompute_seconds(self, tokens: float) -> float:
        return tokens / self.prefill_tokens_per_s

    def recompute_bytes(self, tokens: float) -> float:
        return tokens * self.kv_bytes_per_token

    def breakeven_bw_gbs(self) -> float:
        """Tier bandwidth at which restoring beats recomputing, in GB/s (docs/00 §4).

        restore_time = bytes / BW ; recompute_time = tokens / rate ; bytes = tokens * bpt
        => BW* = bpt * rate
        """
        return self.kv_bytes_per_token * self.prefill_tokens_per_s / 1e9

    def is_worth_tier(self, tier: TierConfig) -> bool:
        return tier.effective_gbs > self.breakeven_bw_gbs()


@dataclass(frozen=True)
class SimConfig:
    hbm_pool_tokens: int = 1_003_197         # live config; 1,263,788 at 17 GiB/card
    block_tokens: int = 816                  # engine-verified: attention block forced to 816 tokens
    cost: CostModel = field(default_factory=CostModel)
    tiers: tuple[TierConfig, ...] = ()
    # Phase-2 reality: the controller does not control HBM eviction, only its own tiers.
    hbm_steerable: bool = False
    patient_slo_ttft_s: float = 1.5
    thrash_window_s: float = 60.0

    @property
    def hbm_bytes(self) -> int:
        return int(self.hbm_pool_tokens * self.cost.kv_bytes_per_token)

    def tier(self, name: str) -> TierConfig | None:
        return next((t for t in self.tiers if t.name == name and t.enabled), None)

    def ordered_tiers(self) -> list[TierConfig]:
        """Cheapest-to-reach first is wrong; nearest-to-GPU first is right."""
        return [t for t in self.tiers if t.enabled]


def default_config() -> SimConfig:
    """The 4xL20 host this project was written on (docs/00 §3)."""
    return SimConfig(
        hbm_pool_tokens=1_003_197,
        block_tokens=816,
        cost=CostModel(),
        tiers=(
            TierConfig("ram", bytes_capacity=160 * 1024**3, bandwidth_gbs=96.0, efficiency=0.55),
            # Present to record the negative result: at 0.41 GB/s it is below break-even (~0.45).
            TierConfig("disk", bytes_capacity=1024 * 1024**3, bandwidth_gbs=0.413, enabled=False),
        ),
    )


def load_config(path: str) -> SimConfig:
    with open(path) as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    base = default_config()
    cost = replace(CostModel(), **(raw.get("cost") or {}))
    tiers = tuple(
        TierConfig(name=t["name"], bytes_capacity=int(t["bytes_gib"]) * 1024**3,
                   bandwidth_gbs=float(t["bandwidth_gbs"]),
                   enabled=bool(t.get("enabled", True)),
                   efficiency=float(t.get("efficiency", 1.0)))
        for t in (raw.get("tiers") or [])
    ) or base.tiers
    sim = SimConfig(
        hbm_pool_tokens=int(raw.get("hbm_pool_tokens", base.hbm_pool_tokens)),
        block_tokens=int(raw.get("block_tokens", base.block_tokens)),
        cost=cost,
        tiers=tiers,
        hbm_steerable=bool(raw.get("hbm_steerable", False)),
        patient_slo_ttft_s=float(raw.get("patient_slo_ttft_s", 1.5)),
        thrash_window_s=float(raw.get("thrash_window_s", 60.0)),
    )
    return sim
