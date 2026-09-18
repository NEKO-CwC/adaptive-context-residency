"""Render the serve-time configuration for a given host.

Kept as code rather than a shell snippet because the numbers come from the cost model:
the host tier is sized against the configured token budget, and the storage tier is emitted
only when it clears the break-even bandwidth test (docs/00 §4) — on this box it does not, so
the default render has no storage tier and that is intentional, not an omission.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from ..config import CostModel, default_config


@dataclass(frozen=True)
class LaunchPlan:
    cpu_bytes_to_use: int
    block_size: int | None
    eviction_policy: str
    cache_policy_module_path: str
    secondary_tiers: tuple[dict, ...]
    token_budget: int
    warnings: tuple[str, ...]

    def kv_transfer_config(self) -> dict:
        extra: dict = {
            "cpu_bytes_to_use": self.cpu_bytes_to_use,
            "eviction_policy": self.eviction_policy,
            "cache_policy_module_path": self.cache_policy_module_path,
            "offload_prompt_only": False,
        }
        if self.block_size:
            extra["block_size"] = self.block_size
        if self.secondary_tiers:
            extra["secondary_tiers"] = [dict(t) for t in self.secondary_tiers]
        return {"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
                "kv_connector_extra_config": extra}


def plan(token_budget: int = 3_000_000,
         ram_cap_gib: float = 160.0,
         storage_gib: float | None = None,
         storage_bw_gbs: float | None = None,
         cost: CostModel | None = None,
         block_size: int | None = None) -> LaunchPlan:
    cfg = default_config()
    cost = cost or cfg.cost
    warnings: list[str] = []

    wanted = int(token_budget * cost.kv_bytes_per_token)
    cap = int(ram_cap_gib * 1024**3)
    if wanted > cap:
        warnings.append(
            f"token budget {token_budget:,} needs {wanted / 1024**3:,.0f} GiB but the host tier "
            f"is capped at {ram_cap_gib:,.0f} GiB — residency will be lossy under pressure")
        cpu_bytes = cap
    else:
        cpu_bytes = wanted

    tiers: list[dict] = []
    if storage_gib and storage_bw_gbs:
        if storage_bw_gbs <= cost.breakeven_bw_gbs():
            warnings.append(
                f"storage tier {storage_bw_gbs:.3f} GB/s <= break-even "
                f"{cost.breakeven_bw_gbs():.3f} GB/s: restore is no cheaper than recompute, "
                f"so the tier is emitted for durability only if you enable it explicitly")
        else:
            tiers.append({"type": "fs", "root_dir": "/data/acr/kv",
                          "n_read_threads": 8, "n_write_threads": 8})

    return LaunchPlan(cpu_bytes_to_use=cpu_bytes, block_size=block_size,
                      eviction_policy="AcrValuePolicy",
                      cache_policy_module_path="acr.vllm.policy",
                      secondary_tiers=tuple(tiers), token_budget=token_budget,
                      warnings=tuple(warnings))


def render(**kwargs) -> str:
    p = plan(**kwargs)
    return json.dumps(p.kv_transfer_config(), indent=2)


if __name__ == "__main__":  # pragma: no cover
    p = plan()
    print(render())
    for w in p.warnings:
        print("WARNING:", w)
