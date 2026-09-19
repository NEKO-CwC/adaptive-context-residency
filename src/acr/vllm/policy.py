"""Out-of-tree `CachePolicy` for vLLM's CPU offload tier.

Loadable by the shipped engine with **no fork and no patch**, purely from the serve command:

    --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
      "kv_connector_extra_config":{"cpu_bytes_to_use":..., "eviction_policy":"AcrValuePolicy",
      "cache_policy_module_path":"acr.vllm.policy"}}'

Contract mirrored from `vllm/v1/kv_offload/cpu/policies/base.py` (read from the installed image,
2026-09-18):

    __init__(cache_capacity: int)          # capacity in BLOCKS, not bytes
    get(key) -> BlockStatus | None
    insert(key, block: BlockStatus)        # also moves the key into the evictable set
    remove(key)
    touch(keys: Iterable[OffloadKey], req_context: ReqContext)
    evict(n: int, protected: set[OffloadKey]) -> list[(key, BlockStatus)] | None   # atomic
    clear()
    mark_evictable(key) / mark_non_evictable(key)

Known API shape limits we design around, not around:
* `evict()` receives no request context, so every signal must be folded into per-block state at
  `touch()` time (`req_context.kv_transfer_params` is available there — that is the channel the
  gateway stamps: role, session, tool ETA, SLO class).
* the built-in manager admits unconditionally, so block-level admission needs either a custom
  `OffloadingManager` (see docs/01 §3) or a ~10-line upstream hook.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Iterable

try:  # engine-side import; absent when running the simulator's test suite on a laptop
    from vllm.v1.kv_offload.base import OffloadKey, ReqContext
    from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy

    HAVE_VLLM = True
except Exception:  # pragma: no cover - exercised only off-engine
    OffloadKey = bytes  # type: ignore[misc,assignment]
    ReqContext = Any   # type: ignore[misc,assignment]
    BlockStatus = Any  # type: ignore[misc,assignment]

    class CachePolicy:  # type: ignore[no-redef]
        def __init__(self, cache_capacity: int) -> None:
            self.cache_capacity = cache_capacity

    HAVE_VLLM = False


DEFAULT_PARAMS_KEY = "acr"
HORIZON_S = 120.0
FLOOR_P = 0.05
BLOCK_BYTES_FALLBACK = 0


class AcrValuePolicy(CachePolicy):
    """Lease-aggregated, cost-difference-weighted residency value with a pressure penalty.

        score(b) = P_b · max(0, T_recompute(b) − T_restore(b))
                   − λ_mem · bytes · pressure
                   − λ_io · bytes · P_b

    `P_b` is estimated from the block's own inter-arrival statistics plus the number of live
    sessions that hold it, and it is *biased by role*: a block last touched by interactive
    patient traffic is trusted to come back sooner than one last touched by batch traffic,
    because that is what the category-conditional reuse literature reports and what our own
    traffic mix shows.
    """

    def __init__(self, cache_capacity: int, params: dict[str, Any] | None = None) -> None:
        # The engine constructs policies as `cls(cache_capacity)`, so operator tuning knobs
        # arrive from the environment rather than a constructor argument.
        super().__init__(cache_capacity)
        cfg = dict(json.loads(os.environ.get("ACR_POLICY_CONFIG", "{}") or "{}"))
        cfg.update(params or {})
        self.horizon_s = float(cfg.get("horizon_s", HORIZON_S))
        self.lambda_mem = float(cfg.get("lambda_mem", 1.0))
        self.lambda_io = float(cfg.get("lambda_io", 0.25))
        self.role_weight = dict(cfg.get("role_weight",
                                        {"patient": 1.6, "reviewer": 1.0, "supervisor": 1.0}))
        self.kv_bytes_per_token = float(cfg.get("kv_bytes_per_token", 57_753.0))  # 56.4 KiB, docs/00 §1
        # 11.1K is the *marginal* cold-prefill rate measured on this engine; the old default of
        # 7.9K was prompt/e2e under contention and made recompute look 1.4x cheaper than it is.
        self.prefill_tokens_per_s = float(cfg.get("prefill_tokens_per_s", 11_100.0))
        self.restore_gbs = float(cfg.get("restore_gbs", 52.8))
        # The engine never hands a policy its geometry, and a zero here silently collapses every
        # score to 0.0 — which turns evict() into "whatever set iteration yields" (measured
        # 2026-09-20: role/ETA/lease variants produced byte-identical replay results and the policy
        # lost 14.8 hit-points to the built-in LRU). Default to this build's attention group.
        self.block_tokens = int(cfg.get("block_tokens", 816))

        self.blocks: dict[OffloadKey, BlockStatus] = {}
        self.evictable: set[OffloadKey] = set()
        self.meta: dict[OffloadKey, dict[str, float]] = {}

    # ---- engine callbacks ---------------------------------------------------

    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self.blocks.get(key)

    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self.blocks[key] = block
        if block.ref_cnt == 0:
            self.evictable.add(key)
        self.meta.setdefault(key, {"last_use": time.monotonic(), "uses": 0,
                                   "iat": 0.0, "leases": 1.0, "weight": 1.0,
                                   "bytes": self._bytes_of(key)})

    def remove(self, key: OffloadKey) -> None:
        self.blocks.pop(key, None)
        self.evictable.discard(key)
        self.meta.pop(key, None)

    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        now = time.monotonic()
        hints = _hints(getattr(req_context, "kv_transfer_params", None))
        for key in keys:
            m = self.meta.get(key)
            if m is None:
                continue
            gap = max(0.0, now - m["last_use"])
            m["iat"] = gap if m["iat"] <= 0 else 0.7 * m["iat"] + 0.3 * gap
            m["last_use"] = now
            m["uses"] += 1
            if hints:
                m["weight"] = self.role_weight.get(hints.get("role", ""), 1.0)
                if hints.get("tool_eta_s") is not None:
                    # An agent that says when it returns is worth believing: fold the ETA into
                    # the hazard by shifting the last-use clock forward to the promised return.
                    m["last_use"] = min(m["last_use"] + float(hints["tool_eta_s"]), now)
                m["leases"] = float(hints.get("session_lease_count", hints.get("leases", m["leases"])))
            if key in self.evictable:
                pass  # recency is captured by last_use; eviction recomputes score

    def mark_evictable(self, key: OffloadKey) -> None:
        self.evictable.add(key)

    def mark_non_evictable(self, key: OffloadKey) -> None:
        self.evictable.discard(key)

    def clear(self) -> None:
        self.blocks.clear()
        self.evictable.clear()
        self.meta.clear()

    def evict(self, n: int, protected: set[OffloadKey]) -> list[tuple[OffloadKey, BlockStatus]] | None:
        if n == 0:
            return []
        now = time.monotonic()
        candidates = [k for k in self.evictable if k not in protected]
        if len(candidates) < n:
            return None                      # atomic: no state change on failure
        # score ties are the norm at cold start (every block has iat<=0 => the floor probability),
        # so without a tiebreak a "value" policy is strictly worse than LRU. Recency then reuse
        # count, both ascending: evict the stalest, least-reused first.
        candidates.sort(key=lambda k: (self.score(k, now), self.meta.get(k, {}).get("last_use", 0.0),
                                       self.meta.get(k, {}).get("uses", 0)))
        out = [(k, self.blocks[k]) for k in candidates[:n]]
        for k, _ in out:
            self.remove(k)
        return out

    # ---- the part that is ours ---------------------------------------------

    def score(self, key: OffloadKey, now: float) -> float:
        m = self.meta.get(key)
        if m is None:
            return 0.0
        p = self.reuse_prob(m, now)
        nbytes = m["bytes"] or self._bytes_of(key)
        tokens = nbytes / max(1.0, self.kv_bytes_per_token)
        t_recompute = tokens / self.prefill_tokens_per_s
        t_restore = nbytes / (self.restore_gbs * 1e9)
        saved = p * max(0.0, t_recompute - t_restore)
        pressure = len(self.blocks) / max(1, self.cache_capacity)
        rent = self.lambda_mem * (nbytes / 1024**3) * pressure * 0.1
        io = self.lambda_io * (nbytes / 1024**3) * p * 0.01
        return (saved - rent - io) * m.get("weight", 1.0)

    def reuse_prob(self, m: dict[str, float], now: float) -> float:
        if m["iat"] <= 0:
            p = FLOOR_P
        else:
            idle = max(0.0, now - m["last_use"])
            p = math.exp(-idle / m["iat"]) * (1.0 - math.exp(-self.horizon_s / m["iat"]))
        leases = max(1.0, m.get("leases", 1.0))
        return p if leases <= 1 else 1.0 - (1.0 - p) ** min(leases, 8)

    def _bytes_of(self, key: OffloadKey) -> float:
        """Bytes for one offloaded block, inferred once from the engine's own geometry."""
        if self.block_tokens <= 0:
            return 0.0
        return self.block_tokens * self.kv_bytes_per_token


def _hints(params: dict[str, Any] | None) -> dict[str, Any]:
    """Read the gateway-stamped `kv_transfer_params["acr"]` blob, tolerating absence."""
    if not params:
        return {}
    blob = params.get(DEFAULT_PARAMS_KEY)
    return blob if isinstance(blob, dict) else {}
