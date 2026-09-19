"""A session-TTL eviction policy, written against the deployed ``CachePolicy`` ABC.

This is the *baseline the idea started from*: "mark a session's context alive for N seconds, then
let it age out". Putting it in the same interface as ``AcrValuePolicy`` and the library's own
``lru``/``arc`` makes the comparison exact — same manager, same capacity, same stream, same call
sequence — instead of a simulator with a self-chosen block size (the F-7b lesson).

Differences from the library policies that matter for the comparison:
* ``lru``/``arc`` order by access history only; this orders by **declared lifetime**, per role, so a
  patient block dies early and a coding block survives regardless of recency;
* a TTL policy can be *wrong* in both directions: it drops a block that was about to be reused, and
  it keeps a block nobody will ever ask for again. The replay measures exactly that trade.

The engine constructs policies as ``cls(cache_capacity)``, so tuning arrives from the environment
(``ACR_TTL_CONFIG``) like it does for ``acr.vllm.policy``.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Iterable

try:  # engine-side import; absent when running the repo's own test suite off-engine
    from vllm.v1.kv_offload.base import OffloadKey, ReqContext
    from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy

    HAVE_VLLM = True
except Exception:  # pragma: no cover
    OffloadKey = bytes  # type: ignore[misc,assignment]
    ReqContext = Any   # type: ignore[misc,assignment]
    BlockStatus = Any  # type: ignore[misc,assignment]

    class CachePolicy:  # type: ignore[no-redef]
        def __init__(self, cache_capacity: int) -> None:
            self.cache_capacity = cache_capacity

    HAVE_VLLM = False

DEFAULT_TTL_S = 300.0


class SessionTtlPolicy(CachePolicy):
    """Evict by declared lifetime: expired first, then soonest-to-expire."""

    def __init__(self, cache_capacity: int, params: dict[str, Any] | None = None) -> None:
        super().__init__(cache_capacity)
        cfg = dict(json.loads(os.environ.get("ACR_TTL_CONFIG", "{}") or "{}"))
        cfg.update(params or {})
        self.ttl = {k: float(v) for k, v in cfg.get("ttl_s",
                                                    {"patient": 60.0, "action_judge": 60.0,
                                                     "coding": 3600.0, "supervisor": 3600.0}).items()}
        self.default_ttl = float(cfg.get("default_ttl_s", DEFAULT_TTL_S))
        self.blocks: dict[OffloadKey, BlockStatus] = {}
        self.expires: dict[OffloadKey, float] = {}
        self.role: dict[OffloadKey, str] = {}
        self.pinned: set[OffloadKey] = set()

    # ---- engine callbacks --------------------------------------------------
    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self.blocks.get(key)

    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self.blocks[key] = block
        self.expires.setdefault(key, time.monotonic() + self.default_ttl)

    def remove(self, key: OffloadKey) -> None:
        self.blocks.pop(key, None)
        self.expires.pop(key, None)
        self.role.pop(key, None)
        self.pinned.discard(key)

    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        now = time.monotonic()
        params = getattr(req_context, "kv_transfer_params", None) or {}
        blob = params.get("acr") if isinstance(params.get("acr"), dict) else {}
        role = blob.get("role", "")
        ttl = self.ttl.get(role, self.default_ttl)
        for key in keys:
            if key in self.blocks:
                # A declared lifetime is a promise about the *future*, so each sighting refreshes
                # it; without refresh a long coding session would age out mid-task.
                self.expires[key] = now + ttl
                if role:
                    self.role[key] = role

    def mark_evictable(self, key: OffloadKey) -> None:
        self.pinned.discard(key)

    def mark_non_evictable(self, key: OffloadKey) -> None:
        self.pinned.add(key)

    def clear(self) -> None:
        self.blocks.clear()
        self.expires.clear()
        self.role.clear()
        self.pinned.clear()

    def evict(self, n: int, protected: set[OffloadKey]) -> list[tuple[OffloadKey, BlockStatus]] | None:
        if n == 0:
            return []
        now = time.monotonic()
        live = [k for k in self.blocks if k not in protected and k not in self.pinned]
        if len(live) < n:
            return None                                   # atomic: no state change on failure
        expired = [k for k in live if self.expires[k] <= now]
        rest = sorted((k for k in live if k not in set(expired)), key=lambda k: self.expires[k])
        victims = (expired + rest)[:n]
        out = [(k, self.blocks[k]) for k in victims]
        for k in victims:
            self.remove(k)
        return out
