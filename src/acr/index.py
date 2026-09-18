"""Prefix block DAG + session leases.

Two modeling decisions carry most of the design:

1. **The cache object is a prefix block; the session is a lease on it.** Team-mode coding
   sessions share a system/tool/repo prefix; if residency were tracked per session, a shared
   320K-token root would be counted (and priced) once per session and would look far more
   valuable — and far more expensive — than it is.

2. **Reuse probability is aggregated over leases, not per session.** A block leased by three
   sessions with independent 30 % reuse probability has P_b = 1 - 0.7^3 = 66 %. That is what
   keeps genuinely shared prefixes sticky in a principled way, and what makes a shared root
   whose only live lease is a dying session evictable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import CostModel


@dataclass
class BlockInfo:
    key: str
    tokens: int
    nbytes: int
    depth: int
    shared: bool
    created_at: float = 0.0
    last_use: float = 0.0
    uses: int = 0
    iat_ema: float | None = None
    refs: set[str] = field(default_factory=set)
    roles: set[str] = field(default_factory=set)
    tier: str | None = None          # where the simulator currently holds it

    def observe_use(self, now: float) -> None:
        if self.last_use:
            gap = max(0.0, now - self.last_use)
            self.iat_ema = gap if self.iat_ema is None else 0.7 * self.iat_ema + 0.3 * gap
        self.uses += 1
        self.last_use = now


class PrefixIndex:
    """Content-addressed block chains per session, with a shared-root namespace."""

    def __init__(self, block_tokens: int, cost: CostModel):
        self.block_tokens = max(1, block_tokens)
        self.cost = cost
        self.blocks: dict[str, BlockInfo] = {}
        self.session_path: dict[str, list[str]] = {}
        self.session_role: dict[str, str] = {}
        self.session_last_seen: dict[str, float] = {}
        self.invalidations: list[str] = []   # keys that must not be reused after an abort

    # ---- chain construction -------------------------------------------------

    def chain(self, session_id: str, prompt_tokens: int, shared_prefix_tokens: int,
              root_id: str = "root") -> list[tuple[str, int, bool]]:
        """(key, tokens, is_shared) in prefix order for one request's prompt.

        A partial trailing block gets a length-qualified key, matching how a real engine
        re-hashes a block when it is completed; the superseded partial key is forgotten by
        `touch_path` so stale blocks do not accumulate leases forever.
        """
        bt = self.block_tokens
        shared = min(shared_prefix_tokens, prompt_tokens)
        private = prompt_tokens - shared
        parts: list[tuple[str, int, bool]] = []
        # Block keys are positional, so a partial trailing block keeps its identity as it
        # fills. A length-qualified key would invent a new block every turn and make the
        # simulator charge recompute for work the engine never redoes.
        full_shared, rem_shared = divmod(shared, bt)
        for i in range(full_shared):
            parts.append((f"{root_id}:b{i}", bt, True))
        if rem_shared:
            parts.append((f"{root_id}:b{full_shared}", rem_shared, True))
        base = len(parts)
        full_priv, rem_priv = divmod(private, bt)
        for i in range(full_priv):
            parts.append((f"{session_id}:p{base + i}", bt, False))
        if rem_priv:
            parts.append((f"{session_id}:p{base + full_priv}", rem_priv, False))
        return parts

    def touch_path(self, session_id: str, role: str, prompt_tokens: int,
                   shared_prefix_tokens: int, now: float,
                   aborted: bool = False) -> list[str]:
        """Register/refresh the chain this request needs; return keys in prefix order."""
        # One shared-prefix namespace per role family. In this workload each role has exactly
        # one canonical prefix (system+tools+repo for coding, persona for patients), so the role
        # is the identity; a real deployment keys this by the hash of the prefix itself.
        parts = self.chain(session_id, prompt_tokens, shared_prefix_tokens,
                           root_id=f"root:{role}")
        keys = [k for k, _, _ in parts]
        old = self.session_path.get(session_id, [])
        if aborted and len(keys) < len(old):
            # Logical rollback: the model must not keep attending discarded state.
            self.invalidations.extend(old[len(keys):])
        stale = [k for k in old if k not in keys]
        self.session_path[session_id] = keys
        self.session_role[session_id] = role
        self.session_last_seen[session_id] = now
        for depth, (key, tokens, shared) in enumerate(parts):
            info = self.blocks.get(key)
            if info is None:
                info = self.blocks[key] = BlockInfo(
                    key=key, tokens=tokens,
                    nbytes=int(tokens * self.cost.kv_bytes_per_token),
                    depth=depth, shared=shared, created_at=now)
            elif tokens > info.tokens:               # partial block filled in as the grew
                info.tokens = tokens
                info.nbytes = int(tokens * self.cost.kv_bytes_per_token)
            info.refs.add(session_id)
            info.roles.add(role)
            info.observe_use(now)
        for key in stale:
            info = self.blocks.get(key)
            if info is not None:
                info.refs.discard(session_id)
        return keys

    # ---- lease queries ------------------------------------------------------

    def release_session(self, session_id: str) -> None:
        """Session closed: leases drop, KV survives into the normal eviction queue."""
        for key in self.session_path.pop(session_id, []):
            self.blocks[key].refs.discard(session_id)
        self.session_last_seen.pop(session_id, None)

    def live_leases(self, key: str, now: float, stale_after: float = 900.0) -> list[str]:
        info = self.blocks.get(key)
        if not info:
            return []
        return [s for s in info.refs
                if now - self.session_last_seen.get(s, math.inf) <= stale_after]

    def live_reuse_prob(self, key: str, now: float, horizon: float) -> float:
        """P_b = 1 - Π_lease (1 - p_s), with p_s from the block's own inter-arrival hazard.

        Single hazard per block for now: we have per-session inter-arrival EWMA, not trained
        per-category predictors. The aggregation is the part worth testing, not the estimator.
        """
        info = self.blocks.get(key)
        if info is None:
            return 0.0
        if info.iat_ema is None or info.iat_ema <= 0:
            p = 0.05                        # never reused: unproven, so unremarkable
        else:
            idle = max(0.0, now - info.last_use)
            p = math.exp(-idle / info.iat_ema) * (1.0 - math.exp(-horizon / info.iat_ema))
        leases = max(1, len(self.live_leases(key, now)))
        return p if leases <= 1 else 1.0 - (1.0 - p) ** min(leases, 8)
