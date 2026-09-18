"""Residency policies for the tier below HBM.

The method names mirror vLLM's `CachePolicy` ABC
(`vllm/v1/kv_offload/cpu/policies/base.py`: get/insert/remove/touch/evict/clear plus
mark_evictable/mark_non_evictable) so that a policy written and measured here can be adapted to
the engine with a thin wrapper instead of a rewrite. Two deliberate additions:

* `admit()` — vLLM's built-in CPU manager admits every block unconditionally
  (`keys_to_store = [k for k in keys if self._policy.get(k) is None]`), so per-block admission
  has to live in a custom manager. We measure whether it earns its keep before proposing it.
* `probe` — the application-semantic features (lease-aggregated reuse probability, role, tool
  ETA). This is the input the engine cannot derive and the reason the project exists.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .config import CostModel, TierConfig
from .index import PrefixIndex


@dataclass
class Entry:
    key: str
    nbytes: int
    inserted_at: float
    last_touch: float
    uses: int = 1
    expires_at: float = math.inf
    next_use: float = math.inf          # oracle only (filled by the replay harness)


@dataclass
class Probe:
    """Read-only application-semantic view a policy may consult."""

    index: PrefixIndex
    cost: CostModel
    tier: TierConfig
    horizon_s: float = 120.0
    now: float = 0.0
    # session -> last reported tool ETA (gateway/agent hint)
    tool_eta: dict[str, float] = field(default_factory=dict)

    def reuse_prob(self, key: str, priors: dict[str, float] | None = None) -> float:
        """Reuse probability, falling back to a role prior for blocks with no history yet.

        Without the prior a first-time block is priced at the "never seen" floor and can never
        enter the tier, which is how the first version of the adaptive policy locked itself out.
        Category-conditional reuse (arXiv/USENIX ATC'25 KVCache trace study) is the justification
        for a per-role prior rather than a global one.
        """
        info = self.index.blocks.get(key)
        p = self.index.live_reuse_prob(key, self.now, self.horizon_s)
        if priors and info is not None and info.iat_ema is None:
            return max(priors.get(min(info.roles or {"?"}), priors.get("default", p)), p)
        return p

    def chain_bytes(self, key: str) -> int:
        """Bytes of the whole prefix chain of the session that last used this block.

        Residency value is chain-scaled because the engine can only serve a contiguous prefix
        (docs/05 F-1); a per-block TTL prices a 256-token block at 32 ms and expires everything.
        """
        info = self.index.blocks.get(key)
        if not info:
            return 0
        biggest = 0
        for session in info.refs:
            path = self.index.session_path.get(session, ())
            total = sum(self.index.blocks[k].nbytes for k in path)
            biggest = max(biggest, total)
        return biggest

    def chain_reuse_prob(self, keys: Iterable[str], priors: dict[str, float] | None = None) -> float:
        total = 0
        weight = 0
        for key in keys:
            info = self.index.blocks.get(key)
            nbytes = info.nbytes if info else 0
            total += self.reuse_prob(key, priors) * nbytes
            weight += nbytes
        return total / max(1, weight)

    def has_history(self, key: str) -> bool:
        info = self.index.blocks.get(key)
        return bool(info and info.iat_ema is not None)

    def recompute_seconds(self, nbytes: int) -> float:
        return self.cost.recompute_seconds(nbytes / self.cost.kv_bytes_per_token)

    def restore_seconds(self, nbytes: int) -> float:
        return self.tier.restore_seconds(nbytes)

    def has_live_lease(self, key: str) -> bool:
        return bool(self.index.live_leases(key, self.now))

    def session_eta(self, key: str) -> float:
        """How long the sessions leasing this block said they would be away.

        Per block, not global: a shared repo prefix leased by one idle and one busy session is
        worth what the *busy* one says, and a global max would let any long-tooling session
        pin unrelated blocks.
        """
        return max((self.tool_eta.get(s, 0.0) for s in self.index.live_leases(key, self.now)),
                   default=0.0)

    def recompute_seconds_per_byte(self) -> float:
        return 1.0 / (self.cost.prefill_tokens_per_s * self.cost.kv_bytes_per_token)

    def roles(self, key: str) -> set[str]:
        info = self.index.blocks.get(key)
        return info.roles if info else set()


class TierPolicy(ABC):
    """Owns the contents of one tier: what is admitted, what is touched, what goes."""

    name = "base"

    def __init__(self, capacity_bytes: int):
        self.capacity = capacity_bytes
        self.entries: dict[str, Entry] = {}

    # ---- bookkeeping shared by all policies ---------------------------------

    @property
    def occupancy(self) -> int:
        return sum(e.nbytes for e in self.entries.values())

    @property
    def pressure(self) -> float:
        return self.occupancy / max(1, self.capacity)

    def get(self, key: str) -> Entry | None:
        return self.entries.get(key)

    def insert(self, key: str, nbytes: int, now: float, probe: Probe) -> None:
        e = self.entries.get(key)
        if e is None:
            self.entries[key] = Entry(key=key, nbytes=nbytes, inserted_at=now, last_touch=now)
        else:
            e.last_touch = now
            e.uses += 1
        self.entries[key].expires_at = self.expiry(key, nbytes, now, probe)

    def touch(self, keys: list[str], now: float, probe: Probe) -> None:
        for key in keys:
            e = self.entries.get(key)
            if e is not None:
                e.last_touch = now
                e.uses += 1
                e.expires_at = self.expiry(key, e.nbytes, now, probe)

    def remove(self, key: str) -> Entry | None:
        return self.entries.pop(key, None)

    def clear(self) -> None:
        self.entries.clear()

    def expiry(self, key: str, nbytes: int, now: float, probe: Probe) -> float:
        return math.inf

    # ---- decisions ----------------------------------------------------------

    def admit(self, key: str, nbytes: int, now: float, probe: Probe) -> bool:
        """Per-block veto hook (not wired into the current simulator path)."""
        return True

    def admit_chain(self, items: list[tuple[str, int]], now: float,
                    probe: Probe) -> list[tuple[str, int]]:
        """Return the part of a freshly-computed prefix chain worth keeping.

        The unit of cache value is a *chain*, not a block: vLLM can only serve a contiguous
        resident prefix, so admitting block-by-block misprices every block. See docs/05 F-1.
        """
        return items

    @abstractmethod
    def evict(self, needed_bytes: int, protected: set[str], now: float,
              probe: Probe) -> list[str] | None:
        """Free at least `needed_bytes`, never touching `protected`.

        Returns the evicted keys, or None if the request cannot be satisfied — the atomicity
        contract the real ABC requires ("if None is returned, no state changes are made").
        """


class RecencyPolicy(TierPolicy):
    """LRU: evict the least recently touched, expiring nothing."""

    name = "lru"

    def evict(self, needed_bytes, protected, now, probe):
        order = sorted((e for e in self.entries.values() if e.key not in protected),
                       key=lambda e: e.last_touch)
        return _take(order, needed_bytes, self.entries)


class FrequencyPolicy(TierPolicy):
    name = "lfu"

    def evict(self, needed_bytes, protected, now, probe):
        order = sorted((e for e in self.entries.values() if e.key not in protected),
                       key=lambda e: (e.uses, e.last_touch))
        return _take(order, needed_bytes, self.entries)


class FixedTTLPolicy(TierPolicy):
    """The original heuristic: TTL as a function of context size, no reuse signal.

    Calibrated to the proposal it is meant to represent ("70K -> 3 minutes"), i.e. a flat
    retention window scaled by GiB.
    """

    name = "fixed_ttl"

    def __init__(self, capacity_bytes: int, base_s: float = 180.0, per_gib_s: float = 60.0):
        super().__init__(capacity_bytes)
        self.base_s = base_s
        self.per_gib_s = per_gib_s

    def expiry(self, key, nbytes, now, probe):
        return now + self.base_s + self.per_gib_s * (nbytes / 1024**3)

    def evict(self, needed_bytes, protected, now, probe):
        _expire(self.entries, now)
        order = sorted((e for e in self.entries.values() if e.key not in protected),
                       key=lambda e: (e.expires_at, e.last_touch))
        return _take(order, needed_bytes, self.entries)


class ContinuumTTLPolicy(TierPolicy):
    """Continuum-style: TTL from recompute/restore cost, extended by the expected tool gap.

    arXiv 2511.02230 sets a per-state expiration from the cost of losing the state and the
    wait it imposes on others; the agent's own tool-call duration is the revisit estimate. We
    implement that shape with the two terms our own measurement gives us.
    """

    name = "continuum_ttl"

    def __init__(self, capacity_bytes: int, k: float = 3.0, min_s: float = 5.0,
                 max_s: float = 900.0):
        super().__init__(capacity_bytes)
        self.k, self.min_s, self.max_s = k, min_s, max_s

    def expiry(self, key, nbytes, now, probe):
        # Chain-scaled: the thing you lose by expiring a block is the whole session's prefix,
        # not one block (docs/05 F-1).
        cost = probe.recompute_seconds(probe.chain_bytes(key) or nbytes)
        ttl = min(self.max_s, max(self.min_s, self.k * cost))
        return now + ttl + probe.session_eta(key)

    def evict(self, needed_bytes, protected, now, probe):
        _expire(self.entries, now)
        order = sorted((e for e in self.entries.values() if e.key not in protected),
                       key=lambda e: (e.expires_at, e.last_touch))
        return _take(order, needed_bytes, self.entries)


class AdaptiveValuePolicy(TierPolicy):
    """Expected-saved-time residency, admitted against the marginal victim.

        value(b) = P_b · (T_recompute(b) − T_restore(b))  −  λ_mem · T_recompute(b) · P_marginal

    Eviction orders by value; admission is decided per **prefix chain** and asks the only
    question that matters under pressure: does this chain save more time than the best block we
    would have to throw away to hold it? When the tier is not full there is no victim, so
    admission is free — which is what F-2 (docs/05) says the model should predict.

    P_b falls back to a per-role prior for blocks with no history yet; without it a first-time
    block is priced at zero and the tier locks itself against all new sessions.
    """

    name = "adaptive_value"

    def __init__(self, capacity_bytes: int, lambda_mem: float = 0.5,
                 horizon_s: float = 120.0,
                 priors: dict[str, float] | None = None):
        super().__init__(capacity_bytes)
        self.lambda_mem = lambda_mem
        self.horizon_s = horizon_s
        self.priors = priors or {"supervisor": 0.5, "patient": 0.6, "reviewer": 0.2,
                                 "rag": 0.05, "default": 0.1}
        self._marginal_density = 0.0     # value/byte of the best block we have evicted lately

    # -- value ----------------------------------------------------------------

    def value(self, e: Entry, now: float, probe: Probe) -> float:
        """Pure ranking function: must not mutate state, or sorting the same set twice in one
        pass would drift the admission gate within a single eviction."""
        probe.horizon_s = self.horizon_s
        p = probe.reuse_prob(e.key, self.priors)
        benefit = p * (probe.recompute_seconds(e.nbytes) - probe.restore_seconds(e.nbytes))
        displacement = self.lambda_mem * probe.recompute_seconds(e.nbytes) * p * self.pressure
        return benefit - displacement

    def insert(self, key, nbytes, now, probe):
        super().insert(key, nbytes, now, probe)
        # A tier with room to spare has no marginal victim; forget the stale gate so it does not
        # permanently block admissions after pressure subsides.
        if self.pressure < 0.9:
            self._marginal_density = 0.0

    # -- decisions ------------------------------------------------------------

    def admit_chain(self, items, now, probe):
        if not items:
            return []
        if self.pressure < 0.9:
            return items                                   # nothing has to be displaced
        probe.horizon_s = self.horizon_s
        total = sum(nbytes for _, nbytes in items)
        benefit = sum(probe.reuse_prob(key, self.priors)
                      * max(0.0, probe.recompute_seconds(nbytes) - probe.restore_seconds(nbytes))
                      for key, nbytes in items)
        return items if benefit / max(1, total) >= self._marginal_density else []

    def evict(self, needed_bytes, protected, now, probe):
        _expire(self.entries, now)
        order = sorted((e for e in self.entries.values() if e.key not in protected),
                       key=lambda e: self.value(e, now, probe))
        before = {e.key: e for e in order}
        freed = _take(order, needed_bytes, self.entries)
        if freed:
            worst = before[freed[-1]]                      # highest-value victim = marginal cost
            probe.horizon_s = self.horizon_s
            gain = probe.reuse_prob(worst.key, self.priors) * max(
                0.0, probe.recompute_seconds(worst.nbytes) - probe.restore_seconds(worst.nbytes))
            self._marginal_density = gain / max(1, worst.nbytes)
        return freed


class OraclePolicy(TierPolicy):
    """Belady: evict the block whose next use is farthest away.

    An upper bound that is allowed to see the future. Reported so the gap of every real policy
    is measurable regret rather than an unqualified claim of improvement.
    """

    name = "oracle"

    def evict(self, needed_bytes, protected, now, probe):
        order = sorted((e for e in self.entries.values() if e.key not in protected),
                       key=lambda e: (-e.next_use, e.last_touch))
        return _take(order, needed_bytes, self.entries)


def _expire(entries: dict[str, Entry], now: float) -> None:
    for key in [k for k, e in entries.items() if e.expires_at <= now]:
        del entries[key]


def _take(order: list[Entry], needed_bytes: int,
          entries: dict[str, Entry]) -> list[str] | None:
    """Greedy best-effort eviction, in the policy's own preference order.

    Deliberately *not* all-or-nothing: the engine contract is atomic per call, but the real
    manager asks for one chunk at a time, so a turn that cannot fit in a single step still
    lands the part that does. Mirrored faithfully by `evict_atomic()` for engine-side use.
    """
    out: list[str] = []
    freed = 0
    for e in order:
        out.append(e.key)
        freed += e.nbytes
        if freed >= needed_bytes:
            break
    if not out:
        return None
    for key in out:
        del entries[key]
    return out


POLICIES: dict[str, type[TierPolicy]] = {
    p.name: p for p in (RecencyPolicy, FrequencyPolicy, FixedTTLPolicy,
                        ContinuumTTLPolicy, AdaptiveValuePolicy, OraclePolicy)
}


def build(name: str, capacity_bytes: int) -> TierPolicy:
    key = name
    if key not in POLICIES:
        raise KeyError(f"unknown policy {name!r}; known: {sorted(POLICIES)}")
    return POLICIES[key](capacity_bytes)
