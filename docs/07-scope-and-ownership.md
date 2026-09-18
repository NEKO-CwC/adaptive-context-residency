# 07 — What we are actually doing: requirements, reuse boundary, ownership

The repository is a **product capability**. The paper is at most one mechanism inside it, and as of
2026-09-19 the honest state is that no mechanism is yet claimable (see
[06 — reference ledger](06-reference-ledger.md) §4). This document separates the three things the
discussion kept blurring: *what the deployment needs*, *what the engine already does*, and *what we
would have to prove to call it research*.

## 1. Requirements (deployment, not research)

| # | requirement | what it really means here | explicitly not |
| --- | --- | --- | --- |
| R1 | long agent sessions should not re-prefill when they return from a tool | keep the *serviceable prefix* somewhere reachable; RAM first | promising N concurrent 1M contexts execute simultaneously |
| R2 | shared prefixes (system+tools+repo) counted and stored once | consume the engine's content-addressed sharing | inventing a second authoritative hash/index |
| R3 | rollback/fork must not strand useful prefix nor resurrect dead state | per-chain lease bookkeeping + invalidation on abort | treating `session_id` as a correctness credential |
| R4 | two tiers: GPU ↔ host RAM | our storage tier measured below break-even (§2) | defaulting to a three-tier design because the literature has one |
| R5 | long-context work must not wreck short-request latency (patient traffic) | admission + promotion budgeted against interference | solving scheduling by growing the KV pool |
| R6 | policy must be model-agnostic | decisions on tokens/bytes/time/leases only | GDN/MoE/n-gram-specific tuning as a design input |
| R7 | switchable, reversible, portable | out-of-tree plugin + one config block; ACR off ⇒ stock policy | maintaining a vLLM fork |

## 2. Reuse boundary — measured on our build, not on the internet

| capability | status in our pinned vLLM | source |
| --- | --- | --- |
| exact-content prefix matching, block hash chain (parent hash + tokens + LoRA/MM/`cache_salt`), full blocks only | **have it** | `v1/core/kv_cache_utils.py:540-570` |
| GPU block allocation and in-use refcounts | **have it** (engine-owned) | KV manager |
| GPU↔RAM async copy, worker, swap kernels | **have it** | `v1/kv_offload/cpu/{spec,manager,gpu_worker,swap_blocks_triton}.py` |
| tiered secondary stores (fs / object / p2p-NIXL) | **have it**, config-selected | `v1/kv_offload/tiering/` |
| pluggable CPU eviction policy, out-of-tree | **have it** — `eviction_policy` + `cache_policy_module_path` | `cpu/policies/{base,factory}.py`, built-ins lru/arc |
| pluggable secondary tier manager / offloading spec, out-of-tree | **have it** | `tiering/factory.py`, `kv_offload/factory.py` |
| per-request **write** cap (`max_offload_tokens`) | **have it** | `offloading/scheduler.py:321,343-353,592-594` |
| per-request **read** cap (`max_load_tokens`) | **absent in our build** | package-wide grep: no hit |
| per-request tier selection (`kv_load_tiers`) | **have it** | `offloading/scheduler.py:475-487` |
| per-request semantic side channel (`kv_transfer_params`) | **have it** | same |
| block-level **admission veto** in the CPU manager | **absent** — admits everything unseen | `cpu/manager.py:176` |
| HBM residency control: pin / protect-until / evict-on-demand | **absent** (no API) | docs/01 §1 |
| transaction-scoped cache invalidation | **absent**; only global `POST /reset_prefix_cache?reset_external=true` | `entrypoints/serve/dev/cache/api_router.py:20-45` |
| KV events for a shadow index | present but off (`--kv-events-config`) | `config/kv_events.py` |
| LMCache / Mooncake / NIXL | installed in the image | `pip list` |

**Consequence for the plan:** phase A is *not* writing a policy. Phase A is turning on the built-in
offload with the built-in LRU/ARC and measuring how much of the problem the library already solves.
Two of our own results (docs/05 F-6, F-7) say the answer may be "not much, on this workload" — which
is exactly the counterfactual we need before claiming anything.

## 3. What we build, and whether it is research

| module | ours? | research value | first deliverable |
| --- | --- | --- | --- |
| Backend adapter (vLLM offload/config, rollback to stock) | yes, trivially | none — plumbing | LRU and ARC both run, metrics flow, kill-switch verified |
| Application signal adapter (gateway → `kv_transfer_params`) | yes | none by itself (P1, P3); **required input** for any claim | real tool-start/stop, session close, rollback, trusted priority; never client-asserted |
| Session/branch → prefix-chain map (leases) | yes | none as a concept (P4, P9); ours is the *combination with tiering + correctness gates* | invalidation-on-abort wired to R3; cancel stale prefetch plans |
| Cost model + calibration (recompute curve, restore bandwidth, interference) | yes | supports claim 1 (granularity/capacity contingency) | close **C-1**, measure **M-1**; publish curves, not constants |
| **Admission / residency decision** | yes | **the only candidate contribution**, and currently unclaimed (P5 owns vLLM admission; P1/P2 own TTL+workflow) | mechanism defined as a *replacement* unit with same-capacity/same-signal baselines |
| Evaluation harness, traces, replay, safety regressions | yes | a dataset + methodology contribution at best | fork/rollback-heavy workload; correctness regression; information-fair baselines |

## 4. The attribution discipline (this is the part that keeps us honest)

Any reported improvement must be split into three separately-measured effects:

```
E_tier    = stock offload + LRU/ARC, no signals            (library benefit)
E_signal  = same policy + application signals              (information benefit)
E_mech    = same signals/capacity/workload, our decision   (mechanism benefit)
```

and compared under **identical** capacity, workload, and information:

| arm | storage | signals given | purpose |
| --- | --- | --- | --- |
| stock LRU / ARC | same | engine-only | E_tier |
| Continuum-style TTL | same | same as ours | information-fair baseline (label "our re-implementation of its shape", never the paper's system) |
| PrefixShield-style admission | same | same | the closest published mechanism — we must include it, not just LRU |
| **ours** | same | same | E_mech |
| Belady | same | future knowledge | headroom only; **not** a p95/TTFT upper bound unless the objective is exactly what Belady optimizes |

Rule: paper claims are allowed to cite **E_mech only**. If E_mech is inside the noise, the honest
output is a technical report and a production system — not a paper.

## 5. Phase order (replaces docs/03's phase numbering for the policy track)

| phase | content | needs engine window? | exit criterion |
| --- | --- | --- | --- |
| **A** | stock GPU↔RAM offload, LRU + ARC, real multi-session load; G-1 byte-exactness and G-3 stability gates ride along | yes (one window) | E_tier measured; correctness gates pass or the tier is abandoned |
| **B** | signals only: gateway stamps role/session/tool/rollback into `kv_transfer_params`; consumption stays simple (e.g. frequency/recency variants) | no (gateway is service-plane) | E_signal measured separately |
| **C** | one mechanism, swapped in with everything else fixed; fork/rollback-heavy benchmark **designed first** | no (simulator) then yes (confirm) | E_mech > noise against an information-fair baseline, with the *reason* explained |
| **D** | decide: paper vs technical report vs "the library plus correct sizing was enough" | — | written verdict with the tables that justify it |

Phase C's benchmark is the real work: a workload with shared prefixes, speculative tool branches,
rollbacks, and aborts, where pure recency demonstrably keeps dead tails and global flush
demonstrably wastes the shared prefix. Until that artifact exists and shows the gap, there is no
paper — and if it shows no gap, we have saved a submission cycle and shipped a better engine config.

## 6. Standing decisions encoded here

1. **No non-prefix/approximate KV reuse** in this product (CacheBlend/KVCOMM class). Our correctness
   bar is "a wrong patient simulation is worse than a slow one". ACR optimizes *residence*, not
   semantic reconstruction.
2. **No persisted tier for latency, and never for patient traffic** (below break-even measured here;
   KV-at-rest is PHI-bearing). Durability-only, opt-in, encrypted, per-tenant salted.
3. **No vLLM fork.** If a hook is missing we add a custom manager class or an upstream PR.
4. **No model-specific tuning as a claim.** Hybrid GDN/MTP/`inc` is our *deployment and evaluation
   platform*, not our mechanism (user direction, 2026-09-19).
