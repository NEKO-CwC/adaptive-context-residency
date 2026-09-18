# 02 — Prior art and where the defensible delta is

Checked 2026-09-18 against arXiv (abstracts read directly, ids given). **The naive version of this
idea is already taken.** Recording that plainly is more useful than pretending otherwise, and it
is what makes the remaining space visible.

## 1. What exists

| work | what it does | overlap with our original pitch |
| --- | --- | --- |
| **Continuum** (arXiv 2511.02230) | Agent-aware **GPU-side KV TTL**: expire based on recompute/restore cost + added queueing risk, paired with FCFS. Llama-3.1 8B/70B, Gemma-3 12B, GLM-4.5 355B; SWE-Bench/BFCL/OpenHand; **>8× mean JCT** | kills "dynamic TTL from agent gaps" as a contribution |
| **TOPAS** (arXiv 2608.25523) | **Joint** prefix-retention + request-scheduling decisions under one shared KV budget; scores DAG critical-path benefit, downstream prefix reuse, prefix movement/preemption cost; aging for starvation. 39.8 %/49.4 % mean/p99 JCT on synthetic DAGs, 9.8 % on MetaGPT-SOP | kills "workflow-aware retention + scheduling jointly" |
| **GraniKV** (arXiv 2608.15584) | Asymmetric paging granularity: long shared opening as one region, short per-request tails as small regions; step-wise execution-path selection | occupies "shared prefix vs tail" structuring (granularity, not tier value) |
| **Dynamic HBM Repartitioning for Multi-Turn MoE Serving** (2609.13537) | Reclaims expert memory as cache capacity when preserving multi-turn state beats removal/preemption | adjacent: HBM budget shaping for sessions |
| **ReCache** (2608.19662), **SemPIC** (2607.28069), **KVShareArena** (2609.10266) | Non-prefix reuse: tool/skill-state caching across reordered schemas; learned position-independent KV; benchmark for repaired shared state | occupies the "drop strict prefix matching" extension we had listed |
| **SGLang session radix cache + HiCache** | Session-reference-aware eviction; GPU L1 → host L2 → SSD/Mooncake/3FS L3 with prefetch (engineering, shipped) | the data plane; we use it, don't reinvent it |
| **LMCache / Mooncake / Dynamo** | Engine-independent KV store w/ admin API (objects, delete, prefetch, pin); KV-centric disaggregation; KV-locality-aware routing with TTL-predicted cache state | components + routing |
| **MTDS** (Complex & Intelligent Systems) | Multi-tier HBM→DRAM→SSD with predicted future hit probability, adaptive eviction, restore-vs-recompute comparison | the closest to "tiering + prediction" as a package |
| **Alibaba KVCache trace study** (USENIX ATC'25) | Reuse behaviour is diffuse globally but **predictable per request category** | supports category-conditional predictors |
| **"Aborted but Not Forgotten"** (2608.15939) | **Rollback consistency**: a logical abort that clears the transcript does not clear retained KV; the model keeps attending discarded content. Retained KV alone flipped a typed protected effect in **25 of 63 audited cells** while attacker tokens were absent from the request in all 63; reproduced on HF Transformers cache-reuse and LangGraph time-travel; **transaction-local cache restoration** closes it without a global flush | (not overlap — a hazard we must handle, and an opening, see §3) |

## 2. What is *not* occupied, in what we checked

Nothing here is claimed as novel until it is defended by measurement; the point is that these are
the gaps left after Continuum/TOPAS/MTDS.

1. **Hybrid-model residency economics.** Every item above assumes context state scales with token
   count. On a GDN/Mamba-hybrid model (our Qwen3.8-Flash-Next; also Qwen3-Next/Jamba/Zamba families)
   most layers carry an **O(1) recurrent state per checkpoint** alongside O(n) attention KV. The two
   have different sharing (state can't be shared by suffixes the way KV blocks can), different
   restore cost, and different invalidation. Nobody in the list above models the split. LMCache
   has started shipping it as "opaque pages", i.e. the industry is one step behind the policy
   question. **This is our strongest claim, and our box is one of very few places it can be
   measured end-to-end on a production-shaped workload.**
2. **Residency as the enforcement point for attended-state integrity.** The rollback paper shows
   correctness, not performance, is the missing guarantee, and proposes transaction-local restore.
   A component that already owns the per-session prefix DAG and tier placement is exactly where
   that guarantee should live — promote, demote, *and invalidate-on-abort*. The KV-tier literature
   treats correctness as out of scope; the agent-framework literature can't fix it because it
   doesn't own the cache. Our own stack has three live abort paths that would need it (Claude Code
   undo/ESC, Core's bounded per-turn repair re-calls, Dify node retries).
3. **A tier-value sizing law with a measured break-even.** `BW* = bytes_per_token × prefill_rate`.
   For our engine: 56.4 KiB × 7.9K tok/s ≈ **445 MB/s**. Measured /data = 413–432 MB/s, i.e. the
   storage tier sits *just under* the line, so on this machine a four-tier design is strictly
   worse than a two-tier one. Small, but it is a real, checkable, and generalizable design rule
   that decides whether L3 belongs in a deployment — and it explains why single-node
   HiCache-style tiers often disappoint.
4. **Heterogeneous mixed-role production trace** (short high-priority patient + medium reviewer +
   100K–1M agentic coding, one engine, prefix sharing, preemption/interference measurement) as an
   artifact. Most papers evaluate synthetic DAGs or open-agent benchmarks; the category-conditional
   predictability result (ATC'25) says a *labeled, heterogeneous, real* trace is the useful object.

**Downgraded to ablation, not headline:** the shared-prefix lease aggregation
`P_b = 1 − ∏_s(1 − p_s)` and generic app-signal-driven TTL. Both are near TOPAS/GraniKV/Continuum
territory; we keep them as mechanisms and score them as ablations.

## 3. Research questions that survive

- **RQ1 (primary).** On a hybrid-attention model, does a residency policy that models attention KV
  and recurrent state separately beat state-of-the-art tier policies (Continuum-style TTL, MTDS-style
  hit-probability placement, LRU) at equal SLO, and by how much does it change with the KV/state
  ratio across model families?
- **RQ2.** Does transaction-scoped invalidation close the rollback-consistency channel at
  acceptable cost (partial rebuild vs global flush), and does it change how much residency one
  is willing to buy?
- **RQ3.** Given measured bandwidths, when is a storage tier worth building at all — validated
  against the break-even law on ≥3 storage classes (NVMe / 3FS-like / remote RAM pool)?
- **RQ4.** With application-semantic signals at the gateway, how much of the policy is achievable
  **without any HBM control** (i.e. is a pin API worth upstreaming)? Continuum needs GPU-side
  control; our own §1 of docs/01 argues most of the win doesn't.

## 4. Publication bar, honestly

| artifact | venue class |
| --- | --- |
| working ACR + our box's numbers | engineering writeup / blog / open-source |
| + trace-driven replay, 4 baselines incl. oracle, ≥3 workloads, 20–30 % p95/JCT over best baseline | arXiv + systems workshop |
| + hybrid-state model validated on ≥2 model families, online experiments, ablations, reproducible artifact | competitive main-conference submission (ATC/SoCC/EuroSys class) |

The gap between row 2 and row 3 is one more GPU, not one more idea. What is genuinely scarce here
is not VRAM: it is **a second model family with a different KV/state ratio**, and the discipline to
report where the policy does not help (e.g. the storage tier).

## 5. Re-verification duties before any novelty claim is written down

1. Read TOPAS, MTDS, GraniKV, Continuum in full (not abstracts) and diff against RQ1/RQ3.
2. Search specifically for hybrid/mamba KV-cache *tiering* (terms: "gated delta net cache",
   "mamba state checkpoint offload", "hybrid attention KV offload", "recurrent state paging").
3. Check whether vLLM upstream has merged an HBM pin/residency API or an `admit()` hook since our
   build (if yes, adopt instead of inventing).
4. Re-run the search at submission time — this area is moving at roughly one paper per week in the
   recent listing we pulled.
