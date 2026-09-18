# 06 — Reference ledger (verified 2026-09-19)

Every external source that our design or our novelty claims depend on, with **what it actually
claims** (read by us, not summarized by a chat export) and **what it removes from our claim set**.

Confidence labels:
- **V** = verified by us against the primary source (abstract/page or installed code).
- **U** = user-supplied via a ChatGPT export; plausible, not independently verified yet.
- **X** = contradicted by what we verified — do not repeat.

## 1. Prior art that constrains our claims

| id | work | what it actually claims (V) | what it kills for ACR | gap it leaves |
| --- | --- | --- | --- | --- |
| **P1** | **Continuum**, arXiv 2511.02230 — *Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live* | GPU-side KV TTL from recompute/reload cost, tool-call wait, queueing impact; paired with FCFS; Llama-3.1 8B/70B, Gemma-3 12B, GLM-4.5 355B; SWE-Bench/BFCL/OpenHand; >8× mean JCT | "dynamic agent-aware KV TTL" is not ours | it retains on GPU; no cross-session lease accounting, no tier demotion/promotion economics |
| **P2** | **TOPAS**, arXiv 2608.25523 | *jointly* decides which agent prefixes to keep in cache and which requests to schedule; scores candidate post-decision states; DAG longest-remaining-path benefit, downstream prefix reuse, movement/preemption cost, aging against starvation; up to 39.8 %/49.4 % mean/p99 JCT on synthetic DAGs, 9.8 % on MetaGPT-SOP | "workflow-aware retention + scheduling jointly" is not ours | single shared cache budget; no host tier, no restore-vs-recompute across memory levels |
| **P3** | **A Policy-Driven Runtime Layer for Agentic LLM Serving**, arXiv 2605.27744 | proposes a third tier *between* framework and engine with **observe / score / predict / act**; includes cross-session KV caching instantiated as **CacheScout** with survival-based eviction and between-step prefetch; architecture + preliminary results | **the middleware framing itself is not novel**, nor is "cross-session eviction + prefetch driven by agent state" | it is an architecture/position contribution; the honest gap is a *deployed* implementation with correctness constraints and measured negative results |
| **P4** | **AAFLOW+**, arXiv 2607.10987 — *Stateful Operator Abstraction with Zero-Copy Distributed KV Cache Orchestration for Multi-Agent Workflows* | makes KV cache a first-class distributed object; operators for **materialization, transfer, fork, composition, eviction**; zero-copy transfer-aware execution; TTFT ↓50.2×, 7.63× compute cost at 16 agents, KV memory ↓1.72–6.10×, throughput ↑7.74× — **from an analytical cost model parameterized by hardware microbenchmarks** | "KV as an object with fork/compose/evict operators" is not ours. Its headline numbers are model-based, not a live engine | **lineage metadata is not mentioned in the abstract** (the user's export claimed it was: **X**), and there is no production engine evaluation |
| **P5** | **PrefixShield**, arXiv 2608.01657 — *Preserving Admission Responsibility in Multi-Tenant LLM Prefix Caches* | names an "admission-responsibility gap": one tenant's newly materialized blocks evict another's reusable state; meters newly materialized full KV blocks as **debt carried across requests**, gates reuse promotion while debt remains, uses projected debt to choose eviction victims; **implemented and evaluated in vLLM**; +9.39 pp victim-cache hit vs LRU, +8.64 vs S3-FIFO under pollution; 4.92 %→84.87 % victim hits | **block-level admission control in vLLM is not ours either** — including the "not everything generated deserves caching" framing we were going to claim | axis is *attribution/accountability across tenants*, not *continuation value of a prefix chain*; no host-tier/restore-bandwidth economics |
| **P6** | **GraniKV**, arXiv 2608.15584 | asymmetric paging granularity: long shared prefix in one contiguous region, per-request tails in small regions; step-wise backend/path selection | "shared prefix and private tail should be treated differently" is not ours | allocation granularity and pools, not residency tiers or value functions |
| **P7** | **"Aborted but Not Forgotten"**, arXiv 2608.15939 | defines **rollback consistency**: clearing the transcript while the serving session retains KV lets the model attend discarded content; same-token/different-cache audit; retained KV alone flipped a typed protected effect in **25 of 63 cells** (tokens absent from the request in all 63); reproduced on HF Transformers cache-reuse and LangGraph time-travel; **transaction-local cache restoration** closes it without a global flush | naming the problem and "rollback must restore attended state" are not ours; **also: the export's claim that our plain APC is exposed is wrong (see §3)** | it offers no multi-tier residency system, and no policy for what to keep when another session legitimately shares the branch |
| **P8** | **CacheBlend** (2405.16444 **U**), **KVCOMM** (2510.12872 **U**), **Prompt Cache** (2311.04934 **U**) | reuse of non-prefix KV with selective recompute / offset correction / schema-defined module assembly | "context assembly beyond strict prefixes" is a whole research area — we explicitly do **not** compete there | n/a: out of scope by decision (correctness bar in a clinical product) |
| **P9** | **SGLang** RadixAttention + **Session Radix Cache** + **HiCache** (docs **U**, not re-read) | radix token tree with automatic fork; `session_id` registers references on reusable leaves, `close_session` releases references into normal eviction (does not delete); separate handling for full-attention prefix, SWA tail, Mamba state; HiRadixTree records per-node residency across L1/L2/L3 | **session-as-lease on a prefix structure is not ours**, and neither is tiered HBM/host/L3 | from the doc text: session refs act as eviction *preference*, not a transactional validity model — needs a full read before we rely on that distinction |
| **P10** | **Mooncake** (2407.00079 **U**), **MemServe** **U**, **NVIDIA Dynamo** KV-aware routing (**U**), **MTDS** (Springer **U**), **LMCache** MP admin API (**U**) | production KV pooling, elastic pools, locality-aware routing cost = uncached prefill + load, tiered hit-probability placement, cache object/prefetch/pin APIs | the data plane and the routing layer are commodity components we may use and must not re-implement | none claimed as ours |
| **P11** | **Alibaba KVCache trace study**, USENIX ATC'25 (**U**) | reuse is diffuse globally but **predictable per request category** | motivates role-conditional priors; also means "use workload category" is a known idea | n/a |

## 2. Verified facts about our own build (primary source = installed code, 2026-09-18/19)

| fact | evidence | consequence |
| --- | --- | --- |
| per-request **`max_offload_tokens`** is honoured (caps how much of a request gets saved) | `…/kv_connector/v1/offloading/scheduler.py:321, 343-353, 592-594` | "selective saving of a prefix" is **not** an engine gap; our policy must not claim it |
| **`max_load_tokens` is absent** in our pinned build | grep over the whole package returns nothing | we can cap writes but **cannot cap reads** → a restore can over-consume PCIe/TTFT; asymmetric control is a real limitation, and either an upstream ask or a custom manager item |
| `kv_transfer_params` reaches `ReqContext`, `kv_load_tiers` honoured | `offloading/scheduler.py:475-487` | gateway can stamp semantics per request without touching engine code |
| out-of-tree `CachePolicy` / `SecondaryTierManager` / `OffloadingSpec` loadable by config | factory docstrings: *"out-of-tree, no vLLM fork/patch required"* | no fork needed for phase 2 |
| built-in CPU manager admits unconditionally | `v1/kv_offload/cpu/manager.py:176` | block-level admission needs a custom manager (or upstream hook) — but note P5: *having* such a hook is not novelty |
| HBM page/prefix granularity = **816 tokens** | boot log (all 4 ranks) + `v1/core/kv_cache_utils.py:612-672` (`hash_block_size` = `prefix_match_unit` else GCD of prefix-cacheable group sizes) | blast radius of a mid-context edit is ~46 MiB/block; and see §4 — granularity changed our experimental ranking |
| `MambaManager` does not set `supports_fine_grained_hash_lookup`; `FullAttentionManager` does | `v1/core/single_type_kv_cache_manager.py:43, 690` | finer `--prefix-match-unit` may be capped by the recurrent group — must measure, not assume |
| engine serves the **stateful** `/v1/responses` API on `:8002` (HTTP 200) | probed live | this, not plain APC, is where server-side retained conversation state can create a stale-attended-state channel |
| our gateway forwards a **closed path list** and excludes `/v1/responses` | `deploy/k8s/ndefy-model-router/main.go:67-72` | product traffic is stateless request/response; exposure exists only via direct `:8002` on the host network |

## 3. Corrections to claims made earlier in this project (mine and the export's)

1. **"Plain APC today is exposed to the rollback-consistency bug" — wrong (my docs/04 R-5, since fixed).**
   Exact-content prefix matching cannot attend tokens that are not in the request: a request for
   `P+Y` will not pull in cached `X`. The paper's failure mode needs a *stateful* path that keeps
   and re-attends inference state across a logical abort (HF per-session cache reuse, LangGraph
   time-travel; in our stack, `/v1/responses`/session APIs — which our gateway does not forward).
   Our real exposures are: direct `:8002` stateful use, **any** ACR-side promotion of superseded
   branches, and any future move to approximate/non-prefix reuse.
2. **"Continuum-style TTL ≈ adaptive value ≈ 5.6× better than LRU" — did not replicate** at the
   engine's real 816-token granularity (docs/05 F-7 now carries both runs). At 816, at a 40 GiB
   host tier, LRU / fixed-TTL / Continuum-style / **our adaptive policy** / oracle were all
   statistically identical, and only LFU improved things 2.2×. The earlier 5.6× was an artifact of
   my 256-token block size. Treat all policy rankings as **granularity-contingent until proven
   otherwise**.
3. **Export claimed AAFLOW+ includes "lineage metadata" in its KV state tuple — not in the abstract**
   (P4). Recorded as **X** because it mattered to our positioning; the full text must be read
   before we treat "lineage" as either taken or open.
4. **Export's "Continuum … MTDS … INFOCOM 2026 adaptive hierarchical KV scheduling"** — MTDS and the
   IEEE item are **U**; do not cite them in anything we publish until read.

## 4. What is actually left for us (stated small on purpose)

Order = our current belief about defensibility, weakest claim last, with the kill test for each.

1. **A negative/measurement result, not a mechanism.** Policy *rankings* for agent KV residency
   depend on engine-internal parameters (prefix-match granularity, block size, tier capacity
   ratio) that papers report as fixed defaults. We have direct evidence the ranking flips
   (LFU vs LRU vs hint-aware, at 256 vs 816 tokens, 40 vs 100 GiB). *Kill test:* reproduce the flip
   on a second engine/config or in a real vLLM run; if it does not reproduce, this is noise, not a
   result.
2. **Restore-vs-recompute economics as a first-class scheduling constraint under a bandwidth
   budget** — `max_offload_tokens` exists, the read-side counterpart does not, and no prior work we
   verified models "promotion consumes the same resource as decode/TTFT". *Kill test:* check
   Continuum/TOPAS/HiCache full text for bandwidth-budgeted promotion; if present, drop the claim.
3. **Deployed-systems evidence**: an implementation on a real hybrid-quantized production engine
   with a real heterogeneous workload, plus the correctness gates (byte-exactness, abort hygiene)
   that performance-only papers skip. This is engineering contribution, and P3/P5 show reviewers
   will not treat the architecture itself as novel.

**Not available to us** (each with its owner): adaptive TTL (P1), joint retention+scheduling (P2),
agent runtime middleware with eviction/prefetch (P3), KV-as-object with fork (P4), vLLM admission
responsibility (P5), prefix/tail granularity (P6), rollback consistency (P7), non-prefix reuse
(P8), session-as-lease and tiered data plane (P9), KV pools and locality routing (P10),
category-conditional reuse prediction (P11). Model-specific tuning is excluded by the user's
explicit direction: the paper must not depend on GDN/MoE/n-gram structure.

## 5. Reading queue before any claim above is used

1. Continuum, TOPAS, PrefixShield, AAFLOW+, Policy-Driven Runtime — **full text**, not abstracts
   (all five decide claim 1–3 above).
2. SGLang `session_radix_cache` + HiCache design docs, and sgl-project/sglang issue #27574 (roadmap
   reportedly heading toward "orchestrator hint → committed page groups → TTL lease" — if true, our
   lease framing is being absorbed by an engine we do not control).
3. Search specifically: bandwidth-aware / interference-aware KV promotion; admission under
   multi-tier; agent branch-aware caching; "rollback" + "prefix cache".
4. Re-verify vLLM upstream for `max_load_tokens`, an `admit()` hook, and any HBM pin/residency API
   — if merged after our pin, adopt rather than invent.
