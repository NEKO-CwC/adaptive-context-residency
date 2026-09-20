# 09 — KV L2 candidate matrix: native OffloadingConnector vs SimpleCPUOffloadConnector vs FlexKV

Purpose: a code-level decision document for the GPU→host (L2) storage layer.
Three columns, every cell cited to a file:line in **our installed image** or to the
cloned FlexKV repo, or to an explicit availability/version check. Read-only task:
no container was touched, no engine restarted, nothing patched.

Legend for citations:
- `[img]` = verified in OUR installed tree, `ncu-supervisor-chat` at
  `/usr/local/lib/python3.12/dist-packages/vllm/…` (image
  `vllm/vllm-openai@sha256:0aea3024…`, `vllm.__version__` reports
  `0.1.dev20073+g8e685d198`; brief states this fork ≈ upstream `6e448d0` / v0.27.1, 2026-08-11,
  + 4 carried patches). Path below shortened to `vllm/…`.
- `[fkv]` = FlexKV `main` @ `738ddc1` (2026-09-20), cloned to `/tmp/FlexKV`. **NOT in our
  image** (`flexkv` import → `ModuleNotFoundError` `[img]`), and NOT version-matched to our
  vLLM. Behaviour here is upstream-main-only and is flagged as such.
- `[meas]` = our own measurement this week (docs/08 N-series), store-works/load-zero.
- `[win]` = window-required: needs a production restart to establish; **not** restarted.

---

## The contradiction I must flag before the matrix

The tasking brief states our root cause as: *with MTP on, the all-groups eagle fallback
(`offloading/scheduler.py:~230`) flags every group eagle, then the volatile-tail pop
collapses the whole-request hit to 0 (`~:807-810`)*, and cites `#52807 (da8ec28, 09-03)` /
`#52771 (4a806d0, 09-07)`.

Two things are wrong in that framing, and our own docs already corrected the first:

1. **MTP is not the executed trigger — the ring is.** `docs/08-phase-a-log.md` N10 (offline
   replay, EXECUTED on the pristine image, `experiments/connector_lookup_veto_probe.py`)
   refuted both the chunk-width hypothesis and the MTP framing: the non-MTP case
   `"ring-sparse non-MTP"` **also** returns hard 0, halting on the exact
   `max_hit_size_tokens - num_computed < tokens_per_chunk → return 0`
   (`offloading/scheduler.py:748-753` `[img]`). With MTP on and the ring *all-complete*
   (`a-SET3`) it returns a nonzero 84,856. The single discriminator is **the ring
   (`CircularBufferSpec`) being a member of `_lookup_groups`**: its
   `_maximal_prefix_lookup` pins one resident chunk, capping `max_hit` below the next
   group's `tokens_per_chunk`. Excluding the ring restores 84,864 (case b / b-SET3).
   I verified the ring is in `_lookup_groups` in our tree: `_lookup_groups =
   full_attention_groups + _sliding_window_groups` (`offloading/scheduler.py:522 [img]`)
   with **no `prefix_cacheable` filter**, and `get_sliding_window_size_in_chunks` has no
   `CircularBufferSpec` branch — it falls through to `assert isinstance(spec,
   FullAttentionSpec)` (`:125 [img]`), i.e. the stock connector cannot even classify the
   ring without our patch-05 shim. This is precisely upstream issue **#54414**
   (open): *"recent-window state groups can never participate in restores … Any lookup
   chain that includes it collapses the min-across-groups hit window to zero — so one such
   group makes the whole model un-restorable."* [web: vllm#54414]

2. **The brief swapped the two PR numbers against their commits/dates.** Verified from
   the GitHub pages: `#52771` = squash `da8ec28`, merged **2026-09-03**, *"Do not let a
   recurrent group's unhashed block truncate the load boundary"* (touches
   `update_state_after_alloc`, changes the boundary scan start, renames
   `num_locally_computed_gpu_blocks → load_start_gpu_block_idx`); `#52807` = merge commit
   `4a806d0`, merged **2026-09-07**, *"OffloadingConnector: stop zeroing offload hits under
   MTP/EAGLE spec decode"* — which is the one whose three faults match our text (leave the
   drafter-group set **empty** instead of flagging all; widen the volatile-tail pop for
   **all** drafter groups, not just SWA; free the tail at finish; validation moves
   `kv_offload_load_bytes_total` 0→1.10 GB). [web: vllm#52771, vllm#52807]
   The commit hashes and dates in the brief are right; the **number↔hash pairing is crossed**.
   I use the corrected mapping throughout.

I confirmed both faults are still **unfixed in our installed tree**:
`eagle_groups = set(range(len(kv_cache_config.kv_cache_groups)))` (`:230-231 [img]`) and the
SWA-only pop widen `if is_eagle_unverified and sliding_window_size_in_chunks is not None:`
(`:762 [img]`). Both PRs postdate the 08-11 fork, so neither is present. [img]

**Why this matters for the choice:** the fix our N10 says actually flips the native
connector's 0→84,864 is *drop the ring from `_lookup_groups`* — which is exactly what
**upstream issue #54414 proposes as `load_skip_groups`** (still open, unmerged) and exactly
what the **`SimpleCPUOffloadConnector` already does by construction** because it routes the
lookup through the HMA coordinator, whose `find_longest_cache_hit` filters every group by
`prefix_cacheable` and `CircularBufferSpec.prefix_cacheable == False`. That single fact is
the hinge of this whole document.

---

## Matrix

| Dimension | 1. `OffloadingConnector` (native) | 2. `SimpleCPUOffloadConnector` | 3. `FlexKV` (`FlexKVConnectorV1`) |
|---|---|---|---|
| **Non-uniform group block sizes (816/16/8)?** | Aware of groups (`tokens_per_block` per group) but reconciles to ONE scalar `max_hit_size_tokens` with no per-group partial (`_lookup_complete_chunks`, `scheduler.py:697-868 [img]`); `supports_partial_tail` is **False** here because it needs uniform `group_block_sizes` AND no eagle group (`:279-290 [img]`). | Yes, cleanly. Derives `scheduler_block_size` = LCM, keeps each group's own `block_size`; asserts `num_external_tokens % g_block_size == 0` **per group** (`manager.py:108-116, 357-366 [img]`) and skips `is_null` padding blocks (`:389-390`). | Single global `self.block_size = tokens_per_block` for the whole cache (`adapter:253, 439-444 [fkv]`); supports heterogeneous **head_dim / compress-ratio layer groups on one block grid** (Gemma4 256-vs-512, DSv4 C4/C128, `docs/gemma4_support.md`, `docs/flexkv_config_reference`), not heterogeneous block sizes. |
| **Recent-window / circular-buffer group without zeroing the hit?** | **No.** Ring lands in `_lookup_groups` (`:522 [img]`), one pinned chunk → `return 0` (`:748-753, 807-810 [img]`). Confirmed by our N10 executed replay `[meas]`; upstream #54414 open. | **By construction it cannot collapse.** Coordinator excludes non-`prefix_cacheable` groups from `attention_groups` (`kv_cache_coordinator.py:693 [img]`) and `CircularBufferSpec.prefix_cacheable == False` (`kv_cache_interface.py:626-627 [img]`) → the ring is **not** in the min-across-groups bound. Same outcome as our N10 "exclude the ring → 84,864". Ring state is simply not restored (recomputed) — matches #54414's proposed `load_skip_groups`. | No representation for a recurrent/recent-window group at all: zero `mamba|recurrent|gdn|linear-attention|circular` references in the entire `flexkv/` package `[fkv]`. It models attention/SWA/compress-indexer pages only. |
| **Per-group partial restore, or all-or-nothing?** | All-or-nothing: one scalar `num_hit_tokens`; any group landing under a chunk returns 0 (`:751-753, 792-793, 807-810 [img]`). `find_longest_cache_hit_per_group` (`kv_cache_coordinator.py:917 [img]`) exists but the offloading path **does not call it**. | Combined hit is `min` across the *prefix-cacheable* groups (`find_longest_cache_hit`, `:783-915 [img]`), but the **transfer is built per-group** so attention blocks and GDN state blocks are DMA'd independently once a boundary is agreed (`manager.py:347-410 [img]`). | Single `matched_mask.sum()` token count over one block grid (`adapter:453-458 [fkv]`); layerwise restore per registered group (`register_to_server(..., gpu_layouts=[...])`, `adapter:1064-1072 [fkv]`). |
| **Who controls GPU↔CPU DMA; sync/async; per-step load budget?** | Engine scheduler drives store/load through the connector's worker; has both sync & async lookup (`LOOKUP_SYNC_DELAY`/`LOOKUP_ASYNC_DELAY` metrics `[img]`). No `max_load_tokens` knob in our tree. | **Connector owns it**: dedicated low-priority CUDA `load_stream`/`store_stream`, pinned host via `cudaHostRegister`, batched `cudaMemcpyAsync` (`worker.py:54-162, cuda_mem_ops.py:23,161 [img]`). Fully async, driven from `get_finished()`. No per-step token budget knob found. | **FlexKV's own `TransferEngine`**, off the scheduler side (`build_connector_meta`→`launch_tasks`, `query_finished_task`; `flexkv_connector.py:82-99 [img]`, README "fully handled asynchronously"). `cudaMemcpy2DAsync` strided fast path, CE memcpy modes (`config_reference [fkv]`). No `max_load_tokens` budget `[fkv]`. |
| **Host memory model; bytes/token; per-rank copy or canonical; sizing** | Own offload pool sized by `cpu_bytes_to_use` (blocks via manager). Per-rank (TP) pages; canonical single-copy only under a narrow MLA/`canonical_layout` gate (`config.py:138-183 [img]`). | Own pinned CPU buffer; `cpu_bytes_to_use` is **server-wide**, divided `// world_size`, overridable by `cpu_bytes_to_use_per_rank` (`connector:74-90 [img]`). Sizing in GB→blocks via `_derive_cpu_config` scaling GPU blocks by the CPU/GPU byte ratio (`manager.py:186-222 [img]`). | `FLEXKV_CPU_CACHE_GB` (default 16) in GB `[fkv]`. For TP it is **not N× replicated**: `FLEXKV_MLA_D2H_MODE` offers `sharded` (each GPU writes 1/N → one canonical KV, **MLA-only**), `rank_rotate`, `rank0_only`, `all_write` (N×) (`config_reference:121 [fkv]`). HugePage host-buffer option; optional `disk`/SSD L3 (`ssd_cache_gb`) in the same component. |
| **Admission + eviction; pluggable from OUTSIDE?** | Both. Eviction is a **module-path plug-in**: `cache_policy_module_path` + `eviction_policy`, documented *"out-of-tree … no vLLM fork/patch required"* (`kv_offload/cpu/policies/factory.py:60-79`, cited docs/01:46). Admission: `store_threshold` reuse-counter + per-request `RequestOffloadingContext(BLOCK_LEVEL|REQUEST_LEVEL)` (`docs/01:85-92,48 [meas-adjacent]`). | **None pluggable.** Eviction is the fixed CPU `BlockPool` LRU free-queue (`manager.py:254, 279, 310, 396-397 [img]`); extra_config keys are only `cpu_bytes_to_use[_per_rank]`, `lazy_offload`, `kv_offload_backend=disk`, `disk_*`, `use_page_cache` (`connector:74-121 [img]`). No policy module path, no secondary tiers, no admission hook. | **Compiled-in C++**: `enum class EvictionPolicy { LRU, LFU, FIFO, MRU, FILO, SLRU }` + `create_eviction_strategy` factory (`csrc/eviction_strategy.h:14-104 [fkv]`). Selected by env/config string, plus knobs `FLEXKV_EVICT_RATIO`, `hit_reward_seconds`, `slru_protected_threshold`. Adding an ACR policy = edit C++ and rebuild → **fork**. Admission is store-on-finish: `should_put = request.is_finished() and (normal_finish or requested_abort_offload)` (`adapter:567 [fkv]`). |
| **Pinning / non-evictable per-block?** | Yes: `CachePolicy.mark_non_evictable` / `mark_evictable` on the policy interface (`docs/01:46 [meas-adjacent]`). | Only transient `touch()` pins during in-flight load/store (`manager.py:279, 396-397 [img]`); no user-facing persistent pin API. | Internal `full_lock`/`swa_lock` ref-counts and SLRU protected segment (`cache/radixtree.py:133-140, 697, 767 [fkv]`) — tied to in-flight loads and hit counts, not a caller-facing per-session pin API. |
| **Session/role-aware residency (per-request metadata hook)?** | **Strong.** `kv_transfer_params` → `Request.kv_transfer_params` → `ReqContext` (`offloading/scheduler.py:475-487`, docs/01:48) already drives a per-request `TierFilter` (`kv_load_tiers` medium/locality). ACR sees per-request metadata natively. | **None.** Lookup keys only on `request.block_hashes` (`manager.py:261-271 [img]`); no `kv_transfer_params`, no namespace read anywhere in the simple package `[img]`. | **Yes, namespace isolation.** `_extract_namespace` reads `lora_request.lora_name`, `cache_salt`, and `request.namespace_info` into a hierarchical cache-isolation key (`adapter:375-421 [fkv]`). Caveat: `namespace_info` is not a stock vLLM `Request` field — plumbing it in is `[win]`/source-required. |
| **What breaks when MTP is on (spec lookahead / volatile tail)?** | Native #52807 fixes a real MTP zeroing (all-groups-eagle fallback `:230 [img]` + SWA-only pop widen `:762 [img]`); both present-unfixed → MTP is a *second* reason the native path under-serves even after the ring is excluded. | Coordinator's all-groups-eagle fallback exists (`kv_cache_coordinator.py:110-111 [img]`) **but** it explicitly does NOT apply the eagle last-block drop to `MambaSpec` (`:851-853, 853 [img]`) and the FA drop only trims one scheduler block, not to 0. MTP multi-module lookahead is guarded (`scheduler_block_size 816 ≥ num_spec_tokens 4`, `:124-133 [img]`). No executed MTP collapse on this path. | Unknown — cannot enable on this engine (see next rows), so MTP behaviour on hybrid is `[win]`/unverifiable here. |
| **Observability we can gate promotion on (exact names)** | Full Prometheus: `vllm:kv_offload_{load,store}_bytes` (Counters), `vllm:kv_offload_{load,store}_size` (Histograms), `…_time`, `vllm:kv_offload_{lookup_sync,lookup_async}_delay_seconds`, `vllm:kv_offload_allocation_failure`, plus deprecated `vllm:kv_offload_total_bytes{transfer_type}` (`offloading/metrics.py:25-148 [img]`). This is exactly the series we gated `[meas]`. | **None.** No Prometheus counters/gauges in `v1/simple_kv_offload/` — only internal `_load_event_counter`/`_store_event_counter` (`manager.py:177-179, 905 [img]`). It emits `KVCacheEvent`s via `take_events` (`connector:298-301 [img]`) but no offload byte/hit metric → **promotion cannot be gated on a load-direction counter today.** | Own stats: `flexkv_stats.record_get/…` on every match (`adapter:364 [fkv]`), `get_kv_connector_stats()` plumbed through the glue (`flexkv_connector.py:254 [img]`), and a `monitoring/` dir `[fkv]`. |
| **Install / ops delta + rollback** | Already in image, 0 new packages; 4 carried patches + revised patch-05 (drop ring from lookup). Rollback = relaunch without `--kv-transfer-config` / drop patch. | Already in image, **0 extra packages, 0 extra processes**; config is one `--kv-transfer-config` blob; optional disk backend needs a path. Lowest ops delta of the three. Rollback = remove the config. | Heavy: `bash build.sh` → CMake + **CUDA** compile of `csrc/` (nvcomp `.so`, hiredis), `libcuda-11.8` stubs in `run_tests.sh [fkv]`; **`flexkv` is not installed** in our image (`ModuleNotFoundError [img]`). Also **not `SupportsHMA`** (`flexkv_connector.py:35 [img]` vs `offloading_connector.py:49`, `simple_cpu_offload_connector.py:54 [img]`), and `factory.py:54-59 [img]` raises *"Connector … does not support HMA but HMA is enabled. Please set `--disable-hybrid-kv-cache-manager`."* → FlexKV is **rejected on our HMA engine** unless we disable HMA (which our mamba-align + multi-group setup depends on). Rollback = drop package + config. |
| **Upstream health / tested?** | Merged, actively patched. Both fixes touched `tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py` `[web]`. Known open issue for our exact shape: **#54414** (recent-window groups un-restorable), plus #52771/#52807 for MTP/recurrent-boundary. | In-image, `SupportsHMA`, but sparse: it is the "one flag, no infra" connector (docs/01:51). Not covered by the offloading unit tests; no evidence in-tree that it is exercised for a GDN+ring hybrid — treat hybrid support as `[win]`. | Most active (last commit **2026-09-20**, `[fkv]`), merged into vLLM mainline since **v0.17.2** (PR #34328, README) and SGLang mainline. But its hybrid support is **DeepSeek-V4 / Gemma4 (MLA, SWA, compress-indexer)** — **no GDN/Mamba/linear-attention** anywhere `[fkv]`. |

---

## What each column would still require us to write

- **Native `OffloadingConnector`** — a library-side fix to **drop the ring from
  `_lookup_groups`/`_sliding_window_groups`** (our N10's minimal fix, == upstream #54414's
  `load_skip_groups`, which is **still open**, so this is a *carried patch*, not a backport)
  **plus** a backport of #52807 (`4a806d0`) to stop the MTP zeroing after the ring is gone.
  ACR's policy itself stays **out-of-tree**: `AcrOffloadingSpec`/`OffloadingManager`
  (`spec_module_path`), `AcrValuePolicy` (`cache_policy_module_path`, `mark_non_evictable`),
  per-request `kv_transfer_params` tier filter. No engine edits if the ring fix lands as an
  upstream-style scheduler change we can carry.
- **`SimpleCPUOffloadConnector`** — we would have to **fork it** to get any policy seam
  (no `cache_policy_module_path`, no `kv_transfer_params`, no admission/pin, no secondary
  tiers), and **add observability** (no offload metrics) before we could gate promotion on
  evidence. The upside: the lookup bug we hit is *structurally absent*, at lower cost than
  the two-part native fix. This is the tension the recommendation resolves.
- **FlexKV** — a CUDA build against a vLLM version we do not have, **plus** an HMA
  accommodation (either FlexKV declaring `SupportsHMA` or us disabling the hybrid manager,
  which our model needs on), **plus** GDN/recurrent-state support FlexKV does not have,
  **plus** a C++ fork for any non-enum eviction policy. Effectively: fork it, then rebuild
  it, then teach it a cache type it has never modelled.

---

## Recommendation

**(a) Default production data plane: native `OffloadingConnector`.**
It is the only column with a real out-of-tree policy surface (`cache_policy_module_path`,
per-request `kv_transfer_params`→tier filter, `mark_non_evictable`, secondary tiers) and the
only one with the Prometheus `kv_offload_load_bytes` / `…_total_bytes{transfer_type}` series
we already gate on `[meas]`. Its two defects are known, localized, and cheap to carry: the
ring must come out of `_lookup_groups` (our N10 fix; upstream #54414 unmerged → carried
patch) and the MTP zeroing must be backported (#52807 `4a806d0`, a ~2-site edit in a file we
already patch).

**(b) The A/B baseline worth one window: `SimpleCPUOffloadConnector`.**
It is already in the image, needs **zero new packages or processes**, declares `SupportsHMA`,
and — the decisive reason — routes lookup through the HMA coordinator, whose
`find_longest_cache_hit` **drops the ring on `prefix_cacheable=False`** (`kv_cache_interface.py:626-627`,
`kv_cache_coordinator.py:693 [img]`). That is exactly the exclusion our offline N10 showed
flips the native path from 0 to 84,864. So SimpleCPU is the cheapest realistic test of
"does a *correct* lookup path actually move `CPU_to_GPU` off zero on this model," independent
of the two-part native fix. **Caveat that keeps it a single window, not a decision:** it has
no load-direction metric, so the window must measure GPU-side prefix-cache hit + restore
latency vs cold re-prefill (weaker evidence), and its GDN-restore correctness on our fork is
unverified.

**(c) Does ACR's contribution survive each choice?**
- Native: **yes.** The connector deliberately leaves the policy layer external — ACR's
  admission/eviction/pin and per-request residency map onto documented plug-ins. ACR is the
  product; the connector is plumbing.
- SimpleCPU: **subsumes / blocks us.** Hard-coded LRU, no policy or per-request seam
  `[img]`. As the *data plane* it would erase the very thing ACR adds; useful only as a
  diagnostic A/B arm, not a target.
- FlexKV: **both subsumes and is infeasible.** Policy is compiled-in C++ (fork to change),
  and it cannot run on this hybrid engine at all today. Not a candidate; a possible future
  L3 *tier* only if ACR ships it behind `SecondaryTierManager`, which still requires FlexKV
  to model GDN.

**Single next step (offline, no window): replay the ring-exclusion on the native connector.**
Our N10 already predicted the number (≈84,856–85,680 hit) but flags that the *deployed* tree
must actually have the ring out of `_lookup_groups` (patch-05 shape), not just out of stores.
The offline discriminator is: instrument `OffloadingConnectorScheduler._lookup_complete_chunks`
over the real three-group config with the ring removed, plus the #52807 two-line MTP fix, and
confirm a nonzero converged `num_hit_tokens` — then, and only then, spend the window.

---

## Window-required (not restarted — the one observation that settles each)

- **SimpleCPU actually serves a nonzero restore on this model** (not just avoids the
  collapse in arithmetic): observation = `vllm:prefix_cache_hits` rising on a re-sent
  post-eviction prompt and TTFT for a warm request beating cold re-prefill, with the
  SimpleCPU config (no offload metric exists → use engine prefix-cache counters).
- **SimpleCPU GDN + attention restore is byte-correct** (the project's central open
  question, docs/08:461-466): observation = a controlled restore-vs-recompute token-identity
  or content-hash diff on one prompt; cannot be answered without running it.
- **Whether FlexKV's external package even matches our image's glue call-surface**
  (`FlexKVConnectorV1Impl(vllm_config, role)` and the methods `flexkv_connector.py` invokes):
  observation = a successful `import flexkv.integration.vllm.vllm_v1_adapter` + a dry
  `get_num_new_matched_tokens` in a GPU-less throwaway container after `build.sh`. Blocked by
  "no HMA" regardless.
- **Does the native connector's ring-exclusion fix hold at runtime** (predicted offline):
  observation = `vllm:kv_offload_total_bytes{transfer_type="CPU_to_GPU"} > 0` after a re-sent
  86,000-token prompt following full HBM eviction (docs/08:542-545's stated gate).

## Cells verified in OUR image vs read from upstream/main (misleading-if-main-only)

- Columns 1 and 2 and the coordinator/spec `find_longest_cache_hit` / `prefix_cacheable`
  logic are **all read from the installed tree** `[img]` — safe to rely on.
- **The two PR fixes (#52771 `da8ec28`, #52807 `4a806d0`) and issue #54414 are upstream-main
  only, dated after the 08-11 fork, and confirmed NOT present in our tree** (`:230`,
  `:762 [img]`). Any statement that "the connector now loads" based on upstream main would
  **mislead us** — our image still zeroes.
- **FlexKV column is entirely `main`-only `[fkv]`**, package not installed `[img]`, and not
  version-matched; its DeepSeek-V4/Gemma4 hybrid support **does not imply** GDN/QSA-ring
  support. Treating that "hybrid support" as applicable to our model would be wrong.

## What I could NOT verify (list)

1. Any **runtime** load-direction success for SimpleCPU or FlexKV on this engine — neither was
   started (window-required).
2. Byte-equivalence of a restored page for any connector (open project question).
3. Whether `request.namespace_info` (FlexKV's session seam) is reachable from a stock vLLM
   `Request`/extra-body on our tree — I did not grep vLLM's request path for it.
4. SimpleCPU **has unit-test coverage** for a GDN+ring hybrid — I saw the connector but no
   test in the wheel (tests aren't shipped in dist-packages); assumed absent/unproven.
5. FlexKV **install success against our vLLM/CUDA** (`build.sh` never run — no GPU-less
   container was built this task).
6. The exact per-token byte cost of SimpleCPU's host pool on our config (I read the sizing
   formula, not the instantiated value — that prints at boot).
7. Whether disabling HMA (the only way to admit FlexKV) leaves our `mamba_cache_mode=align` +
   816-block attention correct at all — engine-config consequence I did not test.

## Authoritative correction (verified against the GitHub API on 2026-09-21)

The PR↔hash mapping stated in the review thread was wrong; ground truth is:
- **#52771** — "OffloadingConnector: stop zeroing offload hits under MTP/EAGLE", merged 2026-09-07, `4a806d0`.
- **#52807** — "Do not let a recurrent group's unhashed block trip eviction", merged 2026-09-03, `da8ec28`.
- **#54414** — "[Feature][KV-offloading]: recent-window state groups can never participate in restores", **still open**.

Reconciling the two seemingly contradictory causal results: they are **two independent sources of the
same zero**, not competing explanations.
1. The all-groups eagle fallback zeroes the hit when MTP/EAGLE is on → fixed upstream by #52771.
2. The QSA `CircularBufferSpec` group sitting in `_lookup_groups` zeroes it **regardless of MTP** (our
   N10 non-MTP case also returned 0) → **no upstream fix exists**; #54414 is the open feature request.
Supporting asymmetry found in our own image: the HBM-side coordinator already excludes
non-`prefix_cacheable` groups from its min-across-groups bound
(`CircularBufferSpec.prefix_cacheable=False`, `kv_cache_interface.py:626-627`; applied at
`kv_cache_coordinator.py:693`), while the offload connector's `_lookup_groups` has no equivalent
filter (`offloading/scheduler.py:522`). So vLLM is internally inconsistent about the same group, and
matching the coordinator's rule is the minimal fix.

Consequence for the stack decision: production needs **both** the #52771 backport and a carried
ring-exclusion patch; and the ring-exclusion — with the coordinator asymmetry as its justification —
is the most defensible thing ACR has to contribute upstream.
