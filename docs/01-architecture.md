# 01 — Architecture and the exact seams we use

All paths below are inside the running image
(`/usr/local/lib/python3.12/dist-packages/vllm/…`, vllm `0.1.dev20073+g8e685d198`), verified by
reading the installed source, not the docs site.

## 1. Layering

```
             clients (Claude Code / Dify / Core patient transport)
                                 │
                    ndefy-model-router (Go, our gateway)
                    stamps kv_transfer_params: session_id, role,
                    tool_eta_ms, slo_class, expected_turns
                                 │
        ┌────────────────────────┼─────────────────────────┐
        │                        │                         │
   vLLM HTTP              /metrics + KV events        ACR controller (out-of-tree
        │                 (shadow index)              python, loaded by the engine)
        ▼                        │                         ▲
  OffloadingConnector ───────────┴──────► ACR Manager ──────┘
   (SupportsHMA)                            │ CachePolicy (ours)
                                            │ SecondaryTierManager (ours / fs / obj / p2p)
                              ┌─────────────┼──────────────┐
                            HBM           RAM            STORAGE
                        (vLLM APC)   (cpu_bytes_to_use)  (root_dir / obj / p2p)
```

The engine still owns HBM allocation and its own LRU-ish prefix cache. ACR owns **everything below
HBM and the timing of promotions into it**. Hard boundary that no HTTP-level trick can cross:
A CR cannot force "keep block X in HBM until 17:03:25" — vLLM exposes no HBM pin API. Three
honest options, in increasing cost:

1. **stay out of HBM policy** (phase 2): only manage RAM/storage. Already covers the 9.2×-interference
   problem, because avoiding cold recompute is most of the win.
2. **influence it**: `kv_load_tiers` + request `priority` + `watermark` + size-aware admission so HBM
   eviction order roughly follows our value order.
3. **change it** (phase 3+): a small upstream patch adding a residency hint / pin API. Worth doing
   as an upstream contribution, not as a private fork of a build we already carry 4 patches on.

## 2. Extension points actually available (verified)

| seam | how it is selected | contract |
| --- | --- | --- |
| `OffloadingSpec` | `kv_connector_extra_config: {spec_name, spec_module_path}` (`vllm/v1/kv_offload/factory.py:26-46`) | `get_manager()`, `get_worker(kv_caches)`, `build_metric_definitions()`; attrs `offload_prompt_only`, `tokens_per_block`, `tokens_per_hash`, `blocks_per_chunk` |
| `CachePolicy` | `extra_config: {eviction_policy, cache_policy_module_path}` (`vllm/v1/kv_offload/cpu/policies/factory.py:60-79`, *"out-of-tree … no vLLM fork/patch required"*) | `__init__(cache_capacity)`, `get/insert/remove/touch/evict/clear`, `mark_evictable/mark_non_evictable`; `evict(n, protected)` must be atomic (return `None` if n cannot be satisfied) |
| `SecondaryTierManager` | `extra_config: {secondary_tiers: [{type, module_path, …}]}` (`vllm/v1/kv_offload/tiering/factory.py:20-30`, same "no fork" wording) | `lookup`, `submit_store`, `submit_load`, `get_finished_jobs`, `touch`, `on_new_request`, `on_request_finished`, `on_schedule_end`, `serve_external_requests`, `drain_jobs`, `shutdown`, `get_stats` |
| per-request signals | request body `kv_transfer_params` → `Request.kv_transfer_params` → `_create_req_context()` (`…/v1/offloading/scheduler.py:475-487`) → `ReqContext.kv_transfer_params` | already honoured today: `kv_load_tiers: [{medium: CPU|STORAGE, locality: LOCAL|REMOTE}]` builds a per-request `TierFilter` |
| manager-side hooks | `OffloadingManager` (`vllm/v1/kv_offload/base.py:220-392`) | `lookup`, `prepare_load`, `touch`, `complete_load`, `prepare_store`, `complete_store`, **`on_new_request`**, **`on_request_finished`**, `take_events`, **`on_schedule_end(ScheduleEndContext{new_req_ids, preempted_req_ids})`**, `has_pending_work`, `reset_cache`, `get_stats` |

Registered connectors in this build: `OffloadingConnector`, `SimpleCPUOffloadConnector`,
`LMCacheConnectorV1`, `LMCacheMPConnector`, `MooncakeStoreConnector`, `HF3FSKVConnector`,
`MultiConnector`, `Nixl*`, `FlexKVConnectorV1`, `DecodeBenchConnector`. Built-in secondary tiers:
`example`, `fs` (`root_dir`, `n_read_threads`, `n_write_threads`, `locality`), `obj`, `p2p`.
Already pip-installed in the image: `lmcache 0.5.4`, `nixl 1.3.2`,
`mooncake-transfer-engine-cuda13 0.3.12`, `pyzmq 27.2`.

Hybrid-model readiness signal: `…/kv_connector/v1/ssm_conv_transfer_utils.py` exists and its
docstring names **"GDN (Gated Delta Net): conv = [Q, K, V] … temporal = (num_v_heads, v_dim, k_dim)"**
— recurrent-state pages have a defined transfer layout in this tree. `OffloadingConnector`
declares `SupportsHMA` (hybrid memory allocator). This is *encouraging*, not proof for our
combination (GDN + MTP + `inc` W4A16 + TP4/EP4 + explicit `kv_cache_memory_bytes` + 4 carried
patches) — see gate G-1 in docs/04.

## 3. The one real gap: block-level admission

`CPUOffloadingManager.prepare_store` admits unconditionally:

```python
keys_to_store = [k for k in keys if self._policy.get(k) is None]   # cpu/manager.py:176
```

There is no per-block "is this worth caching at all" veto. Request-granular admission *does* exist
(`on_new_request` returns `RequestOffloadingContext(policy=BLOCK_LEVEL|REQUEST_LEVEL)`), which is
enough to drop whole throwaway requests, but not enough to refuse single junk blocks inside a
useful request. Two routes:

- **our own manager** (chosen): we ship an `OffloadingSpec` returning our `OffloadingManager`, where
  `prepare_store` consults `policy.admit(key, ctx)` first. Pure out-of-tree code.
- **upstream hook**: a ~10-line `admit()` on `CachePolicy` with a default-True implementation.
  Good upstream candidate; do it after the simulator shows admission matters.

## 4. Configuration we would run with (promotion window only)

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "AcrOffloadingSpec",
    "spec_module_path": "acr.vllm.spec",
    "cpu_bytes_to_use": 171798691840,
    "block_size": 816,
    "eviction_policy": "AcrValuePolicy",
    "cache_policy_module_path": "acr.vllm.policy",
    "offload_prompt_only": false,
    "secondary_tiers": [
      {"type": "AcrDurabilityTier", "module_path": "acr.vllm.tier",
       "root_dir": "/data/acr/kv", "budget_bytes": 1073741824000}
    ]
  }
}
```

Notes: `cpu_bytes_to_use` 160 GiB of the 359 GiB available (≈2.9M tokens) — sized to the §5
requirement in docs/00, not greedily; `block_size` must be reconciled with this tree's unified
816-token page layout (open item **P-1**); `offload_prompt_only=false` because agentic sessions
must also keep their own generated tail; the disk tier exists **only** for warm-boot/durability.

Mounting ACR is a `-v` and a `PYTHONPATH` — no image rebuild, which is what makes the rollback
story as cheap as the current `deploy_qwen.sh` last-good restore.

## 5. Where the controller state lives

Per block: `key`, `bytes`, `refcount` (session leases), `last_use`, `reuse_count`, inter-arrival
EWMA + p95, prefix-hit contribution, `tokens`, referencing roles, owning workflow stage.
Per session: prefix path, current leaf, token count, role, SLO class, last activity, tool ETA
distribution, growth rate, expected remaining turns.

Two independent views of this state must be reconciled, and their divergence is itself a metric:

- **gateway view** (authoritative for app semantics: it sees sessions, roles, tool calls)
- **engine view** (authoritative for physical residency: `take_events()` / KV events / `/metrics`)

The shadow index (§1 diagram) is the diff between them. If the diff is small, the gateway can drive
residency without engine cooperation — that is the finding that makes phase 1 possible at all.

## 6. Native QoS surface in the deployed build (verified 2026-09-20, read-only greps of the live container)

Every line below was read out of `/usr/local/lib/python3.12/dist-packages` **inside the running
production container**, not out of upstream docs, because this build is a fork with patches.

| capability | verdict | evidence |
| --- | --- | --- |
| per-request `priority` on `/v1/chat/completions` | **present** | `entrypoints/openai/chat_completion/protocol.py:380` |
| priority via HTTP header (no body rewrite) | **present** | `entrypoints/generate/base/serving.py:45` (`PRIORITY_HEADER = "X-Vllm-Priority"`), `:249` |
| preemption of the *lowest*-priority running request when blocks run out | **present** | `v1/core/sched/scheduler.py:670-674` (`max(self.running, key=(priority, arrival_time))`) |
| per-request prefill chunk cap | **present but global** | `scheduler.py:591-592`: clamps `num_new_tokens` for *any* request over the threshold; there is no class dimension |
| `watermark` (fraction of blocks kept free), `scheduler_reserve_full_isl` | **present** | `vllm/config/scheduler.py` fields; defaults 0.0 / True |
| per-class KV quota / reservation | **absent** | `grep -l quota vllm/v1/core/ vllm/config/` → no files |
| priority **overrides a full running-slot set** | **NO — the trap** | `scheduler.py:785-786`: `if num_running >= self.max_num_running_reqs: break` before the waiting queue is consulted; `:777` additionally skips the waiting queue entirely on a step where anything was preempted |

Consequences that change the design:

1. **The engine can express KV precedence and cannot express slot precedence.** `max_num_seqs` is a
   hard admission cap that priority does not unlock, so "latency class = priority 0" is *not* an
   interactivity guarantee when all slots are held by long-running agent requests. Slot QoS is
   therefore a control-plane (gateway semaphore) responsibility — this is ACR's job, not a fork job.
2. Preemption is **recompute-based** in V1. Without a working RAM tier, a preempted 1M agent context
   costs ~90 s of cold re-prefill (docs/00 §2). So "medical reclaims capacity" is only cheap if the
   offload path works; the tier is a dependency of the QoS design, not an optimisation of it.
3. Production today runs `scheduling_policy=fcfs` (absent from the boot's `non-default args`), so
   any priority stamping is inert until the engine is booted with `--scheduling-policy priority`.
   A non-zero `priority` is documented to *error* in that case, which is an unverified claim in this
   build → must be tested behaviourally, not trusted.
4. `long_prefill_token_threshold` is bounded below by the page: this build's scheduler keeps a
   mamba-block-aligned split path (`scheduler.py:419-424`, `:319 need_mamba_block_aligned_split`)
   and the live page is **816 tokens**, so budgets like 512/1024 (commonly recommended upstream for
   non-hybrid models) sit *below* one page here. Sweep multiples of 816 instead.
5. Observability for all of the above already exists: `vllm:num_preemptions_total`,
   `vllm:prefix_cache_{queries,hits}_total`, `vllm:kv_cache_usage_perc` (seen in `tune/results/g1-*.json`).

`--prefix-match-unit` (`engine/arg_utils.py:1246`) is exposed and, per
`v1/core/kv_cache_utils.py:608-672`, can be set **finer than the physical block** as long as every
prefix-cacheable group's block size is divisible by it; the live boot log confirms
`Mamba cache mode is set to 'align'`, which is the precondition for that path (otherwise the resolver
silently backs off to the scheduler block size). "It controls matching granularity only, not how
often states are stored" (verbatim, `config/cache.py:63-66`). This is the lever that answers the
rollback-stranding problem without touching capacity, and it is exactly the precondition our patch
01 (`enable_mamba_fine_grained_prefix_cache`) documents.
