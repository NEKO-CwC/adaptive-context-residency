# 08 — Phase A field log (2026-09-18/19): what the engine actually did

The engineering record for the first attempt at turning on stock GPU↔RAM KV offload on the
production engine. Kept separate from the design docs because most of it is about **how to run an
experiment on an engine that is also serving the experimenter**, and because two of my own
conclusions died here.

## Outcome

| attempt | config delta | result |
| --- | --- | --- |
| `PHASEA-lru` (09-18 17:45) | `--kv-transfer-config OffloadingConnector{lru, cpu_bytes_to_use=48GiB}` on the production command | never ready in 900 s; container killed by another agent session; **no forensics kept** |
| `PHASEA-lru-dev` (09-19 08:18) | same + `VLLM_SERVER_DEV_MODE=1` (G-1's flush lever) | never ready in 900 s, **zero vLLM output after the patch loop**, `RestartCount=0` → a silent hang, not a crash |
| control | the identical command without `--kv-transfer-config` | boots in **296–297 s**, `kv=1,003,197 tokens`, serving |

So: **the connector config does not start on this build/configuration**, cause still unknown. The
control run is what makes that statement meaningful — everything except the connector flag is
identical and boots fine.

## Hypotheses raised and how each was settled

1. **`/dev/shm` too small for `cpu_bytes_to_use`** — vLLM really does raise
   `Insufficient space in /dev/shm: 49152 MiB required, 32768 MiB free` (verified by calling
   `check_shm_free_space` directly), **but the production container runs `--ipc host` and sees a
   252 GiB `/dev/shm`**, so this cannot explain the observed hang. My `--shm-size 32g` test
   container reproduced the error and tempted me into reporting a false root cause. Falsified.
2. **Zero-block sizing** (`cpu_bytes_to_use // aligned_kv_bytes_per_chunk == 0` → region of zero
   bytes) — read `cpu/spec.py:86-120,145-175`: with 48 GiB and this model's page geometry the
   division cannot reach zero, and `create_worker` only builds the mmap region when
   `num_blocks > 0`. Not it, though it is a genuine trap for a much smaller tier.
3. **Two TP4 engines cannot coexist** — confirmed the hard way: a `--load-format dummy
   --gpu-memory-utilization 0.06` probe still took **7.78 GiB/card** while production held
   36.9/44.4 GiB, and OOMed with 173 MiB free. Consequence: **there is no way to test the connector
   off-window.** Probing it must happen inside a window, and it must be the *first* thing in it.
4. **`cudaHostRegister` of a 48 GiB file-backed tmpfs region is slow/silent** — untested, and the
   leading remaining explanation (no output at all, `RestartCount=0`, never ready). The
   `--gpu-memory-utilization`-shaped dummy probe above cannot test it either, because it dies at
   weight/shape allocation first.

## The rule that follows

Run the cheap, decisive probe **inside** the window, before committing to a real boot:

```
window budget:   [dummy-weight connector probe ~2 min] → verdict
                 if feasible → real config + G-1 (dev-mode on) → promote or restore
                 if not      → restore known-good, close phase A with a cause
```

Two minutes of a window buys a cause; fifteen minutes of a full boot buys a timeout and a shrug.

## Operational lessons (also in the medical repo's `tune/RESCUE-BRIEF.md` and project memory)

1. **A port probe is not proof of provenance.** On 09-19 08:40:32 my deploy logged a successful
   `seqs=4 patch=first` restore; five seconds later a concurrent recovery script replaced the
   container with `seqs=1 patch=none`. The deploy was *correct about its own work* and *blind to
   what was actually serving*. Fix: keep the `docker run -d` container id and assert
   id-match + `--max-num-seqs` + `/patches` presence before logging success.
2. **One controller owns the engine, decided on disk.** A live process is not proof of progress
   (SIGKILL skips bash traps, so the 09-18 "auto-restore" never ran); and a *dead* process's lock is
   not proof of ownership (it blocked rescue for its whole deadline). Both a live pid **and** an
   unexpired deadline are required, and a second caller must refuse rather than "wait and retry".
3. **Do not split the outage across tool calls.** The agent's own next generation needs the engine:
   return from a call only once the engine is verified serving, and keep that wait inside the call.
4. **Verify a hypothesis before writing it down.** The `/dev/shm` story was plausible, arithmetic
   checked, and wrong for the real container because I tested a different configuration than the
   one I was diagnosing.
5. **Report the parameter set with the number.** My phase-0 headline (revisit signals worth 5.6×)
   was real at a 256-token simulated block size and gone at the engine's actual 816 (docs/05 F-7b).
6. **Cost accounting.** Four outages ≈ 35 minutes of a shared engine, one clinical product offline,
   and two agent sessions interrupted — for a result that is currently a negative: *this connector
   configuration does not boot here*. The durable output is the tooling and the measured constants,
   which is why they are in commits and docs rather than only in chat.

## Correction (2026-09-19, later same day): both "crashes" were CUDA OOM, not the fault family

The premortem capture added an hour earlier produced the traceback that the recovery path had
destroyed twice before:

```
torch.OutOfMemoryError: Tried to allocate 446.00 MiB. GPU 3 has a total capacity of 44.40 GiB
of which 272.81 MiB is free. … 42.03 GiB allocated by PyTorch, 41.88 MiB in private pools
(CUDA Graphs), 1.09 GiB reserved by PyTorch but unallocated.
→ EngineCore fatal → EngineDeadError → HTTP 500 → restarts=10
```

So the `bfloat16`-state run (failed at round ~60) and the `auto`-state run at the same 17 GiB
(failed at round ~188) had the **same** cause: **we over-reserved KV**, not prefix granularity.
The block-432 attribution I wrote an hour earlier was wrong, and the surviving conclusion is the
geometry one (capacity flat, block halved), not a stability one.

### Margin arithmetic this yielded

At `kv = 17 GiB/card`: measured minimum free during a 300×3 growing-prefix load = **111 MiB**, and
the failing allocation needed 446 MiB. Peak non-KV resident is therefore
`44.99 − 20.14 − 17.00 − 0.11 ≈ 7.74 GiB/card`. Because the KV reservation is a fixed allocation,
giving back 1 GiB buys ~1 GiB of peak margin, so the candidate must satisfy

```
kv ≤ 44.99 − 20.14 − 7.74 − margin        margin ≥ 1.5 GiB  ⇒  kv ≤ 15.6 GiB
```

**15.5 GiB** was derived from that inequality (pool 1,152,677 tokens, +14.9 %), but it is **not yet
earned**: the soak that produced the 3.33 GiB margin actually ran on the 13.5 GiB container
(attribution error caught after the fact — see docs/00). 13.5 GiB is therefore the validated
production number, and 15.5 GiB stays a candidate until a soak that can prove it ran on it. A capacity claim without a measured peak
margin is not a claim, it is an outage waiting for round 188.

### Operational findings worth keeping

1. **A fallback target must be labelled so a second responder can tell it from an experiment.**
   Pinning the in-test `GATE15.5` as `known-good` caused another agent session to distrust it and
   fall through to `rollback_supervisor.sh` — leaving production *unpatched*, i.e. removing the
   mitigation for the fault family we spent this whole CR on. The safe fallback is now
   `PROD-PATCHED-CONSERVATIVE` (13.5 GiB, patched) and candidates live in separate files.
2. **Escalation counters must be per-incident.** A lifetime-cumulative `actions` counter meant a
   new incident skipped the pinned (patched) restore and went straight to the unpatched deep
   rollback, because two *successful* recoveries earlier that day had already reached the limit.
3. **Premortem capture is what made this diagnosable.** `deploy_qwen.sh` now dumps status + 200 log
   lines of the *outgoing* container before removing it whenever it is unhealthy/dead. Without it we
   had two "engine died" events with no traceback and were reasoning from memory counters.

## 15.5 GiB earned its gate on the second attempt (2026-09-19 14:1x)

With the attribution tool fixed, the candidate ran its own 300×3 growing-prefix soak and passed:
**0 failures, 671 s, peak 43,506 MiB/card, min free 2,011 MiB**, container id and
`--kv-cache-memory-bytes` identical before and after. It is now the production config and the
auto-restore target (`PROD-PATCHED-15.5`, pool 1,152,677 tokens).

Two numbers worth keeping: the margin inequality over-predicted risk (said ~1.3 GiB, measured
2.0 GiB), and the same soak at 13.5 GiB took 1,229 s while at 15.5 GiB it took 671 s — a
**1.8× throughput difference from 2 GiB of KV reservation**, with co-resident agent traffic being
the common condition. That gap is the honest, measured version of "why capacity matters", and it
came with the box already running nothing but our own harness.

## 2026-09-20 — phase A has a named blocker, and it is a geometry one

Two gates stood in front of the stock `OffloadingConnector`, and they were hit in this order:

1. **Allocator gate (real, cleared without an engine).** `vllm/config/vllm.py:998-1004` refuses any
   connector that pins KV memory while `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set —
   which *our own* `deploy_qwen.sh` had been exporting. Proven with no engine: unset the variable and
   the identical `VllmConfig(kv_transfer_config=OffloadingConnector, TP4/EP4)` constructs in 1.0 s.
   Stock exits: stop setting it, or `--enable-cumem-allocator`.
2. **Block-geometry gate (fatal today).** With (1) cleared, the boot reached engine-core init and
   died on `offloading/config.py:60`:
   `tokens_per_block=8 not divisible by tokens_per_hash=816`. The connector asserts over **every**
   KV-cache group, while the hash unit is derived only from the **prefix-cacheable** groups; this
   hybrid tree has a recurrent-state group at 8 tokens/block and an attention group at 816. `oom=false`,
   and the patchset is not implicated (`01-pr-53945` touches `hash_block_size` but no geometry).

What that means for the project, stated plainly: **vLLM's shipped offload path cannot key a hybrid
GDN+attention cache whose groups disagree with the hash unit** — the library refuses to start rather
than degrading. That is the strongest evidence yet that "hybrid-state residency" is a real gap rather
than a tuning exercise, and the weakest evidence for phase A as originally scoped (measure the stock
tier), which can no longer be done at all on this build without changing group geometry.

Open, and answerable without another window: why `tokens_per_hash` resolves to 816 instead of
`gcd(816, 8) = 8`, i.e. which group is non-prefix-cacheable (`CircularBufferSpec.prefix_cacheable`
is False in this tree). Until that is read out, `--mamba-block-size 816` and `--prefix-match-unit 8`
are *candidate* exits, not a plan: each changes either HBM page geometry or hash cost by ~100×, and
both feed the capacity identity in docs/00 §6.

Operational note, because it is the reason this cost 25 minutes: the deploy trap's auto-restore
refused to run (its child saw the still-alive parent's in-flight marker and stood down) and the
container carried `--restart unless-stopped`, so the failing config looped instead of dying. Both are
fixed in the medical repo (`71fc8189e`); experimental windows now start with `--restart no`.

## 2026-09-20 18:10–18:28Z — W1 got a name: it was never a hang

`--alloc plain` (dropping **our own** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which
`deploy_qwen.sh:232` has set since the capacity work, and which `vllm/config/vllm.py:998-1004`
treats as incompatible with any connector that pins KV) cleared the rejection found in W0. The
engine then proceeded normally — config, weights, workers — and died inside the connector's own
boundary construction:

```
kv_connector/v1/offloading/config.py:60
  assert group.tokens_per_block % tokens_per_hash == 0
  AssertionError: tokens_per_block=8 not divisible by tokens_per_hash=816
  ... EngineCore failed to start
```

Read from that file: `groups` is built from **every** KV-cache group's `kv_cache_spec.block_size`,
and `tokens_per_hash` comes from `resolve_kv_cache_block_sizes`. This model has an **8-token group**
(the GDN/conv state group) and 816-token hash granularity ⇒ `8 % 816 ≠ 0`. The assertion's own hint
("hybrid models need `--enable-prefix-caching` to align block sizes") is already satisfied: our
`align` mode aligns **bytes** (page equality, `interface.py:915/939`), not tokens, which is exactly
why the two groups keep different `tokens_per_block`.

**This is the phase-A result, and it is a statement about the interface, not about our policy:** the
stock `OffloadingConnector` cannot be enabled on this hybrid tree at the engine's default hash
granularity. Two follow-ons, each one flag:

1. **W1c — the unlock to try next window:** `--prefix-match-unit 8`. `resolve_kv_cache_block_sizes`
   lets `prefix_match_unit` override hash granularity and only requires every prefix-cacheable
   group's block size to be divisible by it; 8 divides both 8 and 816. Prediction: the
   `config.py:60` assertion clears and boot continues to G-1 (W2), which remains the correctness
   gate for the layout hazard. If instead it changes the resolved attention block size, the
   capacity identity in docs/00 §6 predicts what we should see.
2. **W5 is now a separate, cheaper question:** patch 01 in our chain adds
   `enable_mamba_fine_grained_prefix_cache` with **default False**, wired next to
   `prefix_match_unit`/`mamba_block_size`. So production today does *not* take a checkpoint at the
   shared-prefix junction, and `prefix_match_unit` smaller than the mamba block size is the
   precondition that makes patch 01 do anything. That is a rollback-granularity win independent of
   whether the RAM tier ever works.

### Process cost of this window (mine)

- `READINESS_SEC=380` was sized from a 294 s production boot; the connector boot is slower, so the
  first attempt was inconclusive-by-timeout rather than failed, and cost a second window.
- The container crash-looped (`RestartCount` 0→3) while `status` read `running` between restarts, so
  `deploy_qwen.sh`'s crash fast-fail never fired and sat in the readiness loop for the whole budget.
  Fast-fail must treat a *growing* RestartCount as crash evidence even when status is `running`.
- A `setsid`-detached watchdog left no log lines after `watchdog start` and was dead while production
  was down. Recovery came from the deploy's own SIGTERM trap (the fixed `marker_owner_live` let the
  restore child proceed: `restore-…-PROD-PATCHED-15.5`, boot 289 s, provenance ok, pool 1,152,677
  tokens) plus the human's external session. The rescuer now runs as a **systemd transient unit**
  (`ncu-a-watchdog`) so it cannot die with an agent session.
- End state verified read-only: healthy, no `--kv-transfer-config` in PID 1, no deploy marker, real
  generate returns.

## 2026-09-20 19:28–19:34Z — W1d: two walls cleared, the third is structural

`--alloc plain --kvtransfer lru --pmu 8` (connector + `--prefix-match-unit 8`), evidence preserved
by the new `preserve_evidence` trap into `tune/results/boot-failure-W1d-pmu8-lru.txt` (59 KB):

- `grep -c "not divisible by tokens_per_hash"` → **0**. The `offloading/config.py:60` assertion is
  gone, i.e. **`--prefix-match-unit 8` does clear it**, and the boot proceeded all the way to
  `factory.py:62 Creating v1 connector with name: OffloadingConnector` →
  `factory.py:57 Creating offloading spec with name: CPUOffloadingSpec`. The tier even created its
  host region (`shared_offload_region.py:304 … /dev/shm/vllm_offload_<uuid>.mmap`).
- It then died at a third, different place, with no message beyond an assertion:

  ```
  offloading/scheduler.py:193-195  for idx, tokens_per_block in enumerate(spec.tokens_per_block):
                                     kv_spec = kv_cache_config.kv_cache_groups[idx].kv_cache_spec
                                     sw = get_sliding_window_size_in_chunks(kv_spec, …)   # unconditional
  offloading/scheduler.py:112-126   handles SlidingWindowSpec / ChunkedLocalAttentionSpec /
                                    MambaSpec (→1), then: assert isinstance(kv_spec, FullAttentionSpec)
  ```

  So the connector's **scheduler-side config cannot represent this model's group set**: one of our
  KV-cache groups carries a spec class outside those four, and the loop has no guard. The
  `CudaIPCTypes.cpp:16 Producer process has been terminated` lines that followed — which I had
  guessed were a PLE/IPC conflict — are cleanup noise from the dying engine, not a cause. That
  hypothesis is dead.

**Consequence for the project, stated plainly:** phase A's question ("can we just use the library?")
is answered **no** for the RAM tier on this model class. Two of the three walls were configuration
we control (`expandable_segments`, hash granularity); the third is code inside the connector's
spec handling. Either it gets an upstream branch (if our group's spec turns out to be a
`MambaSpec`-shaped thing) or ACR ships its own `OffloadingSpec`/manager — which is what docs/01
already reserved as the second extension point.

**Open evidence gap (N7, no restart):** name the offending spec class. `kv_cache_interface` exposes
14 `*Spec` classes and only four are accepted; computing this model's group specs on the CPU in a
throwaway container would say whether the fix is a 3-line branch or a new manager, without another
production boot. That is the next thing to do before asking for another window.

### N7 (no restart) — the offending class is named: `CircularBufferSpec`

Two independent levels of evidence, both first-party:

* the only spec constructed anywhere in this model's package that is **not** in the connector's
  accepted set is `vllm/models/qwen3_8_flash_next/common/qsa_cache.py:785 → CircularBufferSpec(`;
* the same W1d boot names the live subsystem: attention backend `QWEN38_FLASH_NEXT_EXP_QSA_STATE`,
  and production runs `_qsa_mqa_paged_kernel` / `_expand_qsa_indices_kernel` at inference time — the
  group is real, not a dead branch.

Hierarchy read out of the installed package: `CircularBufferSpec(AttentionSpec)` — not
`FullAttentionSpec`, not `SlidingWindowSpec`, not `ChunkedLocalAttentionSpec`, not `MambaSpec`, so
it falls through `get_sliding_window_size_in_chunks` into the bare assert at
`offloading/scheduler.py:125`.

Consequence for W1e, stated as a fork rather than a guess:

1. **~5-line upstream branch** — give `get_sliding_window_size_in_chunks` a `CircularBufferSpec`
   case (a ring of capacity C behaves window-like: `cdiv(C, tokens_per_chunk)`). That clears *config*.
   It says nothing about whether a ring buffer's live slots have stable hashes or a representable
   store/load chunk at all — which is exactly the class of "boots but restores wrong bytes" failure
   G-1 exists to catch. So this path must be measured, not trusted.
2. **Our own `OffloadingSpec` + manager** through the factory seam (`spec_module_path`), where the
   offload unit is **group-typed**: this tree has ≥4 spec kinds with different token granularity
   (attention 816, GDN state group 8/16, MLA-shaped and circular ones). The engine's own log proves
   the heterogeneity: *"Setting attention block size to 816 tokens to ensure that attention page size
   ≥ mamba page size"* + *"Padding mamba page size by 1.62%"*.

Either way the paper claim sharpens into something model-agnostic and testable: **a KV offload tier
whose unit is not typed per KV-cache group cannot serve a hybrid model** — and stock vLLM's tier is
exactly such a library.

### 2026-09-20 ~04:20Z — the tier is reachable with a 5-line patch, validated offline

`deploy/gpu/patches/sets/acr/` = production set + `05-qsa-circular-buffer-offload.patch`, generated
by diffing an edit of the installed file (never hand-written hunks) and validated in a fresh
container off the pinned image digest:

```
APPLIED 01..05                       apply.sh rc=0
ALREADY_APPLIED 01..05               apply.sh --check rc=0   ← the idempotence that crash-looped us on 09-18
circular(8) vs chunk 816 → 1 ; vs chunk 8 → None ; full/mamba/swa unchanged (None/1/6)
SchedulerOffloadConfig.from_spec → completes, groups=3, num_workers=4
```

So the honest position for the paper and for the next window: **the library's RAM tier is one
5-line group-aware branch away from booting on this hybrid tree** — the three walls were (1) our own
`expandable_segments` export, (2) hash granularity vs the 8-token group (`--prefix-match-unit 8`),
(3) `scheduler.py:125`'s unguarded `FullAttentionSpec` assert. None of them is about policy; all of
them are about the tier assuming a single, full-attention KV-cache group.

Still unproven, and it is the only thing that matters for production: whether store/restore of a
ring whose live slots move is **byte-correct** (G-1), and what a real restore costs (M-1). Both need
the window; the command is in `sets/acr/README.md`.

### 2026-09-20 08:54–09:08Z — W1e: all three walls cleared, the tier reached the scheduling loop

`--patchset acr --kvtransfer arc --alloc plain --pmu 8 --devmode on`. The engine **started, loaded,
captured graphs, and entered the busy loop**; it died at 09:03:14 inside the *first* scheduling steps:

```
vllm/v1/core/sched/scheduler.py:999   self._mamba_block_aligned_split(...)
vllm/v1/core/sched/scheduler.py:432   if tail_boundary and self.use_eagle_block_drop:
AttributeError: 'AsyncScheduler' object has no attribute 'use_eagle_block_drop'
```

So: wall 1 (our `expandable_segments` export) ✓, wall 2 (`tokens_per_hash` vs the 8-token group,
cleared by `--pmu 8`) ✓, **wall 3 (`FullAttentionSpec` assert) ✓ — patch 05 works in the real engine,
not just in the offline probe.** `--prefix-match-unit 8` is *required* for the tier (the QSA group is
8 tokens; only a hash granularity ≤ 8 satisfies `config.py:60`), and it activates the fine-grained
`tail_boundary` branch that has never run before — which references an attribute this build never
defines. Full evidence: `tune/results/boot-failure-W1e-acr-arc.txt` (61 KB, saved by the
`preserve_evidence` hook added the same day).

Production restored to `PROD-PATCHED-15.5` (restore start 09:03:48Z, provenance-verified serving
09:08:43Z, boot 295 s, pool 1,152,677 tokens — `tune/results/window-W1e-acr-arc/deploy.log`,
`tune/results/deploy-log.tsv`): no connector, no `--pmu`, healthy, real generate verified, no
marker left.

**Root cause, settled offline before W1f** (`sets/acr/README.md`, patch 06 header): the read at
`scheduler.py:432` was introduced by **our own patch 01**, and its producer exists nowhere in the
tree — `grep -rn block_drop vllm/` returns only patch 01's two read sites. *The guess written an
hour earlier ("a config field the scheduler failed to copy") is retracted: there was nothing to
copy; no assignment ever existed.* Patch 06 (`06-scheduler-use-eagle-block-drop.patch`) defines the
bit as `use_eagle and mamba_fine_grained_prefix_cache`. With the fine-grained opt-in off — every
production boot to date — `tail_boundary` is only non-zero when `prefix_match_unit` is below the
block size, so the branch is unreachable. That makes this a **latent landmine in the shipped
patchset**: production runs `patch=first` ⊃ patch 01, and the day anyone enables fine-grained
prefix caching without patch 06, the first request through the tail-boundary branch kills the
engine. The tier and the landmine are the same piece of code meeting in production for the first
time.

### 2026-09-20 09:22–09:39Z — W1f: the tier booted, served, and stored — the restore never fired

Patch 06 in the set (label `W1f-acr-arc-p6`), command `deploy_qwen.sh --seqs 4 --batched 8192
--kvgib 15.5 --util 0.80 --mtp on --patchset acr --devmode on --kvtransfer arc --alloc plain
--pmu 8` with `NCU_CPU_OFFLOAD_GIB=16` → rendered `cpu_bytes_to_use: 17179869184`,
`eviction_policy: "arc"` (`window-W1f-acr-arc-p6/deploy.log`).

- **It served.** `Creating v1 connector … OffloadingConnector` ×5, boot **298 s**, `restarts=0`,
  provenance ok `serving with seqs=4 patch=acr`, GPU pool unchanged at 1,152,677 tokens. (The
  window's own generate gate still logged `VERDICT=BOoted-BUT-NOT-SERVING`, rc=2 — twelve failed
  probes 09:27:33→09:28:35, after deploy_qwen's own probe had passed and minutes before G-1
  completed every arm. The cause of the gate's false negative is not established from the
  artifacts; recorded so the verdict line is not misread as "engine down".)
- **Write direction works.** G-1 capture from 09:30:07Z (`g1-W1f.json`): offload bytes counter
  +9,689,106,176 B (42,118,637,312 → 51,807,743,488) over +1.062 s of store time ⇒ **9.12 GB/s**.
  Sync lookups did run (`kv_offload_lookup_sync_delay_seconds_count` 11 → 27).
- **Read direction never fired.** `total_bytes` equals `store_bytes` to the byte and
  `cpu_cache_read_usage_perc` stays 0.0 — no load-direction bytes or time appear anywhere in the
  end-of-probe snapshot. Restored-vs-baseline-cold (compare against `g1-pre.json`, anti-vacuity
  bar 1.5×, `g1_correctness.py:204`): **x0.97 / x0.88 / x0.92 / x0.90** at 4K/16K/64K/150K — a
  "restored" request pays full recompute cost. All four arms under the bar ⇒ **INCONCLUSIVE
  (rc 8)**, by construction: the test never engaged the thing it measures.
- **Correctness held trivially.** `exact: true` ×4, worst dlogprob 0.0017 vs within-config noise
  floor 0.00257; greedy text and retrieved secrets identical in cold/warm/restored.
  `reset_ok: true` ×4 — the HBM flush worked; the tier simply never brought anything back.
- Reverted at 09:33:52; `restore-PROD-PATCHED-15.5` serving 09:39:14Z, boot 294 s
  (`W1f-revert.out`).

### 2026-09-20 10:04–10:18Z — W1g: a 64 GiB tier falsifies the eviction hypothesis

Identical to W1f with `NCU_CPU_OFFLOAD_GIB=64` → `cpu_bytes_to_use: 68719476736` (provenance block,
`window-W1g-cpu64/deploy.log`). Boot 313 s, `VERDICT=SERVING`, `g1 rc=8`.

- Store again moved first: probe delta **9,689,106,176 B** (9,689,222,656 − 116,480) over 1.058 s
  ⇒ 9.15 GB/s — byte-identical to W1f's delta because G-1 replays the same seeded content.
- Read again nothing: same counter shape, and restored-vs-cold **x0.97 / x0.59 / x0.92 / x0.90**
  (`probe.txt`).
- **Why this was the right test** — capacity arithmetic at the engine's geometry (docs/00 §1:
  56.4 KiB/token aggregate, one 816-token block ≈ 46 MiB): 16 GiB ≈ **297K tokens**, 64 GiB ≈
  **1.19M tokens**. The probe's largest single context is 150K tokens and all four arms together
  are 234K unique tokens — the 64 GiB tier holds the entire probe working set ~5× over. So
  "16 GiB was too small, the tier self-evicted the answer before reuse" was testable rather than
  obviously false, and W1g kills it: the load path does not engage even with a generous budget.
- Reverted at 10:13:10; restore serving 10:18:31Z, boot 294 s (`W1g-revert.out`).

## Phase-A verdict, as of 2026-09-20: the stock tier is one-way

Stated no stronger than the evidence: **on this hybrid tree the stock `OffloadingConnector`
stores and never restores.** Two windows, two tier sizes (16/64 GiB), four arms each: store fires
at ~9.1 GB/s (20× F-4's 446 MB/s break-even bar — bandwidth is not the problem), restore never
transfers a byte, and every "restored" request pays full recompute. G-1 verdicts are INCONCLUSIVE
— the question "does a restored page come back byte-correct" has still never been brought against
a restoring engine. Falsified along the way: tier size / self-eviction as the explanation.

Open root-cause candidates, none of which needs another production window:

1. the `reset_external=false` semantics of G-1's HBM flush possibly clearing the connector's own
   index — the flush *is* what forces the restore path;
2. `OffloadKey` identity mismatch between store and lookup at `tokens_per_hash=8` (the `--pmu 8`
   path changed hash granularity; a key stored under one identity and looked up under another
   misses silently);
3. load-side scheduling gates (`offload_prompt_only`, `store_threshold`, `HIT_PENDING`) that
   could accept stores and drop loads without moving any counter.

The separating diagnosis is tracked in `forensics/20260920-tier-load-path-diagnosis.md` in the
medical repo (in progress as of this writing; until it lands, the candidates are simply open).
Measured summary recorded as docs/05 F-12.

### Process cost of the three tier windows

- Non-serving time per the window logs and `deploy-log.tsv`: W1e 08:54:04→09:08:43 = **14 min 39 s**
  (the engine never served inside the window), W1f = 10 min 48 s (two boots), W1g = 10 min 40 s
  (two boots) — **≈36 min total**, ~47 min of window span. A "~15 min total" figure would be about
  right for one window and wrong for three.
- Every window ended on a provenance-verified `PROD-PATCHED-15.5` (restores serving 09:08:43 /
  09:39:14 / 10:18:31Z, boot 294–296 s, pool 1,152,677 tokens); the chain in `deploy-log.tsv` is
  unbroken from 09-17 to here.
- `preserve_evidence`, added the same day, is what made W1e's root cause knowable at all: the 61 KB
  traceback (`boot-failure-W1e-acr-arc.txt`) is the artifact behind the landmine finding. Without
  the hook, W1e joins the 09-18/19 list of "engine died, no traceback".
- The same morning's W3 `--lpt` window is already recorded as docs/05 F-11 (probe artifacts
  `tune/results/batch-mech-baseline-lpt0.json`, `batch-mech.json`); not duplicated here.
- One new process fact worth keeping: on the rc=2 path `run_window.sh` does **not** restore (the
  gate says "booted", provenance says "serving"), so a `BOoted-BUT-NOT-SERVING` verdict must be
  re-checked against provenance and a direct probe before anyone acts on it — W1f passed that check
  by luck of a separate G-1 run, not by design.

### 2026-09-20 12:33–12:58Z — W3i/W4: the tier stores, and under a valid test it never reads back

Two windows, both uncontended (measurement ran while no model turn was in flight), with per-arm
counter attribution added to `g1_correctness.py` after W2h showed label-flattened counters had
fooled us.

**W3i** (96 GiB tier, quiet engine): correctness clean (4/4 sizes identical text, `retrieval_ok=True`,
worst dlogprob 0.00199 < noise floor 0.00257). But every `restored` arm **stored again**
(16K→0, 64K→+478 MB, 150K→+1.08 GB) and the load series was absent entirely: `/reset_prefix_cache` +
re-send makes the engine **re-prefill**, so G-1's flush lever does not exercise the restore path at
all. G-1 verdict: INCONCLUSIVE (now correctly reported as "no benefit observed", with the earlier
FAIL reclassified — that FAIL was an empty arm from a refused flush, i.e. missing data).

**W4 / G-1b** (128 GiB tier, `g1b_natural_restore.py` — natural eviction instead of a dev flush):
A cold 86K-token prompt = **10.973 s** (7.8K tok/s, genuinely cold), flood **1,385,248 tokens of
unique content > the 1,152,677-token HBM pool** so P is truly evicted (the tier wrote 59.5 GB during
the flood), then re-send: **10.938 s (x1.00)**, identical text, **load_bytes +0.00 GB**, and it
**stored another 1.51 GB**. `CPU_to_GPU` stayed 0.0 for the whole window.

Read together, and stating the limit of the evidence:
- **Store direction: works**, at multiple GB per request, correctness无反证.
- **Load direction: never observed under a controlled test.** W2h did record 65.7 GB of loads, but
  only while a live multi-turn session was resuming its own context — so the plausible split is
  *continuation of an existing session loads; a brand-new request carrying identical content does not*.
  That is a claim about vLLM's connector semantics that we have not yet pinned to code, and it means
  **byte-equivalence of a restored page remains unverified** — the project's central correctness
  question is still open, not passed.
- Two self-inflicted measurement errors were caught and fixed the same day: a re-used content seed
  that made a "cold" 86K prefill look like 79K tok/s (the previous crashed run had warmed HBM), and a
  flood of 519K tokens that was smaller than the pool, so nothing had actually been evicted.

### Code-level reason the whole request can load nothing (read 2026-09-20, no engine touched)

`_lookup_complete_chunks` converges a single hit boundary across **all** groups, and inside the
per-group loop:

```
max_hit_size_tokens = min(max_hit_size_tokens, len(offload_keys) * tokens_per_chunk)
if max_hit_size_tokens - num_computed_tokens < tokens_per_chunk:
    # We can only load less than a chunk, so skip.
    return 0
```

There is no per-group partial result: one group whose chunk is coarser than the remaining hit makes
the **entire request** load zero. On this tree the groups differ in chunk width by two orders of
magnitude (attention 816, GDN 16, QSA ring 8), and `supports_partial_tail` is False because group
block sizes are not uniform — so the coarsest group's alignment rule alone can veto every load.
That matches W3i/W4 exactly: stores succeed per group, `_lookup` returns 0, `CPU_to_GPU` stays 0.0.

Status of the claim: the predicate is quoted from the installed code; that this (rather than key
identity or tier eviction) is *the* trigger for our workload is still unconfirmed — the trace patch
that would have confirmed it crashed EngineCore (patch 07, quarantined) and will be rebuilt as an
offline replay instead, not another production boot.

### 2026-09-21 — the veto claim, EXECUTED offline on the pristine image (`connector_lookup_veto_probe.py`, N10)

Method (all executed, no engine, no GPU, pinned image `sha256:0aea3024…`): built a real
`OffloadingConnectorScheduler` over the three-group set (attention 816 / QSA-ring 8 / GDN 16,
`tokens_per_hash=8`, `blocks_per_chunk=1`) with the pristine library's own `from_spec` +
`_lookup_complete_chunks` + `_maximal_prefix_lookup` + `_sliding_window_lookup` +
`RequestOffloadState.update_offload_keys`, driven by a controlled HIT/MISS manager (the tier's
residency is the only thing the harness sets — the veto is scheduler arithmetic over that), then
called the **real** `_lookup_complete_chunks` for an 86,000-token prompt, fully-evicted
(`num_locally_computed_tokens=0`). The ring needs a classification shim only so `from_spec` can
build (stock `get_sliding_window_size_in_chunks` asserts on `CircularBufferSpec` at :125); the
**lookup method itself is never patched**. Every case was cross-checked against an independent
reimplementation of the scalar loop — 11/11 executed returns agree with the arithmetic.

**Verdict on the claim: REFUTED as stated, with the real mechanism relocated.**

- Heterogeneous **chunk width** does not veto. With every lookup group complete and
  boundary-consistent, the real method returns a large hit — it does **not** return 0. The
  two-group width-mismatch tree [FA 816 + GDN 16] returns exactly the same hit as the uniform
  tree [FA 816 + 816] (84,864), so a "chunk-width reconciliation" would change nothing.
- The only executed path to a hard 0 is the **ring acting as an un-completable full-attention
  lookup group**: `CircularBufferManager` pins a single block, so the ring's
  `_maximal_prefix_lookup` yields at most one resident chunk, which caps `max_hit` below the next
  group's `tokens_per_chunk` and trips the quoted `return 0`.

| case (EXECUTED)                                            | hit    | where it stops |
|------------------------------------------------------------|-------:|----------------|
| a  [FA816+GDN16] all complete, MTP                         | 84,864 | return num_hit |
| b  = a, ring excluded from lookup                          | 84,864 | return num_hit |
| c  [FA816+GDN16] attn complete, GDN short by one chunk     | 84,864 | return num_hit |
| d  [FA816+GDN16] prompt 86,000 (misaligned), all complete  | 84,864 | return num_hit |
| d' prompt 86,016 (aligned to 816), all complete            | 84,864 | return num_hit |
| a-ctl [FA816+GDN16] all complete, no MTP                   | 85,680 | return num_hit |
| e  uniform width (both 816), all complete                  | 84,864 | return num_hit |
| a-SET3 [FA816+ring8+GDN16] all complete incl ring          | 84,856 | return num_hit |
| **a-real ring as lookup group, only 1 pinned chunk (MTP)** |   **0**| g-ring @808-810|
| **ring-sparse non-MTP**                                    |   **0**| g-ring→g-GDN @748-753 (the quoted line)|
| b-SET3 ring excluded, attn+GDN complete                    | 84,864 | return num_hit |

The last non-zero-vs-zero pair is the discriminator: with the ring present in `_lookup_groups` it
is **0** (and the non-MTP variant halts on the *exact* quoted `max_hit_size_tokens - num_computed
< tokens_per_chunk → return 0`); excluding the ring restores **84,864**.

**Minimal library-side fix (per the case boundary, not the width hypothesis):** do NOT chase
chunk-width reconciliation. Drop the ring group from `_lookup_groups`/`_sliding_window_groups`
exactly as revised patch-05 does (case b → 84,864). The residual risk that case (a-real) is
*also* the deployed failure is why the ring must be excluded from lookup, not just from stores.

**Executable prediction for the next production window:** with the ring excluded from the lookup
groups, a re-sent 86,000-token prompt after full HBM eviction must have `_lookup_complete_chunks`
return ≈84,856–85,680 (not 0), and `vllm:kv_offload_total_bytes{transfer_type="CPU_to_GPU"}` must
become > 0 (restore beats cold re-prefill, x>1.0 vs cold). **If it still returns 0, the deployed
tree has NOT removed the ring from the *lookup* groups** (old-05 shape — ring in
`full_attention_groups`) — the fix is confirmed wrong-or-not-applied, and the width hypothesis is
still not the cause. All labels here are EXECUTED (offline replay); the production numbers are the
next window to confirm.
