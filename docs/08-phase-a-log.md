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
