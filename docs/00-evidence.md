# 00 — Evidence base and cost model (2026-09-18)

Every number used by the simulator or the value function is listed here with its provenance and
its confidence class. **C-measured** = produced by an instrument on this host; **C-derived** =
arithmetic on measured values; **C-est** = labeled estimate pending measurement; **C-conflict** =
two sources disagree and it is unresolved.

The whole design is only as good as this table, so the disagreement is recorded rather than smoothed over.

## 1. Engine resource model (C-measured, from engine's own logs and errors)

| quantity | value | provenance |
| --- | --- | --- |
| single-card capacity | 46,068 MiB (44.97 GiB) | `nvidia-smi` |
| weight memory | 20.14 GiB/card | boot log |
| non-KV steady footprint | ≈6.4 GiB/card | nvidia-smi while serving (40,966 MiB used, util 0.80, kv 13.5) |
| `--kv-cache-memory-bytes` semantics | **per GPU rank**, **skips memory profiling**, **ignores `gpu_memory_utilization`** | worker log verbatim: *"Initial free memory 38.42 GiB, reserved 12.0 GiB … skipped memory profiling. This does not respect the gpu_memory_utilization config."* |
| 1M-context KV floor | 13.45 GiB/card (at 12 GiB the engine refuses: *"estimated maximum model length 890256"*) | engine ValueError verbatim |
| capacity at 13.5 GiB/card | 1,003,197 tokens | boot log |
| capacity at 15 GiB/card | 1,110,107 tokens | boot log (C5) |
| capacity at 17 GiB/card | 1,263,788 tokens (+26 %) | boot log — **not deployable**, see the retraction below |
| capacity at **15.5 GiB/card** | **1,152,677 tokens (+14.9 %)** | **production, gated with proven attribution** (container id + `--kv-cache-memory-bytes` sampled before and after): 300×3 growing-prefix, 0 failures, 671 s, peak 43,506 MiB/card, **min free 2,011 MiB** |
| capacity at 13.5 GiB/card | 1,003,197 tokens | validated manual alternative (300×3 soak, min free 3.33 GiB); costs 12.9 % of the pool |
| capacity linearity | ≈74,300 tokens / GiB / card | 3-point fit of the rows above |
| block / prefix-match granularity | **816 tokens** | boot log verbatim: *"Setting attention block size to 816 tokens to ensure that attention page size is >= mamba page size"* + *"Padding mamba page size by 1.62%…"* (all 4 ranks). `hash_block_size` = `prefix_match_unit` if set, else GCD of prefix-cacheable group sizes (`v1/core/kv_cache_utils.py:612-672`) |

Derived (C-derived):

- KV bytes per token, **aggregate over 4 ranks**: 13.5 GiB × 4 / 1,003,197 ≈ **56.4 KiB/token**
  (≈14.1 KiB/token/rank).
- A 1M-token context = **55 GiB aggregate** (13.8 GiB/rank); 300K = 16.5 GiB; 70K = 3.9 GiB.
- At 816 tokens/block, one block = **46 MiB aggregate** (5.8 MiB/rank) and 70K = ~86 blocks.
  Consequence for invalidation blast radius: a content change at position *x* re-misses every
  block after *x*, so an edit at 10K in a 70K context strands ~73 blocks ≈ 3.4 GiB ≈ **8.5 s** of
  re-prefill at the measured 7.9K tok/s. Coarse matching is a real cost — but it is the cost
  side of the *engine's* hash, not a knob ACR introduces (docs/05 F-9).
- HBM pool is a **single shared budget**, not a per-request reservation. Oversubscription does not
  fail: it causes LRU eviction of idle prefixes, and preemption/recompute when nothing idle remains.

## 2. Throughput and latency (C-measured, read-only probes on the live engine)

| quantity | value | provenance |
| --- | --- | --- |
| cold prefill, marginal rate | **11.1–13.3K tok/s** (4K→16K: 13.3K; 16K→64K: 12.0K; 64K→150K: 11.1K) | `tune/results/g1-pre.json`, differences of TTFT between sizes so fixed overhead and decode cancel out |
| cold prefill, average incl. overhead | 7.2K (4K prompt) → 11.3K (150K prompt) | same; the old "7.9K tok/s" was this kind of number (prompt/**e2e**) measured while other requests were sharing the engine — it is not a rate, and it is superseded |
| decode, single stream, by content | **273 / 161 / 135 tok/s** for easy(counting) / medium(prose) / hard(invented tokens) | `tune/` probe, 256-token greedy generations, repeated within 2% |
| decode, 4 concurrent, medium | 113–120 tok/s per stream, **469 aggregate** | same — this reproduces the older "118 tok/s" figure exactly, which means that number was a *contended* measurement, not a single-stream baseline |
| MTP acceptance, by content | 100 % (4.00/step) / 50 % (2.00) / 37.6 % (1.50); per-position medium = 77/56/36/31 % | `/metrics` `spec_decode_num_accepted_tokens_per_pos_total` deltas around each probe |

**Decode throughput is not a constant on this stack.** With `num_speculative_tokens=4`, speed is set by
draft acceptance, which is set by content predictability: 273 tok/s on trivially predictable output
versus 135 tok/s on invented tokens, on the same engine and the same config. Any latency model that
uses one decode number will be wrong by ~2× in one direction or the other, so the simulator now
takes `decode_tokens_per_s` as the *effective production mix* value and the tables above are the
source. It also means patient-role output (strict JSON with per-turn dynamic enums) behaves like the
**hard** row, not the easy one.
| warm (prefix-cached) short request TTFT | 0.36–0.37 s | same |
| short-request TTFT during a 30K cold storm | **3.42 s (9.2×)** | same |
| prefix-cache hit ratio, cumulative | 73.8 %; 93–94 % in steady multi-session windows | `/metrics` |
| MTP acceptance | 27,904/56,848 tokens ≈ 1.96 of 4 | `/metrics` |
| growing-prefix fault family exposure | 7 harness runs since, incl. 300×3 PASS at 13.5 and 15.5 GiB; **one FAIL at 17 GiB (CUDA OOM, not the fault family)** | `tune/results/soak-*`, `tune/../premortem-oom-20260919.txt` |

### C-conflict: cold-prefill throughput at long context

Recorded earlier in this project: *"400K cold re-prefill 9.6 s"* ⇒ ≈41.7K tok/s.
Measured yesterday: 30K cold ⇒ 7.9K tok/s. These cannot both describe the same code path, and
**cold-prefill cost is the numerator of the entire residency value function** (`C_recompute`).

Possible benign explanations (unverified): the 400K figure was measured on a partially-warm
prefix, or counted only a segment of the request, or came from a different `max_num_batched_tokens`.

**C-1 — RESOLVED 2026-09-19** by a read-only probe on the live engine (`tune/results/g1-pre.json`,
identical deterministic content, thinking off, temp 0):

| fresh prompt | cold TTFT | effective tok/s |
| --- | --- | --- |
| 4,000 | 0.55 s | 6.8 K (overhead-dominated) |
| 16,000 | 1.45 s | 10.8 K |
| 64,000 | 5.47 s | 11.6 K |
| 150,000 | 13.23 s | 11.2 K |

So cold prefill is **~11K tok/s and roughly linear in this range**; the older "400K in 9.6 s"
(41.7K tok/s) cannot describe the same code path and is retired. The simulator's default moves to
`prefill_tokens_per_s = 11_000`; the break-even bandwidth of docs/00 §4 becomes ~624 MB/s, which
puts the measured NVMe tier (413–432 MB/s) even further below the line — the storage-tier verdict
strengthens rather than weakens. The 200K/400K points are still unmeasured, so the quadratic term
remains unknown and long-context recompute is still a *lower*-bound estimate.

## 3. Host and fabric (C-measured today)

| quantity | value | provenance |
| --- | --- | --- |
| RAM | 503 GiB total, **359 GiB available** | `free -g` |
| swap | 39 GiB (irrelevant for latency; not a tier) | `free -g` |
| /data | 3.0 TiB LVM, 1.8 TiB free, non-rotational | `df`/`lsblk` (`sda` LOGICAL VOLUME, ROTA=0) |
| NVMe sequential write | **432 MB/s** (1 GiB, `oflag=direct`) | `dd` |
| NVMe sequential read | **413 MB/s** (1 GiB, `iflag=direct`, caches dropped) | `dd` |
| PCIe link | **Gen4 ×16** (current == max) | `nvidia-smi --query-gpu=pcie.link.gen.current,...` |
| H2D practical bandwidth | ≈24 GB/s/rank (Gen4 ×16 raw 31.5 GB/s × ~0.75 efficiency) | **C-est**, resolution target **M-1** |

## 4. Cost model as instantiated on this box

Per-token cost of recovering context, using the numbers above:

| path | per token | 1M context | vs recompute | verdict |
| --- | --- | --- | --- | --- |
| recompute (cold prefill @7.9K tok/s) | 127 µs | ~127 s (linear lower bound; quadratic term unmeasured) | 1.0× | the baseline we are avoiding |
| **host RAM → HBM** | ≈0.6 µs (56.4 KiB / 96 GB/s aggregate) | ≈0.6–1.5 s | **≈80–200× cheaper** | the whole value of the project |
| NVMe → RAM → HBM | ≈136 µs + 0.6 µs | ≈136 s | **≈1.05× — no win** | durability/warm-boot only, never a latency tier |

Even if M-1 shows the effective H2D bandwidth at half the estimate, RAM restore stays ~40× better
than recompute. The NVMe row is robust in the other direction too: at 0.4 GB/s there is no
context length for which the disk tier beats recomputing, so **we deliberately do not model a
four-tier latency path** — it would be decorative complexity.

## 5. What this implies for the original capacity question

The request "keep 1×1M + 2×500K + several ≤200K resident at once" = 2.2–2.8M tokens needs
~37.7 GiB/card of HBM — impossible on 45 GiB cards. But it **fits in host RAM** (2.8M tokens ≈
154 GiB of the 359 GiB available) at a restore cost of ~1–2 s per promotion. That converts a
hardware-impossible requirement into a scheduling problem, which is the reason this repo exists.
Gated entirely on §5 of docs/04: does the offload path preserve KV bit-exactly for *this* hybrid
GDN + MTP + `inc` + patched build.

## 6. What actually bounds the pool: the page-equality identity (measured 2026-09-19)

Two boots, one variable (`--mamba-ssm-cache-dtype bfloat16`, i.e. the GDN recurrent state
halved), both at `--kv-cache-memory-bytes 17 GiB/card`:

| state dtype | attention block size (log) | mamba page padding | pool tokens |
| --- | --- | --- | --- |
| float32 (`auto`, from HF config) | 816 | 1.62 % | 1,263,788 |
| bfloat16 | **432** | 3.10 % | **1,276,495** (+1 %) |

Halving the state cost nothing in capacity and bought exactly a 2× finer prefix granularity.
That is not a coincidence — it is forced by the engine's own invariant. With one attention group
(A = bytes/token) and one recurrent-state group (S = bytes per checkpoint, independent of block
size), the allocator demands equal pages: `A·B = S`, so `B = S/A` and

```
capacity = pool_bytes / page_bytes × B = pool_bytes / (A·B) × B = pool_bytes / A
```

**Capacity is set by attention bytes per token. Recurrent state is free in capacity and is paid
for in granularity.** The identity reproduces both measured points (predicted B = 413 vs measured
432, within padding), and it makes one falsifiable prediction: `--kv-cache-dtype fp8` should nearly
double the pool (→ ~2.55 M tokens at 17 GiB) while pushing B back to ~825 — unless the bf16 state
change is kept, in which case B stays near 432 at the same doubled capacity.

Corollary for this box: `17 GiB` is *already* "all the VRAM minus a small margin". Measured peak
non-KV resident (after CUDA graph capture and 300×3 harness load) is **7.41 GiB/card**, so the
ceiling is 44.99 − 20.14 − 7.41 ≈ **17.44 GiB**, and the driver reported 23 MiB free at 17 GiB.
Any capacity growth must come from `A`, not from claiming more GiB.

### Retraction: "17 GiB is gate-proven" was wrong

On 09-17 a 300×3 growing-prefix soak passed at `kv=17 GiB, util=0.86`. On 09-19 the same
reservation at `util=0.80` hit `torch.OutOfMemoryError` (446 MiB requested, 272 MiB free) at round
~188 of the *identical* soak, after `restarts=10` — traceback preserved in
`deploy/gpu/validation/premortem-oom-20260919.txt`. The likeliest difference is not the util number
but **what else was running**: the 09-19 soak shared the engine with live agent sessions (this one
included), which raises the non-KV peak and the block churn; its 300 rounds also took 1229 s vs
575 s, i.e. ~2.1× slower under real co-resident load.

Consequence for how capacity is chosen here: headroom must be measured **with production traffic
co-resident**, not by a dedicated soak, and the reservation must be derived from a margin
inequality rather than from "what fits". Hence the reservation is derived from the inequality, not from what fits. Measured at 15.5 GiB:
min free **2,011 MiB** (the inequality predicted ~1.3 GiB, so the estimate was conservative by
~0.7 GiB) — and at 17 GiB it went negative, which is what the retraction above is about.

### Attribution rule for capacity gates (learned by violating it)

The "15.5 GiB gated with 3.33 GiB margin" claim I first wrote here was **mis-attributed**: the
watchdog had restored the 13.5 GiB fallback four minutes before that soak began
(`results/deploy-log.tsv`, 12:17:07Z), so the numbers belonged to a different config. From now on a
capacity gate is only valid if the soak records the live container's id and
`--kv-cache-memory-bytes` before and after and asserts they match — enforced in
`tune/soak_with_attribution.sh`, which **refuses to run** if it cannot read them (an unattributable
measurement is worse than none, because it looks attributable).

## 7. What the real workload footprints are, and which capacity targets are reachable

Medical-side numbers are read out of the project's own evidence, not assumed:
`CR-20260819-h2-patient-simulation-runtime-recovery/evidence/G2/*/preflight.json` records, per
patient turn, `input_tokens` 3,873 and 4,031 with `reserved_output_tokens` 2,800,
`safety_margin_tokens` 256 inside a `context_window_tokens` 9,216 budget (roles enumerated by
`contracts/hospital-simulation/v2/patient-behavior-live-context-budget-v1.schema.json`:
`action_judge`, `patient`, `final_review`, `independent_evaluator`).

| claim | arithmetic | verdict |
| --- | --- | --- |
| "reserve 500K of KV for medicine" | 20 concurrent patient sessions × ~4K = **80K tokens = 6.9 %** of the 1.15 M pool | over-provisioned by ~6×; medicine's scarce resource is **decode slots and ITL isolation**, not KV |
| "grow the pool to 1.5 M tokens" | needs 1.5 M / 74,366 tok·GiB⁻¹ = **20.2 GiB/card**, ceiling is 17.44 GiB/card (§6) | **unreachable** — there is no 1.5 M tier; the only two points are ~1.15 M and ~2.3 M |
| "two 1 M agents at once" | 2 M / 74,366 = **26.9 GiB/card** | unreachable at bf16 attention; reachable only if `A` halves (fp8) |
| fp8 attention KV, pre-registered prediction | `capacity = pool / A`, `B = S/A` | pool 1.15 M → **≈2.3 M**; `A` halved → block `B` doubles to **≈1,632** unless `--mamba-ssm-cache-dtype bfloat16` is combined, which restores `B` ≈ 816 at the doubled capacity |

### Already-measured cost of the granularity knob (bf16 state), from `tune/results/g1-bf16.json`

One capture with `--mamba-ssm-cache-dtype bfloat16` against the fp32 baseline (`g1-pre.json`), same
engine, same content, greedy, thinking off:

| prompt | cold text identical | max Δlogprob | marginal prefill rate |
| --- | --- | --- | --- |
| 4 K / 16 K | yes | 1.7e-4 / 1.8e-4 | contaminated (other load on the engine) — do not use |
| 64 K → 150 K | yes | 2.1e-3 → 3.9e-3 | 11.1 K → **10.4 K tok/s (−9 %)** |

So halving the recurrent state is **not free**: it buys a 2× finer page (816 → 432, measured) for
~9 % of marginal prefill throughput and a small non-zero logprob drift, while capacity moves +1 %
(exactly as the page-equality identity predicts). Any gate for a lossy dtype change must therefore be
a *quality* gate (argmax text + behaviour eval + MTP acceptance), never byte equivalence.

## 8. Batch sharing between the two traffic classes (C-measured 2026-09-20, live engine, no config change)

Instruments: `deploy/gpu/validation/vllm-stability/tune/{batch_interference_probe,batch_mechanism_probe,agent_cost_probe}.py`.
Every arm records `num_requests_running/waiting` and the `waiting_by_reason` labels around itself,
so contamination and mechanism are read out rather than assumed. Patient arm = warm 2,274-token
context + strict-JSON persona reply (~90–150 tokens, the *hard* content class); agent arm = a cold
~73.7K-token prefill with `max_tokens=1`, so it occupies the step budget without decoding.

**Two distinct harms, separated by arrival order — this is the finding:**

| situation | patient TTFT | patient ITL p50 / p95 | agent prefill |
| --- | --- | --- | --- |
| patient alone | 0.48 s | 18.7 / 19.4 ms | — |
| patient admitted **into** an in-flight 73.7K cold prefill | **9.9–10.6 s** | 18.7–24.7 / 19–177 ms | 7,203–6,806 tok/s (1.02–1.08× slower) |
| patient admitted **before** the prefill starts (both running) | 0.50 s | **241 / 1,396 ms** | 7,338 tok/s |
| 4 patient streams + one big prefill | ~10.4 s | 24.3–24.7 / up to 201 ms | 6,806 tok/s |
| same 73.7K prompt **warm** (hit ratio 0.805) | 0.84 s | 18.0 / 19.0 ms | 28,760 tok/s |

Read from this, in order of consequence:

1. **A newly arriving request gets essentially nothing from a step owned by a running prefill.**
   `token_budget = max_num_scheduled_tokens` (8192 here) is consumed by the running request's chunk
   first; the waiting loop only sees the remainder, which is ~0. The engine labels such requests
   `waiting_by_reason{reason="capacity"}` (measured 3.0 with 6 agents + 1 patient; `deferred` stayed
   0) — so the starvation is budget, not MTP deferral. This is the thing
   `long_prefill_token_threshold` addresses, and the prediction is sharp: cap the agent chunk at
   *T* and a waiting patient's 2.3K prefill fits in the following step, so patient TTFT should fall
   from ~10 s to ≈ its own prefill plus one chunk (`T`/rate), while the coding side has already been
   measured to pay only 1.02–1.08× for co-scheduling.
2. **Once both are running, the harm moves from TTFT to ITL:** 241 ms median / 1.4 s worst between
   patient tokens. For a role whose output is watched as a stream, that is the worse product
   failure, and it is invisible in every TTFT-based SLO — including the "0.48 s/call" number this
   project quoted for two days.
3. **Priority cannot fix either of these.** It reorders the waiting queue and preempts on KV
   exhaustion (§6); both harms happen in the allocation among *running* requests.
4. **Warmth removes both** (last row): the same 73.7K context, cached, costs the patient 0.84 s and
   normal ITL, and is 4× cheaper for the agent. Residency is therefore not only a recompute-saving
   mechanism — it is the interference-avoidance mechanism. That reframes ACR's value claim: the tier
   buys latency isolation as well as capacity.
5. `max_num_seqs=4` binds only beyond 4 in-flight requests (6 agents + 1 patient → run≤4, wait≤3).
   With ≤4 requests nothing waits on slots, so the slot cap is a *scaling* limit, not today's
   bottleneck.

### C-conflict (new, unresolved): cold-prefill rate 7.3K vs 11.1K tok/s

Two single-request measurements of the same engine/config disagree by 1.55×: `g1-pre.json` gives
90 µs/token (11.1K tok/s) for 64K→150K, while these probes give 136 µs/token (7.3K tok/s) for a
73.7K `case_prompt`. The likeliest cause is the probe's own content: `case_prompt` builds from a
repeated sentence, so a fresh seed still shares its first ~40 % of blocks with every earlier prompt
(hit ratios of 0.18–0.20 in arms meant to be cold) — i.e. these are *partially warm*, and the
"cold" label is wrong. Consequence for how the numbers may be used: the **ratios** inside one arm
(patient TTFT/ITL, agent slowdown) are valid, the **absolute** prefill rate of these synthetic
prompts is not, and the break-even bandwidth must keep using the G-1 curve. To be closed by
re-measuring with position-unique random filler.

### 8.1 The gate is admission, not the shape of the backlog (same session, follow-up probes)

A request of **8 tokens** launched 3.0 s into a cold 73.7K prefill waited **7.36 s**; launched at 6.0 s
it waited **4.01 s** — in both cases exactly `agent_e2e − offset`, i.e. until the prefill ended. An
8-token request cannot be blocked by a shortage of *its own* size, so what blocks it is the step's
leftover budget: `scheduler.py:779` only enters the waiting-queue loop `while … and token_budget > 0`,
and `token_budget` (init 8192, line 524) is decremented by the running request's chunk at line 728.
Verified in this build that `draft_slots = 0` for plain MTP (`speculative.py:1469-1494`, "MTP / not
parallel / not draft_model → 0"), so the reserve is not what eats the budget.

**Retracted instrument:** `vllm:iteration_tokens_total` is *not* per-step here — a single-request
73.7K prefill recorded **one** observation totalling 59,025 tokens (Δcount=1). Any claim about
"effective chunk size" derived from that histogram (including a ~1.8K-token step implied by the
patient's 241 ms ITL) is unsupported, and the open question is stated instead of answered: how this
build actually splits an 8192-token budget across a hybrid+MTP chunked prefill. The cheap, decisive
experiment is the boot-flag A/B itself — if `long_prefill_token_threshold=2448` lets the mid-prefill
tiny request through, the budget model above is right and the fix is native; if it does not, there is
a policy gate not yet located, and `--scheduler-cls` becomes the only path.
