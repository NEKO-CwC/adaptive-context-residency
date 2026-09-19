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
