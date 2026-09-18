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
| capacity at 17 GiB/card | 1,263,788 tokens (+26 %) | boot log (C4/V1), gate-proven: 300×3 growing-prefix PASS |
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
| decode, single stream | 118 tok/s | `tune/live_scaling_probe.py` |
| decode, 4 concurrent | 132–176 tok/s per stream, aggregate ≈5× | same |
| cold prefill, 30K fresh tokens | 3.8 s → **7.9K tok/s** | same |
| warm (prefix-cached) short request TTFT | 0.36–0.37 s | same |
| short-request TTFT during a 30K cold storm | **3.42 s (9.2×)** | same |
| prefix-cache hit ratio, cumulative | 73.8 %; 93–94 % in steady multi-session windows | `/metrics` |
| MTP acceptance | 27,904/56,848 tokens ≈ 1.96 of 4 | `/metrics` |
| growing-prefix fault family exposure | 6 harness runs, 590+ rounds, 0 crash / 0 NaN / 0 preemption | `tune/*/results.csv` |

### C-conflict: cold-prefill throughput at long context

Recorded earlier in this project: *"400K cold re-prefill 9.6 s"* ⇒ ≈41.7K tok/s.
Measured yesterday: 30K cold ⇒ 7.9K tok/s. These cannot both describe the same code path, and
**cold-prefill cost is the numerator of the entire residency value function** (`C_recompute`).

Possible benign explanations (unverified): the 400K figure was measured on a partially-warm
prefix, or counted only a segment of the request, or came from a different `max_num_batched_tokens`.

Resolution target **C-1**: measure cold prefill at 20K/50K/100K/200K/400K fresh prefixes on the
live engine (each is just a request — no restart, no config change) and fit
`T_recompute(n) = a·n + b·n²`. Until then the simulator's `prefill_tokens_per_s` is a
single configured constant with the pessimistic (7.9K) value as default and a sensitivity sweep.

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
