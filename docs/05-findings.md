# 05 — Findings from the first replay runs

Produced by `./acr replay` on this repo's simulator, with the cost model from docs/00. These are
model outputs, not engine measurements — they are the hypotheses that phase 1/2 must confirm or
kill. Where a finding *contradicts* the original idea, that is stated.

## F-1. The unit of cache value is a prefix chain, not a block

vLLM can only serve a **contiguous** resident prefix: block *n* is usable only if blocks *0…n−1*
are. So a block's value is not its own recompute cost — it is its contribution to a chain.

Consequence for the design (and it broke the first version of the adaptive policy): per-block
admission — the shape vLLM's manager exposes, `keys_to_store = [k for k in keys if policy.get(k)
is None]` — misprices every block, and a per-block "is this worth caching" test that uses a
seconds-scale threshold rejects *everything*, because a single 256-token block is only worth
~32 ms of recompute on this engine.

ACR therefore admits per **chain** (`admit_chain`) and evicts per block:

```
admit(chain)  iff   P̄(chain) ≥ P_incumbent · max(min_lease, pressure)
evict order       by  P_b·(T_recompute(b) − T_restore(b)) − λ_mem·T_recompute(b)·P_incumbent·pressure
```

i.e. **keep context whose reuse probability beats what it displaces, scaled by how full the tier
is**. The break-even test collapses to a comparison of reuse probabilities because both benefit and
displacement scale with bytes — a pleasant simplification, and it means the policy needs no
absolute threshold to tune.

## F-2. Eviction policy only matters inside a capacity band — and the band is the whole point

| host tier vs distinct working set | outcome |
| --- | --- |
| tier ≫ working set | **every policy produces identical numbers** — the tier just keeps everything, and the only decision that mattered was having the tier |
| tier ≈ working set | policies diverge; round-robin over 8×260K-token sessions in a 30 GiB tier: oracle 6.84 M recompute-tokens vs LRU 9.08 M (−25 %) |
| tier ≪ working set | all policies converge again at "recompute everything" — the tier is decorative |

This is a constraint on how the project must report results: **never quote a policy improvement
without the capacity ratio it was measured at.** It is also the honest read on the user's original
pitch ("fit a formula for each session's expected cache lifetime"): at our actual RAM/HBM ratio
(6.7×) and our actual working sets, a lifetime formula buys little; what buys a lot is the tier's
existence and the *promotion* decision (F-3).

## F-3. Most host-tier traffic is churn the GPU side caused

With a 60 GiB host tier under the round-robin trace, LRU restores hundreds of GiB of context that
had left **HBM** less than `thrash_window_s` (60 s) earlier — the engine's own LRU pushed out a
session, and the host tier paid PCIe to put it back minutes later. `thrash / restores` is high
because the round-robin pattern is exactly the pathological case for recency-based HBM eviction.

Read that as the quantitative argument for phase 3: the win available from an HBM residency hint
(pin-until-predicted-return, i.e. what Continuum does) is potentially larger than any host-tier
policy choice. But we cannot test that without engine cooperation, which means a window — so
phase 2 measures the part we *can* do unilaterally, and reports the churn ratio as the case for
the harder part.

## F-4. The storage tier is not a latency tier on this box, and that is a general rule

`BW* = bytes_per_token × prefill_rate` = 57.6 KB × 7.9K tok/s ≈ **446 MB/s**. Measured /data =
413–432 MB/s. Below the line, so `acr.vllm.launch.plan()` refuses to emit a storage tier for
latency purposes and the simulator keeps `disk.enabled: false`. The same arithmetic says a 3 GB/s
NVMe (or a remote RAM pool over the network) *would* clear it — so the rule, not this box's
answer, is the transferable result (docs/02, RQ3).

## F-5. Reproducing the measured interference for free

The simulator has no interference term. It models prefill as one contended server and PCIe as
another, and the measured production effect falls out of the queueing: a single 30K cold prefill
delays a concurrent short request by roughly the prefill's own duration, which is the 0.37 s →
3.42 s (9.2×) we observed live. A model that reproduces a measured number it was not fitted to is
the closest thing to validation available before phase 1, and it is why the two-server structure
is kept rather than replaced by a fancier queue.

## F-6. On our own real session, there is no residency problem at all

Replaying the captured Claude Code trace (343 turns, one session, context growing to 323K
tokens, 66.0 M prompt-tokens offered) through the model at the real 4×L20 sizes:

```
recompute 350.5 Ktok of 66.0 Mtok offered  ->  99.5 % served from HBM
host-tier involvement: 0 bytes               wasted prefill: 44.4 s over 14 h
TTFT p50 0.38 s / p95 0.88 s                 (measured production warm floor: 0.36-0.37 s)
```

A single agentic session fits entirely inside the HBM pool, so nothing ever leaves to the host
tier and every policy is identical. **The cold-start pain is not a single-agent problem — it is
strictly a concurrency/oversubscription problem**, which is worth saying out loud because the
original pitch ("make cold starts rarer") was framed per-session. It also means our measured
cumulative 73.8 % hit ratio (vs 99.5 % for one session alone) is itself the size of the
multi-session interference we are trying to fix.

## F-7. Under team-mode concurrency the host tier is worth 5–9×, and plain LRU captures none of it

> **⚠ DID NOT REPLICATE at the engine's real block granularity — see F-7b. The numbers below were
> produced at a simulated 256-token block size; the engine uses 816. Read F-7b before quoting any
> ratio from this table.**

Five concurrent copies of the real trace (1,715 turns, 339 M prompt-tokens offered), tier sized
below the 82 GiB working set (`experiments/logs/2026-09-18-team-real.log`):

| policy | 40 GiB tier: wasted | p95 TTFT | 100 GiB tier: wasted | p95 TTFT |
| --- | --- | --- | --- | --- |
| LRU | **1166.1 s (0 B restored)** | 35.2 s | 76.4 s | 4.17 s |
| LFU | 485.8 s | 13.3 s | 78.3 s | 4.23 s |
| fixed TTL (size-scaled, no reuse signal) | 1166.1 s | 35.2 s | 81.6 s | 4.28 s |
| Continuum-style TTL (recompute cost + announced gap) | 207.2 s | 4.95 s | 64.5 s | 3.94 s |
| **adaptive value (ours)** | 208.2 s | 4.98 s | **61.9 s** | **3.74 s** |
| Belady oracle | 131.8 s | 4.35 s | 52.6 s | 3.81 s |

Three things to take from it:

1. **Engine-recency tiering can be exactly worthless.** At 40 GiB, LRU and the size-based TTL
   restored *zero* bytes: with five round-robining sessions and room for ~2.5, recency always
   evicts the session that is about to come back. Same tier, right signal: 5.6× less wasted time.
2. **The announced revisit time is worth most of the win; the value function adds the last few
   percent.** Continuum-style TTL and adaptive-value are within 0.5 % at 40 GiB; adaptive only
   pulls ahead (−32 % vs Continuum, −19 % vs LRU) once the tier is big enough for choices to be
   marginal, where it reaches 88 % of the oracle's efficiency. Honest reading: buy the *signal*
   (tool ETA / session state at the gateway) before buying the *sophistication*.
3. **`thrash` is ~30 K of the ~50 K restores** at both sizes: most host-tier traffic is context
   the engine's own HBM LRU had just dropped. The host tier is largely repairing GPU-side
   eviction decisions — the quantitative case for the HBM residency hint (docs/01 §1, phase 3).

## F-8. What the model cannot see, and therefore where these numbers are optimistic

- HBM is modelled as a byte-capped LRU set; the real engine's block allocator, chunked prefill,
  MTP and preemption are absent, so real restore counts will be lower and real TTFT noisier.
- Requests are served by two single servers (prefill, PCIe) with no batching effects on decode;
  the 9.2× interference result is reproduced by queueing, but multi-stream decode contention is
  not modelled at all.
- `tool_eta_s` in the Claude Code loader is the *previous* inter-turn gap, i.e. what an agent
  would have to predict, not a true announcement; results with a real announcement channel
  should be better, and results with an uncooperative client should be worse.
- C-1 is still open (7.9K vs 41.7K tok/s), and it scales every `wasted_s` in the tables above.
  The *ranking* of policies is insensitive to it (it multiplies all recompute terms equally),
  which is why the conclusions are phrased as ratios rather than seconds wherever possible.

## F-7b. The F-7 ranking was an artifact of my own block size, and the replication failed

F-7 was produced with `block_tokens: 256`. The engine's real granularity is **816 tokens** (boot
log, docs/00 §1). Re-running the identical experiment at 816 (`results/exp-real-816.log`,
2026-09-19):

| policy | 40 GiB: wasted | p95 TTFT | 100 GiB: wasted | p95 TTFT |
| --- | --- | --- | --- | --- |
| LRU | 554.5 s (0 B restored) | 20.52 s | 146.8 s | 0.84 s |
| **LFU** | **248.8 s (45.7 GiB restored)** | **1.85 s** | 146.8 s | 0.84 s |
| fixed TTL | 554.5 s | 20.52 s | 146.8 s | 0.84 s |
| Continuum-style TTL | 554.5 s | 20.52 s | 146.8 s | 0.84 s |
| adaptive value (ours) | 554.5 s | 20.52 s | 146.8 s | 0.84 s |
| Belady oracle | 549.9 s | 19.86 s | 146.8 s | 0.84 s |

What changed:

1. **The 5.6× hint-driven win vanished.** At 816 tokens our adaptive policy is *statistically tied
   with LRU* (both restore zero bytes), the oracle buys almost nothing (−0.8 %), and the only policy
   that helps is plain **frequency** (2.2× less wasted time, p95 20.5 s → 1.85 s). "The signal is
   worth most of the win" was a conclusion drawn from my own parameterization, not from the system.
2. **Why**: at 256-token granularity a 285K-token chain is ~1114 blocks, so tier decisions are
   finely graded and revisit-aware eviction can select them one by one. At 816 tokens the same chain
   is 350 blocks of 46 MiB, and with a 40 GiB tier holding ~2.5 of 5 chains, whole-chain-granular
   decisions have nowhere to be clever — retention becomes all-or-nothing per session, so the only
   thing that matters is not throwing away blocks that are referenced often (LFU) instead of blocks
   touched last (LRU).
3. **At 100 GiB every policy is identical** (working set fits) — F-2's capacity-band result
   reproduced, but now it swamps the whole experiment: the earlier "adaptive wins at 100 GiB" was
   also granularity-conditioned.
4. **Single-session and 3-session runs are flat at both sizes** — reconfirming F-6.
5. **Unseparated confound we must not paper over**: our adaptive policy ties LRU (restoring nothing)
   at 40 GiB. That is consistent with "coarse tier + all-equal reuse probabilities leaves no room to
   be clever", but it is *also* consistent with our admission gate rejecting everything under
   pressure. This harness cannot tell those apart, and we did not run the internal sanity arm
   (adaptive with admission disabled) that would. Until that is run, F-7b supports "signals bought
   nothing here", **not** "our policy is fine".

Consequences for how the project runs from now on:
- the simulator's block size is a **first-class experimental variable**, not a config detail, and it
  must equal the engine's real value in any headline table;
- every policy claim now needs the *pair* (granularity, capacity ratio) attached to it;
- the "buy the signal before the sophistication" conclusion is **withdrawn pending phase A** — it may
  turn out the honest ordering is `tier + frequency > recency`, with signals mattering only at finer
  matching;
- this is the second self-inflicted result in this doc (F-1 also came from a bug) — which is an
  argument for measuring on the real engine (phase A) before building more policy logic.

## F-10 — the ranking contribution is real but **conditional on RAM being scarce**, and our box is not

Time-interleaved ward replay (12 coding sessions over a shared 120-block repo prefix + 20 patient
sessions, 40 turns each, coder cadence 8 s / patient cadence 45 s), driven through the deployed
`CPUOffloadingManager`, unique working set ≈ 1,440 blocks of 816 tokens:

| RAM (blocks → tokens) | LRU hit | ACR hit | Δ hit | Δ recompute-seconds |
| --- | --- | --- | --- | --- |
| 1200 → 0.98 M | 98.4 % | 97.4 % | −0.98 pt | +62 s |
| 600 → 0.49 M | 89.9 % | 85.2 % | −4.69 pt | +295 s |
| 500 → 0.41 M | 82.5 % | 85.3 % | **+2.77 pt** | −174 s |
| 450 → 0.37 M | 79.4 % | 84.3 % | +4.90 pt | −308 s |
| 400 → 0.33 M | 76.7 % | 84.0 % | +7.30 pt | −459 s |
| 350 → 0.29 M | 74.3 % | 82.7 % | **+8.36 pt** | −525 s |
| 300 → 0.25 M | 72.4 % | 79.8 % | +7.39 pt | −464 s |
| 240 → 0.20 M | 70.6 % | 76.4 % | +5.78 pt | −363 s |

### Amendment, same night: the baseline is ARC, not LRU

Adding the library's other registered policy (`arc`) and a faithful session-TTL policy
(`acr.vllm.ttl`, role TTLs patient 60 s / coding 3600 s, refreshed on touch) to the same sweep:

| blocks | lru | **arc** | ttl | acr |
| --- | --- | --- | --- | --- |
| 1200 | **98.4 %** | 98.3 % | 96.5 % | 97.4 % |
| 600 | 89.9 % | **93.4 %** | 79.8 % | 85.2 % |
| 500 | 82.5 % | **89.6 %** | 78.9 % | 85.4 % |
| 400 | 76.7 % | 82.6 % | 77.1 % | **83.8 %** |
| 350 | 74.3 % | 79.3 % | 75.9 % | **82.4 %** |
| 240 | 70.6 % | 72.7 % | 72.5 % | **76.8 %** |

So the honest crossover is **against ARC at ≈400 blocks (~28 % of the unique working set)**, not the
~40 % originally written from the LRU-only comparison, and the practical default at every capacity we
would actually run is `eviction_policy: "arc"`. Two further results:

* **Session TTL — the idea this whole line started from — is falsified as a mechanism**: worst or
  second-worst at every capacity, and at 600 blocks it is 13.6 points under ARC. Its failure mode is
  exactly the one a lifetime promise invites: it drops blocks that were about to be reused and keeps
  blocks nobody asks for.
* ARC's edge is largest precisely where ACR's role/ETA signal should matter least (mild pressure), and
  ACR's edge grows monotonically as the cap tightens — consistent with "declared future value only
  beats recency when recency's memory is too short to cover the reuse tail".

Below it, value ranking wins and the
latency class wins most: at 350 blocks patient hit goes 22.4 % → 40.5 % *and* coding 88.2 % → 93.9 %
— a Pareto move, not a trade. Above it, recency is the better prior and our ranking actively hurts.

Consequence, stated against ourselves: this box has 359 GiB of host RAM ≈ 8,200 blocks ≈ 6.7 M
tokens, i.e. **an order of magnitude above the crossover for any plausible session mix**. So the
production configuration of the tier is the library's `lru`/`arc` + `store_threshold`, and the value
function earns its keep only where RAM is deliberately capped (co-tenancy, or a residency budget
chosen to bound host-memory pressure). Any claim of the form "signals buy N×" must be quoted with
its capacity condition or it is false at our own operating point.

Also corrected on the way here, because each was measured rather than assumed:
* the first mixed stream was **two sequential phases** (all coders, then all patients) — under that
  arrival order recency is optimal by construction and the policy lost 14.8 pt. That measured the
  generator, not the policy; interleaving it changed the sign of the result.
* the replay **had no clock** (turns processed back-to-back), which made every recency/ETA term
  meaningless; it now advances a virtual clock from the trace's own gaps.
* the policy's geometry defaulted to `block_tokens=0` with a comment claiming inference that was
  never implemented, so `score()` returned 0.0 for every block and eviction degraded to set
  iteration order. See docs/08 / commit bf47d23.

## F-11 — `long_prefill_token_threshold` is not a latency lever on this hybrid+MTP tree (measured, reverted)

Window `W3-lpt2448` at 2026-09-20 08:28→08:33Z (boot 295 s, provenance verified, then reverted at
08:42:30Z to `restore-PROD-PATCHED-15.5`). Same probe, same arms, baseline kept at
`tune/results/batch-mech-baseline-lpt0.json`:

| arm | lpt=0 | lpt=2448 | verdict |
| --- | --- | --- | --- |
| cold-73K, patient arriving simultaneously | TTFT 0.502 s · ITL p50 241 ms | TTFT **4.699 s** · ITL p50 **384 ms** | both worse |
| cold-73K agent prefill | 10.37 s (≈7.1K tok/s) | **18.75 s (≈3.9K tok/s)** | −45 % throughput |
| cold-3×18K + patient | TTFT 5.551 s | 5.426 s | no gain |
| cold-6×18K + patient | TTFT 8.903 s | 8.202 s | −8 % |
| warm-73K + patient | 0.84 s · 18 ms | 0.505 s · 18.6 ms | warmth still dominates everything |
| KV capacity | 1,152,677 tokens · 1.15× | **identical** | the flag does not touch geometry, as predicted |

**The hypothesis was that a waiting latency request starves because one big chunk consumes the step
budget (`scheduler.py:728/779`), so capping the chunk leaves room.** It is falsified: capping made the
patient's TTFT 9× worse in the co-arrival case and cost the agent 45 % of its prefill throughput.
The code localizes *where*, not yet *which*: `num_new_tokens == 0` makes the scheduler `continue`
(skip a request for that step) for five documented reasons, and two of them are specific to this
tree — "insufficient budget for a block-aligned chunk in hybrid models with mamba cache mode align"
and "insufficient budget to keep a **multi-module MTP** prefill chunk out of the prefill-lookahead
window" (`scheduler.py:636-652`, with `:591-592` applying the cap). A sub-8192 chunk therefore drops
small requests below the alignment/lookahead floor instead of admitting them. Separating cause 4 from
cause 5 needs an `--mtp off` boot, which we will not spend on a knob we have already ruled out.

Consequences we act on:
1. **No `--lpt` on this stack.** Patient protection cannot come from chunk sizing.
2. The remaining levers are the **control plane** (cap concurrent agent prefills at the gateway —
   deployable without engine changes) and **`--scheduler-cls`** (verified present in this build,
   `arg_utils.py:1554`), which is now the only in-engine path to a real per-step reservation.
3. Warmth beat every scheduling knob in all five arms — which is the strongest argument yet for
   finishing the tier (W1e), not for tuning around it.

## F-12 — the stock offload tier is **one-way**: store fires at ~9 GB/s, restore never fires (measured, W1f/W1g)

Engine measurements, not model output. G-1 against the patched engine with a live
`OffloadingConnector` (`arc` eviction), 2026-09-20:

| window | tier | probe store Δ | store Δt | rate | load bytes | restored/cold ×4 arms | G-1 status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| W1f (16 GiB) | 17,179,869,184 B | 9,689,106,176 B | 1.062 s | 9.12 GB/s | 0 | 0.97 / 0.88 / 0.92 / 0.90 | INCONCLUSIVE (rc 8) |
| W1g (64 GiB) | 68,719,476,736 B | 9,689,106,176 B | 1.058 s | 9.15 GB/s | 0 | 0.97 / 0.59 / 0.92 / 0.90 | INCONCLUSIVE (rc 8) |

(`g1-W1f.json`, `g1-W1g-cpu64.json`, `window-W1g-cpu64/probe.txt`; ratios vs the `g1-pre.json`
cold baseline — restored arms below the 1.5× anti-vacuity bar, `g1_correctness.py:204`.)

Three things this settles:

1. **The store path is real and fast.** 9.1 GB/s is ~20× F-4's 446 MB/s latency break-even — the
   host region, the offload scheduler branch (patch 05), and the DMA all work.
2. **The load path is dead, and not for lack of budget.** At the engine's geometry (56.4 KiB/token
   aggregate, 46 MiB per 816-token block — docs/00 §1), 16 GiB ≈ 297K tokens and 64 GiB ≈ 1.19M,
   vs a 234K-token four-arm probe working set. If W1f left room to blame self-eviction, W1g
   falsifies it: the tier holds the whole probe ~5× over and still restores nothing.
3. **Byte-correctness of restore remains unmeasured.** `exact: true` ×4 in both windows holds only
   because a never-restored page is trivially consistent — the model recomputed it. Every `T_restore`
   term in this document is hypothetical against stock vLLM on this tree; the tier as shipped here
   is a write-only memory.

Root-cause candidates and the separating diagnosis live in docs/08 ("Phase-A verdict") and the
medical repo's `forensics/20260920-tier-load-path-diagnosis.md` (in progress): flush clearing the
connector's index, `OffloadKey` identity at `tokens_per_hash=8`, or load-side scheduling gates.
This is also the finding that converts the paper's claim from "hybrid trees need group-typed
offload" (docs/08, W1d/N7) into something sharper: even *after* the config walls and the spec
assert are cleared, stock's load path does not engage on this tree at all.
