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
