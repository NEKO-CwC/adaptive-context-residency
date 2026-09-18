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
