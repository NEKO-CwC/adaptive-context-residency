# 03 — Roadmap, phases, gates

The ordering is constrained by one fact: **the engine that would host ACR is the engine serving the
session that is writing this project.** Every phase below is labeled with whether it can be done
without touching a running engine.

## Phase 0 — offline, zero risk (this repo, now)

| task | output | needs engine? |
| --- | --- | --- |
| T-0.1 trace schema + codec | `src/acr/trace.py` | no |
| T-0.2 real agentic traces from Claude Code session JSONL (`input_tokens`/`output_tokens`/`thinking_tokens`/timestamps per turn) | `trace/ncu-cc-*.jsonl` | no |
| T-0.3 mixed synthetic workload generator (coding + patient + reviewer, shared prefix, tool gaps) | `src/acr/trace.py::synthetic_mix` | no |
| T-0.4 multi-tier replay simulator, policy-pluggable, baselines + oracle | `src/acr/replay.py` | no |
| T-0.5 cost-model calibration on the live engine (read-only requests): cold prefill 20K/50K/100K/200K/400K fresh prefixes → fit `T_recompute(n)`; resolves **C-1** | `bench/cold_prefill_curve.py` | read-only API only |
| T-0.6 H2D bandwidth (resolves **M-1**) — needs a GPU alloc; do it in the window or on another box | `bench/h2d_bw.py` | yes (small alloc) |

Exit: a comparison table (LRU / LFU / fixed-TTL / Continuum-TTL / adaptive / oracle) on a real
trace, with the sensitivity of the conclusion to the C-1 disagreement shown.

## Phase 1 — shadow controller on production, still no engine change

Put the residency index **next to the gateway** and let it predict what the engine currently holds;
never act.

| task | output |
| --- | --- |
| S-1.1 gateway-side prefix trie over request bodies (content-addressed blocks), session leases, role/tool fields | `src/acr/shadow/` |
| S-1.2 reconcile with engine truth: `/metrics` prefix hit counters, `nvidia-smi`, (if a window enables it) `--kv-events-config` ZMQ events → measure **shadow-index error** | `acr/shadow/diff.py` |
| S-1.3 1–2 weeks of traces: patient + coding + reviewer, real inter-arrival and growth | `trace/prod-*.jsonl` |
| S-1.4 counterfactual replay of the captured stream against every policy | replay reports |

Exit criteria: shadow index explains engine hit-rate movement within a stated tolerance (target:
predict the cumulative hit ratio to ±3 pp, and per-request first-hit-block position to ±1 block
median — to be re-targeted once real error is measured, not guessed), and at least one policy
beats LRU by a margin that survives the C-1 sensitivity sweep.

This is the phase that decides whether the project has a result or only a mechanism. It is also
entirely safe, and it is the phase an application engineer can own end to end.

## Phase 2 — connector on, in a maintenance window (gated)

One restart, one config block (docs/01 §4), one rollback target (`last-good`).

| gate | test | pass condition |
| --- | --- | --- |
| **G-1 correctness** | byte-exact round trip: store→evict→restore the same prompt, compare logits/first-token distribution at temp 0 across ≥50 prompts spanning 4K/70K/300K/1M contexts, plus a checksum of the host buffer | identical outputs; zero NaN; 0 diffs |
| **G-2 abort hygiene** | the "same-token/different-cache" audit from arXiv 2608.15939 on our own stack: branch → abort → rebuild vs retain | no protected-effect flip when retention crosses an abort boundary |
| **G-3 stability** | the existing growing-prefix harness, conc 4 and 8, ≥300 rounds, with connector on | 0 crash / 0 NaN / 0 preemption, same as the patched baseline |
| **G-4 interference** | the 30K cold-storm probe, with transfers saturating PCIe concurrently with decode | decode tok/s degradation ≤10 %; short-request TTFT ≤2× baseline |
| **G-5 benefit** | eviction-pressure soak: force HBM oversubscription (many long sessions), compare connector-on vs off | measured TTFT/JCT improvement matching the phase-0 prediction within a stated factor |

If G-1 or G-4 fails, the project's honest conclusion is "this build cannot carry a KV tier" — which
is a publishable negative result and immediately useful to the parent project (it means stop
spending windows on KV).

## Phase 3 — policy that needs HBM cooperation

Only if phase 2 shows the residual bottleneck is GPU-side retention (predicted: HBM is 6.7×
smaller than RAM, so it likely will be).

- Upstream `CachePolicy.admit()` hook and/or a residency-hint API (small PRs; we already carry 4
  patches, so keep the private delta near zero and prefer upstreamable form).
- `kv_transfer_params`-driven promotion hints (`acr.promote`, `acr.ttl`, `acr.invalidate_tx`).
- Possibly the `p2p` tier for a second engine, which turns residency into routing (Dynamo-style).

## Phase 4 — paper

RQ1 (hybrid-state residency economics) is the headline; RQ2 (rollback consistency) is the
correctness section that makes reviewers care; RQ3 is the sizing-law section. Needs: ≥2 model
families with different KV/state ratios, ≥3 workloads, online + replay, baselines including
Continuum-style and oracle, ablations for each signal, artifact + traces.

## Sequencing against the medical project

Nothing here blocks the patient-model consolidation, and the consolidation does not need ACR:
patient traffic is noise-level load (≈13 req/min, <0.5 s/call). Where they touch:

- the **gateway** is the shared piece of plumbing — it becomes the signal source in phase 1, so the
  S1–S5 serving-layer changes (role-class stamping, `chat_template_kwargs`, priority) should be
  written so they can also emit `kv_transfer_params` later without a redesign.
- **role labels** (patient / reviewer / coding) are the categories the reuse predictor needs
  (ATC'25 says category-conditional reuse is predictable) — the medical workload is the trace that
  demonstrates it, which is also the paper's heterogeneous-mix argument.
- the abort paths (Core bounded repair, Dify retry) are G-2's test cases.
