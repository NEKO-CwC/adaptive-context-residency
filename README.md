# Adaptive Context Residency (ACR)

A policy layer that decides **which computed LLM context lives where, and when it moves** —
GPU HBM, host RAM, or storage — using application/session signals the inference engine cannot see.

Target engine: vLLM (our fork `0.1.dev20073+g8e685d198`, Qwen3.8-Flash-Next hybrid GDN+MTP,
TP4/EP4 on 4×L20, 1M ctx). The design is engine-agnostic; the vLLM binding is out-of-tree plugins,
**no fork, no patch**.

## Why this exists

The engine is not compute-bound or even memory-bound in the way it looks. Measured on the host
(this is our evidence base, see [`docs/00-evidence.md`](docs/00-evidence.md)):

| quantity | value | source |
| --- | --- | --- |
| KV pool in HBM | 1,003,197 tokens (13.5 GiB/card) — 1,263,788 at 17 GiB | engine boot logs |
| KV bytes per token (aggregate over 4 ranks) | ≈56.4 KiB | derived from the two rows above |
| cold prefill throughput | ≈7.9K tok/s (30K fresh prefix → 3.8 s) | read-only live probe |
| host RAM free | 359 GiB ≈ **6.7M tokens ≈ 6.7× the HBM pool** | `free -g` |
| H2D bandwidth | PCIe **Gen4 ×16** per GPU (≈24 GB/s practical, to be measured) | `nvidia-smi --query-gpu` |
| NVMe /data | 432 MB/s write / 413 MB/s read (direct) | `dd` measured here |

Consequences, in order of how much they change the plan:

1. **Restoring a context from host RAM is ~200× cheaper than recomputing it (⚠ prefill rate is
   contested, docs/00 §2 C-conflict — the ratio is 10× softer if the 41.7K tok/s figure is right)****
   (0.6 µs/token vs 127 µs/token), and RAM holds 6.7× more tokens than the GPU pool.
   The "how much VRAM can we carve out" question is the wrong question.
2. **The storage (NVMe) tier buys nothing for latency on this box** — 136 µs/token to restore
   is *worse* than recomputing. It only buys durability across restarts. So this is a
   **two-tier residency problem**, not a four-tier one.
3. **The scarce resource is placement timing, not capacity**: one 30K cold prefill storm raised
   short-request TTFT by **9.2×** (0.37 s → 3.42 s). Every eviction mistake turns into a
   prefill storm that hurts unrelated traffic.

So: given ~6.7M tokens of RAM-tier context and 1.0M tokens of HBM, **which blocks get promoted
into HBM, when, and which never leave RAM** — decided from session/role/tool/workflow state that
only the application layer has.

## What already exists in the engine (verified, file:line in docs/01)

vLLM in this build ships the entire data plane and an explicitly out-of-tree plugin API:

- `TieringOffloadingSpec` + `OffloadingConnector` (`SupportsHMA`, i.e. hybrid-memory-allocator aware)
- `CachePolicy` ABC with `eviction_policy` / `cache_policy_module_path` —
  docstring: *"out-of-tree, no vLLM fork/patch required"*
- `SecondaryTierManager` ABC with `secondary_tiers[].type/module_path`; built-in `fs`, `obj`,
  `p2p` (ZMQ control + NIXL data) tiers
- per-request side channel: `kv_transfer_params` → `ReqContext.kv_transfer_params`, plus the
  already-honoured `kv_load_tiers: [{medium, locality}]` filter
- `Medium.{CPU,STORAGE}` × `Locality.{LOCAL,REMOTE}` tier taxonomy, `OffloadPolicy.{BLOCK,REQUEST}_LEVEL`
- lmcache 0.5.4 / nixl 1.3.2 / mooncake-transfer-engine already installed in the image

What ACR adds is the part nobody ships: **the value function and its signal plumbing.**

## Components

| path | what |
| --- | --- |
| `src/acr/trace.py` | trace schema + loaders (Claude Code session JSONL, harness CSV, synthetic mixes) |
| `src/acr/index.py` | prefix block DAG, session leases, shared-prefix accounting |
| `src/acr/policies.py` | LRU / LFU / fixed-TTL / Continuum-style-TTL / **adaptive-value (ours)** / oracle |
| `src/acr/replay.py` | trace-driven multi-tier replay simulator → policy comparison table |
| `src/acr/vllm/` | the real artifacts: out-of-tree `CachePolicy` + `OffloadingSpec` for vLLM |
| `bench/` | cost-model calibration (prefill curve, H2D bandwidth, round-trip correctness) |
| `bench/` | cost-model calibration (cold-prefill curve; read-only against a live engine) |
| `docs/` | [evidence](docs/00-evidence.md) · [architecture](docs/01-architecture.md) · [prior art](docs/02-novelty-and-related-work.md) · [roadmap](docs/03-roadmap.md) · [risks](docs/04-risks.md) · [findings](docs/05-findings.md) · **[reference ledger](docs/06-reference-ledger.md)** · [scope & ownership](docs/07-scope-and-ownership.md) · **[phase A field log](docs/08-phase-a-log.md)** |

## Quickstart

```bash
pip install -e .          # or run from a checkout with ./acr (no install needed)
pytest -q
# replay a real agentic session trace through the tier model
./acr replay --config examples/ncu-4xl20.yaml \
    --trace <path.jsonl> --trace-format cc-transcript \
    --policies lru,lfu,fixed_ttl,continuum_ttl,adaptive_value,oracle
```

Early results and the places where the model contradicts the original idea:
[`docs/05-findings.md`](docs/05-findings.md).

## What this project currently believes about itself

The **repository is the deliverable**; the paper is unsettled. Every mechanism we had queued —
adaptive TTL, workflow-aware retention, agent-runtime middleware, KV-as-object with fork, block-level
admission, session-as-lease, rollback consistency — turns out to be already published (see
[`docs/06`](docs/06-reference-ledger.md)). And our own headline policy result **failed to replicate**
when the simulator was set to the engine's real 816-token granularity
([`docs/05` F-7b](docs/05-findings.md)): at that granularity LRU, fixed-TTL, Continuum-style TTL and
our adaptive value policy are identical and the only winner is plain frequency.

So the next step is not more policy code. It is **phase A** ([`docs/07`](docs/07-scope-and-ownership.md)):
run the *stock* vLLM GPU↔RAM offload with LRU/ARC on a real multi-session load, measure how much the
library alone buys, and only then decide whether an information-fair baseline leaves any room for a
mechanism we could call ours.

Status: **V0 — simulator + policy layer + vLLM plugin skeleton. No engine has been touched.**
Phase A needs a maintenance window and passes a byte-exactness gate first, because a wrong KV restore
in a clinical simulation is silently wrong, not loudly wrong.
