"""`acr` CLI: build traces, replay them through the tier model, compare policies."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import default_config, load_config
from .replay import compare, markdown_table
from .trace import (Turn, from_jsonl, load_cc_transcript, load_harness_csv, synthetic_mix,
                    to_jsonl)

DEFAULT_POLICIES = "lru,lfu,fixed_ttl,continuum_ttl,adaptive_value,oracle"


def _turns(args: argparse.Namespace) -> list[Turn]:
    if args.trace == "-":
        return [Turn(**{k: v for k, v in json.loads(line).items()})
                for line in sys.stdin if line.strip()]
    path = Path(args.trace)
    if args.trace_format == "cc-transcript":
        return load_cc_transcript(path, shared_prefix_tokens=args.shared_prefix_tokens or 0)
    if args.trace_format == "harness-csv":
        return load_harness_csv(path)
    return from_jsonl(path)


def cmd_replay(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="acr replay")
    ap.add_argument("--trace", required=True, help="jsonl | cc-transcript jsonl | harness csv | -")
    ap.add_argument("--trace-format", default="jsonl",
                    choices=["jsonl", "cc-transcript", "harness-csv"])
    ap.add_argument("--config", default=None, help="YAML cost/tier config (default: this host)")
    ap.add_argument("--policies", default=DEFAULT_POLICIES)
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--ram-gib", type=float, default=None, help="override RAM tier size")
    ap.add_argument("--hbm-tokens", type=int, default=None)
    ap.add_argument("--prefill-tok-s", type=float, default=None,
                    help="override C_recompute (the C-1 contested constant)")
    ap.add_argument("--shared-prefix-tokens", type=int, default=0,
                    help="stamp onto cc-transcript turns (estimate, not measurement)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config) if args.config else default_config()
    from dataclasses import replace

    if args.prefill_tok_s:
        cfg = replace(cfg, cost=replace(cfg.cost, prefill_tokens_per_s=args.prefill_tok_s))
    if args.hbm_tokens:
        cfg = replace(cfg, hbm_pool_tokens=args.hbm_tokens)
    if args.ram_gib is not None:
        cfg = replace(cfg, tiers=tuple(
            replace(t, bytes_capacity=int(args.ram_gib * 1024**3)) if t.name == "ram" else t
            for t in cfg.tiers))

    turns = _turns(args)
    if not turns:
        print("empty trace", file=sys.stderr)
        return 2
    reports = compare(cfg, turns, [p.strip() for p in args.policies.split(",") if p.strip()])

    print(f"# trace: {len(turns)} turns over "
          f"{turns[-1].t - turns[0].t:,.0f} s, "
          f"{len({t.session_id for t in turns})} sessions, "
          f"{sum(t.prompt_tokens for t in turns) / 1e6:,.1f} M prompt-tokens offered")
    print(f"# hbm pool {cfg.hbm_pool_tokens:,} tokens | ram {cfg.tier('ram').bytes_capacity / 1024**3:,.0f} GiB"
          f" | break-even BW {cfg.cost.breakeven_bw_gbs() * 1000:,.0f} MB/s"
          f" | ram tier {cfg.tier('ram').effective_gbs * 1000:,.0f} MB/s")
    print()
    print(markdown_table(reports))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"config": {"hbm_pool_tokens": cfg.hbm_pool_tokens,
                        "block_tokens": cfg.block_tokens,
                        "cost": cfg.cost.__dict__,
                        "tiers": [t.__dict__ for t in cfg.tiers]},
             "reports": [r.as_row() for r in reports]}, indent=1))
    return 0


def cmd_synth(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="acr synth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--horizon", type=float, default=1800.0)
    ap.add_argument("--coding-sessions", type=int, default=12)
    ap.add_argument("--patient-sessions", type=int, default=20)
    args = ap.parse_args(argv)
    turns = synthetic_mix(seed=args.seed, horizon_s=args.horizon,
                          coding_sessions=args.coding_sessions,
                          patient_sessions=args.patient_sessions)
    to_jsonl(turns, args.out)
    print(f"wrote {len(turns)} turns to {args.out}")
    return 0


def cmd_trace_cc(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="acr trace-cc")
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--role", default="supervisor")
    ap.add_argument("--shared-prefix-tokens", type=int, default=0,
                    help="estimated common system/tools prefix across sessions on this host")
    args = ap.parse_args(argv)
    turns = load_cc_transcript(args.src, role=args.role,
                               shared_prefix_tokens=args.shared_prefix_tokens)
    to_jsonl(turns, args.out)
    print(f"wrote {len(turns)} turns to {args.out}")
    return 0


COMMANDS = {"replay": cmd_replay, "synth": cmd_synth, "trace-cc": cmd_trace_cc}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("commands: " + ", ".join(COMMANDS))
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"unknown command {cmd!r}", file=sys.stderr)
        return 2
    return COMMANDS[cmd](rest)


if __name__ == "__main__":
    raise SystemExit(main())
