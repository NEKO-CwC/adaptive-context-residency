"""Adaptive Context Residency.

Public surface (stable-ish): `load_config`, `default_config`, `synthetic_mix`,
`load_cc_transcript`, `run_policy`, `compare`, `markdown_table`.
"""
from __future__ import annotations

from .config import CostModel, SimConfig, TierConfig, default_config, load_config
from .index import BlockInfo, PrefixIndex
from .policies import (AdaptiveValuePolicy, ContinuumTTLPolicy, FixedTTLPolicy,
                       FrequencyPolicy, OraclePolicy, Probe, RecencyPolicy, TierPolicy, build)
from .replay import Report, Simulator, compare, markdown_table, run_policy
from .trace import Turn, from_jsonl, group_by_session, load_cc_transcript, synthetic_mix, to_jsonl

__version__ = "0.0.1"

__all__ = [
    "CostModel", "SimConfig", "TierConfig", "default_config", "load_config",
    "BlockInfo", "PrefixIndex",
    "TierPolicy", "RecencyPolicy", "FrequencyPolicy", "FixedTTLPolicy",
    "ContinuumTTLPolicy", "AdaptiveValuePolicy", "OraclePolicy", "Probe", "build",
    "Report", "Simulator", "run_policy", "compare", "markdown_table",
    "Turn", "to_jsonl", "from_jsonl", "load_cc_transcript", "synthetic_mix", "group_by_session",
    "__version__",
]
