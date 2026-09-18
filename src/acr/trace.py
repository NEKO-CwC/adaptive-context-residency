"""Session traces: schema, real-data loaders, and a mixed-workload generator.

A `Turn` is deliberately engine-agnostic: it is what a gateway or an agent transcript can
actually report. Anything the engine knows (block ids, hit ratios) is derived downstream.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Turn:
    """One model call: what was sent, what came back, and when it happened."""

    t: float                     # seconds since trace start (request arrival)
    session_id: str
    role: str                    # supervisor | patient | reviewer | rag | ...
    prompt_tokens: int
    completion_tokens: int
    shared_prefix_tokens: int = 0    # bytes of the prompt shared with other sessions
    tool_eta_s: float | None = None  # agent's own estimate of when it will come back
    aborted: bool = False            # previous branch was logically discarded
    slo_class: str = "interactive"

    @property
    def decode_seconds(self) -> float:  # informational; the simulator owns timing
        return self.completion_tokens / 118.0


@dataclass
class Session:
    session_id: str
    role: str
    shared_prefix_tokens: int
    turns: list[Turn] = field(default_factory=list)


def load_cc_transcript(path: str | Path, role: str = "supervisor",
                       shared_prefix_tokens: int = 0) -> list[Turn]:
    """Claude Code session JSONL -> turns.

    Real field evidence (verified on this host): assistant messages carry
    `message.usage.{input_tokens,output_tokens,thinking_tokens}` and an ISO `timestamp`;
    the gap between consecutive assistant messages is tool execution + thinking + queueing,
    i.e. exactly the idle window a residency policy must predict.

    Note: `cache_read_input_tokens` is 0 in our gateway path even when the engine's prefix
    cache is hitting — the Anthropic-compatible endpoint does not surface APC accounting to
    the client. That is part of why this project keeps its own shadow index.
    """
    out: list[Turn] = []
    prev: tuple[int, int] | None = None
    prev_t: float | None = None
    t0: datetime | None = None
    session_id = Path(path).stem[:12]
    for line in Path(path).read_text(errors="ignore").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") != "assistant":
            continue
        msg = rec.get("message") or {}
        usage = msg.get("usage") if isinstance(msg, dict) else None
        if not usage:
            continue
        ts = rec.get("timestamp")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            continue
        if t0 is None:
            t0 = dt
        t = (dt - t0).total_seconds()
        prompt = int(usage.get("input_tokens") or 0)
        completion = int(usage.get("output_tokens") or 0)
        if prompt <= 0:
            continue                     # bookkeeping record, not a model call
        if prev is not None and prev == (prompt, completion) and (prev_t is None or t - prev_t < 1.5):
            continue                     # the same API call logged twice by the client
        gap = None if prev_t is None else max(0.0, t - prev_t)
        prev = (prompt, completion)
        out.append(Turn(t=t, session_id=session_id, role=role,
                        prompt_tokens=prompt, completion_tokens=completion,
                        # The transcript cannot tell us how much of the prompt is shared with
                        # other sessions; the caller stamps the estimate (system+tools for this
                        # host). It is metadata, not measurement, and is labelled as such.
                        shared_prefix_tokens=min(shared_prefix_tokens, prompt),
                        tool_eta_s=gap))
        prev_t = t
    return out


def load_harness_csv(path: str | Path, role: str = "supervisor") -> list[Turn]:
    """Growing-prefix/cold-prefill harness CSV -> turns (approx_ctx_tokens + ttft_s)."""
    import csv

    out: list[Turn] = []
    t = 0.0
    with open(path) as fh:
        for row in csv.DictReader(fh):
            ctx = int(float(row.get("approx_ctx_tokens") or 0))
            ttft = float(row.get("ttft_s") or 0.0)
            out.append(Turn(t=t, session_id=f"{Path(path).parent.name}", role=role,
                            prompt_tokens=ctx, completion_tokens=1))
            t += ttft + 1.0
    return out


def to_jsonl(turns: Iterable[Turn], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        fh.write(json.dumps({"schema_version": SCHEMA_VERSION}) + "\n")
        for turn in turns:
            fh.write(json.dumps(asdict(turn)) + "\n")


def from_jsonl(path: str | Path) -> list[Turn]:
    out: list[Turn] = []
    for line in Path(path).read_text().splitlines():
        rec = json.loads(line)
        if "schema_version" in rec:
            continue
        out.append(Turn(**rec))
    return out


def _lognormal(rng: random.Random, median: float, sigma: float) -> float:
    return median * math.exp(rng.gauss(0.0, sigma))


def synthetic_mix(seed: int = 7, horizon_s: float = 1800.0,
                  coding_sessions: int = 12, patient_sessions: int = 20,
                  shared_root_tokens: int = 320_000,
                  coding_grow_tokens: int = 24_000,
                  coding_turn_tokens: int = 2_200) -> list[Turn]:
    """The workload mix this project exists for: long-lived agentic coding + short
    high-priority patient turns + a few reviewer calls on one engine.

    Design intent for the experiment, not realism per number (real traces come from
    load_cc_transcript): the *distinct* working set deliberately oversubscribes the HBM
    pool (coding_sessions x (root + growth) >> 1M tokens) so tiering has to choose, and
    every coding session ends somewhere, so retention that ignores dead contexts thrashes.
    """
    rng = random.Random(seed)
    turns: list[Turn] = []

    for i in range(coding_sessions):
        sid = f"code-{i}"
        start = rng.uniform(0, horizon_s * 0.6)
        # Team-mode shape: everyone starts from the same repo/system/tools prefix and then
        # accumulates a private conversation tail on top of it.
        ctx = shared_root_tokens + rng.uniform(20_000, 120_000)
        n_turns = int(rng.uniform(18, 55))
        t = start
        for k in range(n_turns):
            long_tool = rng.random() < 0.18
            gap = _lognormal(rng, 150.0 if long_tool else 9.0, 0.5)
            turns.append(Turn(t=round(t, 3), session_id=sid, role="supervisor",
                              prompt_tokens=int(ctx),
                              completion_tokens=int(rng.uniform(120, 1400)),
                              shared_prefix_tokens=shared_root_tokens,
                              tool_eta_s=None if k == 0 else round(gap, 3)))
            ctx += coding_turn_tokens * rng.uniform(0.4, 1.8)
            if rng.random() < 0.06:  # agent aborts a branch (ESC/undo)
                ctx -= rng.uniform(2_000, 9_000)
                turns[-1] = Turn(**{**asdict(turns[-1]), "aborted": True})
            t += gap + rng.uniform(1.0, 6.0)
            if t > horizon_s:
                break

    for i in range(patient_sessions):
        sid = f"patient-{i}"
        t = rng.uniform(0, horizon_s * 0.85)
        for k in range(int(rng.uniform(6, 14))):
            turns.append(Turn(t=round(t, 3), session_id=sid, role="patient",
                              prompt_tokens=int(rng.uniform(3_500, 5_000)),
                              completion_tokens=int(rng.uniform(80, 512)),
                              shared_prefix_tokens=4_000,
                              tool_eta_s=None if k == 0 else round(_lognormal(rng, 25.0, 0.6), 3),
                              slo_class="interactive"))
            t += _lognormal(rng, 55.0, 0.5)
            if t > horizon_s:
                break

    for i in range(4):
        sid = f"reviewer-{i}"
        t = rng.uniform(0, horizon_s * 0.7)
        for k in range(6):
            turns.append(Turn(t=round(t, 3), session_id=sid, role="reviewer",
                              prompt_tokens=int(rng.uniform(28_000, 46_000)),
                              completion_tokens=int(rng.uniform(800, 2_048)),
                              shared_prefix_tokens=12_000,
                              tool_eta_s=None if k == 0 else round(_lognormal(rng, 400.0, 0.4), 3),
                              slo_class="batch"))
            t += _lognormal(rng, 600.0, 0.4)

    turns.sort(key=lambda x: x.t)
    return turns


def group_by_session(turns: Iterable[Turn]) -> Iterator[Session]:
    sessions: dict[str, Session] = {}
    for turn in sorted(turns, key=lambda x: x.t):
        s = sessions.get(turn.session_id)
        if s is None:
            s = sessions[turn.session_id] = Session(turn.session_id, turn.role,
                                                    turn.shared_prefix_tokens)
        s.turns.append(turn)
    yield from sessions.values()
