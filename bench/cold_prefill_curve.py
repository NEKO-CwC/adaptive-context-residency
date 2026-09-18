#!/usr/bin/env python3
"""T-0.5 — calibrate C_recompute: cold prefill cost vs context length (docs/00 C-conflict).

Why this exists: the residency value function is
`P_b · (T_recompute − T_restore)`, so every policy decision is scaled by the prefill curve. We
currently hold two incompatible measurements (30K cold → 7.9K tok/s vs a recorded 400K → 9.6 s,
i.e. 41.7K tok/s — a 5.3× spread) and the simulator's conclusions must be shown to survive it.

Read-only against a running engine: it sends fresh, non-overlapping prompts (APC contributes
nothing) and never changes configuration. It does *load* the engine, so run it when the box is
allowed to be busy, not silently in the background.

    python bench/cold_prefill_curve.py --sizes 20000,50000,100000,200000 --out results/cold.json

Reports per-size wall time plus the fitted `T(n) = a + b·n + c·n²`; the quadratic term is the
attention contribution and is the part that decides whether long contexts are disproportionately
expensive to lose (which is what makes residency worth engineering at all).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request

FILLER = ("Section {k}: the clinical training platform coordinates patient simulation, objective "
          "truth confirmation, workflow usage analytics, and diagnostic report contracts across "
          "hospital scenarios. ")


def fresh_prompt(tokens: int, salt: int) -> str:
    chars = int(tokens * 3.6)
    # salt makes every prompt distinct so the engine's prefix cache cannot help us
    body = "".join(FILLER.format(k=f"{salt}-{i}") for i in range(chars // len(FILLER) + 1))
    return (body[:chars] + "\n\nReply with exactly: OK").strip()


def timed_request(url: str, model: str, prompt: str, timeout: float) -> dict:
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 8, "temperature": 0, "stream": True,
               "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    ttft = None
    usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        while True:
            line = resp.readline()
            if not line:
                break
            text = line.decode(errors="ignore").strip()
            if not text.startswith("data:"):
                continue
            data = text[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if ttft is None and (chunk.get("choices") or [{}])[0].get("delta"):
                ttft = time.monotonic() - t0
            if chunk.get("usage"):
                usage = chunk["usage"]
    return {"wall_s": round(time.monotonic() - t0, 3), "ttft_s": round(ttft or 0.0, 3),
            "prompt_tokens": usage.get("prompt_tokens"), "usage": usage}


def fit_quadratic(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Least-squares T(n) = a + b·n + c·n² on (tokens, seconds); pure stdlib."""
    rows = [(1.0, n, n * n, t) for n, t in points]
    # normal equations for 3 unknowns
    XtX = [[sum(r[i] * r[j] for r in rows) for j in range(3)] for i in range(3)]
    Xty = [sum(r[i] * r[3] for r in rows) for i in range(3)]
    m = [XtX[i] + [Xty[i]] for i in range(3)]
    for col in range(3):                       # gaussian elimination with partial pivoting
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(3):
            if r == col:
                continue
            factor = m[r][col] / m[col][col]
            m[r] = [a - factor * b for a, b in zip(m[r], m[col])]
    return tuple(m[i][3] / m[i][i] for i in range(3))  # type: ignore[return-value]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8002/v1/chat/completions")
    ap.add_argument("--model", default="ncu-supervisor-chat-qwen38-flash-next-q4xl")
    ap.add_argument("--sizes", default="20000,50000,100000,200000")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", default="results/cold-prefill.json")
    args = ap.parse_args()

    samples: list[dict] = []
    for i, size in enumerate(int(x) for x in args.sizes.split(",")):
        runs = []
        for rep in range(args.reps):
            try:
                runs.append(timed_request(args.url, args.model, fresh_prompt(size, i * 100 + rep),
                                          args.timeout))
            except Exception as exc:                       # keep going, report the gap
                runs.append({"error": str(exc)})
        good = [r for r in runs if "wall_s" in r]
        if not good:
            samples.append({"requested_tokens": size, "error": runs[0]["error"]})
            continue
        med = statistics.median(r["wall_s"] for r in good)
        measured = [r["prompt_tokens"] or size for r in good]
        samples.append({"requested_tokens": size, "prompt_tokens": measured,
                        "wall_median_s": round(med, 3),
                        "tok_per_s": round(statistics.mean(measured) / med, 1),
                        "ttft_median_s": round(statistics.median(r["ttft_s"] for r in good), 3)})
        print(json.dumps(samples[-1]))

    points = [(s["prompt_tokens"][0], s["wall_median_s"]) for s in samples
              if "wall_median_s" in s]
    fit = fit_quadratic(points) if len(points) >= 3 else None
    out = {"samples": samples, "fit_a_b_c": fit,
            "linear_tok_per_s_at_median": (
                statistics.median(s["tok_per_s"] for s in samples if "tok_per_s" in s)
                if any("tok_per_s" in s for s in samples) else None)}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({"fit_a_b_c": fit, "note": "c>0 means losing long context is superlinear"},))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
