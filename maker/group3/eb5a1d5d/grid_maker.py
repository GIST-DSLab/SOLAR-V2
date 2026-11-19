"""
ARC Task: eb5a1d5d (RE-ARC) — LLM-generated grid_maker
"""
from __future__ import annotations

import inspect

import sys
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

SOLAR_ROOT = Path(__file__).resolve().parents[3]
if str(SOLAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLAR_ROOT))

REARC_ROOT = SOLAR_ROOT / "re-arc"
_rs = str(REARC_ROOT)
while _rs in sys.path:
    sys.path.remove(_rs)
sys.path.insert(0, _rs)

from maker.base_grid_maker import BaseGridMaker

import importlib
for _m in ["utils", "dsl", "generators"]:
    if _m in sys.modules:
        del sys.modules[_m]
from utils import *  # noqa: F401,F403  (unifint, choice, sample, etc.)
from dsl import *    # noqa: F401,F403

# ── LLM-generated: sample_colors / generate / derive_operations ───────────────
import random
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # The generator draws a new set of ring colours for every instance. The rule
    # (squeeze the stretched nested boxes back to rings one cell thick) does not
    # depend on which colours are used, so no colour role has to be fixed.
    return {}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, **color_kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    def transpose(g):
        return tuple(tuple(r) for r in zip(*g))

    lim = min(max_h, max_w)
    dmax = max(2, min(10, (lim + 1) // 2))
    d = unifint(diff_lb, diff_ub, (2, dmax))
    d = max(2, min(d, dmax))
    n = 2 * d - 1
    colss = random.sample(range(10), d)
    go = [[-1] * n for _ in range(n)]
    for j, cc in enumerate(colss):
        a, b = j, 2 * d - 2 - j
        for k in range(a, b + 1):
            go[a][k] = cc
            go[b][k] = cc
            go[k][a] = cc
            go[k][b] = cc
    go = tuple(tuple(r) for r in go)

    nvenl = unifint(diff_lb, diff_ub, (0, max(0, max_h - n)))
    nhenl = unifint(diff_lb, diff_ub, (0, max(0, max_w - n)))
    enl = [nvenl, nhenl]
    gi = go
    while enl[0] > 0 or enl[1] > 0:
        h, w = len(gi), len(gi[0])
        opts = []
        if enl[0] > 0 and h < max_h:
            opts.append(0)
        if enl[1] > 0 and w < max_w:
            opts.append(1)
        if not opts:
            break
        ch = random.choice(opts)
        g = gi if ch == 0 else transpose(gi)
        idx = random.randint(0, len(g) - 1)
        g = g[:idx + 1] + g[idx:]
        gi = g if ch == 0 else transpose(g)
        enl[ch] -= 1
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O, examples=None):
    # Rule: the input is a set of nested boxes with some rows and columns repeated.
    # The output is the same nesting with every ring one cell thick.
    # Unit of work here: one RING (one colour) at a time, painted onto an n x n
    # canvas, outermost ring first and each next ring layered inside it.
    # The ring order and the colours come from I alone: drop the repeated rows,
    # drop the repeated columns, then read down the diagonal.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    def dedupe(lines):
        out = []
        for ln in lines:
            if not out or out[-1] != ln:
                out.append(ln)
        return out

    rows = dedupe([tuple(r) for r in I.tolist()])
    squeezed = np.array(dedupe([tuple(c) for c in np.array(rows).T.tolist()])).T
    n = squeezed.shape[0]          # 2d-1, read off the squeezed figure
    d = (n + 1) // 2
    ring_cols = [int(squeezed[j, j]) for j in range(d)]

    G = I.copy()
    # 1) Resize the canvas to the n x n figure: the full rectangle [0,0,n,n].
    #    This is a transparent copy, so zero cells are not carried over.
    if (hi, wi) != (n, n):
        ops.append(33)
        sels.append([0, 0, n - 1, n - 1])
        newG = np.zeros((n, n), dtype=int)
        sub = I[:n, :n]
        newG[sub != 0] = sub[sub != 0]
        G = newG

    # 2) Paint the rings from the outside in. Ring j fills the square
    #    [j..n-1-j]^2 with its colour. Inner rings later cover the middle,
    #    which is ordinary layering. Cells that ALREADY hold the colour on the
    #    current grid are skipped.
    for j, cc in enumerate(ring_cols):
        cells = [(r, c) for r in range(j, n - j) for c in range(j, n - j)
                 if G[r, c] != cc]
        if not cells:
            continue
        ops.append(cc)
        sels.append(sel_of(cells))
        for r, c in cells:
            G[r, c] = cc

    ops.append(34)
    sels.append([0, 0, n - 1, n - 1])
    return ops, sels


# ── GridMaker ─────────────────────────────────────────────────────────────────

class GridMaker(BaseGridMaker):

    def parse(self, **kwargs) -> List[Tuple[
        List[NDArray], List[NDArray],
        List[NDArray], List[NDArray],
        Dict[str, Any],
    ]]:
        num_samples  = kwargs.get("num_samples", 1)
        num_examples = kwargs.get("num_examples", 3)
        max_h, max_w = kwargs.get("max_grid_dim", [30, 30])
        dataset = []

        for _sn in range(num_samples):
            # Episode-level retry: if 10 attempts at some instance all fail, that's
            # transient (bad luck with the generator's randomness) — retry the WHOLE
            # episode from scratch (fresh colors/instance plan) up to 5 times, rather
            # than silently continuing with a partial episode (fewer examples than
            # requested, or a missing test instance with operations=[]/selections=[]
            # quietly appended as if it were a normal sample).
            for _episode_attempt in range(5):
                pr_in:  List[NDArray] = []
                pr_out: List[NDArray] = []
                ex_in:  List[NDArray] = []
                ex_out: List[NDArray] = []
                ops:  List[int]       = []
                sels: List[List[int]] = []

                # sample color roles once per episode → consistent across all instances
                # sample_colors() may optionally accept num_examples (to pre-plan
                # per-instance categories) — call it either way for compatibility
                # with grid_makers generated before this parameter existed.
                if "num_examples" in inspect.signature(sample_colors).parameters:
                    colors = sample_colors(num_examples=num_examples)
                else:
                    colors = sample_colors()

                # Plans are consumed by INDEX, not mutated: retries for instance j
                # must receive the same variant. category_plan is retained as a
                # backwards-compatible single-key form; new makers use kwargs dict entries.
                category_plan = colors.pop("category_plan", None) if isinstance(colors, dict) else None
                instance_plan = colors.pop("instance_plan", None) if isinstance(colors, dict) else None
                if category_plan is not None and instance_plan is not None:
                    raise ValueError(
                        "sample_colors must return only one of category_plan/instance_plan"
                    )
                if category_plan is not None and len(category_plan) != num_examples + 1:
                    # A wrong plan length is a deterministic bug in sample_colors(),
                    # not bad luck — retrying the episode won't fix it. Fail loudly
                    # instead of clamping the index and silently reusing an entry.
                    raise ValueError(
                        f"category_plan length {len(category_plan)} != "
                        f"num_examples+1 ({num_examples + 1}) for task eb5a1d5d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task eb5a1d5d"
                        )
                    if any(not isinstance(entry, dict) for entry in instance_plan):
                        raise ValueError("every instance_plan entry must be a kwargs dict")
                    if instance_plan[-1] not in instance_plan[:-1]:
                        raise ValueError(
                            "instance_plan test variant must appear among the examples"
                        )

                try:
                    j = 0
                    while j < num_examples + 1:
                        ok = False
                        for _ in range(10):
                            try:
                                call_kwargs = dict(colors)
                                if instance_plan is not None:
                                    call_kwargs.update(instance_plan[j])
                                elif category_plan is not None:
                                    call_kwargs["category"] = category_plan[j]
                                r = generate(
                                    random.uniform(0.2, 0.5),
                                    random.uniform(0.5, 0.8),
                                    max_h, max_w,
                                    **call_kwargs,
                                )
                                I = np.array(r["input"],  dtype=np.uint8)
                                O = np.array(r["output"], dtype=np.uint8)
                                # enforce max_grid_dim — skip oversized grids
                                if I.shape[0] > max_h or I.shape[1] > max_w:
                                    continue
                                if O.shape[0] > max_h or O.shape[1] > max_w:
                                    continue
                                ok = True
                                break
                            except (IndexError, ValueError, KeyError):
                                continue
                        if not ok:
                            raise RuntimeError(
                                f"Failed to generate instance {j} after 10 attempts "
                                f"for task eb5a1d5d"
                            )
                        if j == num_examples:
                            pr_in.append(I)
                            pr_out.append(O)
                            ops, sels = derive_operations(I, O)
                        else:
                            ex_in.append(I)
                            ex_out.append(O)
                        j += 1
                    break  # episode complete
                except RuntimeError:
                    continue
            else:
                raise RuntimeError(
                    f"Failed to build a complete episode for task eb5a1d5d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"eb5a1d5d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
