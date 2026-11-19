"""
ARC Task: 9d9215db (RE-ARC) — LLM-generated grid_maker
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
    # Only the background is fixed; the rule depends on dot positions, not their colours.
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    if bgc is None:
        bgc = random.choice(range(10))

    def unifint(lo, hi):
        if hi < lo:
            hi = lo
        a = lo + int(round(diff_lb * (hi - lo)))
        b = lo + int(round(diff_ub * (hi - lo)))
        return random.randint(min(a, b), max(a, b))

    uh = max(2, min(14, (max_h - 1) // 2))
    uw = max(2, min(14, (max_w - 1) // 2))
    h = unifint(min(5, uh), uh) * 2 + 1
    w = unifint(min(5, uw), uw) * 2 + 1
    remcols = [c for c in range(10) if c != bgc]
    ub = min(h, w) // 4
    nrings = unifint(1, ub)
    onlinesbase = [(2 * k + 1, 2 * k + 1) for k in range(ub)]
    gi = np.full((h, w), bgc, dtype=int)
    for (i, j) in random.sample(onlinesbase, nrings):
        gi[i, j] = random.choice(remcols)
    linsbase = [(a, b + 2) for (a, b) in onlinesbase[:-1]]
    go = gi.copy()
    if linsbase:
        nlines = unifint(1, len(linsbase))
        for (i, j) in random.sample(linsbase, nlines):
            col = random.choice(remcols)
            gi[i, j] = col
            gi[j, i] = col
            go[i, j] = col
            go[j, i] = col
            for c in range(j, w - j):
                if c % 2 == 1:
                    go[i, c] = col
            for r in range(j, h - j):
                if r % 2 == 1:
                    go[r, i] = col
    base = go.copy()
    for g in (base[:, ::-1], base[::-1, :], base[::-1, ::-1]):
        m = g != bgc
        go[m] = g[m]
    k = random.randint(0, 3)
    return {"input": np.rot90(gi, k).tolist(), "output": np.rot90(go, k).tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    # Background: generator paints a canvas and places only a handful of dots on it.
    vals, cnts = np.unique(I, return_counts=True)
    bgc = int(vals[np.argmax(cnts)])
    G = I.copy()
    ops, sels = [], []
    pts = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]

    # 1) Mirror the corner pattern into the other three corners (Color ops, since
    #    colour 0 can be real content and Copy/Paste would drop it).
    for f in (lambda r, c: (r, w - 1 - c),
              lambda r, c: (h - 1 - r, c),
              lambda r, c: (h - 1 - r, w - 1 - c)):
        bycol = {}
        for (r, c) in pts:
            rr, cc = f(r, c)
            if G[rr, cc] != I[r, c]:
                bycol.setdefault(int(I[r, c]), []).append((rr, cc))
        for col in sorted(bycol):
            cells = sorted(set(bycol[col]))
            ops.append(col); sels.append(sel_of(cells))
            for rr, cc in cells:
                G[rr, cc] = col

    # 2) Each line dot (next to ring k's corner) extends along all four sides of
    #    ring k at odd positions, strictly between the ring corners.
    ub = min(h, w) // 4
    for k in range(ub - 1):
        a = 2 * k + 1
        col = int(G[a, a + 2])
        if col == bgc:
            continue
        sides = [
            [(a, c) for c in range(a + 2, w - a - 1, 2)],
            [(r, w - 1 - a) for r in range(a + 2, h - a - 1, 2)],
            [(h - 1 - a, c) for c in range(w - a - 3, a, -2)],
            [(r, a) for r in range(h - a - 3, a, -2)],
        ]
        for side in sides:
            cells = [p for p in side if G[p] != col]
            if cells:
                ops.append(col); sels.append(sel_of(cells))
                for p in cells:
                    G[p] = col

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 9d9215db"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 9d9215db"
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
                                f"for task 9d9215db"
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
                    f"Failed to build a complete episode for task 9d9215db "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"9d9215db-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
