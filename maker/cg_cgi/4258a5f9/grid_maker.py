"""
ARC Task: 4258a5f9 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    colopts = [c for c in range(10) if c != 1]
    bgc = random.choice(colopts)
    fgc = random.choice([c for c in colopts if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, **kwargs) -> dict:
    colopts = [c for c in range(10) if c != 1]
    if bgc is None:
        bgc = random.choice(colopts)
    if fgc is None or fgc == bgc:
        fgc = random.choice([c for c in colopts if c != bgc])

    def unif(lo, hi):
        lo_ = lo + int(round(diff_lb * (hi - lo)))
        hi_ = lo + int(round(diff_ub * (hi - lo)))
        lo_, hi_ = max(lo, min(lo_, hi)), max(lo, min(hi_, hi))
        if hi_ < lo_:
            lo_, hi_ = hi_, lo_
        return random.randint(lo_, hi_)

    h = unif(2, max(2, max_h))
    w = unif(2, max(2, max_w))
    gi = np.full((h, w), bgc, dtype=int)
    mp = (h * w) // 2 if (h * w) % 2 == 1 else (h * w) // 2 - 1
    mp = max(1, mp)
    ndots = unif(1, mp)
    inds = [(r, c) for r in range(h) for c in range(w)]
    dots = random.sample(inds, ndots)
    go = gi.copy()
    for (r, c) in dots:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    go[rr, cc] = 1
    for (r, c) in dots:
        go[r, c] = fgc
        gi[r, c] = fgc
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # Rule: the rarer colour marks dots; the 3x3 halo around each dot becomes 1.
    # Reading (from the separable view): a 3x3 dilation is a horizontal 1x3
    # dilation followed by a vertical 3x1 dilation. So: first a row pass that
    # paints each dot's left/right background neighbours, then a column pass
    # that paints the background cells above/below anything already marked.
    cnt = Counter(I.flatten().tolist())
    fgc = min(cnt.keys(), key=lambda c: (cnt[c], c))
    others = [c for c in cnt if c != fgc]
    bgc = max(others, key=lambda c: cnt[c]) if others else None
    if bgc is None:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    cur = I.copy()

    # Pass 1: per row, widen each dot sideways.
    for r in range(h):
        cells = set()
        for c in range(w):
            if I[r, c] == fgc:
                for cc in (c - 1, c + 1):
                    if 0 <= cc < w and cur[r, cc] == bgc:
                        cells.add((r, cc))
        if cells:
            cells = sorted(cells)
            ops.append(1); sels.append(sel_of(cells))
            for (rr, cc) in cells:
                cur[rr, cc] = 1

    # Pass 2: per column, extend every marked cell (dot or new 1) up and down.
    marked = cur != bgc
    for c in range(w):
        cells = set()
        for r in range(h):
            if marked[r, c]:
                for rr in (r - 1, r + 1):
                    if 0 <= rr < h and cur[rr, c] == bgc:
                        cells.add((rr, c))
        if cells:
            cells = sorted(cells)
            ops.append(1); sels.append(sel_of(cells))
            for (rr, cc) in cells:
                cur[rr, cc] = 1

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 4258a5f9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4258a5f9"
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
                                f"for task 4258a5f9"
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
                    f"Failed to build a complete episode for task 4258a5f9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4258a5f9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
