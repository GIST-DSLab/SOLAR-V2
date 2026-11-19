"""
ARC Task: d4f3cd78 (RE-ARC) — LLM-generated grid_maker
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
    # 8 is hardcoded as the fill colour, so neither bgc nor fgc can be 8
    cols = [c for c in range(10) if c != 8]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    lo_i = int(round(lo))
    hi_i = int(round(hi))
    if hi_i < lo_i:
        hi_i = lo_i
    return random.randint(max(a, lo_i), min(b, hi_i))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, **kwargs) -> dict:
    if bgc is None or fgc is None:
        cols = [c for c in range(10) if c != 8]
        bgc, fgc = random.sample(cols, 2)
    H = max(10, min(30, max_h))
    W = max(10, min(30, max_w))
    h = _unifint(diff_lb, diff_ub, (10, H))
    w = _unifint(diff_lb, diff_ub, (10, W))
    ih = _unifint(diff_lb, diff_ub, (3, h // 3 * 2))
    iw = _unifint(diff_lb, diff_ub, (3, w // 3 * 2))
    loci = random.randint(1, h - ih - 1)
    locj = random.randint(1, w - iw - 1)
    r0, c0 = loci, locj
    r1, c1 = loci + ih - 1, locj + iw - 1

    box = set()
    for c in range(c0, c1 + 1):
        box.add((r0, c)); box.add((r1, c))
    for r in range(r0, r1 + 1):
        box.add((r, c0)); box.add((r, c1))
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    opts = sorted(box - corners)
    nholes = _unifint(diff_lb, diff_ub, (1, len(opts)))
    holes = random.sample(opts, nholes)

    gi = np.full((h, w), bgc, dtype=int)
    for (r, c) in box - set(holes):
        gi[r, c] = fgc

    go = gi.copy()
    A, B, C, D = r0 + 1, c0 + 1, r1 - 1, c1 - 1
    go[A:C + 1, B:D + 1] = 8
    for (r, c) in holes:
        dr = 1 if r > C else (-1 if r < A else 0)
        dc = 1 if c > D else (-1 if c < B else 0)
        rr, cc = r, c
        while 0 <= rr < h and 0 <= cc < w:
            go[rr, cc] = 8
            rr += dr; cc += dc
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # background = canvas colour; the box outline is a tiny minority of cells
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # The box's four corners are never holes, so the fg bbox is the box.
    fg = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]
    r0 = min(r for r, _ in fg); r1 = max(r for r, _ in fg)
    c0 = min(c for _, c in fg); c1 = max(c for _, c in fg)

    ops, sels = [], []

    # Reading: the box is a container of 8 "liquid". Each interior ROW is one
    # line of liquid held between the left and right walls; where a wall is
    # missing on that row, the same line spills straight through to the grid
    # edge. So each interior row becomes one horizontal stroke whose ends are
    # the walls or the grid edges.
    for r in range(r0 + 1, r1):
        start = 0 if I[r, c0] == bgc else c0 + 1
        end = w - 1 if I[r, c1] == bgc else c1 - 1
        ops.append(8)
        sels.append(sel_of([(r, c) for c in range(start, end + 1)]))

    # Interior columns then leak vertically through holes in the top/bottom
    # walls: the liquid in that column runs out through the hole to the edge.
    # The interior part of the column is already liquid; only the spill
    # through the hole is new.
    for c in range(c0 + 1, c1):
        if I[r0, c] == bgc:
            ops.append(8)
            sels.append(sel_of([(r, c) for r in range(0, r0 + 1)]))
        if I[r1, c] == bgc:
            ops.append(8)
            sels.append(sel_of([(r, c) for r in range(r1, h)]))

    ops.append(34)
    sels.append(sel_of([(r, c) for r in range(h) for c in range(w)]))
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
                        f"num_examples+1 ({num_examples + 1}) for task d4f3cd78"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d4f3cd78"
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
                                f"for task d4f3cd78"
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
                    f"Failed to build a complete episode for task d4f3cd78 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d4f3cd78-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
