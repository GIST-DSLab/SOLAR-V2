"""
ARC Task: 508bd3b6 (RE-ARC) — LLM-generated grid_maker
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

from collections import Counter


def sample_colors(num_examples=None) -> dict:
    # colour 3 is hardcoded as the paint colour; bgc, bar and ray colours are sampled from the rest
    cols = [c for c in range(10) if c != 3]
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    barc = random.choice(rem)
    rem = [c for c in rem if c != barc]
    linc = random.choice(rem)
    return {"bgc": bgc, "barc": barc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, barc, linc) -> dict:
    from utils import unifint
    from dsl import (canvas, fill, connect, shoot, ofcolor, sfilter, order, first, last,
                     identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    m = min(max_h, max_w, 30)
    h = unifint(diff_lb, diff_ub, (5, m))
    w = unifint(diff_lb, diff_ub, (h, m))
    barh = unifint(diff_lb, diff_ub, (1, h // 2))
    barloci = unifint(diff_lb, diff_ub, (2, h - barh))
    gi = canvas(bgc, (h, w))
    for j in range(barloci, barloci + barh):
        gi = fill(gi, barc, connect((j, 0), (j, w - 1)))
    dotlociinv = unifint(diff_lb, diff_ub, (0, barloci - 1))
    dotloci = min(max(0, barloci - 2 - dotlociinv), barloci - 1)
    ln1 = shoot((dotloci, 0), (1, 1))
    ofbgc = ofcolor(gi, bgc)
    ln1 = sfilter(ln1 & ofbgc, lambda ij: ij[0] < barloci)
    ln1 = order(ln1, first)
    ln2 = shoot(ln1[-1], (-1, 1))
    ln2 = sfilter(ln2 & ofbgc, lambda ij: ij[0] < barloci)
    ln2 = order(ln2, last)[1:]
    ln = ln1 + ln2
    k = len(ln1)
    lineleninv = unifint(diff_lb, diff_ub, (0, k - 2))
    linelen = k - lineleninv
    givenl = ln[:linelen]
    reml = ln[linelen:]
    gi = fill(gi, linc, givenl)
    go = fill(gi, 3, reml)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = random.choice((1, 2))
    for fn in random.sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading: the ray of colour linc is extended (as one diagonal segment) up to the
    # bar, then reflected off the bar into a second diagonal segment of the opposite
    # slope that runs until it leaves the grid. The golf entry reads the task the same
    # way (two diagonal lines forming a V). Each segment is one line object, so each
    # one is painted with a single Color3 op.
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    colors = sorted(set(I.flatten().tolist()))

    # bar: a solid rectangle spanning a whole grid axis
    barc, box = None, None
    for c in colors:
        cells = np.argwhere(I == c)
        r0, c0 = cells.min(0)
        r1, c1 = cells.max(0)
        bh, bw = int(r1 - r0 + 1), int(c1 - c0 + 1)
        if len(cells) == bh * bw and (bh == h or bw == w):
            barc, box = int(c), (int(r0), int(c0), int(r1), int(c1))
            break

    rest = [c for c in colors if c != barc]
    # ray: a colour with no 4-adjacent twin (a pure diagonal)
    linc = None
    for c in rest:
        s = {(int(r), int(cc)) for r, cc in np.argwhere(I == c)}
        if all((r + 1, cc) not in s and (r, cc + 1) not in s for (r, cc) in s):
            linc = int(c)
            break
    bgc = [c for c in rest if c != linc][0]

    r0, c0, r1, c1 = box
    vertical = (r1 - r0 + 1) == h
    bar = {(int(r), int(c)) for r, c in np.argwhere(I == barc)}
    line = [(int(r), int(c)) for r, c in np.argwhere(I == linc)]

    def dbar(p):
        return (max(r0 - p[0], 0) + max(p[0] - r1, 0) +
                max(c0 - p[1], 0) + max(p[1] - c1, 0))

    head = min(line, key=dbar)
    tail = max(line, key=lambda p: abs(p[0] - head[0]) + abs(p[1] - head[1]))
    dr = (head[0] > tail[0]) - (head[0] < tail[0])
    dc = (head[1] > tail[1]) - (head[1] < tail[1])

    def march(start, d):
        seg = []
        r, c = start
        while True:
            nr, nc = r + d[0], c + d[1]
            if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in bar:
                return (r, c), seg
            seg.append((nr, nc))
            r, c = nr, nc

    # segment 1: continue the ray from its tip to the cell touching the bar
    bounce, seg1 = march(head, (dr, dc))
    # segment 2: reflected off the bar, running until it leaves the grid
    refl = (dr, -dc) if vertical else (-dr, dc)
    _, seg2 = march(bounce, refl)

    for seg in (seg1, seg2):
        cells = [p for p in seg if I[p[0], p[1]] == bgc]
        if cells:
            ops.append(3)
            sels.append(sel_of(cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task 508bd3b6"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 508bd3b6"
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
                                f"for task 508bd3b6"
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
                    f"Failed to build a complete episode for task 508bd3b6 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"508bd3b6-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
