"""
ARC Task: 4612dd53 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter

import numpy as np
from maker.sel_helpers import sel_of


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return int(min(max(a, round(a + (b - a) * d)), b))


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 2]          # 2 is the rule's fixed fill colour
    bgc, col = random.sample(cols, 2)
    return {"bgc": bgc, "col": col}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, col=None, **kwargs) -> dict:
    if bgc is None or col is None:
        bgc, col = random.sample([c for c in range(10) if c != 2], 2)
    max_h = max(8, min(30, max_h))
    max_w = max(8, min(30, max_w))
    h = _unifint(diff_lb, diff_ub, (8, max_h))
    w = _unifint(diff_lb, diff_ub, (8, max_w))
    ih = _unifint(diff_lb, diff_ub, (5, h - 1))
    iw = _unifint(diff_lb, diff_ub, (5, w - 1))
    loci = random.randint(0, h - ih)
    locj = random.randint(0, w - iw)
    r0, c0, r1, c1 = loci, locj, loci + ih - 1, locj + iw - 1
    bx = set()
    for c in range(c0, c1 + 1):
        bx.add((r0, c)); bx.add((r1, c))
    for r in range(r0, r1 + 1):
        bx.add((r, c0)); bx.add((r, c1))
    if random.choice((True, False)):
        locc = random.randint(loci + 2, loci + ih - 3)
        br = {(locc, c) for c in range(locj + 1, locj + iw - 1)}
    else:
        locc = random.randint(locj + 2, locj + iw - 3)
        br = {(r, locc) for r in range(loci + 1, loci + ih - 1)}
    crns = random.sample([(r0, c0), (r0, c1), (r1, c0), (r1, c1)], 3)
    rembx = sorted(bx - set(crns))
    onbr = random.sample(sorted(br), 2)
    rembr = sorted(br - set(onbr))
    noccbx = _unifint(diff_lb, diff_ub, (0, len(rembx)))
    noccbr = _unifint(diff_lb, diff_ub, (0, len(rembr)))
    occbx = random.sample(rembx, noccbx)
    occbr = random.sample(rembr, noccbr)
    c = [[bgc] * w for _ in range(h)]
    for (r, cc) in bx | br:
        c[r][cc] = col
    gi = [row[:] for row in c]
    go = [row[:] for row in c]
    for (r, cc) in list(occbx) + list(occbr):
        gi[r][cc] = bgc
        go[r][cc] = 2
    if random.choice((True, False)):
        for (r, cc) in br:
            gi[r][cc] = bgc
            go[r][cc] = bgc
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O, examples=None):
    """
    Reading (taken from the line-by-line golf solution): the figure is a set of
    full straight lines -- the rectangle's two vertical sides, its two horizontal
    sides, and optionally one inner bar running side to side.  Every line is
    treated the same way: its gaps (background cells) along its whole span
    become colour 2.  Lines are handled by orientation: first all vertical lines
    of the figure in one go, then all horizontal lines (re-read from the
    current grid, so a missing corner that the vertical pass already filled is
    not selected twice).
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    FILL = 2

    counts = Counter(I.flatten().tolist())
    col = min(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]     # leastcolor
    bgc = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    ops, sels = [], []
    pts = np.argwhere(I == col)
    if len(pts) == 0:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    r0, c0 = int(pts[:, 0].min()), int(pts[:, 1].min())
    r1, c1 = int(pts[:, 0].max()), int(pts[:, 1].max())

    # lines of the figure, by orientation
    vcols = [c0, c1]
    hrows = [r0, r1]
    inner = [(int(r), int(c)) for r, c in pts if r0 < r < r1 and c0 < c < c1]
    if inner:
        rows = {r for r, _ in inner}
        cols = {c for _, c in inner}
        if len(rows) == 1 and len(cols) == 1:
            horizontal = (r1 - r0 + 1) > (c1 - c0 + 1)
        else:
            horizontal = (len(rows) == 1)
        if horizontal:
            hrows.append(inner[0][0])
        else:
            vcols.append(inner[0][1])

    G = I.copy()

    # pass 1: vertical lines (each spans rows r0..r1)
    vcells = [(r, c) for c in vcols for r in range(r0, r1 + 1) if G[r, c] == bgc]
    if vcells:
        ops.append(FILL); sels.append(sel_of(vcells))
        for r, c in vcells:
            G[r, c] = FILL

    # pass 2: horizontal lines (each spans cols c0..c1), on the current grid
    hcells = [(r, c) for r in hrows for c in range(c0, c1 + 1) if G[r, c] == bgc]
    if hcells:
        ops.append(FILL); sels.append(sel_of(hcells))
        for r, c in hcells:
            G[r, c] = FILL

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])      # full-grid rectangle: submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 4612dd53"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4612dd53"
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
                                f"for task 4612dd53"
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
                    f"Failed to build a complete episode for task 4612dd53 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4612dd53-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
