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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 2]          # 2 is the fixed fill colour
    bgc, col = random.sample(cols, 2)
    return {"bgc": bgc, "col": col}


def _unif(diff_lb, diff_ub, lo, hi):
    if hi <= lo:
        return lo
    d = random.uniform(diff_lb, diff_ub)
    return max(lo, min(hi, lo + int(round(d * (hi - lo)))))


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, col=1) -> dict:
    H = max(8, max_h)
    W = max(8, max_w)
    h = _unif(diff_lb, diff_ub, 8, H)
    w = _unif(diff_lb, diff_ub, 8, W)
    ih = _unif(diff_lb, diff_ub, 5, h - 1)
    iw = _unif(diff_lb, diff_ub, 5, w - 1)
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
        br = [(locc, c) for c in range(locj + 1, locj + iw - 1)]
    else:
        locc = random.randint(locj + 2, locj + iw - 3)
        br = [(r, locc) for r in range(loci + 1, loci + ih - 1)]
    crns = random.sample([(r0, c0), (r0, c1), (r1, c0), (r1, c1)], 3)
    rembx = sorted(bx - set(crns))
    onbr = random.sample(br, 2)
    rembr = sorted(set(br) - set(onbr))
    noccbx = _unif(diff_lb, diff_ub, 0, len(rembx))
    noccbr = _unif(diff_lb, diff_ub, 0, len(rembr))
    occbx = random.sample(rembx, noccbx)
    occbr = random.sample(rembr, noccbr)
    c = [[bgc] * w for _ in range(h)]
    for (r, cc) in list(bx) + br:
        c[r][cc] = col
    gi = [row[:] for row in c]
    go = [row[:] for row in c]
    for (r, cc) in occbx + occbr:
        gi[r][cc] = bgc
        go[r][cc] = 2
    if random.choice((True, False)):
        for (r, cc) in br:
            gi[r][cc] = bgc
            go[r][cc] = bgc
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Reading (shared with the golf entry): the figure is a set of straight LINES.
    Each line is a row or column of the rectangle that should be solid in the
    foreground colour. The rule completes each line by painting its gaps with 2.
    The golf entry handles one orientation at a time (rotate, fix every dense row,
    rotate again), so this version groups the work the same way:
      1. horizontal lines from top to bottom (top edge, the bar if it is
         horizontal, bottom edge), each over the full rectangle width;
      2. then vertical lines from left to right (left edge, the bar if it is
         vertical, right edge), each over the interior rows only, because the
         corners already belong to the horizontal lines.
    Each line is one unit: one Color2 on that line's background cells, which
    can be read from I.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    FILL = 2

    counts = Counter(I.flatten().tolist())
    col = min(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]   # leastcolor
    pts = np.argwhere(I == col)
    ops, sels = [], []
    if len(pts) == 0:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    r0, c0 = int(pts[:, 0].min()), int(pts[:, 1].min())
    r1, c1 = int(pts[:, 0].max()), int(pts[:, 1].max())

    # inner bar: foreground cells strictly inside the rectangle (0 or >=2 on one line)
    inner = [(int(r), int(c)) for r, c in pts if r0 < r < r1 and c0 < c < c1]
    h_rows = [r0, r1]
    v_cols = [c0, c1]
    if inner:
        rows = {r for r, _ in inner}
        cols = {c for _, c in inner}
        if len(rows) == 1 and len(cols) > 1:
            h_rows.append(next(iter(rows)))
        elif len(cols) == 1 and len(rows) > 1:
            v_cols.append(next(iter(cols)))
        else:   # one cell left (not produced by the generator): use the rule's tie-break
            rr, cc = inner[0]
            if (r1 - r0 + 1) > (c1 - c0 + 1):
                h_rows.append(rr)
            else:
                v_cols.append(cc)

    # pass 1: horizontal lines, top to bottom
    for r in sorted(h_rows):
        lo, hi_c = (c0, c1) if r in (r0, r1) else (c0 + 1, c1 - 1)
        gaps = [(r, c) for c in range(lo, hi_c + 1) if I[r, c] != col]
        if gaps:
            ops.append(FILL); sels.append(sel_of(gaps))

    # pass 2: vertical lines, left to right (interior rows; corners are already done)
    for c in sorted(v_cols):
        gaps = [(r, c) for r in range(r0 + 1, r1)
                if I[r, c] != col and r not in h_rows]
        if gaps:
            ops.append(FILL); sels.append(sel_of(gaps))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])   # whole grid: submit
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
