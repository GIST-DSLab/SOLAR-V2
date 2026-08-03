"""
ARC Task: 32597951 (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c != 3]
    bgc, noisec, fgc = random.sample(cols, 3)
    return {"bgc": bgc, "noisec": noisec, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec, fgc) -> dict:
    mh = max(10, min(30, int(max_h)))
    mw = max(10, min(30, int(max_w)))

    def bbox_area(grid, col, hh, ww):
        rs = [i for i in range(hh) for j in range(ww) if grid[i][j] == col]
        cs = [j for i in range(hh) for j in range(ww) if grid[i][j] == col]
        if not rs:
            return None
        return (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)

    while True:
        h = unifint(diff_lb, diff_ub, (10, mh))
        w = unifint(diff_lb, diff_ub, (10, mw))
        ih = unifint(diff_lb, diff_ub, (2, h // 2))
        iw = unifint(diff_lb, diff_ub, (2, w // 2))

        g = [[bgc for _ in range(w)] for _ in range(h)]
        ndev = unifint(diff_lb, diff_ub, (1, (h * w) // 2))
        num = random.choice((ndev, h * w - ndev))
        # keep both bgc and noisec plentiful so their bboxes stay grid-sized
        num = min(max(num, (h * w) // 4), (3 * h * w) // 4)
        allcells = [(i, j) for i in range(h) for j in range(w)]
        for (i, j) in random.sample(allcells, num):
            g[i][j] = noisec
        # anchor both scattered colors at opposite grid corners
        g[0][0] = bgc
        g[h - 1][w - 1] = bgc
        g[0][w - 1] = noisec
        g[h - 1][0] = noisec

        loci = random.randint(0, h - ih)
        locj = random.randint(0, w - iw)
        block = [(i, j) for i in range(loci, loci + ih) for j in range(locj, locj + iw)]
        # opposite block corners become fgc -> fgc bbox is exactly the block
        g[loci][locj] = bgc
        g[loci + ih - 1][locj + iw - 1] = bgc

        fg = [(i, j) for (i, j) in block if g[i][j] == bgc]
        rest = [(i, j) for (i, j) in block if g[i][j] != bgc]
        if not rest:
            cand = [p for p in block
                    if p != (loci, locj) and p != (loci + ih - 1, locj + iw - 1)]
            pi, pj = random.choice(cand)
            g[pi][pj] = noisec
            fg = [(i, j) for (i, j) in block if g[i][j] == bgc]
            rest = [(i, j) for (i, j) in block if g[i][j] != bgc]

        gi = [row[:] for row in g]
        for (i, j) in fg:
            gi[i][j] = fgc

        # fgc must be the unique minimal-bbox-area color (the rule's anchor)
        a_f = bbox_area(gi, fgc, h, w)
        ok = a_f is not None
        if ok:
            for col in set(v for row in gi for v in row):
                if col == fgc:
                    continue
                a = bbox_area(gi, col, h, w)
                if a is not None and a <= a_f:
                    ok = False
                    break
        if not ok:
            continue

        go = [row[:] for row in gi]
        for (i, j) in rest:
            go[i][j] = 3

        return {
            'input': tuple(tuple(r) for r in gi),
            'output': tuple(tuple(r) for r in go),
        }


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # Rule (measured from I alone): the color with the smallest bounding-box area
    # is the compact foreground block; every cell inside that bbox which is NOT
    # that color becomes 3.
    best = None
    for col in np.unique(I):
        rs, cs = np.where(I == col)
        r0, r1 = int(rs.min()), int(rs.max())
        c0, c1 = int(cs.min()), int(cs.max())
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if best is None or area < best[0]:
            best = (area, int(col), r0, r1, c0, c1)

    _, fgc, r0, r1, c0, c1 = best

    # bbox-minus-object mask, derived from I
    targets = [(r, c)
               for r in range(r0, r1 + 1)
               for c in range(c0, c1 + 1)
               if I[r, c] != fgc]

    ops, sels = [], []
    if targets:
        ops.append(3)
        sels.append(sel_of(targets))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 32597951"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 32597951"
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
                                f"for task 32597951"
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
                    f"Failed to build a complete episode for task 32597951 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"32597951-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
