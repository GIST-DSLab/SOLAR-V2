"""
ARC Task: ec883f72 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)


def sample_colors(num_examples=None) -> dict:
    bgc, sqc, linc = random.sample(range(10), 3)
    return {"bgc": bgc, "sqc": sqc, "linc": linc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, sqc=1, linc=2, **kwargs) -> dict:
    h = _unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = _unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    ohi = _unifint(diff_lb, diff_ub, (0, h - 6))
    owi = _unifint(diff_lb, diff_ub, (0, w - 6))
    oh = h - 5 - ohi
    ow = w - 5 - owi
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    gi = [[bgc] * w for _ in range(h)]
    for r in range(loci, loci + oh):
        for c in range(locj, locj + ow):
            gi[r][c] = sqc
    # outbox(outbox(square)) frame
    r0, r1, c0, c1 = loci - 2, loci + oh + 1, locj - 2, locj + ow + 1
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r in (r0, r1) or c in (c0, c1)) and 0 <= r < h and 0 <= c < w:
                gi[r][c] = linc
    go = [row[:] for row in gi]
    for (sr, sc, dr, dc) in ((r1, c1, 1, 1), (r0, c0, -1, -1), (r1, c0, 1, -1), (r0, c1, -1, 1)):
        r, c = sr, sc
        while -60 <= r <= 60 and -60 <= c <= 60:
            if 0 <= r < h and 0 <= c < w and gi[r][c] == bgc:
                go[r][c] = sqc
            r += dr
            c += dc
    return {"input": tuple(tuple(x) for x in gi), "output": tuple(tuple(x) for x in go)}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # The square is the one colour whose cells fill their bounding box exactly;
    # the (possibly clipped) frame is always at least an L shape, and the
    # background surrounds everything, so neither is a solid rectangle.
    sqc = None
    best = -1
    for col in np.unique(I).tolist():
        rs, cs = np.nonzero(I == col)
        area = (rs.max() - rs.min() + 1) * (cs.max() - cs.min() + 1)
        if area == len(rs) and area < hi * wi and len(rs) > best:
            sqc, best = col, len(rs)
    rs, cs = np.nonzero(I == sqc)
    sr0, sr1, sc0, sc1 = rs.min(), rs.max(), cs.min(), cs.max()
    # frame corners lie 2 cells outside the square; diagonals start one beyond
    fr0, fr1, fc0, fc1 = sr0 - 2, sr1 + 2, sc0 - 2, sc1 + 2
    ops, sels = [], []
    for (r, c, dr, dc) in ((fr0, fc0, -1, -1), (fr0, fc1, -1, 1),
                           (fr1, fc0, 1, -1), (fr1, fc1, 1, 1)):
        cells = []
        r += dr
        c += dc
        while 0 <= r < hi and 0 <= c < wi:
            cells.append((int(r), int(c)))
            r += dr
            c += dc
        if cells:
            # paint this whole diagonal ray in the square's colour
            ops.append(int(sqc)); sels.append(sel_of(cells))
    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task ec883f72"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ec883f72"
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
                                f"for task ec883f72"
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
                    f"Failed to build a complete episode for task ec883f72 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ec883f72-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
