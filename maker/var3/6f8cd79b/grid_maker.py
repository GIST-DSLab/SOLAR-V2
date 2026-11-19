"""
ARC Task: 6f8cd79b (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # Generator samples: bgc (canvas colour) and a palette of foreground colours.
    # The rule (draw an 8-box around the whole grid) is colour-independent, but we
    # keep one episode-wide background and one episode-wide palette ordering so all
    # instances look like the same puzzle.  8 is hardcoded by the task -> not sampled.
    cols = [c for c in range(10) if c != 8]
    bgc = random.choice(cols)
    pool = [c for c in cols if c != bgc]
    random.shuffle(pool)
    return {"bgc": bgc, "colpool": pool}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, colpool=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    hmax = max(3, max_h)
    wmax = max(3, max_w)
    h = unifint(diff_lb, diff_ub, (3, hmax))
    w = unifint(diff_lb, diff_ub, (3, wmax))

    remcols = [c for c in cols if c != bgc]
    if colpool:
        pool = [c for c in colpool if c != bgc and c != 8]
        if not pool:
            pool = list(remcols)
    else:
        pool = list(remcols)
        random.shuffle(pool)

    gi = canvas(bgc, (h, w))
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ncols = max(1, min(ncols, len(pool)))
    ccols = pool[:ncols]

    ncells = unifint(diff_lb, diff_ub, (0, h * w))
    inds = asindices(gi)
    cells = random.sample(totuple(inds), ncells)
    obj = {(random.choice(ccols), ij) for ij in cells}
    gi = paint(gi, obj)

    brd = box(inds)
    go = fill(gi, 8, brd)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """Rule: draw a frame (colour 8) around the border of the grid.

    The frame is four border lines; which line a solver draws first, and whether
    they trace the ring corner-to-corner or lay the two full rows down and then
    fill the two column gaps, is a real choice with no bearing on the result.
    That choice is taken here from the input alone (its size, its colour count,
    its cell sum), so the same input always yields the same route and the answer
    grid is never consulted.  Every stroke paints a disjoint, non-empty run of
    cells that do not already hold the frame colour -- nothing is painted twice.
    """
    import numpy as np
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Frame colour is a convention of the task: read it from a demonstration.
    bcol = 8
    if examples:
        try:
            eo = np.asarray(examples[0][1], dtype=int)
            bcol = int(eo[0][0])
        except Exception:
            bcol = 8

    cur = I.copy()
    ops, sels = [], []

    def stroke(cells):
        cells = [(r, c) for (r, c) in cells if 0 <= r < hi and 0 <= c < wi]
        if not cells:
            return
        if all(cur[r, c] == bcol for (r, c) in cells):
            return                      # nothing would change -> emit nothing
        ops.append(int(bcol))
        sels.append(sel_of(cells))
        for (r, c) in cells:
            cur[r, c] = bcol

    if hi < 3 or wi < 3:
        # Degenerate: the frame is the whole grid, one stroke.
        ring = [(r, c) for r in range(hi) for c in range(wi)
                if r in (0, hi - 1) or c in (0, wi - 1)]
        stroke(ring)
    else:
        tot = int(I.sum())
        ncolors = len(set(I.reshape(-1).tolist()))
        style = (tot + ncolors + hi * wi) % 4
        start = (tot + hi + wi) % 4

        if style == 0:
            # Trace the ring clockwise; each stroke owns its starting corner.
            segs = [
                [(0, c) for c in range(0, wi - 1)],              # top,  TL -> TR
                [(r, wi - 1) for r in range(0, hi - 1)],         # right,TR -> BR
                [(hi - 1, c) for c in range(wi - 1, 0, -1)],     # bot,  BR -> BL
                [(r, 0) for r in range(hi - 1, 0, -1)],          # left, BL -> TL
            ]
            for i in range(4):
                stroke(segs[(start + i) % 4])

        elif style == 1:
            # Trace the ring counter-clockwise.
            segs = [
                [(0, c) for c in range(wi - 1, 0, -1)],          # top,  TR -> TL
                [(r, 0) for r in range(0, hi - 1)],              # left, TL -> BL
                [(hi - 1, c) for c in range(0, wi - 1)],         # bot,  BL -> BR
                [(r, wi - 1) for r in range(hi - 1, 0, -1)],     # right,BR -> TR
            ]
            for i in range(4):
                stroke(segs[(start + i) % 4])

        elif style == 2:
            # Both full rows first, then the two column gaps between them.
            fulls = [
                [(0, c) for c in range(wi)],
                [(hi - 1, c) for c in range(wi)],
            ]
            gaps = [
                [(r, 0) for r in range(1, hi - 1)],
                [(r, wi - 1) for r in range(1, hi - 1)],
            ]
            if start % 2:
                fulls.reverse()
            if (start // 2) % 2:
                gaps.reverse()
            for seg in fulls + gaps:
                stroke(seg)

        else:
            # Both full columns first, then the two row gaps between them.
            fulls = [
                [(r, 0) for r in range(hi)],
                [(r, wi - 1) for r in range(hi)],
            ]
            gaps = [
                [(0, c) for c in range(1, wi - 1)],
                [(hi - 1, c) for c in range(1, wi - 1)],
            ]
            if start % 2:
                fulls.reverse()
            if (start // 2) % 2:
                gaps.reverse()
            for seg in fulls + gaps:
                stroke(seg)

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])   # bbox == exactly the whole grid
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
                        f"num_examples+1 ({num_examples + 1}) for task 6f8cd79b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6f8cd79b"
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
                                f"for task 6f8cd79b"
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
                    f"Failed to build a complete episode for task 6f8cd79b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6f8cd79b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
