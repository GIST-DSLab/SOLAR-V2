"""
ARC Task: 2013d3e2 (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    # Only the background color needs to be shared across the episode: the rule
    # ("find the pinwheel of non-background cells, read out its generating tile")
    # depends on which color is background, not on the object's palette.
    bgc = choice(interval(1, 10, 1))
    return {"bgc": bgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int) -> dict:
    cols = interval(1, 10, 1)
    hmax = min(10, max(3, min(max_h, max_w) // 2))
    h = unifint(diff_lb, diff_ub, (3, hmax))
    w = h
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (2, h * w - 1))
    bx = asindices(canv)
    obj = {(choice(remcols), choice(totuple(bx)))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, toindices(obj))
        cnds = totuple(bx & dns)
        if len(cnds) == 0:
            break
        ch = choice(cnds)
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    gi = paint(canv, obj)
    gi1 = hconcat(gi, rot90(gi))
    gi2 = hconcat(rot270(gi), rot180(gi))
    gi = vconcat(gi1, gi2)
    fullh = unifint(diff_lb, diff_ub, (2 * h, max_h))
    fullw = unifint(diff_lb, diff_ub, (2 * w, max_w))
    gio = asobject(gi)
    gic = canvas(bgc, (fullh, fullw))
    loci = randint(0, fullh - 2 * h)
    locj = randint(0, fullw - 2 * w)
    gi = paint(gic, shift(gio, (loci, locj)))
    reminds = difference(asindices(gi), ofcolor(gi, bgc))
    go = lefthalf(tophalf(subgrid(reminds, gi)))
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    import numpy as np
    from collections import Counter

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # --- Find the pinwheel: the block of non-background cells.
    # The background is the color whose complement has a SQUARE bounding box that
    # is invariant under a quarter turn (that invariance is the whole point of the
    # picture: four rotated copies of one tile). Testing candidates by that
    # property is robust even when the background is not the majority color.
    bgc, box = None, None
    for cand, _ in Counter(I.flatten().tolist()).most_common():
        mask = (I != cand)
        if not mask.any():
            continue
        rs = np.where(mask.any(axis=1))[0]
        cs = np.where(mask.any(axis=0))[0]
        r0, r1, c0, c1 = int(rs[0]), int(rs[-1]), int(cs[0]), int(cs[-1])
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        if bh != bw or bh % 2:
            continue
        win = I[r0:r0 + bh, c0:c0 + bw]
        if not np.array_equal(np.rot90(win), win):
            continue
        bgc, box = cand, (r0, c0, bh)
        break
    if box is None:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
        mask = (I != bgc)
        rs = np.where(mask.any(axis=1))[0]
        cs = np.where(mask.any(axis=0))[0]
        box = (int(rs[0]), int(cs[0]), int(rs[-1]) - int(rs[0]) + 1)

    r0, c0, size = box
    m = size // 2
    if m != ho:                      # uniform border rows are trimmed symmetrically
        k = m - ho
        r0 += k
        c0 += k
        m = ho
        size = 2 * m

    # 1) Crop the canvas down to the pinwheel itself.
    #    Intended cells ARE exactly this full square rectangle -> bbox selection ok.
    if not (r0 == 0 and c0 == 0 and size == hi and size == wi):
        ops.append(33)
        sels.append([int(r0), int(c0), int(size) - 1, int(size) - 1])

    blk = I[r0:r0 + size, c0:c0 + size]
    bl = blk[m:, :m]                 # bottom-left blade = tile turned a quarter turn CCW
    upright = np.rot90(bl, k=3)      # turning it CW puts the tile upright

    if not np.array_equal(bl, upright):
        # 2) Take one blade of the pinwheel (bottom-left), full rectangle -> bbox ok.
        ops.append(33)
        sels.append([int(m), 0, int(m) - 1, int(m) - 1])
        # 3) Turn that blade upright: quarter turn clockwise (op25) on the whole
        #    square grid. This is the rotation the task is built on, performed.
        ops.append(25)
        sels.append([0, 0, int(m) - 1, int(m) - 1])
    else:
        # Degenerate case: the tile is itself quarter-turn symmetric, so rotating
        # would change nothing (a no-op). Read the upright blade directly.
        ops.append(33)
        sels.append([0, 0, int(m) - 1, int(m) - 1])

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 2013d3e2"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 2013d3e2"
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
                                f"for task 2013d3e2"
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
                    f"Failed to build a complete episode for task 2013d3e2 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"2013d3e2-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
