"""
ARC Task: ff805c23 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------------------
# 1) colors: canvas background + the palette the mosaic is drawn from.
#    The rule (fill the punched-out rectangle by mirror symmetry, output the
#    patch) does not depend on which colors are used, only on bgc being fixed
#    and the hole color (0 here) never occurring as ordinary content.
# ---------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    cols = list(range(1, 10))               # 0 is reserved for the punched hole
    bgc = random.choice(cols)
    rest = [c for c in cols if c != bgc]
    random.shuffle(rest)
    return {"bgc": bgc, "palette": rest}


# ---------------------------------------------------------------------------
# 2) generator: 4/8-fold symmetric mosaic, a rectangle punched out with 0,
#    output = the punched-out content. Whole thing randomly rotated.
# ---------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, palette=None, **kwargs) -> dict:
    cols = interval(1, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    if palette is None:
        palette = list(remove(bgc, cols))

    hmax = min(15, max_h // 2, max_w // 2)
    if hmax < 3:
        hmax = 3
    h = unifint(diff_lb, diff_ub, (3, hmax))
    w = h

    remcols = list(palette)
    numcols = unifint(diff_lb, diff_ub, (1, min(8, len(remcols))))
    remcols = sample(remcols, numcols)

    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (1, h * w))
    bx = asindices(canv)
    obj = {(choice(remcols), choice(totuple(bx)))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, toindices(obj))
        cands = totuple(bx & dns)
        if len(cands) == 0:
            break
        ch = choice(cands)
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    gi = paint(canv, obj)

    # make the tile diagonally symmetric, then mirror it out to 2h x 2w
    tr = sfilter(asobject(dmirror(gi)), lambda cij: cij[1][1] >= cij[1][0])
    gi = paint(gi, tr)
    gi = hconcat(gi, vmirror(gi))
    gi = vconcat(gi, hmirror(gi))

    locidev = unifint(diff_lb, diff_ub, (1, 2 * h))
    locjdev = unifint(diff_lb, diff_ub, (1, w))
    loci = 2 * h - locidev
    locj = w - locjdev
    loci2 = unifint(diff_lb, diff_ub, (loci, 2 * h - 1))
    locj2 = unifint(diff_lb, diff_ub, (locj, w - 1))
    bd = backdrop(frozenset({(loci, locj), (loci2, locj2)}))
    go = subgrid(bd, gi)
    gi = fill(gi, 0, bd)

    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# ---------------------------------------------------------------------------
# 3) derive_operations
#
# Rule: the grid is a mirror-symmetric mosaic with ONE solid rectangle painted
# over in a single "hole" color (0 in the RE-ARC instances, an arbitrary color
# in the original ARC pairs -- detected dynamically, never assumed).
# Mirroring the whole grid across an axis whose reflection of the hole lands on
# intact mosaic brings the missing content INTO the hole's own position
# (and pushes the hole away).  Then crop that rectangle: it is the answer.
#
# Copy/Paste is unusable here: the answer may legitimately contain 0 cells and
# Paste treats 0 as transparent -- a geometric flip carries them correctly.
# ---------------------------------------------------------------------------
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    ho, wo = O.shape

    def cands():
        out = []
        for col in np.unique(I).tolist():
            cells = np.argwhere(I == col)
            r0 = int(cells[:, 0].min()); r1 = int(cells[:, 0].max())
            c0 = int(cells[:, 1].min()); c1 = int(cells[:, 1].max())
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            if cells.shape[0] != hh * ww:          # not a solid rectangle -> not the hole
                continue
            for op in (27, 26):                    # 27 = FlipV (up/down), 26 = FlipH (left/right)
                if op == 27:
                    mr0, mr1, mc0, mc1 = H - 1 - r1, H - 1 - r0, c0, c1
                else:
                    mr0, mr1, mc0, mc1 = r0, r1, W - 1 - c1, W - 1 - c0
                # the mirror source must be intact mosaic, not the hole itself
                if not (mr1 < r0 or mr0 > r1 or mc1 < c0 or mc0 > c1):
                    continue
                block = I[mr0:mr1 + 1, mc0:mc1 + 1]
                patch = np.flipud(block) if op == 27 else np.fliplr(block)
                real = 1 if not np.array_equal(patch, I[r0:r1 + 1, c0:c1 + 1]) else 0
                out.append((op, r0, c0, hh, ww, patch, real))
        return out

    all_c = cands()
    exact = [x for x in all_c if x[5].shape == (ho, wo) and np.array_equal(x[5], O)]
    pool = exact if exact else all_c
    if pool:
        # prefer a rectangle whose mirror really differs (a genuine hole), then the largest
        pool.sort(key=lambda x: (-x[6], -(x[3] * x[4])))
        flip_op, r0, c0, hh, ww = pool[0][0], pool[0][1], pool[0][2], pool[0][3], pool[0][4]
    else:                                          # degenerate safety net
        flip_op, r0, c0, hh, ww = 27, 0, 0, ho, wo

    ops, sels = [], []

    # 1. mirror the whole mosaic: the symmetric counterpart of the missing
    #    rectangle slides onto the hole's own position. Selection is the whole
    #    grid rectangle (background included) -- exactly the intended cells.
    ops.append(flip_op); sels.append([0, 0, H - 1, W - 1])

    # 2. crop the rectangle that used to be the hole: it now holds the answer.
    #    Selection is exactly that full rectangle.
    ops.append(33); sels.append([r0, c0, hh - 1, ww - 1])

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task ff805c23"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ff805c23"
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
                                f"for task ff805c23"
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
                    f"Failed to build a complete episode for task ff805c23 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ff805c23-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
