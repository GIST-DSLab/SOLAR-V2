"""
ARC Task: b94a9452 (RE-ARC) — LLM-generated grid_maker
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
    # bgc / outer / inner are the three colours the generator samples.
    # The rule ("swap the two colours of the framed rectangle") is colour
    # agnostic, but the background and both object colours are kept fixed
    # across the whole episode so every instance shares one palette.
    cols = list(range(10))
    bgc, outer, inner = random.sample(cols, 3)
    return {"bgc": bgc, "outer": outer, "inner": inner}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, outer=None, inner=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or outer is None or inner is None:
        bgc, outer, inner = sample(cols, 3)

    hmax = max(4, min(30, int(max_h)))
    wmax = max(4, min(30, int(max_w)))

    h = unifint(diff_lb, diff_ub, (4, hmax))
    w = unifint(diff_lb, diff_ub, (4, wmax))
    c = canvas(bgc, (h, w))

    oh = unifint(diff_lb, diff_ub, (3, h - 1))
    ow = unifint(diff_lb, diff_ub, (3, w - 1))
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)

    oh2d = unifint(diff_lb, diff_ub, (0, oh // 2))
    ow2d = unifint(diff_lb, diff_ub, (0, ow // 2))
    oh2 = choice((oh2d, oh - oh2d))
    oh2 = min(max(1, oh2), oh - 2)
    ow2 = choice((ow2d, ow - ow2d))
    ow2 = min(max(1, ow2), ow - 2)
    loci2 = randint(loci + 1, loci + oh - oh2 - 1)
    locj2 = randint(locj + 1, locj + ow - ow2 - 1)

    obj1 = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    obj2 = backdrop(frozenset({(loci2, locj2), (loci2 + oh2 - 1, locj2 + ow2 - 1)}))
    gi = fill(c, outer, obj1)
    gi = fill(gi, inner, obj2)
    go = compress(gi)
    go = switch(go, outer, inner)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule (read off the input alone): the canvas holds one rectangle of a frame
    colour with a smaller solid block of a second colour strictly inside it.
    Zoom in on that rectangle and swap the two colours.

    Two people carrying out that same rule differ in three places, and each
    choice below is read off the INPUT geometry, never off O:

      * when to zoom in -- crop first if the rectangle is a small part of the
        canvas (<= 1/3 of it), otherwise recolour it where it lies and crop at
        the end;
      * whether to lay a base colour across the whole rectangle and draw the
        other region on top of it (natural when one region clearly dominates),
        or to paint the two regions straight onto the grid;
      * when painting the ring, which of its four bands to start from (the
        thickest one) and which way to sweep round (clockwise when the block
        sits on the right half, anticlockwise otherwise), and which of ring /
        block to take first (the larger one).
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # --- background -------------------------------------------------------
    # The generator guarantees oh <= h-1 and ow <= w-1, so at least one whole
    # row (and one whole column) of the canvas is pure background; no object
    # row can be uniform because it never spans the full width.  This pins bgc
    # without any majority assumption (the rectangle may cover most of the
    # canvas).
    bgc = None
    for r in range(hi):
        if len(set(I[r].tolist())) == 1:
            bgc = int(I[r, 0])
            break
    if bgc is None:
        for c in range(wi):
            if len(set(I[:, c].tolist())) == 1:
                bgc = int(I[0, c])
                break
    if bgc is None:
        bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    # --- the framed rectangle --------------------------------------------
    cells = np.argwhere(I != bgc)
    r0, c0 = int(cells[:, 0].min()), int(cells[:, 1].min())
    r1, c1 = int(cells[:, 0].max()), int(cells[:, 1].max())
    ho, wo = r1 - r0 + 1, c1 - c0 + 1
    sub = I[r0:r1 + 1, c0:c1 + 1]

    # the block sits strictly inside, so the rectangle's own corner is the
    # frame colour (this is geometric -- it does not assume which colour is
    # the more frequent one, and the block may well be the bigger region)
    frame_col = int(sub[0, 0])
    rest = sorted(v for v in set(sub.flatten().tolist()) if v != frame_col)
    ops, sels = [], []

    if not rest:
        # degenerate: a single colour, nothing to swap -- just zoom in
        ops.append(33); sels.append([r0, c0, ho - 1, wo - 1])
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    block_col = int(rest[0])
    bl = np.argwhere(sub == block_col)
    br0, bc0 = int(bl[:, 0].min()), int(bl[:, 1].min())
    br1, bc1 = int(bl[:, 0].max()), int(bl[:, 1].max())

    bbox_area = ho * wo
    block_area = (br1 - br0 + 1) * (bc1 - bc0 + 1)
    ring_area = bbox_area - block_area

    # choice 1: zoom in first, or recolour in place and zoom in afterwards
    crop_first = 3 * bbox_area <= hi * wi
    off_r, off_c = (0, 0) if crop_first else (r0, c0)

    def rect(rr0, rr1, cc0, cc1):
        return [(r + off_r, c + off_c)
                for r in range(rr0, rr1 + 1) for c in range(cc0, cc1 + 1)]

    full = rect(0, ho - 1, 0, wo - 1)          # the whole rectangle
    block = rect(br0, br1, bc0, bc1)           # the inner block
    bands = {
        'top':    rect(0, br0 - 1, 0, wo - 1),
        'right':  rect(br0, br1, bc1 + 1, wo - 1),
        'bottom': rect(br1 + 1, ho - 1, 0, wo - 1),
        'left':   rect(br0, br1, 0, bc0 - 1),
    }

    # choice 2: where to start sweeping the ring and which way round
    cyc = ['top', 'right', 'bottom', 'left']
    thick = {'top': br0, 'right': wo - 1 - bc1,
             'bottom': ho - 1 - br1, 'left': bc0}
    start = max(range(4), key=lambda i: thick[cyc[i]])
    clockwise = (bc0 + bc1) >= (wo - 1)        # block centred on the right half
    seq = [cyc[(start + (i if clockwise else -i)) % 4] for i in range(4)]
    ring_parts = [bands[k] for k in seq if bands[k]]   # empty bands emit nothing

    if crop_first:
        # bbox selection: the intended cells ARE exactly this full rectangle
        ops.append(33); sels.append([r0, c0, ho - 1, wo - 1])

    # choice 3: base-then-draw, or the two regions straight on
    if block_area * 3 <= bbox_area:
        # ring dominates -> lay the block colour right across the rectangle,
        # then redraw the block on top of it
        ops.append(block_col); sels.append(sel_of(full))
        ops.append(frame_col); sels.append(sel_of(block))
    elif ring_area * 3 <= bbox_area:
        # block dominates -> lay the frame colour across the rectangle, then
        # draw the thin ring back on, band by band
        ops.append(frame_col); sels.append(sel_of(full))
        for part in ring_parts:
            ops.append(block_col); sels.append(sel_of(part))
    elif block_area > ring_area:
        # comparable sizes -> paint each region once, larger one first
        ops.append(frame_col); sels.append(sel_of(block))
        for part in ring_parts:
            ops.append(block_col); sels.append(sel_of(part))
    else:
        for part in ring_parts:
            ops.append(block_col); sels.append(sel_of(part))
        ops.append(frame_col); sels.append(sel_of(block))

    if not crop_first:
        ops.append(33); sels.append([r0, c0, ho - 1, wo - 1])

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
                        f"num_examples+1 ({num_examples + 1}) for task b94a9452"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b94a9452"
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
                                f"for task b94a9452"
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
                    f"Failed to build a complete episode for task b94a9452 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b94a9452-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
