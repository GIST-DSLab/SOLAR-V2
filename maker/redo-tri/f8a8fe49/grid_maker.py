"""
ARC Task: f8a8fe49 (RE-ARC) — LLM-generated grid_maker
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


# ----------------------------------------------------------------------------
# Task f8a8fe49
#
# Structure (canonical orientation): a "box" made of two full lines of colour
# boxc (rows L1 and L2), each with two short end-stubs one row inside the box.
# In the INPUT a pattern of colour objc sits INSIDE the box, in two zones:
#   * rows L1+2 .. L1+objh+1   (just below the top line)
#   * rows L2-objh-1 .. L2-2   (just above the bottom line)
# The OUTPUT is the same grid with each zone REFLECTED ACROSS ITS OWN LINE
# (distance preserving), i.e. row L1+k -> row L1-k.
# The whole thing may be dmirrored (lines vertical instead of horizontal).
#
# The generator is patched so the pattern always occupies the row farthest
# from the line (uppermost(cells) == 0).  Without that pin, DSL `hmirror`
# mirrors inside the pattern's own bbox and the offset is no longer a
# reflection about the box line -> contradictory examples.
# ----------------------------------------------------------------------------


VARIANTS = [{"transposed": False}, {"transposed": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    # bgc and objc must be non-zero: ARCLE's object ops treat 0 as "nothing",
    # so a 0 background or a 0 pattern colour would break the band flip.
    bgc = random.choice([c for c in cols if c != 0])
    objc = random.choice([c for c in cols if c not in (0, bgc)])
    boxc = random.choice([c for c in cols if c not in (bgc, objc)])

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "objc": objc, "boxc": boxc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, boxc, transposed=None) -> dict:
    if transposed is None:
        transposed = random.choice([True, False])

    # a dmirror at the end swaps the dimensions -> swap the limits beforehand
    hlim = max_w if transposed else max_h
    wlim = max_h if transposed else max_w
    hlim = max(10, min(30, int(hlim)))
    wlim = max(4, min(30, int(wlim)))

    h = unifint(diff_lb, diff_ub, (10, hlim))
    w = unifint(diff_lb, diff_ub, (4, wlim))
    fullh = unifint(diff_lb, diff_ub, (10, h))
    fullw = unifint(diff_lb, diff_ub, (3, w))

    bcanv = canvas(bgc, (h, w))
    loci = random.randint(0, h - fullh)
    locj = random.randint(0, w - fullw)
    loc = (loci, locj)
    canvi = canvas(bgc, (fullh, fullw))
    canvo = canvas(bgc, (fullh, fullw))
    objh = (fullh // 2 - 3) // 2
    br = connect((objh + 1, 0), (objh + 1, fullw - 1))
    br = br | {(objh + 2, 0), (objh + 2, fullw - 1)}
    cands = backdrop(frozenset({(0, 1), (objh - 1, fullw - 2)}))
    toprow = totuple(sfilter(cands, lambda ij: ij[0] == 0))
    botrow = totuple(sfilter(cands, lambda ij: ij[0] == objh - 1))

    for k in range(2):
        canvi = fill(canvi, boxc, br)
        canvo = fill(canvo, boxc, br)
        ncellsd = unifint(diff_lb, diff_ub, (0, (objh * (fullw - 2)) // 2))
        ncells = random.choice((ncellsd, objh * (fullw - 2) - ncellsd))
        ncells = min(max(1, ncells), objh * (fullw - 2))
        cells = frozenset(random.sample(totuple(cands), ncells))
        # pin BOTH extreme rows of the pattern band: this keeps hmirror's axis
        # equal to the band's full extent, so the shift is exactly a
        # distance-preserving reflection about the box line.
        cells = insert(random.choice(botrow), cells)
        cells = insert(random.choice(toprow), cells)
        canvi = fill(canvi, objc, cells)
        canvo = fill(canvo, objc, shift(hmirror(cells), (objh + 3, 0)))
        canvi = hmirror(canvi)
        canvo = hmirror(canvo)

    gi = paint(bcanv, shift(asobject(canvi), loc))
    go = paint(bcanv, shift(asobject(canvo), loc))
    if transposed:
        gi = dmirror(gi)
        go = dmirror(go)
    go, gi = gi, go
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    ops, sels = [], []

    # (0,0) of the canvas is always outside the box -> background colour
    bgc = int(I[0, 0])

    palette = sorted(set(int(v) for v in I.flatten().tolist()) - {bgc})
    if len(palette) < 2:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    boxes = {}
    for c in palette:
        rs, cs = np.where(I == c)
        boxes[c] = (int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max()))
    # the frame encloses the pattern -> strictly larger bounding box
    boxc = max(palette, key=lambda c: (boxes[c][1] - boxes[c][0] + 1) *
                                      (boxes[c][3] - boxes[c][2] + 1))
    objc = [c for c in palette if c != boxc][0]

    r0, r1, c0, c1 = boxes[boxc]
    # the two long frame lines are rows iff the top row of the frame bbox is
    # entirely frame colour (otherwise the frame lines run vertically)
    horizontal = bool(np.all(I[r0, c0:c1 + 1] == boxc))

    ors, ocs = np.where(I == objc)
    pattern = list(zip(ors.tolist(), ocs.tolist()))

    if horizontal:
        L1, L2 = r0, r1
        span = list(range(c0, c1 + 1))
        limits = (h - 1)
        coord = lambda rc: rc[0]
        flip_op = 27  # FlipV: up<->down, reflection across a horizontal line
        make = lambda line, k, s, o: (line + s * k, o)
    else:
        L1, L2 = c0, c1
        span = list(range(r0, r1 + 1))
        limits = (w - 1)
        coord = lambda rc: rc[1]
        flip_op = 26  # FlipH: left<->right, reflection across a vertical line
        make = lambda line, k, s, o: (o, line + s * k)

    # each interior pattern cell belongs to the line it is closer to
    near_1, near_2 = [], []
    for cell in pattern:
        p = coord(cell)
        if L1 < p < L2:
            if (p - L1) <= (L2 - p):
                near_1.append(p - L1)
            else:
                near_2.append(L2 - p)

    for line, dists in ((L1, near_1), (L2, near_2)):
        if not dists:
            continue
        d = max(dists)
        d = min(d, line, limits - line)          # stays on the canvas
        if d < 2:
            continue
        # band symmetric about the line: the pattern strip and the empty strip
        # on the other side.  The line itself and its stub row/column
        # (distance 1) are left out, so the frame is not disturbed.
        cells = []
        for k in range(2, d + 1):
            for s in (-1, 1):
                for o in span:
                    cells.append(make(line, k, s, o))
        ops.append(flip_op)
        sels.append(sel_of(sorted(set(cells))))

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
                        f"num_examples+1 ({num_examples + 1}) for task f8a8fe49"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f8a8fe49"
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
                                f"for task f8a8fe49"
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
                    f"Failed to build a complete episode for task f8a8fe49 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f8a8fe49-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
