"""
ARC Task: 9aec4887 (RE-ARC) — LLM-generated grid_maker
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
    """All six roles (bgc, pc, c1..c4) are sampled randomly by the original generator,
    so all six are fixed once per episode.  Extra constraint: pc != 0, because the blob
    is carried into the box with CopyI/Paste and 0 is transparent to those ops."""
    cols = list(range(10))
    picked = random.sample(cols, 6)
    if picked[1] == 0:                      # keep pc non-zero
        for i in range(2, 6):
            if picked[i] != 0:
                picked[1], picked[i] = picked[i], picked[1]
                break
    bgc, pc, c1, c2, c3, c4 = picked
    return {"bgc": bgc, "pc": pc, "c1": c1, "c2": c2, "c3": c3, "c4": c4}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, pc=None, c1=None, c2=None, c3=None, c4=None) -> dict:
    if bgc is None:
        ck = sample_colors()
        bgc, pc = ck["bgc"], ck["pc"]
        c1, c2, c3, c4 = ck["c1"], ck["c2"], ck["c3"], ck["c4"]

    # a final rot90/rot270 can swap the grid dimensions -> bound both by the smaller cap
    lim = max(12, min(max_h, max_w))
    h = unifint(diff_lb, diff_ub, (12, lim))
    w = unifint(diff_lb, diff_ub, (12, lim))
    oh = unifint(diff_lb, diff_ub, (4, h // 2 - 2))
    ow = unifint(diff_lb, diff_ub, (4, w // 2 - 2))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (oh, ow))
    ln1 = connect((1, 0), (oh - 2, 0))
    ln2 = connect((1, ow - 1), (oh - 2, ow - 1))
    ln3 = connect((0, 1), (0, ow - 2))
    ln4 = connect((oh - 1, 1), (oh - 1, ow - 2))
    go = fill(go, c1, ln1)
    go = fill(go, c2, ln2)
    go = fill(go, c3, ln3)
    go = fill(go, c4, ln4)
    objB = asobject(go)
    bounds = asindices(canvas(-1, (oh - 2, ow - 2)))
    objA = {choice(totuple(bounds))}
    ncells = unifint(diff_lb, diff_ub, (1, ((oh - 2) * (ow - 2)) // 2))
    for k in range(ncells - 1):
        objA.add(choice(totuple((bounds - objA) & mapply(neighbors, objA))))
    while shape(objA) != (oh - 2, ow - 2):
        objA.add(choice(totuple((bounds - objA) & mapply(neighbors, objA))))
    fullinds = asindices(gi)
    loci = randint(0, h - 2 * oh + 2)
    locj = randint(0, w - ow)
    plcdB = shift(objB, (loci, locj))
    plcdi = toindices(plcdB)
    rems = sfilter(fullinds - plcdi,
                   lambda ij: loci + oh <= ij[0] <= h - oh + 2 and ij[1] <= w - ow + 2)
    loc = choice(totuple(rems))
    plcdA = shift(objA, loc)
    gi = paint(gi, plcdB)
    gi = fill(gi, pc, plcdA)
    objA = shift(objA, (1, 1))
    objs = objects(go, T, F, T)
    for ij in objA:
        manhs = {obj: manhattan(obj, {ij}) for obj in objs}
        manhsl = list(manhs.values())
        mmh = min(manhsl)
        if manhsl.count(mmh) == 1:
            col = color([o for o, mnh in manhs.items() if mmh == mnh][0])
        else:
            col = pc
        go = fill(go, col, {ij})
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """Rule (measured from I only):
       I holds a rectangular frame made of 4 straight one-cell-wide lines of 4 distinct
       colours (corners empty), plus one free-form blob of a 5th colour whose bounding box
       is exactly the size of the frame's interior.
       O = the frame box, with the blob carried inside (its bbox onto the interior) and
       every blob cell recoloured with the colour of the STRICTLY nearest frame line
       (manhattan, min over that line's cells); ties keep the blob colour.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    cells_by_color = {}
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v != bgc:
                cells_by_color.setdefault(v, []).append((r, c))

    # the blob is the only non-bg colour that is not a 1-cell-wide straight line
    blob_color = None
    for col in sorted(cells_by_color):
        pts = cells_by_color[col]
        rs = [p[0] for p in pts]
        cs = [p[1] for p in pts]
        if (max(rs) - min(rs)) >= 1 and (max(cs) - min(cs)) >= 1:
            blob_color = col
            break

    blob = cells_by_color[blob_color]
    a0 = min(p[0] for p in blob); a1 = max(p[0] for p in blob)
    b0 = min(p[1] for p in blob); b1 = max(p[1] for p in blob)
    ih, iw = a1 - a0 + 1, b1 - b0 + 1

    # the 4 frame lines -> the box region (measured from I, not from O)
    lines = {col: pts for col, pts in cells_by_color.items() if col != blob_color}
    frame_cells = [p for pts in lines.values() for p in pts]
    br = min(p[0] for p in frame_cells); br2 = max(p[0] for p in frame_cells)
    bc = min(p[1] for p in frame_cells); bc2 = max(p[1] for p in frame_cells)
    bh, bw = br2 - br + 1, bc2 - bc + 1

    ops, sels = [], []

    # 1) pick up the blob: its bounding box is exactly the frame's interior size,
    #    and the selection IS that full rectangle, so bbox form is exact here.
    ops.append(28); sels.append([a0, b0, ih - 1, iw - 1])

    # 2) crop the canvas down to the frame box (extent from the frame lines in I).
    #    Full-rectangle region -> bbox form is exact.
    ops.append(33); sels.append([br, bc, bh - 1, bw - 1])

    # 3) drop the blob onto the interior, top-left corner (1,1)
    ops.append(30); sels.append([1, 1, 0, 0])

    # 4) recolour each blob cell with its strictly-nearest frame line.
    #    Work in box coordinates.
    lines_box = {col: [(r - br, c - bc) for (r, c) in pts] for col, pts in lines.items()}
    blob_box = [(r - a0 + 1, c - b0 + 1) for (r, c) in blob]

    targets = {}
    for (r, c) in blob_box:
        dists = []
        for col, pts in lines_box.items():
            d = min(abs(r - pr) + abs(c - pc_) for (pr, pc_) in pts)
            dists.append((d, col))
        dmin = min(d for d, _ in dists)
        winners = [col for d, col in dists if d == dmin]
        if len(winners) == 1:                       # strict nearest -> that line's colour
            targets.setdefault(winners[0], []).append((r, c))
        # ties -> cell keeps the blob colour, already placed by the Paste

    # emit one Color op per frame line, ordered top / left / right / bottom
    def line_key(col):
        pts = lines_box[col]
        rs = [p[0] for p in pts]; cs = [p[1] for p in pts]
        horizontal = (max(rs) == min(rs))
        if horizontal:
            return (0, 0) if min(rs) == 0 else (3, 0)
        return (1, 0) if min(cs) == 0 else (2, 0)

    for col in sorted(targets, key=line_key):
        ops.append(int(col)); sels.append(sel_of(targets[col]))

    ops.append(34); sels.append([0, 0, bh - 1, bw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 9aec4887"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 9aec4887"
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
                                f"for task 9aec4887"
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
                    f"Failed to build a complete episode for task 9aec4887 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"9aec4887-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
