"""
ARC Task: 7f4411dc (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
from collections import deque, Counter

try:
    from maker.sel_helpers import sel_of
except Exception:  # lightweight stand-in with the same mask semantics
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for r, c in cells]}


# ----------------------------------------------------------------------------
# 1. colors
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    # generator does: bgc, fgc = sample(cols, 2)
    # fgc must be non-zero: the trajectory replicates the squares with
    # CopyI/Paste and colour 0 is "nothing" to the clipboard.
    cols = list(range(10))
    fgc = random.choice([c for c in cols if c != 0])
    bgc = random.choice([c for c in cols if c != fgc])
    return {"bgc": bgc, "fgc": fgc}


# ----------------------------------------------------------------------------
# 2. generator (RE-ARC generate_7f4411dc with max_h/max_w and injected colors)
# ----------------------------------------------------------------------------
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, fgc: int) -> dict:
    hub = max(10, min(30, int(max_h)))
    wub = max(10, min(30, int(max_w)))
    h = unifint(diff_lb, diff_ub, (10, hub))
    w = unifint(diff_lb, diff_ub, (10, wub))
    nsq = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 15)))
    maxtr = 4 * nsq
    tr = 0
    succ = 0
    go = canvas(bgc, (h, w))
    inds = asindices(go)
    while tr < maxtr and succ < nsq:
        oh = randint(2, 6)
        ow = randint(2, 6)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        loci, locj = loc
        obj = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        obj = shift(obj, loc)
        if obj.issubset(inds):
            go = fill(go, fgc, obj)
            succ += 1
            inds = (inds - obj) - outbox(obj)
        tr += 1
    inds = ofcolor(go, bgc)
    nnoise = unifint(diff_lb, diff_ub, (0, max(0, len(inds) // 2 - 1)))
    gi = tuple(e for e in go)
    for k in range(nnoise):
        if len(inds) == 0:
            break
        loc = choice(totuple(inds))
        inds = inds - dneighbors(loc)
        inds = inds - {loc}
        gi = fill(gi, fgc, {loc})
    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------
# 3. trajectory
#
# Rule (verifier: occurrences of a 2x2 block of the object colour):
#   the kept cells are exactly the cells covered by some solid 2x2 square of
#   the object colour; everything else becomes background.  Route:
#     * lay the background canvas over the whole grid (wipes the noise),
#     * then REPLICATE each solid square back onto it: CopyI its region from
#       the untouched input, Paste it at that region's origin.
# ----------------------------------------------------------------------------
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    def cover(c):
        """cells belonging to at least one solid 2x2 block of colour c in I"""
        m = (I == c)
        cov = np.zeros((h, w), dtype=bool)
        for r in range(h - 1):
            for cc in range(w - 1):
                if m[r, cc] and m[r, cc + 1] and m[r + 1, cc] and m[r + 1, cc + 1]:
                    cov[r, cc] = cov[r, cc + 1] = True
                    cov[r + 1, cc] = cov[r + 1, cc + 1] = True
        return cov

    def components(mask):
        seen = np.zeros((h, w), dtype=bool)
        out = []
        for r in range(h):
            for c in range(w):
                if mask[r, c] and not seen[r, c]:
                    seen[r, c] = True
                    q = deque([(r, c)])
                    cells = []
                    while q:
                        y, x = q.popleft()
                        cells.append((y, x))
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True
                                q.append((ny, nx))
                    out.append(cells)
        return out

    def solid_rects(cl):
        boxes = []
        for cells in cl:
            r0 = min(r for r, _ in cells); r1 = max(r for r, _ in cells)
            c0 = min(c for _, c in cells); c1 = max(c for _, c in cells)
            if (r1 - r0 + 1) * (c1 - c0 + 1) != len(cells):
                return None
            boxes.append((r0, c0, r1 - r0 + 1, c1 - c0 + 1))
        return boxes

    # --- which colour is the object colour? -------------------------------
    fgc = None
    fg_boxes = None
    fallback = None
    for c in sorted(set(I.flatten().tolist())):
        cov = cover(c)
        if not cov.any():
            continue
        boxes = solid_rects(components(cov))
        if boxes is None:            # its 2x2-covered set is not a set of squares
            continue
        if fallback is None:
            fallback = (c, boxes)
        if np.array_equal(cov, O == c):
            fgc, fg_boxes = c, boxes
            break
    if fgc is None and fallback is not None:
        fgc, fg_boxes = fallback

    # background colour = the other colour of the (2-colour) grid
    cnt = Counter(O.flatten().tolist())
    bgc = None
    for col, _ in cnt.most_common():
        if col != fgc:
            bgc = col
            break
    if bgc is None:
        bgc = int(cnt.most_common(1)[0][0])

    if fgc is None:
        # degenerate safety net (no 2x2 square anywhere): plain repaint
        for col in sorted(set(O.flatten().tolist())):
            cells = [(r, c) for r in range(h) for c in range(w)
                     if O[r, c] == col and I[r, c] != col]
            if cells:
                ops.append(int(col)); sels.append(sel_of(cells))
        ops.append(34); sels.append(sel_of([(r, c) for r in range(h) for c in range(w)]))
        return ops, sels

    all_cells = [(r, c) for r in range(h) for c in range(w)]

    # 1. lay the background canvas over the whole grid (the noise disappears)
    #    selection = the entire grid, a full rectangle
    ops.append(int(bgc)); sels.append(sel_of(all_cells))

    # 2. replicate every solid square: copy it from the (unchanged) input and
    #    paste it back onto the fresh canvas at its own origin
    for (r0, c0, rh, rw) in fg_boxes:
        region = [(r, c) for r in range(r0, r0 + rh) for c in range(c0, c0 + rw)]
        if fgc == 0:
            # colour 0 is invisible to the clipboard -> paint the square
            ops.append(0); sels.append(sel_of(region))
        else:
            ops.append(28); sels.append(sel_of(region))       # CopyI: full rectangle
            ops.append(30); sels.append(sel_of([(r0, c0)]))   # Paste at its origin

    ops.append(34); sels.append(sel_of(all_cells))
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
                        f"num_examples+1 ({num_examples + 1}) for task 7f4411dc"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7f4411dc"
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
                                f"for task 7f4411dc"
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
                    f"Failed to build a complete episode for task 7f4411dc "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7f4411dc-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
