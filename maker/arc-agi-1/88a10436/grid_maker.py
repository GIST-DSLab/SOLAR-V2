"""
ARC Task: 88a10436 (RE-ARC) — LLM-generated grid_maker
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


# ----------------------------------------------------------------------------
# 1) episode-level colors
#    Roles: bgc (canvas), fgc (the single-cell markers), ccols_pool (the colours
#    the stamped shape is painted from).  The rule itself ("replace every marker
#    dot by the big shape, centred on the dot") is colour independent, but the
#    marker role must stay stable across the episode, so bgc/fgc are fixed here.
#    No discrete structural variants exist (one uniform rule), so no plan needed.
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, fgc = random.sample(cols, 2)
    pool = [c for c in cols if c != bgc and c != fgc]
    random.shuffle(pool)
    return {"bgc": bgc, "fgc": fgc, "ccols_pool": pool}


# ----------------------------------------------------------------------------
# 2) generator (RE-ARC generate_88a10436 with max_h/max_w and injected colours)
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, ccols_pool) -> dict:
    hlo = min(8, max_h)
    hhi = max(hlo, max_h)
    wlo = min(8, max_w)
    whi = max(wlo, max_w)
    h = unifint(diff_lb, diff_ub, (hlo, hhi))
    w = unifint(diff_lb, diff_ub, (wlo, whi))

    objh = unifint(diff_lb, diff_ub, (0, 2))
    objw = unifint(diff_lb, diff_ub, (0 if objh > 0 else 1, 2))
    objh = objh * 2 + 1
    objw = objw * 2 + 1
    while objh > h:
        objh -= 2
    while objw > w:
        objw -= 2
    if objh == 1 and objw == 1:                 # never a 1x1 "shape"
        if w >= 3:
            objw = 3
        elif h >= 3:
            objh = 3

    bb = asindices(canvas(-1, (objh, objw)))
    sp = (objh // 2, objw // 2)
    obj = {sp}
    bb = remove(sp, bb)
    ncells = unifint(diff_lb, diff_ub, (max(objh, objw), objh * objw))
    for k in range(ncells - 1):
        cands = totuple((bb - obj) & mapply(dneighbors, obj))
        if len(cands) == 0:
            break
        obj.add(choice(cands))
    while height(obj) != objh or width(obj) != objw:
        cands = totuple((bb - obj) & mapply(dneighbors, obj))
        if len(cands) == 0:
            break
        obj.add(choice(cands))
    ncells = max(1, len(obj))

    ncols = unifint(diff_lb, diff_ub, (1, len(ccols_pool)))
    ccols = ccols_pool[:ncols]
    obj = {(choice(ccols), ij) for ij in obj}
    obj = normalize(obj)

    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    loci = randint(0, h - objh)
    locj = randint(0, w - objw)
    loc = (loci, locj)
    plcd = shift(obj, loc)
    gi = paint(gi, plcd)
    go = paint(go, plcd)

    inds = (asindices(gi) - toindices(plcd)) - mapply(neighbors, toindices(plcd))
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (2 * ncells))))
    maxtrials = 4 * nobjs
    tr = 0
    succ = 0
    while succ < nobjs and tr <= maxtrials:
        if len(inds) == 0:
            break
        loc = choice(totuple(inds))
        plcd = shift(obj, loc)
        plcdi = toindices(plcd)
        if plcdi.issubset(inds):
            go = paint(go, plcd)
            gi = fill(gi, fgc, {center(plcdi)})
            succ += 1
            inds = (inds - plcdi) - mapply(dneighbors, plcdi)
        tr += 1

    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------
# 3) derive_operations
#    Rule: the one multi-cell component is a stamp; every isolated single cell
#    is a marker.  Each marker is replaced by a copy of the stamp placed so the
#    stamp's bbox-centre cell lands exactly on the marker.  The stamp may
#    contain colour 0 and the background may be non-zero, so Copy/Paste is
#    unsafe (0 is "nothing" to the clipboard, and the stamp's bbox holes must
#    NOT be repainted -- another copy's cell can legally sit in one).  We paint
#    each stamp copy, one Color op per colour of that copy.
# ----------------------------------------------------------------------------
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    ops, sels = [], []

    # background = the canvas colour the generator fills before placing anything
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- 4-connected non-background components -------------------------------
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] \
                                and I[nr, nc] != bgc:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                comps.append(sorted(cells))
    if not comps:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    # --- the stamp = the single multi-cell component -------------------------
    stamp = max(comps, key=len)
    rs = [r for r, _ in stamp]
    cs = [c for _, c in stamp]
    r0, c0 = min(rs), min(cs)
    oh = max(rs) - r0 + 1
    ow = max(cs) - c0 + 1
    ar, ac = oh // 2, ow // 2          # anchor cell inside the stamp's bbox

    pattern = [(r - r0 - ar, c - c0 - ac, int(I[r, c])) for (r, c) in stamp]
    colour_order = []
    for _, _, col in pattern:
        if col not in colour_order:
            colour_order.append(col)

    # --- markers = the isolated single cells, in reading order ---------------
    markers = sorted([cl[0] for cl in comps if len(cl) == 1])

    G = I.copy()                        # working grid, kept in sync with ARCLE
    for (mr, mc) in markers:
        by_col = {}
        for dr, dc, col in pattern:
            rr, cc = mr + dr, mc + dc
            if 0 <= rr < hi and 0 <= cc < wi:
                by_col.setdefault(col, []).append((rr, cc))
        for col in colour_order:
            cells = by_col.get(col, [])
            # skip cells that ALREADY hold this colour right now (no-op cells)
            cells = [(rr, cc) for (rr, cc) in cells if G[rr, cc] != col]
            if not cells:
                continue
            ops.append(int(col))
            sels.append(sel_of(cells))
            for (rr, cc) in cells:
                G[rr, cc] = col

    # full-grid rectangle: submitting the whole canvas
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
                        f"num_examples+1 ({num_examples + 1}) for task 88a10436"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 88a10436"
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
                                f"for task 88a10436"
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
                    f"Failed to build a complete episode for task 88a10436 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"88a10436-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
