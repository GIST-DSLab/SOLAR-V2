"""
ARC Task: 045e512c (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    objc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "objc": objc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int, objc: int) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (11, max_h))
    w = unifint(diff_lb, diff_ub, (11, max_w))
    while True:
        oh = unifint(diff_lb, diff_ub, (2, min(4, (h - 2) // 3)))
        ow = unifint(diff_lb, diff_ub, (2, min(4, (w - 2) // 3)))
        bounds = asindices(canvas(-1, (oh, ow)))
        c1 = choice(totuple(connect((0, 0), (oh - 1, 0))))
        c2 = choice(totuple(connect((0, 0), (0, ow - 1))))
        c3 = choice(totuple(connect((oh - 1, ow - 1), (oh - 1, 0))))
        c4 = choice(totuple(connect((oh - 1, ow - 1), (0, ow - 1))))
        obj = {c1, c2, c3, c4}
        remcands = totuple(bounds - obj)
        ncells = unifint(diff_lb, diff_ub, (0, len(remcands)))
        for k in range(ncells):
            loc = choice(remcands)
            obj.add(loc)
            remcands = remove(loc, remcands)
        objt = normalize(obj)
        cc = canvas(0, shape(obj))
        cc = fill(cc, 1, objt)
        if len(colorfilter(objects(cc, T, T, F), 1)) == 1:
            break
    loci = randint(oh + 1, h - 2 * oh - 1)
    locj = randint(ow + 1, w - 2 * ow - 1)
    loc = (loci, locj)
    remcols = remove(bgc, remove(objc, cols))
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, ncols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    obj = shift(recolor(objc, obj), loc)
    gi = paint(gi, obj)
    go = paint(go, obj)
    options = totuple(neighbors((0, 0)))
    ndirs = unifint(diff_lb, diff_ub, (1, 8))
    dirs = sample(options, ndirs)
    dcols = [choice(ccols) for k in range(ndirs)]
    hbars = hfrontier((loci - 2, 0)) | hfrontier((loci + oh + 1, 0))
    vbars = vfrontier((0, locj - 2)) | vfrontier((0, locj + ow + 1))
    bars = hbars | vbars
    ofs = increment((oh, ow))
    for direc, col in zip(dirs, dcols):
        indicatorobj = shift(obj, multiply(direc, increment((oh, ow))))
        indicatorobj = sfilter(indicatorobj, lambda cij: cij[1] in bars)
        nindsd = unifint(diff_lb, diff_ub, (0, len(indicatorobj) - 1))
        ninds = len(indicatorobj) - nindsd
        indicatorobj = set(sample(totuple(indicatorobj), ninds))
        if len(indicatorobj) > 0 and len(indicatorobj) < len(obj):
            gi = fill(gi, col, indicatorobj)
            for k in range(1, 10):
                go = fill(go, col, shift(obj, multiply(multiply(k, direc), ofs)))
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """Rule (measured from I only):
       - largest 8-connected same-colour component = the master shape, size (oh, ow)
       - step offset = (oh+1, ow+1)
       - for each of the 8 directions, look at the master shape's footprint shifted
         by ONE step: if I holds a partial (indicator) copy there, that direction is
         live and its colour is the indicator colour
       - paint a FULL copy of the master shape at every multiple k of that step,
         outward, in that colour.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # 8-connected same-colour components of non-background cells
    seen = set()
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == bgc or (r, c) in seen:
                continue
            col = int(I[r, c])
            stack = [(r, c)]
            seen.add((r, c))
            comp = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in seen and I[ny, nx] == col:
                            seen.add((ny, nx))
                            stack.append((ny, nx))
            comps.append(comp)

    ops, sels = [], []
    if comps:
        best = max(comps, key=len)                    # master shape (indicators are strict subsets)
        rs = [p[0] for p in best]
        cs = [p[1] for p in best]
        oh = max(rs) - min(rs) + 1
        ow = max(cs) - min(cs) + 1
        sr, sc = oh + 1, ow + 1                       # step between consecutive copies
        K = max(h // oh, w // ow)                     # how many copies can reach across the grid

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                # is this direction live?  read the indicator sitting one step away in I
                probe = [(y + dr * sr, x + dc * sc) for y, x in best]
                vals = [int(I[y, x]) for y, x in probe
                        if 0 <= y < h and 0 <= x < w and I[y, x] != bgc]
                if not vals:
                    continue
                col = Counter(vals).most_common(1)[0][0]
                # replicate the master shape outward along this direction
                for k in range(1, K + 1):
                    cells = [(y + k * dr * sr, x + k * dc * sc) for y, x in best]
                    cells = [(y, x) for y, x in cells if 0 <= y < h and 0 <= x < w]
                    if not cells:
                        continue
                    ops.append(int(col))
                    sels.append(sel_of(cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task 045e512c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 045e512c"
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
                                f"for task 045e512c"
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
                    f"Failed to build a complete episode for task 045e512c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"045e512c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
