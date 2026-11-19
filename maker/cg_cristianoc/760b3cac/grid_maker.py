"""
ARC Task: 760b3cac (RE-ARC) — LLM-generated grid_maker
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
    bgc, objc, indc = random.sample(range(10), 3)
    return {"bgc": bgc, "objc": objc, "indc": indc}


def _unifint(lb, ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(lb, ub)
    return min(max(a, round(a + (b - a) * d)), b)


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, objc=1, indc=2) -> dict:
    lim = min(max_h, max_w, 30)   # the final rotation may swap h and w
    h = _unifint(diff_lb, diff_ub, (5, lim))
    w = 2 * _unifint(diff_lb, diff_ub, (3, min(14, (lim - 1) // 2))) + 1
    objh = _unifint(diff_lb, diff_ub, (1, h - 3))
    objw = 2 * _unifint(diff_lb, diff_ub, (1, w // 6)) + 1
    gi = np.full((h, w), bgc, dtype=int)
    objL = [(0, 0), (1, 0), (1, 1), (1, 2), (2, 1)]
    if random.random() < 0.5:
        obj, sgn = objL, -1
    else:
        obj, sgn = [(r, 2 - c) for r, c in objL], 1
    for r, c in obj:
        gi[h - 3 + r, w // 2 - 1 + c] = indc
    inds = {(i, j) for i in range(objh) for j in range(objw)}
    objx = {random.choice(sorted(inds))}

    def frontier():
        nb = {(i + di, j + dj) for (i, j) in objx
              for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj}
        return sorted((nb & inds) - objx)

    numcd = _unifint(diff_lb, diff_ub, (0, (objh * objw) // 2))
    numc = random.choice((numcd, objh * objw - numcd))
    numc = min(max(1, numc), objh * objw)
    for _ in range(numc - 1):
        objx.add(random.choice(frontier()))
    while max(j for _, j in objx) - min(j for _, j in objx) + 1 != objw:
        objx.add(random.choice(frontier()))
    mi = min(i for i, _ in objx); mj = min(j for _, j in objx)
    objx = {(i - mi, j - mj) for i, j in objx}
    objh = max(i for i, _ in objx) + 1
    objw = max(j for _, j in objx) + 1
    loci = random.randint(0, h - 3 - objh)
    locj = w // 2 - objw // 2
    plcd = {(i + loci, j + locj) for i, j in objx}
    for i, j in plcd:
        gi[i, j] = objc
    go = gi.copy()
    lo, hi_ = min(j for _, j in plcd), max(j for _, j in plcd)
    for i, j in plcd:
        jj = lo + hi_ - j + objw * sgn
        if 0 <= jj < w:
            go[i, jj] = objc
    k = random.randint(0, 3)
    gi = np.rot90(gi, k); go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def _arrow_like(cells):
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    return (len(cells) == 5 and max(rs) - min(rs) == 2 and max(cs) - min(cs) == 2)


def derive_operations(I, O, examples=None):
    # Reading shared by the maker and the golf entry (they AGREE here): the
    # object's whole bounding strip is reflected across the edge the arrow points
    # at (golf: rows[3:6][::-1] written beside it). Both treat the bbox strip as
    # the unit, so that carving is kept: copy the strip, paste it beside the
    # object, mirror it in place. One refinement: the copied strip is cut by the
    # grid edge (only the part of the source whose reflection lands on the
    # canvas), so a reflection running off the grid is still a copy+mirror.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    cnt = Counter(I.flatten().tolist())
    for ex_i, _ in (examples or []):
        cnt.update(np.asarray(ex_i, dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    groups = {}
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v != bgc:
                groups.setdefault(v, set()).add((r, c))
    if not groups:
        return [34], [[0, 0, hi - 1, wi - 1]]

    # the object's colour = the one the demonstrations grow
    main_col = None
    grown = Counter()
    for ex_i, ex_o in (examples or []):
        ex_i = np.asarray(ex_i, dtype=int); ex_o = np.asarray(ex_o, dtype=int)
        if ex_i.shape != ex_o.shape:
            continue
        for v in np.unique(ex_o).tolist():
            if int((ex_o == v).sum()) > int((ex_i == v).sum()):
                grown[int(v)] += 1
    for v, _ in grown.most_common():
        if v in groups:
            main_col = v
            break
    if main_col is None:
        rest = [k for k in groups if not _arrow_like(groups[k])]
        pool = rest or list(groups)
        main_col = max(pool, key=lambda k: len(groups[k]))
    others = [k for k in groups if k != main_col]
    if not others:
        return [34], [[0, 0, hi - 1, wi - 1]]
    arrow = groups[others[0]]
    main = groups[main_col]

    # direction: the side of the arrow's 3x3 box that holds two of its cells
    ar = [r for r, _ in arrow]; ac = [c for _, c in arrow]
    if sum(1 for r in ar if r == max(ar)) == 2:
        d = (1, 0)
    elif sum(1 for r in ar if r == min(ar)) == 2:
        d = (-1, 0)
    elif sum(1 for c in ac if c == max(ac)) == 2:
        d = (0, 1)
    else:
        d = (0, -1)

    r0 = min(r for r, _ in main); r1 = max(r for r, _ in main)
    c0 = min(c for _, c in main); c1 = max(c for _, c in main)

    if d[0] == 0:
        if d[1] < 0:
            a, b = c0, min(c1, 2 * c0 - 1)
            mir = lambda c: 2 * c0 - 1 - c
        else:
            a, b = max(c0, 2 * c1 + 2 - wi), c1
            mir = lambda c: 2 * c1 + 1 - c
        if b < a:
            return [34], [[0, 0, hi - 1, wi - 1]]
        dst = min(mir(a), mir(b))
        src_box = [r0, a, r1 - r0, b - a]
        dst_box = [r0, dst, r1 - r0, b - a]
        block = I[r0:r1 + 1, a:b + 1]
        sym = np.array_equal(block, block[:, ::-1])
        flip_op = 26
        mirrored = [(r, mir(c)) for r, c in main if a <= c <= b]
    else:
        if d[0] < 0:
            a, b = r0, min(r1, 2 * r0 - 1)
            mir = lambda r: 2 * r0 - 1 - r
        else:
            a, b = max(r0, 2 * r1 + 2 - hi), r1
            mir = lambda r: 2 * r1 + 1 - r
        if b < a:
            return [34], [[0, 0, hi - 1, wi - 1]]
        dst = min(mir(a), mir(b))
        src_box = [a, c0, b - a, c1 - c0]
        dst_box = [dst, c0, b - a, c1 - c0]
        block = I[a:b + 1, c0:c1 + 1]
        sym = np.array_equal(block, block[::-1, :])
        flip_op = 27
        mirrored = [(mir(r), c) for r, c in main if a <= r <= b]

    if main_col != 0:
        # full rectangles on purpose: the strip (background included) is the unit
        ops.append(28); sels.append(src_box)                          # CopyI the strip
        ops.append(30); sels.append([dst_box[0], dst_box[1], 0, 0])   # Paste beside it
        if not sym:
            ops.append(flip_op); sels.append(dst_box)                 # mirror it in place
    else:
        # colour 0 is transparent to Copy/Paste/Flip: draw the reflection directly
        ops.append(0); sels.append(sel_of(sorted(mirrored)))

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
                        f"num_examples+1 ({num_examples + 1}) for task 760b3cac"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 760b3cac"
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
                                f"for task 760b3cac"
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
                    f"Failed to build a complete episode for task 760b3cac "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"760b3cac-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
