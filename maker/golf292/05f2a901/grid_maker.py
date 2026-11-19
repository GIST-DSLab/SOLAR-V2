"""
ARC Task: 05f2a901 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return min(b, max(a, int(round(a + (b - a) * d))))


def sample_colors(num_examples=None) -> dict:
    bgc, fgc, destc = random.sample(range(10), 3)
    return {"bgc": bgc, "fgc": fgc, "destc": destc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, fgc=1, destc=2, **kw) -> dict:
    H = max(8, min(30, max_h)); W = max(8, min(30, max_w))
    h = _unifint(diff_lb, diff_ub, (8, H))
    w = _unifint(diff_lb, diff_ub, (8, W))
    objh = _unifint(diff_lb, diff_ub, (2, min(w // 2, h // 2)))
    objw = _unifint(diff_lb, diff_ub, (objh, w // 2))
    bb = {(i, j) for i in range(objh) for j in range(objw)}
    sp = random.choice(sorted(bb))
    obj = {sp}
    bb.discard(sp)
    ncells = _unifint(diff_lb, diff_ub, (objh + objw, objh * objw))
    for _ in range(ncells - 1):
        cand = sorted({(r + dr, c + dc) for r, c in obj
                       for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))} & (bb - obj))
        obj.add(random.choice(cand))
    rs = [r for r, _ in obj]; cs = [c for _, c in obj]
    if (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1) == len(obj):
        obj.discard(random.choice(sorted(obj)))
    r0 = min(r for r, _ in obj); c0 = min(c for _, c in obj)
    obj = {(r - r0, c - c0) for r, c in obj}
    objh = max(r for r, _ in obj) + 1
    objw = max(c for _, c in obj) + 1
    loci = _unifint(diff_lb, diff_ub, (3, h - objh))
    locj = _unifint(diff_lb, diff_ub, (0, w - objw))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    obj = {(r + loci, c + locj) for r, c in obj}
    for r, c in obj:
        gi[r, c] = fgc
    sqd = random.randint(1, min(w, loci - 1))
    locisq = random.randint(0, loci - sqd - 1)
    # keep the square fully on the grid (still sharing a column with the shape)
    locjsq = random.randint(max(0, locj - sqd + 1), min(w - sqd, locj + objw - 1))
    sq = {(r, c) for r in range(locisq, locisq + sqd) for c in range(locjsq, locjsq + sqd)}
    for r, c in sq:
        gi[r, c] = destc
        go[r, c] = destc
    while not (obj & sq):
        obj = {(r - 1, c) for r, c in obj}
    obj = {(r + 1, c) for r, c in obj}
    if any(r < 0 for r, _ in obj):
        # the shape would slide off the grid before meeting the square: resample
        return generate(diff_lb, diff_ub, max_h, max_w, bgc=bgc, fgc=fgc, destc=destc, **kw)
    for r, c in obj:
        go[r, c] = fgc
    mfs = [lambda g: g, lambda g: g.T, lambda g: np.rot90(g, 2).T, np.fliplr, np.flipud,
           lambda g: np.rot90(g, 3), lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 1)]
    for fn in random.sample(mfs, random.choice((1, 2))):
        gi = fn(gi); go = fn(go)
    return {"input": np.ascontiguousarray(gi).tolist(), "output": np.ascontiguousarray(go).tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    full = [0, 0, hi - 1, wi - 1]

    # Per-colour cell sets: background, the solid square (anchor) and the
    # ragged shape (traveller).
    cells = {}
    for r in range(hi):
        for c in range(wi):
            cells.setdefault(int(I[r, c]), set()).add((r, c))

    def bbox(s):
        rs = [r for r, _ in s]; cs = [c for _, c in s]
        return min(rs), max(rs), min(cs), max(cs)

    def solid(s):
        r0, r1, c0, c1 = bbox(s)
        return len(s) == (r1 - r0 + 1) * (c1 - c0 + 1) and len(s) < hi * wi

    # background: the non-solid colour shared by every demonstration input,
    # else the most frequent non-solid colour
    cand = [k for k in cells if not solid(cells[k])]
    bgc = None
    if examples:
        common = set(cells)
        for ex in examples:
            common &= set(np.asarray(ex[0], dtype=int).flatten().tolist())
        cc = [k for k in cand if k in common]
        if len(cc) == 1:
            bgc = cc[0]
    if bgc is None:
        bgc = max(cand, key=lambda k: len(cells[k])) if cand else int(I[0, 0])

    fg = [k for k in cells if k != bgc]
    sq_cols = [k for k in fg if solid(cells[k])]
    ob_cols = [k for k in fg if not solid(cells[k])]
    if not sq_cols or not ob_cols:
        ops.append(34); sels.append(full)
        return ops, sels
    rect = cells[sq_cols[0]]
    obj_col = ob_cols[0]
    obj = cells[obj_col]

    orr0, orr1, occ0, occ1 = bbox(obj)
    rr0, rr1, rc0, rc1 = bbox(rect)
    if orr0 > rr1:
        dr, dc = -1, 0
    elif orr1 < rr0:
        dr, dc = 1, 0
    elif occ0 > rc1:
        dr, dc = 0, -1
    elif occ1 < rc0:
        dr, dc = 0, 1
    else:
        ops.append(34); sels.append(full)
        return ops, sels

    # gravitate (in Python): slide until the next step would touch the square
    cur = set(obj)
    steps = 0
    while True:
        nxt = {(r + dr, c + dc) for r, c in cur}
        if nxt & rect or not all(0 <= r < hi and 0 <= c < wi for r, c in nxt):
            break
        cur = nxt
        steps += 1

    if steps > 0:
        if obj_col != 0:
            # one copy of the shape: lift it off its old place, set it down at the new one
            ops.append(28); sels.append(sel_of(sorted(obj)))        # CopyI: the shape's own cells
            ops.append(int(bgc)); sels.append(sel_of(sorted(obj)))  # erase the shape where it was
            d0r, _, d0c, _ = bbox(cur)
            ops.append(30); sels.append([d0r, d0c, 0, 0])           # paste at its landing corner
        else:
            # colour-0 shapes cannot be copied (0 = transparent): repaint
            ops.append(int(bgc)); sels.append(sel_of(sorted(obj)))
            ops.append(0); sels.append(sel_of(sorted(cur)))

    ops.append(34); sels.append(full)  # full-grid rectangle: submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 05f2a901"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 05f2a901"
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
                                f"for task 05f2a901"
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
                    f"Failed to build a complete episode for task 05f2a901 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"05f2a901-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
