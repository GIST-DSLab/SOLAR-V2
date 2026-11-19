"""
ARC Task: 5168d44c (RE-ARC) — LLM-generated grid_maker
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

DIRS = [(1, 0), (0, 1), (1, 1)]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    dotcol = random.choice([c for c in cols if c != bgc])
    # box colour 0 cannot be grabbed by Move (ARCLE treats 0 as empty)
    boxcol = random.choice([c for c in cols if c not in (bgc, dotcol, 0)])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(DIRS):
        ex = [{"direc": d} for d in DIRS]
        ex += [{"direc": random.choice(DIRS)} for _ in range(n_ex - len(DIRS))]
        random.shuffle(ex)
    else:
        ex = [{"direc": d} for d in random.sample(DIRS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "dotcol": dotcol, "boxcol": boxcol, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, dotcol, boxcol,
             direc=None, instance_plan=None) -> dict:
    if direc is None:
        direc = random.choice(DIRS)
    direc = tuple(direc)
    h = unifint(diff_lb, diff_ub, (7, max(7, max_h)))
    w = unifint(diff_lb, diff_ub, (7, max(7, max_w)))
    doth = unifint(diff_lb, diff_ub, (1, h // 3))
    dotw = unifint(diff_lb, diff_ub, (1, w // 3))
    borderh = unifint(diff_lb, diff_ub, (1, h // 4))
    borderw = unifint(diff_lb, diff_ub, (1, w // 4))
    dotloci = randint(0, h - doth - 1 if direc == RIGHT else h - doth - borderh - 1)
    dotlocj = randint(0, w - dotw - 1 if direc == DOWN else w - dotw - borderw - 1)
    dotloc = (dotloci, dotlocj)
    gi = canvas(bgc, (h, w))
    dotshap = (doth, dotw)
    starterdot = backdrop(frozenset({dotloc, add(dotloc, decrement(dotshap))}))
    bordershap = (borderh, borderw)
    offset = add(multiply(direc, dotshap), multiply(direc, bordershap))
    itv = interval(-15, 16, 1)
    itv = apply(lbind(multiply, offset), itv)
    dots = mapply(lbind(shift, starterdot), itv)
    gi = fill(gi, dotcol, dots)
    protobx = backdrop(frozenset({
        (dotloci - borderh, dotlocj - borderw),
        (dotloci + doth + borderh - 1, dotlocj + dotw + borderw - 1),
    }))
    bx = protobx - starterdot
    bxshifted = shift(bx, offset)
    go = fill(gi, boxcol, bxshifted)
    gi = fill(gi, boxcol, bx)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    def comps(color):
        seen = set()
        out = []
        for r in range(hi):
            for c in range(wi):
                if I[r, c] != color or (r, c) in seen:
                    continue
                stack = [(r, c)]
                seen.add((r, c))
                cc = []
                while stack:
                    a, b = stack.pop()
                    cc.append((a, b))
                    for na, nb in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                        if 0 <= na < hi and 0 <= nb < wi and (na, nb) not in seen and I[na, nb] == color:
                            seen.add((na, nb))
                            stack.append((na, nb))
                out.append(cc)
        return out

    colors = sorted(set(I.flatten().tolist()))
    # background: the canvas colour -- spans the full grid, most cells
    def spans(col):
        rr, cc = np.where(I == col)
        return rr.min() == 0 and cc.min() == 0 and rr.max() == hi - 1 and cc.max() == wi - 1
    cand = [c for c in colors if spans(c)] or colors
    bgc = max(cand, key=lambda c: int((I == c).sum()))
    others = [c for c in colors if c != bgc]
    # box = colour forming fewest components (a single frame); dots repeat
    boxcol = min(others, key=lambda c: len(comps(c)))
    dotcol = [c for c in others if c != boxcol][0]
    ring = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == boxcol]
    r0 = min(r for r, _ in ring); r1 = max(r for r, _ in ring)
    c0 = min(c for _, c in ring); c1 = max(c for _, c in ring)
    dots = comps(dotcol)
    inner = [d for d in dots if all(r0 <= r <= r1 and c0 <= c <= c1 for r, c in d)][0]
    dr0 = min(r for r, _ in inner); dr1 = max(r for r, _ in inner)
    dc0 = min(c for _, c in inner); dc1 = max(c for _, c in inner)
    doth, dotw = dr1 - dr0 + 1, dc1 - dc0 + 1
    borderh, borderw = r1 - dr1, c1 - dc1   # far-side borders along a moving axis are never clipped
    # direction from the dot chain
    same_col = all(min(c for _, c in d) == dc0 for d in dots) and all(max(c for _, c in d) == dc1 for d in dots)
    same_row = all(min(r for r, _ in d) == dr0 for d in dots) and all(max(r for r, _ in d) == dr1 for d in dots)
    vr = 0 if same_row else 1
    vc = 0 if same_col else 1
    dr = vr * (doth + borderh)
    dc = vc * (dotw + borderw)

    # frame extent: along a moving axis the far border is always fully visible,
    # so the frame there is dot +/- border; along a still axis the visible extent is it.
    fr0, fr1 = (dr0 - borderh, dr1 + borderh) if vr else (r0, r1)
    fc0, fc1 = (dc0 - borderw, dc1 + borderw) if vc else (c0, c1)
    full_shift = set()
    for r in range(fr0, fr1 + 1):
        for c in range(fc0, fc1 + 1):
            if dr0 <= r <= dr1 and dc0 <= c <= dc1:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < hi and 0 <= cc < wi:
                full_shift.add((rr, cc))

    ops, sels = [], []
    src = list(ring)
    if boxcol != 0:
        cur = list(src)
        first = True
        steps = [(21, 1, 0)] * dr + [(22, 0, 1)] * dc
        for op, a, b in steps:
            if not cur:
                break
            ops.append(op)
            sels.append(sel_of(cur) if first else sel_of([]))   # grab once, then continue
            first = False
            cur = [(r + a, c + b) for r, c in cur if 0 <= r + a < hi and 0 <= c + b < wi]
        curset = set(cur)
        hole = sorted(set(src) - curset)
        if bgc != 0 and hole:
            ops.append(bgc); sels.append(sel_of(hole))
        # frame parts that were clipped off the grid in I re-enter at the new spot
        missing = sorted(full_shift - curset)
        if missing:
            ops.append(boxcol); sels.append(sel_of(missing))
    else:
        # colour-0 frame cannot be grabbed by Move: erase old frame, draw new one
        ops.append(bgc); sels.append(sel_of(src))
        ops.append(0); sels.append(sel_of(sorted(full_shift)))
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
                        f"num_examples+1 ({num_examples + 1}) for task 5168d44c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5168d44c"
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
                                f"for task 5168d44c"
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
                    f"Failed to build a complete episode for task 5168d44c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5168d44c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
