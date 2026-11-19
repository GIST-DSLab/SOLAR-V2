"""
ARC Task: 3ac3eb23 (RE-ARC) — LLM-generated grid_maker
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


# ----------------------------------------------------------------------------- #
# episode-level colour / orientation plan
# ----------------------------------------------------------------------------- #
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        a, b = b, a
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    lo = max(a, min(b, lo))
    hi = max(a, min(b, hi))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    # the palette the seeds are drawn from is fixed for the whole episode;
    # it may legitimately contain colour 0 when bgc != 0.
    fgcols = random.sample(rem, random.randint(2, 4))

    # discrete structural variant: which border the seeds sit on (the generator's
    # identity / rot90 / rot180 / rot270).  Cover every orientation when possible.
    variants = [{"orient": k} for k in range(4)]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(variants):
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(variants, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "fgcols": fgcols, "instance_plan": plan}


# ----------------------------------------------------------------------------- #
# generator
# ----------------------------------------------------------------------------- #
def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgcols=None, orient=None) -> dict:
    if orient is None:
        orient = random.randint(0, 3)
    if fgcols is None:
        fgcols = [c for c in range(10) if c != bgc]
    fgcols = [c for c in fgcols if c != bgc]
    if not fgcols:
        fgcols = [c for c in range(10) if c != bgc]

    # after an odd number of 90 degree turns h and w swap -> respect the caps
    lim_h = max_w if orient % 2 else max_h
    lim_w = max_h if orient % 2 else max_w
    h = _unifint(diff_lb, diff_ub, (3, max(3, min(30, lim_h))))
    w = _unifint(diff_lb, diff_ub, (3, max(3, min(30, lim_w))))

    nlocs = _unifint(diff_lb, diff_ub, (1, max(1, (w - 2) // 3)))
    locopts = list(range(1, w - 1))

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    for _ in range(nlocs):
        if not locopts:
            break
        locj = random.choice(locopts)
        locopts = [x for x in locopts if not (locj - 2 <= x <= locj + 2)]
        col = random.choice(fgcols)
        gi[0][locj] = col
        for p in range(0, h, 2):
            go[p][locj] = col
        for p in range(1, h, 2):
            go[p][locj - 1] = col
            go[p][locj + 1] = col

    for _ in range(orient % 4):                       # rot90 (CW), as in the original
        gi = [list(r) for r in zip(*gi[::-1])]
        go = [list(r) for r in zip(*go[::-1])]

    return {"input": gi, "output": go}


# ----------------------------------------------------------------------------- #
# trajectory
# ----------------------------------------------------------------------------- #
def derive_operations(I, O, examples=None):
    """
    Rule, read off I alone: every non-background cell of I is a seed sitting on ONE
    border.  Each seed throws a fan straight inward: its own lane at even depth, the
    two neighbouring lanes at odd depth (period 2 in depth).

    The rule is always the same; the ROUTE taken to draw it is chosen from what the
    input shows -- how many seeds, whether a colour repeats, how deep the strip is,
    which end of the border the outermost seed is nearest.  O is never consulted.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # ---- background: the canvas colour, agreed across every demonstration ----
    cnt = {}
    grids = [I]
    if examples:
        for pair in examples:
            try:
                grids.append(np.asarray(pair[0], dtype=int))
            except Exception:
                pass
    for g in grids:
        for val in g.ravel().tolist():
            cnt[val] = cnt.get(val, 0) + 1
    bgc = max(cnt.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    seeds = [(r, c) for r in range(hi) for c in range(wi) if int(I[r, c]) != bgc]
    if not seeds:
        return [34], [[0, 0, hi - 1, wi - 1]]

    # ---- which border carries the seeds -> u = inward, v = along the border ----
    if all(r == 0 for r, _ in seeds):
        u, v, D = (1, 0), (0, 1), hi
    elif all(r == hi - 1 for r, _ in seeds):
        u, v, D = (-1, 0), (0, 1), hi
    elif all(c == 0 for _, c in seeds):
        u, v, D = (0, 1), (1, 0), wi
    else:
        u, v, D = (0, -1), (1, 0), wi

    horizontal_border = (v == (0, 1))
    L = wi if horizontal_border else hi

    def along(s):
        return s[1] if horizontal_border else s[0]

    def cell(s, d, t):
        return (s[0] + u[0] * d + v[0] * t, s[1] + u[1] * d + v[1] * t)

    def inside(p):
        return 0 <= p[0] < hi and 0 <= p[1] < wi

    # ---- sweep direction: begin at the end of the border a seed is nearest to ----
    ordered = sorted(seeds, key=along)
    if along(ordered[0]) > (L - 1 - along(ordered[-1])):
        ordered.reverse()
    cols_list = [int(I[r, c]) for (r, c) in ordered]

    def centres(s, start=2):
        return [cell(s, d, 0) for d in range(start, D, 2) if inside(cell(s, d, 0))]

    def wings(s):
        out = []
        for d in range(1, D, 2):
            for t in (-1, 1):
                p = cell(s, d, t)
                if inside(p):
                    out.append(p)
        return out

    # ---- pick the route from the input's own shape ----
    has_dup = len(set(cols_list)) < len(cols_list)
    n = len(ordered)
    stamp_ok = (all(c != 0 for c in cols_list) and D >= 6
                and n * (D // 2 + 2) <= 60 and (D + n) % 2 == 0)
    fan_ok = (n * D <= 60) and ((sum(cols_list) + n) % 2 == 0)

    if has_dup:
        method = "bycolor"          # a repeated colour -> do all of it in one stroke
    elif stamp_ok:
        method = "stamp"            # long strip, printable unit -> copy it down
    elif fan_ok:
        method = "fan"              # short strip -> grow it outward depth by depth
    else:
        method = "lane"             # otherwise -> lay the three lanes down

    wings_first = ((cols_list[0] + n) % 2 == 0)

    ops, sels = [], []

    if method == "bycolor":
        groups, seen = [], {}
        for s, col in zip(ordered, cols_list):
            if col not in seen:
                seen[col] = len(groups)
                groups.append((col, []))
            groups[seen[col]][1].append(s)
        for col, group in groups:
            cs, ws = [], []
            for s in group:
                cs.extend(centres(s))
                ws.extend(wings(s))
            block = [(ws, col), (cs, col)] if wings_first else [(cs, col), (ws, col)]
            for cells, c in block:
                if cells:
                    ops.append(c)               # Color<col>: 0 is an ordinary colour here
                    sels.append(sel_of(cells))

    elif method == "lane":
        for s, col in zip(ordered, cols_list):
            cs, ws = centres(s), wings(s)
            block = [ws, cs] if wings_first else [cs, ws]
            for cells in block:
                if cells:
                    ops.append(col)
                    sels.append(sel_of(cells))

    elif method == "fan":
        for s, col in zip(ordered, cols_list):
            for d in range(1, D):
                cells = [cell(s, d, t) for t in ((0,) if d % 2 == 0 else (-1, 1))]
                cells = [p for p in cells if inside(p)]
                if cells:
                    ops.append(col)
                    sels.append(sel_of(cells))

    else:  # "stamp": finish one 2-deep unit, then print it down the strip
        for s, col in zip(ordered, cols_list):
            unit_wings = [cell(s, 1, t) for t in (-1, 1) if inside(cell(s, 1, t))]
            if unit_wings:
                ops.append(col)
                sels.append(sel_of(unit_wings))
            unit = [cell(s, 0, 0)] + unit_wings
            r0 = min(p[0] for p in unit)
            c0 = min(p[1] for p in unit)
            r1 = max(p[0] for p in unit)
            c1 = max(p[1] for p in unit)
            P = D // 2                              # complete 2-deep units in the strip
            if P > 1:
                ops.append(29)                      # CopyO: the unit as it now stands
                sels.append([r0, c0, r1 - r0, c1 - c0])
                for k in range(1, P):
                    ops.append(30)
                    sels.append([r0 + u[0] * 2 * k, c0 + u[1] * 2 * k, 0, 0])
            if D % 2 == 1:                          # lone even depth at the far border
                p = cell(s, D - 1, 0)
                if inside(p):
                    ops.append(col)
                    sels.append(sel_of([p]))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 3ac3eb23"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3ac3eb23"
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
                                f"for task 3ac3eb23"
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
                    f"Failed to build a complete episode for task 3ac3eb23 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3ac3eb23-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
