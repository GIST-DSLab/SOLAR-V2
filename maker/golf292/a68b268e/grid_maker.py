"""
ARC Task: a68b268e (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    bgc, linc, c1, c2, c3, c4 = random.sample(range(10), 6)
    return {"bgc": bgc, "linc": linc, "c1": c1, "c2": c2, "c3": c3, "c4": c4}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, c1, c2, c3, c4) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round(lb * (b - a))
        hi = a + round(ub * (b - a))
        return random.randint(min(lo, hi), max(lo, hi))
    hmax = max(2, min(14, (max_h - 1) // 2))
    wmax = max(2, min(4, (max_w - 1) // 2))
    h = unifint(diff_lb, diff_ub, (2, hmax))
    w = unifint(diff_lb, diff_ub, (2, wmax))
    inds = [(i, j) for i in range(h) for j in range(w)]

    def ncount():
        d = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
        n = random.choice((d, h * w - d))
        return min(max(1, n), h * w - 1)
    cells = [random.sample(inds, ncount()) for _ in range(4)]
    go = [[bgc] * w for _ in range(h)]
    for col, cs in zip((c1, c2, c3, c4), cells):
        for i, j in cs:
            go[i][j] = col

    def quad(col, cs):
        g = [[bgc] * w for _ in range(h)]
        for i, j in cs:
            g[i][j] = col
        return g
    LR, LL, UR, UL = (quad(c, cs) for c, cs in zip((c1, c2, c3, c4), cells))
    gi = [[linc] * (2 * w + 1) for _ in range(2 * h + 1)]
    for q, (dr, dc) in ((LR, (h + 1, w + 1)), (LL, (h + 1, 0)), (UR, (0, w + 1)), (UL, (0, 0))):
        for i in range(h):
            for j in range(w):
                gi[i + dr][j + dc] = q[i][j]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    h, w = hi // 2, wi // 2
    origins = {"UL": (0, 0), "UR": (0, w + 1), "LL": (h + 1, 0), "LR": (h + 1, w + 1)}
    pals = [set(I[r:r + h, c:c + w].flatten().tolist()) for r, c in origins.values()]
    common = pals[0] & pals[1] & pals[2] & pals[3]
    bgc = min(common)                      # the one colour every quadrant shares
    quad_cols = set().union(*pals) - {bgc}

    def cells_of(g, col):
        return [(r, c) for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r, c] == col]

    def rect(r, c):
        return sel_of([(r + i, c + j) for i in range(h) for j in range(w)])

    # plan steps are evaluated against the grid as it stands
    plan = []
    spare = None
    if bgc != 0:
        if 0 in quad_cols:
            spare = min(set(range(10)) - set(I.flatten().tolist()))
            plan.append(("color", (0, spare)))      # park real colour-0 cells
        plan.append(("color", (bgc, 0)))            # all background cells -> transparent
    lr = origins["LR"]
    for name in ("LL", "UR", "UL"):                 # stack layers in rising priority onto LR
        plan.append(("copy", origins[name]))
        plan.append(("paste", lr))
    plan.append(("crop", lr))
    if bgc != 0:
        plan.append(("color", (0, bgc)))            # remaining holes -> background
        if spare is not None:
            plan.append(("color", (spare, 0)))

    def run(plan):
        g = I.copy()
        clip = None
        ops, sels = [], []
        for kind, p in plan:
            if kind == "color":
                src, dst = p
                cs = cells_of(g, src)
                for r, c in cs:
                    g[r, c] = dst
                ops.append(dst); sels.append(sel_of(cs))
            elif kind == "copy":
                r, c = p
                clip = g[r:r + h, c:c + w].copy()
                ops.append(29); sels.append(rect(r, c))   # full quadrant rectangle
            elif kind == "paste":
                r, c = p
                if clip is not None:
                    sub = g[r:r + h, c:c + w]
                    sub[clip != 0] = clip[clip != 0]
                ops.append(30); sels.append(sel_of([(r, c)]))
            elif kind == "crop":
                r, c = p
                g = g[r:r + h, c:c + w].copy()
                ops.append(33); sels.append(rect(r, c))   # full quadrant rectangle
        return g, ops, sels

    target, ops, sels = run(plan)

    def replay(ops, sels):
        # replay fixed selections exactly as the environment would
        g = I.copy()
        clip = None
        for op, sel in zip(ops, sels):
            cs = [tuple(x) for x in sel["cells"]]
            if op <= 9:
                for r, c in cs:
                    g[r, c] = op
            elif op == 29:
                r0, c0 = min(r for r, _ in cs), min(c for _, c in cs)
                clip = g[r0:r0 + h, c0:c0 + w].copy()
            elif op == 30:
                if clip is not None:
                    r0, c0 = cs[0]
                    sub = g[r0:r0 + h, c0:c0 + w]
                    sub[clip != 0] = clip[clip != 0]
            elif op == 33:
                r0, c0 = min(r for r, _ in cs), min(c for _, c in cs)
                g = g[r0:r0 + h, c0:c0 + w].copy()
        return g

    # drop any op whose removal leaves the (input-derived) result unchanged
    changed = True
    while changed:
        changed = False
        for i in range(len(ops)):
            if ops[i] == 33:
                continue
            g = replay(ops[:i] + ops[i + 1:], sels[:i] + sels[i + 1:])
            if g.shape == target.shape and (g == target).all():
                ops, sels = ops[:i] + ops[i + 1:], sels[:i] + sels[i + 1:]
                changed = True
                break
    ops.append(34); sels.append(sel_of([(r, c) for r in range(h) for c in range(w)]))
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
                        f"num_examples+1 ({num_examples + 1}) for task a68b268e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a68b268e"
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
                                f"for task a68b268e"
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
                    f"Failed to build a complete episode for task a68b268e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a68b268e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
