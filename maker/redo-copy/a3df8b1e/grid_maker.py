"""
ARC Task: a3df8b1e (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover - fallback if helper unavailable
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for (r, c) in cells]}


# ---------------------------------------------------------------- variants ---
# Structural dimensions of this task:
#   numlins  : how many corners emit a bouncing ray (1..4)
#   landscape: whether the grid is wide (bounce off top/bottom) or tall
VARIANTS = [{"numlins": n, "landscape": b} for n in (1, 2, 3, 4) for b in (False, True)]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    linc = random.choice([c for c in cols if c != bgc])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]  # test variant was shown
    return {"bgc": bgc, "linc": linc, "instance_plan": plan}


# --------------------------------------------------------------- generator ---
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    if hi < lo:
        hi = lo
    return random.randint(lo, hi)


def _hm(g):
    return [list(r) for r in g[::-1]]


def _vm(g):
    return [list(r)[::-1] for r in g]


def _tr(g):
    return [list(col) for col in zip(*g)]


def _overlay(dst, src, c):
    for i, row in enumerate(src):
        for j, v in enumerate(row):
            if v == c:
                dst[i][j] = c


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, numlins=None, landscape=None) -> dict:
    if numlins is None:
        numlins = random.choice((1, 2, 3, 4))
    if landscape is None:
        landscape = random.choice((False, True))

    lim = min(int(max_h), int(max_w))
    if lim < 3:
        lim = 3
    wub = min(10, lim - 1)
    if wub < 2:
        wub = 2
    w = _unifint(diff_lb, diff_ub, (2, wub))
    hub = min(30, lim)
    if hub < w + 1:
        hub = w + 1
    h = _unifint(diff_lb, diff_ub, (w + 1, hub))

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]

    # source mark at bottom-left corner; ray shoots up-right, bouncing off side walls
    gi[h - 1][0] = linc
    P = 2 * (w - 1)
    for r in range(h):
        t = (h - 1 - r) % P
        c = t if t < w else P - t
        go[r][c] = linc

    # random dihedral placement of the source corner
    if random.choice((False, True)):
        gi, go = _hm(gi), _hm(go)
    if random.choice((False, True)):
        gi, go = _vm(gi), _vm(go)

    gix = [list(r) for r in gi]
    gox = [list(r) for r in go]
    if numlins > 1:
        _overlay(gi, _hm(gix), linc)
        _overlay(go, _hm(gox), linc)
    if numlins > 2:
        _overlay(gi, _vm(gix), linc)
        _overlay(go, _vm(gox), linc)
    if numlins > 3:
        _overlay(gi, _hm(_vm(gix)), linc)
        _overlay(go, _hm(_vm(gox)), linc)

    if landscape:
        gi, go = _tr(gi), _tr(go)

    return {
        "input": tuple(tuple(r) for r in gi),
        "output": tuple(tuple(r) for r in go),
    }


# ---------------------------------------------------------------- derivation ---
def derive_operations(I, O):
    """Every parameter below is measured from I only.

    Rule (read off the generator): the grid is a billiard table whose short sides are
    the reflecting walls.  Each CORNER of I that carries the line colour launches a
    diagonal ray that bounces between the two short sides forever.  One Color op per
    ray = one visible object drawn.
    """
    import numpy as np

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    submit_sel = [0, 0, hi - 1, wi - 1]

    corner_pos = {(0, 0), (0, wi - 1), (hi - 1, 0), (hi - 1, wi - 1)}
    noncorner = [int(I[r, c]) for r in range(hi) for c in range(wi) if (r, c) not in corner_pos]
    if noncorner:
        bgc = Counter(noncorner).most_common(1)[0][0]
    else:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    others = [c for c in sorted(set(int(v) for v in I.flatten().tolist())) if c != bgc]
    if not others:
        ops.append(34)
        sels.append(submit_sel)
        return ops, sels
    fg = others[0]

    # normalise to portrait (tall) orientation: walls are the left/right columns there
    portrait = hi > wi
    if portrait:
        H, W = hi, wi

        def to_orig(r, c):
            return (r, c)
    else:
        H, W = wi, hi

        def to_orig(r, c):
            return (c, r)

    if W < 2:
        ops.append(34)
        sels.append(submit_sel)
        return ops, sels

    P = 2 * (W - 1)

    def fold(t):
        t %= P
        return t if t < W else P - t

    # build one ray per lit corner (portrait coordinates)
    rays = []
    for (cr, cc) in ((0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)):
        orr, occ = to_orig(cr, cc)
        if int(I[orr, occ]) != fg:
            continue
        cells = []
        for r in range(H):
            t = r if cr == 0 else (H - 1 - r)
            c = fold(t) if cc == 0 else (W - 1 - fold(t))
            cells.append(to_orig(r, c))
        rays.append(cells)

    painted = {(r, c) for r in range(hi) for c in range(wi) if int(I[r, c]) == fg}
    for cells in rays:
        if all(p in painted for p in cells):
            continue  # this ray coincides with one already drawn -> would be a no-op
        ops.append(int(fg))
        sels.append(sel_of(cells))
        painted.update(cells)

    ops.append(34)
    sels.append(submit_sel)
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
                        f"num_examples+1 ({num_examples + 1}) for task a3df8b1e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a3df8b1e"
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
                                f"for task a3df8b1e"
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
                    f"Failed to build a complete episode for task a3df8b1e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a3df8b1e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
