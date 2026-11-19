"""
ARC Task: e179c5f4 (RE-ARC) — LLM-generated grid_maker
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
    return min(max(a, round(a + (b - a) * d)), b)


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc) -> dict:
    lim = min(max_h, max_w, 30)
    w = _unifint(diff_lb, diff_ub, (2, min(10, lim - 1)))
    h = _unifint(diff_lb, diff_ub, (w + 1, lim))
    gi = np.full((h, w), bgc, dtype=int)
    sp = (h - 1, 0)
    gi[sp] = linc
    go = gi.copy()
    direc = 1
    while True:
        sp = (sp[0] - 1, sp[1] + direc)
        if sp[1] == w - 1 or sp[1] == 0:
            direc *= -1
        if sp[0] < 0:
            break
        go[sp] = linc
    mfs = [
        lambda g: g,
        lambda g: g.T,
        lambda g: np.rot90(g, 2).T,
        lambda g: g[:, ::-1],
        lambda g: g[::-1, :],
        lambda g: np.rot90(g, -1),
        lambda g: np.rot90(g, 2),
        lambda g: np.rot90(g, 1),
    ]
    nmfs = random.choice((1, 2))
    for fn in random.sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    gi = np.array(gi)
    go = np.array(go)
    gix, gox = gi.copy(), go.copy()
    numlins = _unifint(diff_lb, diff_ub, (1, 4))
    variants = [lambda g: g[::-1, :], lambda g: g[:, ::-1], lambda g: g[::-1, ::-1]]
    for k in range(numlins - 1):
        f = variants[k]
        gi[f(gix) == linc] = linc
        go[f(gox) == linc] = linc
    go[go == bgc] = 8
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    corners = [(0, 0), (0, wi - 1), (hi - 1, 0), (hi - 1, wi - 1)]
    cset = set(corners)
    # background: the colour of every non-corner cell (the generator's canvas)
    inner = [int(I[r, c]) for r in range(hi) for c in range(wi) if (r, c) not in cset]
    bgc = max(set(inner), key=inner.count)
    marks = [p for p in corners if I[p] != bgc]
    grid = I.copy()
    ops, sels = [], []

    # 1. the whole background region (one connected region) turns 8
    seed = (1, 0) if hi >= 3 else (0, 1)
    ops.append(18); sels.append([seed[0], seed[1], 0, 0])
    grid[grid == bgc] = 8

    # 2. from every marked corner a line bounces wall-to-wall along the long axis;
    #    draw it one diagonal run at a time, skipping cells that already hold the line colour
    portrait = hi > wi
    H, W = (hi, wi) if portrait else (wi, hi)
    for (r0, c0) in marks:
        linc = int(I[r0, c0])
        pr, pc = (r0, c0) if portrait else (c0, r0)
        dr = -1 if pr == H - 1 else 1
        dc = 1 if pc == 0 else -1
        path = [(pr, pc)]
        r, c = pr, pc
        for _ in range(H - 1):
            r += dr
            c += dc
            path.append((r, c))
            if c == 0 or c == W - 1:
                dc = -dc
        segs, cur = [], [path[0]]
        for p in path[1:]:
            cur.append(p)
            if p[1] == 0 or p[1] == W - 1:
                segs.append(cur)
                cur = [p]
        if len(cur) > 1:
            segs.append(cur)
        for seg in segs:
            cells = [(a, b) if portrait else (b, a) for a, b in seg]
            todo = [p for p in cells if grid[p] != linc]
            if not todo:
                continue  # this run is already drawn (coincides with an earlier line)
            ops.append(linc); sels.append(sel_of(todo))
            for p in todo:
                grid[p] = linc

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
                        f"num_examples+1 ({num_examples + 1}) for task e179c5f4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e179c5f4"
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
                                f"for task e179c5f4"
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
                    f"Failed to build a complete episode for task e179c5f4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e179c5f4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
