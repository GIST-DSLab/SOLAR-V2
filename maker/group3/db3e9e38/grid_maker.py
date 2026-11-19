"""
ARC Task: db3e9e38 (RE-ARC) — LLM-generated grid_maker
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

ROTS = [0, 1, 2, 3]  # number of clockwise quarter-turns applied to the canonical (bar-from-top) grid


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": r} for r in ex] + [{"rot": random.choice(ex)}]
    return {"bgc": bgc, "fgc": fgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)

    def unifint(lb, ub):
        if ub < lb:
            ub = lb
        a = lb + (ub - lb) * diff_lb
        b = lb + (ub - lb) * diff_ub
        return int(round(random.uniform(a, b)))

    mh, mw = (max_w, max_h) if rot in (1, 3) else (max_h, max_w)
    h = unifint(3, max(3, mh))
    w = unifint(3, max(3, mw))
    barth = unifint(1, max(1, w // 5))
    loci = unifint(1, h - 2)
    locj = random.randint(1, w - barth - 1)
    gi = np.full((h, w), bgc, dtype=int)
    gi[0:loci + 1, locj:locj + barth] = fgc
    go = np.full((h, w), bgc, dtype=int)

    def fillshift(g, col, dr, dc):
        for r in range(0, loci + 1):
            for c in range(locj, locj + barth):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    g[rr, cc] = col

    for k in range(16):
        fillshift(go, fgc, -2 * k, 2 * k * barth)
        fillshift(go, fgc, -2 * k, -2 * k * barth)
        fillshift(go, 8, -(2 * k + 1), (2 * k + 1) * barth)
        fillshift(go, 8, -(2 * k + 1), -(2 * k + 1) * barth)
    gi = np.rot90(gi, -rot)
    go = np.rot90(go, -rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = int(I[0, 0])  # the bar never touches a grid corner
    mask = I != bgc
    if not mask.any():
        return [34], [[0, 0, hi - 1, wi - 1]]
    rs, cs = np.where(mask)
    r0, r1, c0, c1 = int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())
    fgc = int(I[r0, c0])
    # the bar touches exactly one grid edge
    if r0 == 0:
        edge = 'top'
    elif r1 == hi - 1:
        edge = 'bottom'
    elif c0 == 0:
        edge = 'left'
    else:
        edge = 'right'

    # Staircase stripes: the k-th neighbour stripe (on both sides) is one bar-thickness
    # further out and one cell shorter; odd k -> 8, even k -> bar colour.
    # Group the cells by LINE parallel to the touched edge (distance from that edge).
    line_cells = {}
    for k in range(1, max(hi, wi) + 1):
        for s in (1, -1):
            if edge in ('top', 'bottom'):
                bw = c1 - c0 + 1
                cc0, cc1 = c0 + s * k * bw, c1 + s * k * bw
                rr0, rr1 = (0, r1 - k) if edge == 'top' else (r0 + k, hi - 1)
            else:
                bh = r1 - r0 + 1
                rr0, rr1 = r0 + s * k * bh, r1 + s * k * bh
                cc0, cc1 = (0, c1 - k) if edge == 'left' else (c0 + k, wi - 1)
            for r in range(max(0, rr0), min(hi - 1, rr1) + 1):
                for c in range(max(0, cc0), min(wi - 1, cc1) + 1):
                    if edge == 'top':
                        ln = r
                    elif edge == 'bottom':
                        ln = hi - 1 - r
                    elif edge == 'left':
                        ln = c
                    else:
                        ln = wi - 1 - c
                    line_cells.setdefault(ln, {1: [], 0: []})[k % 2].append((r, c))

    ops, sels = [], []
    # Work one line at a time, starting at the touched edge and moving inward:
    # paint that line's 8-stripe cells, then its bar-coloured stripe cells.
    for ln in sorted(line_cells):
        for par, col in ((1, 8), (0, fgc)):
            cells = sorted(line_cells[ln][par])
            if cells:
                ops.append(col)
                sels.append(sel_of(cells))
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
                        f"num_examples+1 ({num_examples + 1}) for task db3e9e38"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task db3e9e38"
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
                                f"for task db3e9e38"
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
                    f"Failed to build a complete episode for task db3e9e38 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"db3e9e38-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
