"""
ARC Task: b548a754 (RE-ARC) — LLM-generated grid_maker
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
    lo = a + int(round(diff_lb * (b - a)))
    hi = a + int(round(diff_ub * (b - a)))
    return random.randint(min(lo, hi), max(lo, hi))


def sample_colors(num_examples=None) -> dict:
    bgc, boxc, inc, dotc = random.sample(range(10), 4)
    return {"bgc": bgc, "boxc": boxc, "inc": inc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0, boxc=1, inc=2, dotc=3) -> dict:
    h = _unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = _unifint(diff_lb, diff_ub, (4, max(4, max_w)))
    hi = _unifint(diff_lb, diff_ub, (4, h - 1))
    wi = _unifint(diff_lb, diff_ub, (3, w - 1))
    loci = random.randint(0, h - hi)
    locj = random.randint(0, w - wi)
    go = np.full((h, w), bgc, dtype=int)
    go[loci:loci + hi, locj:locj + wi] = boxc
    go[loci + 1:loci + hi - 1, locj + 1:locj + wi - 1] = inc
    cutoff = random.randint(loci + 2, loci + hi - 2)
    gi = np.full((h, w), bgc, dtype=int)
    gi[loci:cutoff + 1, locj:locj + wi] = boxc
    gi[loci + 1:cutoff, locj + 1:locj + wi - 1] = inc
    lc = random.randint(locj, locj + wi - 1)
    gi[loci + hi - 1, lc] = dotc
    k = random.choice((0, 1, 2, 3))
    gi = np.rot90(gi, -k)
    go = np.rot90(go, -k)
    return {"input": [list(map(int, r)) for r in gi],
            "output": [list(map(int, r)) for r in go]}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    # find the box: a colour whose cells are exactly the outline of their bbox (>=3x3),
    # with a single-colour interior
    box = None
    for col in np.unique(I):
        rs, cs = np.where(I == col)
        r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        sub = I[r0:r1 + 1, c0:c1 + 1]
        outline = np.ones_like(sub, dtype=bool)
        outline[1:-1, 1:-1] = False
        if not np.all(sub[outline] == col):
            continue
        if len(rs) != outline.sum():
            continue
        inner = sub[1:-1, 1:-1]
        if len(np.unique(inner)) != 1 or inner[0, 0] == col:
            continue
        box = (int(col), int(inner[0, 0]), int(r0), int(r1), int(c0), int(c1))
        break
    boxc, inc, r0, r1, c0, c1 = box
    # background = colour of cells outside the box; the dot = the lone odd cell out there
    outside = np.ones((H, W), dtype=bool)
    outside[r0:r1 + 1, c0:c1 + 1] = False
    vals, cnts = np.unique(I[outside], return_counts=True)
    bgc = int(vals[np.argmax(cnts)])
    dr_, dc_ = [(int(r), int(c)) for r, c in zip(*np.where(outside & (I != bgc)))][0]

    # choose the box edge facing the dot and the direction it slides
    if dr_ > r1:
        edge = [(r1, c) for c in range(c0, c1 + 1)]; vec = (1, 0); op = 21; d = dr_ - r1
    elif dr_ < r0:
        edge = [(r0, c) for c in range(c0, c1 + 1)]; vec = (-1, 0); op = 20; d = r0 - dr_
    elif dc_ > c1:
        edge = [(r, c1) for r in range(r0, r1 + 1)]; vec = (0, 1); op = 22; d = dc_ - c1
    else:
        edge = [(r, c0) for r in range(r0, r1 + 1)]; vec = (0, -1); op = 23; d = c0 - dc_

    g = I.copy()
    ops, sels = [], []
    # slide the far edge of the box out to the dot (grab once, then empty selections)
    snap = g.copy()
    for r, c in edge:
        snap[r, c] = 0
    cur = list(edge)
    # ARCLE cannot carry colour-0 cells in a Move, so a colour-0 edge is drawn instead
    n_moves = d if boxc != 0 else 0
    for k in range(n_moves):
        ops.append(op)
        sels.append(sel_of(cur) if k == 0 else sel_of([]))
        cur = [(r + vec[0], c + vec[1]) for r, c in cur]
        g = snap.copy()
        for (r, c), (sr, sc) in zip(cur, edge):
            g[r, c] = I[sr, sc]
    # the strip between the old edge position and the new one: interior -> inc, sides -> boxc
    strip_in, strip_side = [], []
    if boxc == 0:
        strip_side += [(r + vec[0] * d, c + vec[1] * d) for r, c in edge]
    for k in range(d):
        for i, (r, c) in enumerate(edge):
            rr, cc = r + vec[0] * k, c + vec[1] * k
            (strip_side if i in (0, len(edge) - 1) else strip_in).append((rr, cc))
    for col, cells in ((inc, strip_in), (boxc, strip_side)):
        cells = [(r, c) for r, c in cells if g[r, c] != col]
        if cells:
            ops.append(col); sels.append(sel_of(cells))
            for r, c in cells:
                g[r, c] = col
    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task b548a754"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b548a754"
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
                                f"for task b548a754"
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
                    f"Failed to build a complete episode for task b548a754 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b548a754-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
